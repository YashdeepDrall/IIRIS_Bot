import difflib
import re
import traceback
from typing import Dict, List

from dotenv import load_dotenv

from analysis import analyze_response
from database import QueryRequest, store_chat
from gemini_api import GeminiClient
from vector_store import GeminiVectorStore


load_dotenv()


CONDENSE_SYSTEM_PROMPT = """
You rewrite follow-up user messages into clear standalone questions for knowledge-base search.

Rules:
- Use the chat history to resolve references like "it", "that", "them", or "yes".
- If the user replies with "yes", "sure", "okay", or similar, convert it into the specific question implied by the assistant's last message.
- Preserve the user's intent.
- Output only the rewritten standalone question.
""".strip()


ANSWER_SYSTEM_PROMPT = """
You are a professional consultant at IIRIS Consulting.

Rules:
- Answer strictly from the provided Knowledge Base Context.
- Use only the information present in the context. Do not use outside knowledge.
- If the context does not support the answer, say: "I do not have enough information in the available IIRIS knowledge base to answer that."
- If the user asks something outside the available IIRIS knowledge base, say: "I can only answer questions based on the available IIRIS knowledge base."
- Do not guess or hallucinate.
- Do not mention internal chunk numbers, similarity scores, or implementation details.
- Keep the response professional, practical, and well-formatted.
- Use only standard Markdown links in the form [label](url).
- When including website links, format them as clickable Markdown links.
- When including email addresses, format them as clickable Markdown mailto links like [contactus@iirisconsulting.com](mailto:contactus@iirisconsulting.com).
- When including phone numbers, format them as clickable Markdown tel links like [+91 9205595358](tel:+919205595358).
- Never use custom link formats like [url|label].
- Never return raw HTML anchor tags.
""".strip()


def is_greeting(question: str) -> bool:
    greeting_words = [
        "hello",
        "hi",
        "hey",
        "greetings",
        "good morning",
        "good afternoon",
        "good evening",
    ]
    cleaned_input = question.lower().strip("!.,? '\"")
    if cleaned_input in greeting_words or any(cleaned_input.startswith(g + " ") for g in greeting_words):
        return True
    return bool(difflib.get_close_matches(cleaned_input, greeting_words, n=1, cutoff=0.8))


def is_signoff(question: str) -> bool:
    signoff_words = ["bye", "goodbye", "see you", "thank you", "thanks", "thankyou", "exit", "quit"]
    cleaned_input = question.lower().strip("!.,? '\"")
    if cleaned_input in signoff_words or any(cleaned_input.startswith(s + " ") for s in signoff_words):
        return True
    return bool(difflib.get_close_matches(cleaned_input, signoff_words, n=1, cutoff=0.8))


def correct_typos(question: str) -> str:
    known_terms = [
        "IIRIS",
        "IntelliRisk",
        "IntelliFuture",
        "IIRIS Knowledge",
        "Consulting",
        "Cybersecurity",
        "Forensics",
    ]
    words = question.split()
    corrected_words = []

    for word in words:
        clean_word = word.strip("!.,? '\"").lower()
        if not clean_word:
            corrected_words.append(word)
            continue

        matches = difflib.get_close_matches(
            clean_word,
            [term.lower() for term in known_terms],
            n=1,
            cutoff=0.8,
        )
        if not matches:
            corrected_words.append(word)
            continue

        match = matches[0]
        corrected_term = next(term for term in known_terms if term.lower() == match)
        lower_word = word.lower()
        start_index = lower_word.find(clean_word)

        if start_index == -1:
            corrected_words.append(corrected_term)
            continue

        prefix = word[:start_index]
        suffix = word[start_index + len(clean_word) :]
        corrected_words.append(prefix + corrected_term + suffix)

    return " ".join(corrected_words)


def _linkify_url(match: re.Match[str]) -> str:
    url = match.group(0)
    return f"[{url}]({url})"


def _linkify_email(match: re.Match[str]) -> str:
    email = match.group(0)
    return f"[{email}](mailto:{email})"


def _linkify_phone(match: re.Match[str]) -> str:
    raw_phone = match.group(0).strip()
    digits_only = re.sub(r"[^\d+]", "", raw_phone)
    digit_count = sum(character.isdigit() for character in digits_only)

    if digit_count < 8:
        return raw_phone

    tel_value = re.sub(r"[^\d+]", "", raw_phone)
    return f"[{raw_phone}](tel:{tel_value})"


def _normalize_custom_link(match: re.Match[str]) -> str:
    target = match.group(1).strip()
    label = match.group(2).strip()
    return f"[{label}]({target})"


def _normalize_html_link(match: re.Match[str]) -> str:
    href = match.group(1).strip()
    label = re.sub(r"<[^>]+>", "", match.group(2)).strip() or href
    return f"[{label}]({href})"


def format_clickable_links(text: str) -> str:
    text = re.sub(
        r"\[(https?://[^\|\]]+|mailto:[^\|\]]+|tel:[^\|\]]+)\|([^\]]+)\]",
        _normalize_custom_link,
        text,
    )
    text = re.sub(
        r"<a\s+[^>]*href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>",
        _normalize_html_link,
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    parts = re.split(r"(\[[^\]]+\]\([^)]+\))", text)
    formatted_parts = []

    for part in parts:
        if re.fullmatch(r"\[[^\]]+\]\([^)]+\)", part):
            formatted_parts.append(part)
            continue

        part = re.sub(r"https?://[^\s)]+", _linkify_url, part)
        part = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", _linkify_email, part)
        part = re.sub(r"(?<!\w)\+?\d[\d ()-]{7,}\d", _linkify_phone, part)
        formatted_parts.append(part)

    return "".join(formatted_parts)


class RagSystem:
    def __init__(self):
        print("Initializing Gemini RAG system...")
        self.client = None
        self.vector_store = None

    def _get_client(self) -> GeminiClient:
        if self.client is None:
            print("Initializing GeminiClient...")
            self.client = GeminiClient()
        return self.client

    def _get_vector_store(self) -> GeminiVectorStore:
        if self.vector_store is None:
            print("Initializing GeminiVectorStore...")
            self.vector_store = GeminiVectorStore(client=self._get_client())
        return self.vector_store

    @staticmethod
    def _update_usage(total_usage: Dict[str, int], usage: Dict[str, int]) -> None:
        total_usage["completion_tokens"] += usage.get("completion_tokens", 0)
        total_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
        total_usage["total_tokens"] += usage.get("total_tokens", 0)

    @staticmethod
    def _format_history(history: List[Dict[str, str]]) -> str:
        formatted = []
        for message in history:
            role = "User" if message.get("role") == "user" else "Consultant"
            formatted.append(f"{role}: {message.get('content', '')}")
        return "\n".join(formatted)

    def _rewrite_question(
        self,
        history_text: str,
        question: str,
        total_usage: Dict[str, int],
    ) -> str:
        if not history_text:
            return question

        prompt = (
            f"Chat History:\n{history_text}\n\n"
            f"Follow Up Input: {question}\n\n"
            "Standalone Question:"
        )
        rewritten_question, usage = self._get_client().generate_text(
            prompt=prompt,
            system_instruction=CONDENSE_SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=128,
        )
        self._update_usage(total_usage, usage)
        return rewritten_question.strip() or question

    @staticmethod
    def _should_rewrite_question(history: List[Dict[str, str]], question: str) -> bool:
        if not history:
            return False

        cleaned_question = question.strip().lower()
        if not cleaned_question:
            return False

        follow_up_patterns = (
            r"^(and|also|what about|how about|tell me more|more about|yes|yeah|yep|sure|okay|ok|then)\b",
            r"^(he|she|it|they|them|that|this|those|these|him|her)\b",
            r"^(is|are|was|were)\s+(he|she|it|they|that|this|those|these)\b",
        )
        if any(re.match(pattern, cleaned_question) for pattern in follow_up_patterns):
            return True

        tokens = re.findall(r"[a-z0-9]+", cleaned_question)
        contextual_tokens = {"he", "she", "it", "they", "them", "that", "this", "those", "these"}
        return bool(tokens) and len(tokens) <= 6 and any(token in contextual_tokens for token in tokens)

    async def answer(self, request: QueryRequest):
        try:
            total_usage = {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}
            corrected_question = correct_typos(request.question)

            if is_greeting(corrected_question) and len(corrected_question.split()) <= 3:
                response = "Hello! How may I help you today?"
                chat_id = store_chat(
                    request.question,
                    response,
                    "",
                    flags=["greeting"],
                    token_usage=total_usage,
                )
                return {
                    "answer": response,
                    "context": "",
                    "docs": [],
                    "chat_id": chat_id,
                    "usage": total_usage,
                }

            if is_signoff(corrected_question):
                response = "Happy to help you! I am here for any assistance. Is there anything else I can help you with?"
                chat_id = store_chat(
                    request.question,
                    response,
                    "",
                    flags=["signoff"],
                    token_usage=total_usage,
                )
                return {
                    "answer": response,
                    "context": "",
                    "docs": [],
                    "chat_id": chat_id,
                    "usage": total_usage,
                }

            history_text = self._format_history(request.history)
            search_query = corrected_question
            if self._should_rewrite_question(request.history, corrected_question):
                search_query = self._rewrite_question(history_text, corrected_question, total_usage)
            vector_store = self._get_vector_store()
            docs = vector_store.search(search_query, k=request.k)

            if not docs or docs[0]["score"] < vector_store.score_threshold:
                response = (
                    "I do not have enough information in the available IIRIS knowledge base "
                    "to answer that. Please ask about IIRIS, its services, leadership, "
                    "locations, or related topics covered in the data."
                )
                flags = ["irrelevant_question"]
                chat_id = store_chat(
                    request.question,
                    response,
                    "",
                    flags=flags,
                    token_usage=total_usage,
                )
                return {
                    "answer": response,
                    "context": "",
                    "docs": [],
                    "chat_id": chat_id,
                    "usage": total_usage,
                }

            context = "\n\n".join(
                f"Source: {doc['metadata']['source']}\n{doc['page_content']}" for doc in docs
            )
            prompt = (
                f"Chat History:\n{history_text or 'None'}\n\n"
                f"Knowledge Base Context:\n{context}\n\n"
                f"Client Question:\n{search_query}\n\n"
                "Consultant Answer:"
            )
            response, usage = self._get_client().generate_text(
                prompt=prompt,
                system_instruction=ANSWER_SYSTEM_PROMPT,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )
            self._update_usage(total_usage, usage)

            flags = analyze_response(corrected_question, response)
            if "irrelevant_question" not in flags and "greeting" not in flags and "signoff" not in flags:
                if list(set(flags)):
                    response += "\n\nFor further assistance, please contact support at: contactus@iirisconsulting.com"

            response = format_clickable_links(response)

            serialized_docs = [
                {
                    "page_content": doc["page_content"],
                    "metadata": doc["metadata"],
                    "score": doc["score"],
                }
                for doc in docs
            ]
            chat_id = store_chat(
                request.question,
                response,
                context,
                flags=flags,
                token_usage=total_usage,
            )
            return {
                "answer": response,
                "context": context,
                "docs": serialized_docs,
                "chat_id": chat_id,
                "usage": total_usage,
            }

        except Exception as error:
            print(f"Error in Gemini RAG handler: {error}\n{traceback.format_exc()}")
            raise


_rag_system = None


def get_rag_system():
    global _rag_system
    if _rag_system is None:
        _rag_system = RagSystem()
    return _rag_system
