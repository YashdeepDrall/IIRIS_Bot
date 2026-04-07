import difflib
import re
import traceback
from pathlib import Path
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
- If the question asks about leadership, leaders, executives, team members, or board members, list every distinct person supported by the context.
- For broad leadership/team questions, prefer concise bullets with Name: Role and group by unit or function when possible.
- Never replace supported names with vague summaries like "other leaders", "various VPs", or "Senior Managers".
""".strip()


LEADERSHIP_QUERY_TOKENS = {
    "advisory",
    "board",
    "coo",
    "ceo",
    "cfo",
    "cto",
    "executive",
    "executives",
    "leader",
    "leaders",
    "leadership",
    "management",
    "president",
    "team",
    "vp",
}

LEADERSHIP_ROSTER_TOKENS = {
    "advisory",
    "board",
    "executives",
    "leader",
    "leaders",
    "leadership",
    "management",
    "team",
}

LEADERSHIP_SOURCE_HINTS = [
    "our_leaders",
    "teamfaq",
    "board_of_advisory",
    "faq",
]

LEADERSHIP_ROSTER_SOURCE_HINTS = [
    "our_leaders",
    "board_of_advisory",
]

TITLE_LOOKUP_SOURCE_HINTS = [
    "our_leaders",
    "teamfaq",
]

TITLE_CATEGORY_PATTERNS = {
    "assistant vice president": (
        r"\bassistant[ -]?vice[ -]?presidents?\b",
        r"\bavps?\b",
    ),
    "vice president": (
        r"\bvice[ -]?presidents?\b",
        r"\bvps?\b",
        r"\bvice president\b",
        r"\bvp\b",
    ),
    "president": (
        r"\bpresidents?\b",
    ),
    "ceo": (
        r"\bceos?\b",
        r"\bchief executive officers?\b",
    ),
    "coo": (
        r"\bcoos?\b",
        r"\bchief operating officers?\b",
    ),
    "cfo": (
        r"\bcfos?\b",
        r"\bchief financial officers?\b",
    ),
    "cto": (
        r"\bctos?\b",
        r"\bchief technology officers?\b",
    ),
}

TITLE_DISPLAY_LABELS = {
    "assistant vice president": "Assistant Vice Presidents",
    "vice president": "Vice Presidents",
    "president": "Presidents",
    "ceo": "CEOs",
    "coo": "COOs",
    "cfo": "CFOs",
    "cto": "CTOs",
}


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

    @staticmethod
    def _question_tokens(question: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", question.lower()))

    def _is_leadership_query(self, question: str) -> bool:
        return bool(self._question_tokens(question) & LEADERSHIP_QUERY_TOKENS)

    def _is_broad_leadership_query(self, question: str) -> bool:
        tokens = self._question_tokens(question)
        if not (tokens & LEADERSHIP_ROSTER_TOKENS):
            return False

        person_indicators = {
            "garry",
            "gurpawan",
            "kunal",
            "sagarika",
            "paul",
            "ashok",
            "prashant",
            "satyagopal",
        }
        return not bool(tokens & person_indicators)

    @staticmethod
    def _merge_docs(primary_docs: List[Dict], secondary_docs: List[Dict]) -> List[Dict]:
        merged: List[Dict] = []
        seen: set[tuple[str, int]] = set()

        for doc in primary_docs + secondary_docs:
            key = (
                str(doc.get("metadata", {}).get("source", "")),
                int(doc.get("metadata", {}).get("chunk_index", -1)),
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(doc)

        return merged

    def _expand_docs_for_leadership_query(
        self,
        question: str,
        vector_store: GeminiVectorStore,
        docs: List[Dict],
    ) -> List[Dict]:
        if not self._is_leadership_query(question):
            return docs

        leadership_docs = vector_store.get_chunks_by_source_hints(LEADERSHIP_SOURCE_HINTS)
        if self._is_broad_leadership_query(question):
            roster_docs = vector_store.get_chunks_by_source_hints(LEADERSHIP_ROSTER_SOURCE_HINTS)
            return roster_docs or docs
        return self._merge_docs(docs, leadership_docs[:6])

    @staticmethod
    def _normalize_title_query_text(question: str) -> str:
        return re.sub(r"[-_/]", " ", question.lower())

    def _extract_requested_title_categories(self, question: str) -> List[str]:
        normalized_question = self._normalize_title_query_text(question)
        requested_categories: List[str] = []

        for category in (
            "assistant vice president",
            "vice president",
            "ceo",
            "coo",
            "cfo",
            "cto",
        ):
            if any(re.search(pattern, normalized_question) for pattern in TITLE_CATEGORY_PATTERNS[category]):
                requested_categories.append(category)
                normalized_question = re.sub(
                    "|".join(TITLE_CATEGORY_PATTERNS[category]),
                    " ",
                    normalized_question,
                )

        if any(re.search(pattern, normalized_question) for pattern in TITLE_CATEGORY_PATTERNS["president"]):
            requested_categories.append("president")

        return requested_categories

    def _is_title_lookup_query(self, question: str) -> bool:
        requested_categories = self._extract_requested_title_categories(question)
        if not requested_categories:
            return False

        normalized_question = self._normalize_title_query_text(question)
        if re.match(r"^(is|are|was|were|does|do|did)\b", normalized_question):
            return False
        lookup_patterns = (
            r"\blist\b",
            r"\bwho is\b",
            r"\bwho are\b",
            r"\bshow\b",
            r"\bname\b",
            r"\ball\b",
            r"\bwhich\b",
            r"\bof iiris\b",
        )
        return any(re.search(pattern, normalized_question) for pattern in lookup_patterns)

    @staticmethod
    def _load_title_lookup_texts(data_dir: Path) -> List[Dict[str, str]]:
        documents: List[Dict[str, str]] = []
        for path in sorted(data_dir.glob("*.txt")):
            source_name = path.name.lower()
            if not any(hint in source_name for hint in TITLE_LOOKUP_SOURCE_HINTS):
                continue
            documents.append(
                {
                    "source": path.name,
                    "text": path.read_text(encoding="utf-8"),
                }
            )
        return documents

    @staticmethod
    def _normalize_person_name(name: str) -> str:
        cleaned_name = re.sub(
            r"^(mainly known as|also known as)\s+",
            "",
            re.sub(r"\s+", " ", name.strip()),
            flags=re.IGNORECASE,
        )
        alias_match = re.fullmatch(r"(.+?)\s+(?:or|/)\s+(.+)", cleaned_name, flags=re.IGNORECASE)
        if alias_match:
            primary_name = alias_match.group(1).strip()
            alternate_name = alias_match.group(2).strip()
            return f"{primary_name} ({alternate_name})"
        return cleaned_name

    @staticmethod
    def _canonical_name_key(name: str) -> str:
        canonical_name = re.sub(r"\s*\(.*?\)\s*", " ", name)
        canonical_name = re.sub(
            r"^(col\.?|maj\.?|lt\.?\s*gen\.?|maj\.?\s*general|dr\.?|ambassador)\s+",
            "",
            canonical_name,
            flags=re.IGNORECASE,
        )
        return re.sub(r"\s+", " ", canonical_name).strip().lower()

    @classmethod
    def _same_person_name(cls, left_name: str, right_name: str) -> bool:
        left_key = cls._canonical_name_key(left_name)
        right_key = cls._canonical_name_key(right_name)
        if left_key == right_key:
            return True

        left_tokens = left_key.split()
        right_tokens = right_key.split()
        if not left_tokens or not right_tokens:
            return False
        if left_tokens[0] != right_tokens[0]:
            return False

        left_tail = " ".join(left_tokens[1:]) or left_tokens[0]
        right_tail = " ".join(right_tokens[1:]) or right_tokens[0]
        tail_similarity = difflib.SequenceMatcher(None, left_tail, right_tail).ratio()

        if tail_similarity >= 0.9:
            return True

        if len(left_tokens) == len(right_tokens) and len(left_tokens) >= 2:
            return difflib.SequenceMatcher(None, left_tokens[-1], right_tokens[-1]).ratio() >= 0.88

        return False

    @staticmethod
    def _looks_like_person_name(name: str) -> bool:
        cleaned_name = name.strip()
        if not cleaned_name or any(character.isdigit() for character in cleaned_name):
            return False

        lowered_name = cleaned_name.lower()
        if lowered_name.startswith(("who ", "what ", "how ", "where ", "why ", "check out")):
            return False

        word_count = len(cleaned_name.split())
        return 1 <= word_count <= 8 and bool(re.search(r"[a-zA-Z]", cleaned_name))

    @staticmethod
    def _role_categories(role: str) -> set[str]:
        normalized_role = re.sub(r"[-_/]", " ", role.lower())
        categories: set[str] = set()

        avp_pattern = r"\bassistant vice president\b|\bavp\b"
        vp_pattern = r"\bvice president\b|\bvp\b"

        if re.search(avp_pattern, normalized_role):
            categories.add("assistant vice president")
            normalized_role = re.sub(avp_pattern, " ", normalized_role)

        if re.search(vp_pattern, normalized_role):
            categories.add("vice president")
            normalized_role = re.sub(vp_pattern, " ", normalized_role)

        if re.search(r"\bpresident\b", normalized_role):
            categories.add("president")
        if re.search(r"\bceo\b|\bchief executive officer\b", normalized_role):
            categories.add("ceo")
        if re.search(r"\bcoo\b|\bchief operating officer\b", normalized_role):
            categories.add("coo")
        if re.search(r"\bcfo\b|\bchief financial officer\b", normalized_role):
            categories.add("cfo")
        if re.search(r"\bcto\b|\bchief technology officer\b", normalized_role):
            categories.add("cto")

        return categories

    @staticmethod
    def _line_to_title_entry(line: str, source: str, line_index: int) -> Dict | None:
        stripped_line = line.strip().lstrip("*-").strip()
        if not stripped_line:
            return None

        lowered = stripped_line.lower()
        if lowered.startswith(("q", "check out", "board of advisory", "what ", "how ", "where ", "why ")):
            return None

        stripped_line = re.sub(r"^\d+\.\s*", "", stripped_line).strip()
        patterns = (
            r"^(?P<name>[^:]{2,120}?)\s*:\s*(?P<role>.+)$",
            r"^(?P<name>[^,]{2,120}?),\s*(?P<role>(?:President|VP|AVP|CEO|COO|CFO|CTO)\b.+)$",
            r"^(?P<name>[^–—-]{2,120}?)\s*[–—-]\s*(?P<role>.+)$",
        )

        for pattern in patterns:
            match = re.match(pattern, stripped_line, flags=re.IGNORECASE)
            if not match:
                continue

            name = RagSystem._normalize_person_name(match.group("name"))
            if not RagSystem._looks_like_person_name(name):
                continue
            role = re.sub(r"\s+", " ", match.group("role").strip())
            categories = RagSystem._role_categories(role)
            if not categories:
                continue

            return {
                "name": name,
                "role": role.split("|", 1)[0].strip(),
                "source": source,
                "line_index": line_index,
                "categories": categories,
            }

        return None

    def _extract_title_entries(self, vector_store: GeminiVectorStore) -> List[Dict]:
        documents = self._load_title_lookup_texts(vector_store.data_dir)
        def preferred_source_rank(source: str) -> int:
            normalized_source = source.lower()
            if "our_leaders" in normalized_source:
                return 0
            if "teamfaq" in normalized_source:
                return 1
            return 99

        deduped_entries: Dict[tuple[str, str], Dict] = {}

        for document in documents:
            for line_index, line in enumerate(document["text"].splitlines()):
                entry = self._line_to_title_entry(line, document["source"], line_index)
                if entry is None:
                    continue

                key = (entry["name"].lower(), entry["role"].lower())
                existing_entry = deduped_entries.get(key)
                if existing_entry is None:
                    deduped_entries[key] = entry
                    continue

                existing_rank = preferred_source_rank(existing_entry["source"])
                current_rank = preferred_source_rank(entry["source"])
                if current_rank < existing_rank or len(entry["role"]) > len(existing_entry["role"]):
                    deduped_entries[key] = entry

        return list(deduped_entries.values())

    @staticmethod
    def _source_rank(source: str) -> int:
        normalized_source = source.lower()
        if "our_leaders" in normalized_source:
            return 0
        if "teamfaq" in normalized_source:
            return 1
        return 99

    @staticmethod
    def _name_quality_penalty(name: str) -> int:
        penalty = 0
        if re.search(r"([a-zA-Z])\1\1", name):
            penalty += 2
        return penalty

    @staticmethod
    def _role_quality_penalty(role: str) -> int:
        penalty = 0
        comma_match = re.match(r"^[A-Z]{2,},\s+([a-z].*)$", role)
        if comma_match:
            penalty += 1
        return penalty

    def _entry_preference_key(self, entry: Dict) -> tuple[int, int, int, int, int]:
        return (
            self._name_quality_penalty(entry["name"]),
            self._role_quality_penalty(entry["role"]),
            self._source_rank(entry["source"]),
            -len(entry["role"]),
            entry["line_index"],
        )

    def _build_title_lookup_response(
        self,
        question: str,
        vector_store: GeminiVectorStore,
    ) -> Dict | None:
        if not self._is_title_lookup_query(question):
            return None

        requested_categories = self._extract_requested_title_categories(question)
        title_entries = self._extract_title_entries(vector_store)
        matched_entries = [
            entry
            for entry in title_entries
            if any(category in entry["categories"] for category in requested_categories)
        ]

        if not matched_entries:
            return None

        best_entries_by_category: Dict[str, List[Dict]] = {}
        for entry in matched_entries:
            for category in requested_categories:
                if category not in entry["categories"]:
                    continue

                category_entries = best_entries_by_category.setdefault(category, [])
                replaced_existing = False
                for index, existing_entry in enumerate(category_entries):
                    if not self._same_person_name(entry["name"], existing_entry["name"]):
                        continue

                    if self._entry_preference_key(entry) < self._entry_preference_key(existing_entry):
                        category_entries[index] = entry
                    replaced_existing = True
                    break

                if not replaced_existing:
                    category_entries.append(entry)

        matched_entries = [
            entry
            for category in requested_categories
            for entry in best_entries_by_category.get(category, [])
        ]

        def category_sort_key(category: str) -> int:
            return requested_categories.index(category) if category in requested_categories else len(requested_categories)

        def source_sort_key(source: str) -> tuple[int, str]:
            normalized_source = source.lower()
            return (self._source_rank(source), normalized_source)

        matched_entries.sort(
            key=lambda entry: (
                min(category_sort_key(category) for category in entry["categories"] if category in requested_categories),
                source_sort_key(entry["source"]),
                entry["line_index"],
                entry["name"].lower(),
            )
        )

        answer_lines = [
            "Based on the available IIRIS knowledge base, these are the requested title-holders at IIRIS:",
            "",
        ]
        context_lines: List[str] = []
        docs: List[Dict] = []

        for category in requested_categories:
            category_entries = [
                entry
                for entry in matched_entries
                if category in entry["categories"]
            ]
            if not category_entries:
                continue

            answer_lines.append(f"{TITLE_DISPLAY_LABELS.get(category, category.title())}")
            for entry in category_entries:
                answer_lines.append(f"- {entry['name']}: {entry['role']}")
                context_lines.append(f"Source: {entry['source']}\n{entry['name']}: {entry['role']}")
                docs.append(
                    {
                        "page_content": f"{entry['name']}: {entry['role']}",
                        "metadata": {
                            "source": entry["source"],
                            "chunk_index": entry["line_index"],
                        },
                        "score": 1.0,
                    }
                )
            answer_lines.append("")

        answer = "\n".join(answer_lines).strip()
        context = "\n\n".join(context_lines)
        return {
            "answer": answer,
            "context": context,
            "docs": docs,
        }

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
            title_lookup_response = self._build_title_lookup_response(search_query, vector_store)
            if title_lookup_response is not None:
                response = format_clickable_links(title_lookup_response["answer"])
                chat_id = store_chat(
                    request.question,
                    response,
                    title_lookup_response["context"],
                    flags=[],
                    token_usage=total_usage,
                )
                return {
                    "answer": response,
                    "context": title_lookup_response["context"],
                    "docs": title_lookup_response["docs"],
                    "chat_id": chat_id,
                    "usage": total_usage,
                }
            docs = vector_store.search(search_query, k=request.k)
            docs = self._expand_docs_for_leadership_query(search_query, vector_store, docs)

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
            question_guidance = ""
            if self._is_broad_leadership_query(search_query):
                question_guidance = (
                    "Special Instruction:\n"
                    "- This is a leadership roster query.\n"
                    "- List all distinct people supported by the context.\n"
                    "- Use concise bullets with name and role only unless the user asks for biographies.\n"
                    "- Group the answer by business unit, region, or advisory board when possible.\n"
                    "- Do not omit names in favor of summaries.\n\n"
                )
            prompt = (
                f"Chat History:\n{history_text or 'None'}\n\n"
                f"{question_guidance}"
                f"Knowledge Base Context:\n{context}\n\n"
                f"Client Question:\n{search_query}\n\n"
                "Consultant Answer:"
            )
            response, usage = self._get_client().generate_text(
                prompt=prompt,
                system_instruction=ANSWER_SYSTEM_PROMPT,
                temperature=request.temperature,
                max_tokens=max(request.max_tokens, 1024) if self._is_broad_leadership_query(search_query) else request.max_tokens,
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
