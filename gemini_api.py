import os
import time
from typing import Dict, Iterable, List, Tuple

import requests
from dotenv import load_dotenv


load_dotenv()


class GeminiAPIError(RuntimeError):
    """Raised when the Gemini API returns an error or malformed payload."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


class GeminiClient:
    def __init__(
        self,
        api_key: str | None = None,
        generation_model: str | None = None,
        embedding_model: str | None = None,
        timeout: int = 60,
        max_retries: int | None = None,
        retry_backoff_seconds: float | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise GeminiAPIError("GEMINI_API_KEY not found.")

        self.generation_model = generation_model or os.getenv(
            "LLM_MODEL_NAME", "gemini-3.1-flash-lite-preview"
        )
        self.embedding_model = embedding_model or os.getenv(
            "EMBEDDING_MODEL_NAME", "gemini-embedding-2-preview"
        )
        self.timeout = timeout
        self.max_retries = max(0, int(max_retries or os.getenv("GEMINI_MAX_RETRIES", "2")))
        self.retry_backoff_seconds = max(
            0.1,
            float(retry_backoff_seconds or os.getenv("GEMINI_RETRY_BACKOFF_SECONDS", "1.5")),
        )
        self.base_url = os.getenv(
            "GEMINI_API_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta/models",
        ).rstrip("/")
        self.session = requests.Session()

    @staticmethod
    def _is_retryable_error(status_code: int | None, message: str) -> bool:
        retryable_status_codes = {408, 429, 500, 502, 503, 504}
        if status_code in retryable_status_codes:
            return True

        lowered = message.lower()
        retryable_markers = [
            "high demand",
            "temporarily unavailable",
            "unavailable",
            "timed out",
            "timeout",
            "connection aborted",
            "connection reset",
        ]
        return any(marker in lowered for marker in retryable_markers)

    def _post(self, endpoint: str, payload: Dict) -> Dict:
        last_error: GeminiAPIError | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.post(
                    f"{self.base_url}/{endpoint}",
                    headers={
                        "x-goog-api-key": self.api_key,
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.timeout,
                )
            except requests.RequestException as error:
                message = f"Gemini API request failed: {error}"
                last_error = GeminiAPIError(
                    message,
                    retryable=True,
                )
            else:
                if not response.ok:
                    message = response.text.strip() or response.reason
                    last_error = GeminiAPIError(
                        f"Gemini API request failed ({response.status_code}): {message}",
                        status_code=response.status_code,
                        retryable=self._is_retryable_error(response.status_code, message),
                    )
                else:
                    try:
                        data = response.json()
                    except ValueError as error:
                        last_error = GeminiAPIError(
                            f"Gemini API returned invalid JSON: {error}",
                            status_code=response.status_code,
                        )
                    else:
                        if "error" in data:
                            error_payload = data["error"]
                            error_message = str(error_payload)
                            error_status = (
                                error_payload.get("code")
                                if isinstance(error_payload, dict)
                                else response.status_code
                            )
                            last_error = GeminiAPIError(
                                error_message,
                                status_code=error_status,
                                retryable=self._is_retryable_error(error_status, error_message),
                            )
                        else:
                            return data

            if last_error is None:
                break

            should_retry = attempt < self.max_retries and last_error.retryable
            if not should_retry:
                raise last_error

            delay_seconds = self.retry_backoff_seconds * (2**attempt)
            print(
                f"Gemini API request issue on attempt {attempt + 1}/{self.max_retries + 1}: "
                f"{last_error}. Retrying in {delay_seconds:.1f}s..."
            )
            time.sleep(delay_seconds)

        raise last_error or GeminiAPIError("Gemini API request failed unexpectedly.")

    @staticmethod
    def _extract_usage(data: Dict) -> Dict[str, int]:
        usage = data.get("usageMetadata", {})
        return {
            "prompt_tokens": int(usage.get("promptTokenCount", 0) or 0),
            "completion_tokens": int(usage.get("candidatesTokenCount", 0) or 0),
            "total_tokens": int(usage.get("totalTokenCount", 0) or 0),
        }

    def generate_text(
        self,
        prompt: str,
        system_instruction: str,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> Tuple[str, Dict[str, int]]:
        payload = {
            "system_instruction": {
                "parts": [{"text": system_instruction}],
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "responseMimeType": "text/plain",
            },
        }

        data = self._post(f"{self.generation_model}:generateContent", payload)
        usage = self._extract_usage(data)

        candidates = data.get("candidates", [])
        if not candidates:
            raise GeminiAPIError("Gemini API returned no candidates.")

        parts = candidates[0].get("content", {}).get("parts", [])
        text = "\n".join(part.get("text", "") for part in parts if part.get("text")).strip()
        if not text:
            raise GeminiAPIError("Gemini API returned an empty response.")

        return text, usage

    def embed_texts(
        self,
        texts: Iterable[str],
        task_type: str = "RETRIEVAL_DOCUMENT",
        batch_size: int = 20,
    ) -> List[List[float]]:
        cleaned_texts = [text.strip() or " " for text in texts]
        if not cleaned_texts:
            return []

        embeddings: List[List[float]] = []

        for start in range(0, len(cleaned_texts), batch_size):
            batch = cleaned_texts[start : start + batch_size]
            payload = {
                "requests": [
                    {
                        "model": f"models/{self.embedding_model}",
                        "content": {
                            "parts": [{"text": text}],
                        },
                        "taskType": task_type,
                    }
                    for text in batch
                ]
            }

            data = self._post(f"{self.embedding_model}:batchEmbedContents", payload)
            batch_embeddings = data.get("embeddings", [])
            if len(batch_embeddings) != len(batch):
                raise GeminiAPIError("Gemini API returned an unexpected embedding count.")

            embeddings.extend(item.get("values", []) for item in batch_embeddings)

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        embeddings = self.embed_texts([text], task_type="RETRIEVAL_QUERY", batch_size=1)
        if not embeddings:
            raise GeminiAPIError("Gemini API returned no query embedding.")
        return embeddings[0]
