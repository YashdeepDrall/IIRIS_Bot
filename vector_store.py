import hashlib
import json
import math
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np
from dotenv import load_dotenv

from gemini_api import GeminiClient


load_dotenv()


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "what",
    "when",
    "where",
    "who",
}

TITLE_TOKENS = {
    "advisory",
    "advisor",
    "avp",
    "board",
    "ceo",
    "cfo",
    "coo",
    "cto",
    "leader",
    "leaders",
    "leadership",
    "president",
    "team",
    "vp",
}

LEADERSHIP_SOURCE_HINTS = (
    "board_of_advisory",
    "faq",
    "leaders",
    "teamfaq",
)


@dataclass
class ChunkRecord:
    chunk_id: str
    source: str
    chunk_index: int
    text: str
    text_hash: str
    file_hash: str


class GeminiVectorStore:
    def __init__(
        self,
        client: GeminiClient | None = None,
        data_dir: str | None = None,
        store_path: str | None = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
    ) -> None:
        self.client = client or GeminiClient()
        self.embedding_model = self.client.embedding_model
        self.data_dir = Path(data_dir or os.getenv("DATA_DIR", "data"))
        self.store_path = Path(store_path or os.getenv("VECTOR_STORE_PATH", "vectorstore"))
        self.chunk_size = int(os.getenv("CHUNK_SIZE", chunk_size))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", chunk_overlap))
        self.score_threshold = float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.25"))

        self.index_path = self.store_path / "index.faiss"
        self.records_path = self.store_path / "records.json"
        self.vectors_path = self.store_path / "vectors.npy"
        self.config_path = self.store_path / "config.json"

        self.index: faiss.Index | None = None
        self.records: List[Dict] = []
        self.vectors: np.ndarray | None = None
        self._normalized_record_texts: List[str] = []
        self._record_term_counts: List[Counter[str]] = []
        self._record_terms: List[set[str]] = []
        self._inverse_document_frequency: Dict[str, float] = {}

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(re.findall(r"[a-z0-9]+", text.lower()))

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [
            token
            for token in re.findall(r"[a-z0-9]+", text.lower())
            if token and token not in STOPWORDS
        ]

    def _build_lexical_index(self) -> None:
        self._normalized_record_texts = []
        self._record_term_counts = []
        self._record_terms = []

        document_frequencies: Counter[str] = Counter()

        for record in self.records:
            source_text = str(record.get("source", "")).replace("_", " ").replace(".txt", " ")
            search_text = f"{source_text} {record.get('text', '')}"
            normalized_text = self._normalize_text(search_text)
            term_counts = Counter(self._tokenize(search_text))

            self._normalized_record_texts.append(normalized_text)
            self._record_term_counts.append(term_counts)
            self._record_terms.append(set(term_counts))
            document_frequencies.update(set(term_counts))

        total_records = max(len(self.records), 1)
        self._inverse_document_frequency = {
            token: math.log((total_records + 1) / (frequency + 1)) + 1.0
            for token, frequency in document_frequencies.items()
        }

    def _lexical_score(
        self,
        query: str,
        query_terms: set[str],
        leadership_intent: bool,
        record_index: int,
    ) -> float:
        if not query_terms:
            return 0.0

        record_terms = self._record_terms[record_index]
        overlap = record_terms & query_terms
        if not overlap:
            return 0.0

        total_weight = sum(self._inverse_document_frequency.get(term, 1.0) for term in query_terms) or 1.0
        overlap_weight = sum(self._inverse_document_frequency.get(term, 1.0) for term in overlap)
        coverage = overlap_weight / total_weight

        term_frequency = sum(
            min(self._record_term_counts[record_index].get(term, 0), 2)
            for term in overlap
        )
        density = term_frequency / max(len(query_terms) * 2, 1)

        normalized_query = self._normalize_text(query)
        phrase_bonus = 0.35 if normalized_query and normalized_query in self._normalized_record_texts[record_index] else 0.0

        title_overlap = overlap & TITLE_TOKENS
        leadership_boost = 0.0
        if leadership_intent and title_overlap:
            leadership_boost += 0.15

        source = str(self.records[record_index].get("source", "")).lower()
        if leadership_intent and any(hint in source for hint in LEADERSHIP_SOURCE_HINTS):
            leadership_boost += 0.05

        return min(1.5, (0.75 * coverage) + (0.2 * density) + phrase_bonus + leadership_boost)

    def _chunk_text(self, text: str) -> List[str]:
        text = text.strip()
        if not text:
            return []

        chunks: List[str] = []
        start = 0
        length = len(text)

        while start < length:
            end = min(start + self.chunk_size, length)

            if end < length:
                paragraph_break = text.rfind("\n\n", start, end)
                if paragraph_break > start + (self.chunk_size // 2):
                    end = paragraph_break
                else:
                    line_break = text.rfind("\n", start, end)
                    if line_break > start + (self.chunk_size // 2):
                        end = line_break
                    else:
                        word_break = text.rfind(" ", start, end)
                        if word_break > start + (self.chunk_size // 2):
                            end = word_break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            if end >= length:
                break

            next_start = max(end - self.chunk_overlap, start + 1)
            start = next_start

        return chunks

    def _collect_records(self) -> List[ChunkRecord]:
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        records: List[ChunkRecord] = []
        for path in sorted(self.data_dir.glob("*.txt")):
            text = path.read_text(encoding="utf-8").strip()
            if not text:
                continue

            relative_source = path.relative_to(self.data_dir).as_posix()
            file_hash = self._hash_text(text)

            for chunk_index, chunk in enumerate(self._chunk_text(text)):
                text_hash = self._hash_text(chunk)
                chunk_id = self._hash_text(f"{relative_source}:{chunk_index}:{text_hash}")
                records.append(
                    ChunkRecord(
                        chunk_id=chunk_id,
                        source=relative_source,
                        chunk_index=chunk_index,
                        text=chunk,
                        text_hash=text_hash,
                        file_hash=file_hash,
                    )
                )

        if not records:
            raise RuntimeError("No usable .txt content found in the data directory.")

        return records

    def _load_existing_store(self) -> tuple[List[Dict], Dict[str, np.ndarray]]:
        if not (
            self.records_path.exists()
            and self.vectors_path.exists()
            and self.config_path.exists()
        ):
            return [], {}

        config = json.loads(self.config_path.read_text(encoding="utf-8"))
        compatible = (
            config.get("embedding_model") == self.embedding_model
            and int(config.get("chunk_size", 0)) == self.chunk_size
            and int(config.get("chunk_overlap", 0)) == self.chunk_overlap
            and int(config.get("dimension", 0)) > 0
        )
        if not compatible:
            return [], {}

        stored_records = json.loads(self.records_path.read_text(encoding="utf-8"))
        stored_vectors = np.load(self.vectors_path)
        if len(stored_records) != len(stored_vectors) or stored_vectors.ndim != 2 or stored_vectors.shape[1] == 0:
            return [], {}

        return stored_records, {
            record["chunk_id"]: stored_vectors[index]
            for index, record in enumerate(stored_records)
        }

    def _save_store(self, records: List[Dict], vectors: np.ndarray) -> None:
        self.store_path.mkdir(parents=True, exist_ok=True)

        np.save(self.vectors_path, vectors)
        self.records_path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.config_path.write_text(
            json.dumps(
                {
                    "embedding_model": self.embedding_model,
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                    "dimension": int(vectors.shape[1]),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        faiss.write_index(index, str(self.index_path))

        self.index = index
        self.records = records
        self.vectors = vectors
        self._build_lexical_index()

    def _ensure_loaded(self) -> None:
        if self.index is not None and self.records:
            return
        self.sync()

    def sync(self) -> Dict[str, int]:
        desired_records = self._collect_records()
        desired_record_dicts = [asdict(record) for record in desired_records]
        stored_records, existing_vectors = self._load_existing_store()

        if stored_records:
            desired_ids = [record["chunk_id"] for record in desired_record_dicts]
            stored_ids = [record["chunk_id"] for record in stored_records]
            if desired_ids == stored_ids and self.index_path.exists():
                self.records = stored_records
                self.vectors = np.load(self.vectors_path)
                self.index = faiss.read_index(str(self.index_path))
                self._build_lexical_index()
                return {
                    "total_chunks": len(stored_records),
                    "reused_chunks": len(stored_records),
                    "new_chunks": 0,
                }

        final_records: List[Dict] = []
        final_vectors: List[np.ndarray | None] = []
        texts_to_embed: List[str] = []
        embed_positions: List[int] = []
        reused = 0

        for record_dict, record in zip(desired_record_dicts, desired_records):
            final_records.append(record_dict)

            if record.chunk_id in existing_vectors:
                final_vectors.append(existing_vectors[record.chunk_id].astype(np.float32))
                reused += 1
            else:
                final_vectors.append(None)
                texts_to_embed.append(record.text)
                embed_positions.append(len(final_vectors) - 1)

        if texts_to_embed:
            embedded_vectors = self.client.embed_texts(
                texts_to_embed,
                task_type="RETRIEVAL_DOCUMENT",
            )
            for position, vector in zip(embed_positions, embedded_vectors):
                final_vectors[position] = np.asarray(vector, dtype=np.float32)

        matrix = np.vstack([vector for vector in final_vectors if vector is not None]).astype(
            np.float32
        )
        faiss.normalize_L2(matrix)
        self._save_store(final_records, matrix)

        return {
            "total_chunks": len(final_records),
            "reused_chunks": reused,
            "new_chunks": len(texts_to_embed),
        }

    def search(self, query: str, k: int = 15) -> List[Dict]:
        self._ensure_loaded()
        if not self.records or self.index is None:
            return []

        query_terms = set(self._tokenize(query))
        leadership_intent = bool(query_terms & TITLE_TOKENS)
        query_vector = np.asarray([self.client.embed_query(query)], dtype=np.float32)
        faiss.normalize_L2(query_vector)

        dense_limit = max(1, min(max(k * 8, 50), len(self.records)))
        scores, indices = self.index.search(query_vector, dense_limit)
        dense_scores = {
            int(index): float(score)
            for score, index in zip(scores[0], indices[0])
            if index >= 0
        }

        scored_results: List[tuple[float, float, float, int]] = []
        for index, record in enumerate(self.records):
            dense_score = dense_scores.get(index, 0.0)
            lexical_score = self._lexical_score(query, query_terms, leadership_intent, index)
            if dense_score <= 0 and lexical_score <= 0:
                continue

            hybrid_score = dense_score
            if lexical_score > 0:
                hybrid_score = max(dense_score, (0.35 * dense_score) + (0.85 * lexical_score))

            scored_results.append((hybrid_score, lexical_score, dense_score, index))

        scored_results.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)

        results: List[Dict] = []
        for hybrid_score, _, _, index in scored_results[: max(1, min(k, len(scored_results)))]:
            record = self.records[index]
            results.append(
                {
                    "page_content": record["text"],
                    "metadata": {
                        "source": record["source"],
                        "chunk_index": record["chunk_index"],
                    },
                    "score": float(hybrid_score),
                }
            )

        return results
