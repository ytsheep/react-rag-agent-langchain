import math
import re
from collections import Counter
from dataclasses import dataclass

from langchain_core.documents import Document

from rag.vector_store import VectorStoreService
from utils.config_handler import chroma_conf

TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+")


def normalize_text(text: str) -> str:
    return "".join(text.lower().split())


def tokenize_text(text: str) -> list[str]:
    tokens: list[str] = []
    for part in TOKEN_RE.findall(text.lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", part):
            if len(part) == 1:
                tokens.append(part)
                continue

            for size in (2, 3):
                if len(part) >= size:
                    tokens.extend(part[idx: idx + size] for idx in range(len(part) - size + 1))

            if len(part) <= 6:
                tokens.append(part)
        else:
            tokens.append(part)
    return tokens


class SimpleBM25Index:
    def __init__(self, tokenized_documents: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.tokenized_documents = tokenized_documents
        self.k1 = k1
        self.b = b
        self.doc_lengths = [len(doc) for doc in tokenized_documents]
        self.avg_doc_length = sum(self.doc_lengths) / max(len(self.doc_lengths), 1)
        self.term_frequencies = [Counter(doc) for doc in tokenized_documents]
        self.document_frequencies: Counter[str] = Counter()

        for doc in tokenized_documents:
            for term in set(doc):
                self.document_frequencies[term] += 1

        total_docs = max(len(tokenized_documents), 1)
        self.idf = {
            term: math.log(1 + (total_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in self.document_frequencies.items()
        }

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores = [0.0] * len(self.tokenized_documents)

        for doc_index, frequencies in enumerate(self.term_frequencies):
            doc_length = self.doc_lengths[doc_index]
            score = 0.0

            for term in query_tokens:
                if term not in frequencies:
                    continue

                tf = frequencies[term]
                idf = self.idf.get(term, 0.0)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * doc_length / max(self.avg_doc_length, 1e-9)
                )
                score += idf * numerator / max(denominator, 1e-9)

            scores[doc_index] = score

        return scores


@dataclass
class RetrievedCandidate:
    doc_id: str
    document: Document
    vector_rank: int | None = None
    bm25_rank: int | None = None
    rerank_score: float = 0.0


class HybridRetriever:
    def __init__(self):
        self.vector_k = chroma_conf.get("hybrid_vector_k", 8)
        self.bm25_k = chroma_conf.get("hybrid_bm25_k", 8)
        self.final_k = chroma_conf.get("hybrid_final_k", 3)
        self.vector_weight = chroma_conf.get("hybrid_vector_weight", 0.45)
        self.bm25_weight = chroma_conf.get("hybrid_bm25_weight", 0.35)
        self.keyword_weight = chroma_conf.get("hybrid_keyword_weight", 0.20)
        self.rrf_k = chroma_conf.get("hybrid_rrf_k", 60)

        self.vector_store_service = VectorStoreService()
        if self.vector_store_service.vector_store._collection.count() == 0:
            self.vector_store_service.load_document()

        self.vector_store = self.vector_store_service.vector_store
        self.documents = self._load_all_documents()
        self.doc_key_to_id = {
            self._document_key(doc): doc.metadata["doc_id"]
            for doc in self.documents
        }
        self.doc_tokens = [tokenize_text(doc.page_content) for doc in self.documents]
        self.bm25_index = SimpleBM25Index(self.doc_tokens)

    def _load_all_documents(self) -> list[Document]:
        payload = self.vector_store.get(include=["documents", "metadatas"])
        documents = payload.get("documents", [])
        metadatas = payload.get("metadatas", [])
        ids = payload.get("ids", [])

        loaded_docs: list[Document] = []
        for idx, page_content in enumerate(documents):
            metadata = dict(metadatas[idx] or {})
            metadata["doc_id"] = ids[idx] if idx < len(ids) else f"doc-{idx}"
            loaded_docs.append(Document(page_content=page_content, metadata=metadata))
        return loaded_docs

    def vector_search(self, query: str) -> list[RetrievedCandidate]:
        results = self.vector_store.similarity_search_with_score(query, k=self.vector_k)
        candidates: list[RetrievedCandidate] = []

        for rank, (document, _score) in enumerate(results, start=1):
            doc_id = document.metadata.get("doc_id") or self.doc_key_to_id.get(self._document_key(document))
            if doc_id is None:
                doc_id = self._fallback_doc_id(document)
            candidates.append(
                RetrievedCandidate(
                    doc_id=doc_id,
                    document=document,
                    vector_rank=rank,
                )
            )

        return candidates

    def bm25_search(self, query: str) -> list[RetrievedCandidate]:
        query_tokens = tokenize_text(query)
        scores = self.bm25_index.get_scores(query_tokens)
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)

        candidates: list[RetrievedCandidate] = []
        for rank, (doc_index, _score) in enumerate(ranked[: self.bm25_k], start=1):
            document = self.documents[doc_index]
            candidates.append(
                RetrievedCandidate(
                    doc_id=document.metadata["doc_id"],
                    document=document,
                    bm25_rank=rank,
                )
            )

        return candidates

    def retrieve(self, query: str) -> list[Document]:
        vector_candidates = self.vector_search(query)
        bm25_candidates = self.bm25_search(query)
        merged = self._merge_candidates(vector_candidates, bm25_candidates)
        reranked = self._rerank(query, merged)
        return reranked[: self.final_k]

    def _merge_candidates(
        self,
        vector_candidates: list[RetrievedCandidate],
        bm25_candidates: list[RetrievedCandidate],
    ) -> list[RetrievedCandidate]:
        merged: dict[str, RetrievedCandidate] = {}

        for candidate in vector_candidates + bm25_candidates:
            existing = merged.get(candidate.doc_id)
            if existing is None:
                merged[candidate.doc_id] = candidate
                continue

            if candidate.vector_rank is not None:
                existing.vector_rank = candidate.vector_rank
            if candidate.bm25_rank is not None:
                existing.bm25_rank = candidate.bm25_rank

        return list(merged.values())

    def  _rerank(self, query: str, candidates: list[RetrievedCandidate]) -> list[Document]:
        query_tokens = set(tokenize_text(query))
        normalized_query = normalize_text(query)

        for candidate in candidates:
            vector_component = self._rrf_score(candidate.vector_rank)
            bm25_component = self._rrf_score(candidate.bm25_rank)
            keyword_component = self._keyword_coverage(query_tokens, candidate.document.page_content)
            exact_bonus = self._exact_match_bonus(normalized_query, candidate.document.page_content)

            candidate.rerank_score = (
                self.vector_weight * vector_component
                + self.bm25_weight * bm25_component
                + self.keyword_weight * keyword_component
                + 0.1 * exact_bonus
            )

        candidates.sort(key=lambda item: item.rerank_score, reverse=True)

        reranked_docs: list[Document] = []
        for rank, candidate in enumerate(candidates, start=1):
            metadata = dict(candidate.document.metadata)
            metadata.update(
                {
                    "vector_rank": candidate.vector_rank,
                    "bm25_rank": candidate.bm25_rank,
                    "rerank_score": round(candidate.rerank_score, 6),
                    "hybrid_rank": rank,
                }
            )
            reranked_docs.append(Document(page_content=candidate.document.page_content, metadata=metadata))

        return reranked_docs

    def _rrf_score(self, rank: int | None) -> float:
        if rank is None:
            return 0.0
        return 1.0 / (self.rrf_k + rank)

    def _keyword_coverage(self, query_tokens: set[str], doc_text: str) -> float:
        if not query_tokens:
            return 0.0
        doc_tokens = set(tokenize_text(doc_text))
        return len(query_tokens & doc_tokens) / len(query_tokens)

    def _exact_match_bonus(self, normalized_query: str, doc_text: str) -> float:
        normalized_doc = normalize_text(doc_text)
        if normalized_query and normalized_query in normalized_doc:
            return 1.0
        return 0.0

    def _fallback_doc_id(self, document: Document) -> str:
        normalized = normalize_text(document.page_content[:100])
        return f"{document.metadata.get('source', 'unknown')}::{normalized}"

    def _document_key(self, document: Document) -> str:
        source = document.metadata.get("source", "unknown")
        content = normalize_text(document.page_content[:200])
        return f"{source}::{content}"
