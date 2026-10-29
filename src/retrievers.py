"""Common transcript retrieval interface and RAG strategies."""

from dataclasses import dataclass
import hashlib
import math
import re
from functools import lru_cache
from typing import Any, Protocol, Sequence

from src.chunking import TranscriptChunk
from src.config import get_settings


@dataclass(frozen=True)
class RetrievedChunk:
    """Retrieved transcript text, relevance score, and source timestamps."""

    text: str
    score: float
    start: float
    end: float
    chunk_id: int


class Retriever(Protocol):
    """Interface implemented by each retrieval approach."""

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        """Return the top-k chunks for a query."""


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class BaselineRetriever:
    """No-retrieval baseline returning the full transcript as one context."""

    def __init__(self, chunks: Sequence[TranscriptChunk], max_chars: int = 24000):
        self.chunks = list(chunks)
        self.max_chars = max_chars

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        del query, k
        words: list[str] = []
        for chunk in self.chunks:
            chunk_words = chunk.text.split()
            overlap = min(len(words), len(chunk_words))
            while overlap and words[-overlap:] != chunk_words[:overlap]:
                overlap -= 1
            words.extend(chunk_words[overlap:])
        text = " ".join(words)[: self.max_chars]
        if not text:
            return []
        return [
            RetrievedChunk(
                text=text,
                score=1.0,
                start=self.chunks[0].start,
                end=self.chunks[-1].end,
                chunk_id=-1,
            )
        ]


class BM25Retriever:
    """Sparse BM25 retrieval using rank_bm25."""

    def __init__(self, chunks: Sequence[TranscriptChunk]):
        if not chunks:
            raise ValueError("Cannot create a retriever without transcript chunks")
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise RuntimeError("Install rank-bm25 to use sparse retrieval") from exc
        self.chunks = list(chunks)
        self._index = BM25Okapi([_tokens(chunk.text) for chunk in self.chunks])

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        if k <= 0:
            return []
        scores = self._index.get_scores(_tokens(query))
        maximum = max(scores, default=0.0)
        indices = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
        return [
            RetrievedChunk(
                self.chunks[index].text,
                float(scores[index] / maximum) if maximum > 0 else 0.0,
                self.chunks[index].start,
                self.chunks[index].end,
                self.chunks[index].chunk_id,
            )
            for index in indices[:k]
        ]


class DenseRetriever:
    """Dense Chroma retrieval backed by HuggingFace sentence embeddings."""

    def __init__(
        self,
        chunks: Sequence[TranscriptChunk],
        persist_directory: str,
        embedding_model: str,
    ):
        if not chunks:
            raise ValueError("Cannot create a retriever without transcript chunks")
        try:
            from langchain_chroma import Chroma
            from langchain_core.documents import Document
        except ImportError as exc:
            raise RuntimeError(
                "Install langchain-chroma and langchain-huggingface for dense retrieval"
            ) from exc

        self.chunks = list(chunks)
        embeddings = _get_embeddings(embedding_model)
        docs = [
            Document(
                page_content=chunk.text,
                metadata={
                    "start": chunk.start,
                    "end": chunk.end,
                    "chunk_id": chunk.chunk_id,
                },
            )
            for chunk in self.chunks
        ]
        chunk_ids = [f"chunk-{chunk.chunk_id}" for chunk in self.chunks]
        self._store = Chroma(
            collection_name="transcript",
            embedding_function=embeddings,
            persist_directory=persist_directory,
        )
        existing_ids = set(self._store.get(include=["metadatas"])["ids"])
        if existing_ids != set(chunk_ids):
            self._store.delete_collection()
            self._store = Chroma.from_documents(
                docs,
                embedding=embeddings,
                persist_directory=persist_directory,
                collection_name="transcript",
                ids=chunk_ids,
            )

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        if k <= 0:
            return []
        matches = self._store.similarity_search_with_relevance_scores(query, k=k)
        return [
            RetrievedChunk(
                text=document.page_content,
                score=float(max(0.0, min(1.0, score))),
                start=float(document.metadata["start"]),
                end=float(document.metadata["end"]),
                chunk_id=int(document.metadata["chunk_id"]),
            )
            for document, score in matches
        ]


@lru_cache(maxsize=4)
def _get_embeddings(model_name: str) -> Any:
    """Reuse an in-process embedding model across retriever strategies."""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as exc:
        raise RuntimeError(
            "Install langchain-huggingface for dense retrieval"
        ) from exc
    return HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": "cpu"})


class HybridRetriever:
    """BM25 and dense retrieval merged by RRF and cross-encoder reranking."""

    def __init__(
        self,
        sparse: Retriever,
        dense: Retriever,
        reranker_model: str,
        rrf_constant: int | None = None,
    ):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "Install sentence-transformers to use the hybrid reranker"
            ) from exc
        self.sparse = sparse
        self.dense = dense
        self.rrf_constant = rrf_constant or get_settings().rrf_constant
        self._reranker = CrossEncoder(reranker_model)

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        if k <= 0:
            return []
        unique: dict[int, RetrievedChunk] = {}
        reciprocal: dict[int, float] = {}
        for retriever in (self.sparse, self.dense):
            for rank, chunk in enumerate(retriever.retrieve(query, max(k * 3, 10)), 1):
                unique.setdefault(chunk.chunk_id, chunk)
                reciprocal[chunk.chunk_id] = reciprocal.get(chunk.chunk_id, 0.0) + (
                    1.0 / (self.rrf_constant + rank)
                )
        ranked_ids = sorted(reciprocal, key=reciprocal.get, reverse=True)
        shortlist = [unique[chunk_id] for chunk_id in ranked_ids[: max(k * 3, 10)]]
        logits = self._reranker.predict([(query, item.text) for item in shortlist])
        scored = [
            RetrievedChunk(
                item.text,
                (
                    1 / (1 + math.exp(-float(logit)))
                    if float(logit) >= 0
                    else math.exp(float(logit)) / (1 + math.exp(float(logit)))
                ),
                item.start,
                item.end,
                item.chunk_id,
            )
            for item, logit in zip(shortlist, logits)
        ]
        return sorted(scored, key=lambda item: item.score, reverse=True)[:k]


def create_retriever(
    strategy: str,
    chunks: Sequence[TranscriptChunk],
    video_id: str,
    cache_dir: str = "cache",
) -> Retriever:
    """Create one configured retrieval strategy for a video transcript."""
    from pathlib import Path

    settings = get_settings()
    if strategy not in {"baseline", "sparse", "dense", "hybrid"}:
        raise ValueError(f"Unknown retrieval strategy: {strategy}")
    if strategy == "baseline":
        return BaselineRetriever(chunks, max_chars=settings.max_context_chars)
    if strategy == "sparse":
        return BM25Retriever(chunks)
    cache_id = hashlib.sha256(
        "\n".join(chunk.text for chunk in chunks).encode("utf-8")
    ).hexdigest()[:12]
    dense = DenseRetriever(
        chunks,
        persist_directory=str(Path(cache_dir) / "vectors" / f"{video_id}-{cache_id}"),
        embedding_model=settings.embedding_model,
    )
    if strategy == "dense":
        return dense
    if strategy == "hybrid":
        return HybridRetriever(
            BM25Retriever(chunks), dense, reranker_model=settings.reranker_model
        )
    raise AssertionError("Retrieval strategy dispatch was incomplete")
