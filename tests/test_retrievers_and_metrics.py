import sys
from types import ModuleType, SimpleNamespace

import pytest

from evaluation.metrics import retrieval_metrics, rouge_l
from src.chunking import TranscriptChunk
from src.generator import get_llm
from src.ingest import identify_source
from src.retrievers import BaselineRetriever, RetrievedChunk


def test_baseline_implements_retrieve_interface():
    retriever = BaselineRetriever(
        [TranscriptChunk("intro", 0, 5, 0), TranscriptChunk("answer", 5, 10, 1)],
        max_chars=100,
    )

    result = retriever.retrieve("question", 4)

    assert len(result) == 1
    assert isinstance(result[0], RetrievedChunk)
    assert result[0].start == 0
    assert result[0].end == 10
    assert result[0].text == "intro answer"


def test_retrieval_metrics_use_timestamp_overlap():
    chunks = [
        RetrievedChunk("irrelevant", 0.8, 0, 5, 0),
        RetrievedChunk("answer", 0.7, 10, 15, 1),
    ]

    result = retrieval_metrics(chunks, [(12, 14)])

    assert result == {"recall@3": 1.0, "recall@5": 1.0, "mrr": 0.5, "hit_rate": 1.0}


def test_retrieval_metrics_require_ground_truth():
    with pytest.raises(ValueError):
        retrieval_metrics([], [])


def test_rouge_l_exact_match():
    assert rouge_l("A short answer", "A short answer") == 1.0


def test_identify_source_rejects_non_youtube_urls():
    with pytest.raises(ValueError):
        identify_source("https://example.com/watch?v=abcdefghijk")


def test_identify_source_accepts_youtube_shorts():
    assert identify_source("https://youtube.com/shorts/abcdefghijk")[0] == "abcdefghijk"


def test_gemini_requires_an_api_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    get_llm.cache_clear()

    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
        get_llm()


def test_gemini_uses_configured_google_api_key(monkeypatch):
    calls = {}

    class FakeChatGoogleGenerativeAI:
        def __init__(self, **kwargs):
            calls.update(kwargs)

    fake_module = ModuleType("langchain_google_genai")
    fake_module.ChatGoogleGenerativeAI = FakeChatGoogleGenerativeAI
    monkeypatch.setitem(sys.modules, "langchain_google_genai", fake_module)
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    get_llm.cache_clear()

    try:
        model = get_llm()
    finally:
        get_llm.cache_clear()

    assert isinstance(model, FakeChatGoogleGenerativeAI)
    assert calls["google_api_key"] == "test-key"
    assert calls["model"] == "gemini-2.5-flash"
