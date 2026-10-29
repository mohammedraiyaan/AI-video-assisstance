"""Gemini-backed title, summary, extraction, and question-answer generation."""

import json
import os
import re
import time
from functools import lru_cache
import threading
from typing import Any

from src.config import get_settings
from src.retrievers import RetrievedChunk

_REQUEST_LOCK = threading.Lock()
_LAST_REQUEST_TIME = 0.0


@lru_cache(maxsize=1)
def get_llm() -> Any:
    """Create the configured Gemini chat model with explicit credential checks."""
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY (or GEMINI_API_KEY) is missing; configure it in .env"
        )
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as exc:
        raise RuntimeError(
            "Install langchain-google-genai to generate responses"
        ) from exc
    settings = get_settings()
    return ChatGoogleGenerativeAI(
        model=settings.llm_model, google_api_key=api_key, temperature=0.2
    )


def generate_text(prompt: str, attempts: int | None = None) -> str:
    """Call Gemini with bounded retries for transient API failures."""
    global _LAST_REQUEST_TIME
    model = get_llm()
    retry_limit = attempts if attempts is not None else get_settings().retry_attempts
    last_error: Exception | None = None
    rate_limit_attempts = 8
    for attempt in range(max(retry_limit, rate_limit_attempts)):
        if attempt >= retry_limit and not _is_rate_limit_error(last_error):
            break
        with _REQUEST_LOCK:
            delay = (
                get_settings().llm_min_request_interval
                - (time.monotonic() - _LAST_REQUEST_TIME)
            )
            if delay > 0:
                time.sleep(delay)
            _LAST_REQUEST_TIME = time.monotonic()
        try:
            response = model.invoke(prompt)
            return str(response.content).strip()
        except Exception as exc:
            last_error = exc
            is_rate_limited = _is_rate_limit_error(exc)
            max_attempts = (
                max(retry_limit, rate_limit_attempts)
                if is_rate_limited
                else retry_limit
            )
            if attempt + 1 >= max_attempts:
                break
            retry_delay = _retry_after_seconds(exc)
            if retry_delay is None:
                retry_delay = (
                    min(5 * (2**attempt), 60)
                    if is_rate_limited
                    else min(2**attempt, 8)
                )
            time.sleep(retry_delay)
    raise RuntimeError("Gemini request failed after retry attempts") from last_error


def _is_rate_limit_error(error: Exception | None) -> bool:
    """Detect HTTP 429 responses from the provider or its wrapped client."""
    current: BaseException | None = error
    while current is not None:
        response = getattr(current, "response", None)
        if getattr(response, "status_code", None) == 429:
            return True
        if getattr(current, "status_code", None) == 429 or getattr(current, "code", None) == 429:
            return True
        if "429" in str(current) or "RESOURCE_EXHAUSTED" in str(current).upper():
            return True
        current = current.__cause__ or current.__context__
    return False


def _retry_after_seconds(error: Exception) -> float | None:
    """Read the provider's retry-after hint when present."""
    current: BaseException | None = error
    while current is not None:
        response = getattr(current, "response", None)
        headers = getattr(response, "headers", {})
        retry_after = headers.get("retry-after") or headers.get("Retry-After")
        if retry_after:
            try:
                return max(0.0, min(float(retry_after), 120.0))
            except (TypeError, ValueError):
                pass
        current = current.__cause__ or current.__context__
    return None


def _transcript_windows(transcript: str) -> list[str]:
    """Split long transcripts on line or word boundaries within the model budget."""
    limit = get_settings().max_context_chars
    windows: list[str] = []
    start = 0
    while start < len(transcript):
        end = min(start + limit, len(transcript))
        if end < len(transcript):
            newline = transcript.rfind("\n", start, end)
            whitespace = transcript.rfind(" ", start, end)
            boundary = newline if newline > start + limit // 2 else whitespace
            if boundary > start:
                end = boundary
        windows.append(transcript[start:end].strip())
        start = end
        while start < len(transcript) and transcript[start].isspace():
            start += 1
    return [window for window in windows if window]


def _extract_across_transcript(transcript: str, instruction: str) -> str:
    windows = _transcript_windows(transcript)
    if len(windows) <= 1:
        return generate_text(f"{instruction}\n\n{windows[0] if windows else transcript}")
    partials = [
        generate_text(f"{instruction}\n\nTranscript section:\n{window}")
        for window in windows
    ]
    return generate_text(
        f"{instruction}\nConsolidate and deduplicate these section-level results. "
        "Retain only items supported by the transcript.\n\n"
        + "\n\n".join(partials)[: get_settings().max_context_chars]
    )


def generate_title(transcript: str) -> str:
    """Generate a concise title from a transcript."""
    if not transcript.strip():
        raise ValueError("Cannot generate a title from an empty transcript")
    return generate_text(
        "Give this video a professional title of at most 8 words. Return only the title.\n\n"
        + transcript[:4000]
    )


def summarize(transcript: str) -> str:
    """Summarize transcript text in concise, source-grounded bullets."""
    if not transcript.strip():
        raise ValueError("Cannot summarize an empty transcript")
    instruction = (
        "Summarize the video transcript in concise, informative bullets. "
        "Do not add unsupported facts."
    )
    windows = _transcript_windows(transcript)
    if len(windows) <= 1:
        return generate_text(f"{instruction}\n\n{windows[0]}")
    partials = [
        generate_text(f"{instruction}\n\nTranscript section:\n{window}")
        for window in windows
    ]
    return generate_text(
        "Combine these partial summaries into one concise, professional video summary. "
        "Do not add unsupported facts.\n\n"
        + "\n\n".join(partials)[: get_settings().max_context_chars]
    )


def extract_key_decisions(transcript: str) -> str:
    """Extract decisions and commitments from transcript text."""
    return _extract_across_transcript(
        transcript,
        "List key decisions and commitments. If none, say none found.",
    )


def extract_action_items(transcript: str) -> str:
    """Extract tasks, owners, and deadlines from transcript text."""
    return _extract_across_transcript(
        transcript,
        "List action items. Include owner and deadline when mentioned, otherwise mark "
        "them not specified. If there are none, say none found.",
    )


def extract_questions(transcript: str) -> str:
    """Extract open questions and follow-ups from transcript text."""
    return _extract_across_transcript(
        transcript,
        "List unresolved questions or follow-ups. If none, say none found.",
    )


def answer_question(query: str, chunks: list[RetrievedChunk]) -> str:
    """Answer only from supplied retrieval context; abstain when no context exists."""
    if not chunks:
        return "I could not find this in the video."
    context = "\n\n".join(
        f"[{chunk.start:.1f}s-{chunk.end:.1f}s] {chunk.text}" for chunk in chunks
    )
    return generate_text(
        "Answer the question using only the video transcript context. If it does not "
        "contain the answer, reply exactly: I could not find this in the video.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}"
    )


def generate_qa_pairs(transcript: str, count: int = 15) -> str:
    """Generate timestamped QA pairs as JSON for human review."""
    if not 15 <= count <= 20:
        raise ValueError("count must be between 15 and 20")
    windows = _transcript_windows(transcript)
    if not windows:
        raise ValueError("Cannot create questions from an empty transcript")
    if len(windows) > count:
        positions = [
            round(index * (len(windows) - 1) / (count - 1))
            for index in range(count)
        ]
        windows = [windows[position] for position in positions]
    per_window_budget = max(800, get_settings().max_context_chars // len(windows))
    excerpts = []
    for window in windows:
        if len(window) > per_window_budget:
            window = window[:per_window_budget]
        excerpts.append(window)
    prompt = (
        f"Create exactly {count} diverse question-answer pairs grounded in these "
        "timestamped transcript excerpts sampled across the video. Return only a JSON "
        "array with objects containing question, answer, start, and end in seconds. "
        "Use timestamp ranges shown in the excerpts; do not invent timestamps. "
        "Questions must cover different topics where possible.\n\n"
        + "\n\n".join(excerpts)
    )
    response = generate_text(prompt).strip()
    if response.startswith("```"):
        response = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            response,
            flags=re.IGNORECASE,
        )
    try:
        pairs = json.loads(response)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Gemini returned invalid JSON for timestamped QA") from exc
    if not isinstance(pairs, list):
        raise RuntimeError("Gemini QA response must be a JSON array")
    if len(pairs) < 15 or len(pairs) > 20:
        raise RuntimeError(f"Gemini generated {len(pairs)} QA pairs; expected 15-20")
    return json.dumps(pairs, ensure_ascii=False)
