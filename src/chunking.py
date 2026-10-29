"""Timestamp-preserving transcript chunking."""

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class TranscriptSegment:
    """A transcript segment with its audio-relative time range."""

    start: float
    end: float
    text: str


@dataclass(frozen=True)
class TranscriptChunk:
    """A retrievable text window and its approximate source time range."""

    text: str
    start: float
    end: float
    chunk_id: int


def chunk_transcript(
    segments: Iterable[TranscriptSegment],
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[TranscriptChunk]:
    """Create overlapping word-aligned chunks and retain segment timestamps.

    Chunks are limited by character count; timestamps span the first and last
    source segments represented in each chunk.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    cleaned = [
        TranscriptSegment(float(segment.start), float(segment.end), segment.text.strip())
        for segment in segments
        if segment.text and segment.text.strip()
    ]
    if any(segment.start < 0 or segment.end < segment.start for segment in cleaned):
        raise ValueError("segment timestamps must be non-negative and ordered")

    words: list[tuple[str, float, float]] = []
    for segment in cleaned:
        tokens = segment.text.split()
        if not tokens:
            continue
        duration = max(0.0, segment.end - segment.start)
        for index, word in enumerate(tokens):
            start = segment.start + duration * index / len(tokens)
            end = segment.start + duration * (index + 1) / len(tokens)
            words.append((word, start, end))

    chunks: list[TranscriptChunk] = []
    position = 0
    while position < len(words):
        end_position = position
        text_length = 0
        while end_position < len(words):
            next_length = len(words[end_position][0]) + (1 if end_position > position else 0)
            if end_position > position and text_length + next_length > chunk_size:
                break
            text_length += next_length
            end_position += 1

        selected = words[position:end_position]
        chunks.append(
            TranscriptChunk(
                text=" ".join(word for word, _, _ in selected),
                start=selected[0][1],
                end=selected[-1][2],
                chunk_id=len(chunks),
            )
        )
        if end_position == len(words):
            break
        next_position = end_position
        overlap_length = 0
        while next_position > position and overlap_length < overlap:
            next_position -= 1
            overlap_length += len(words[next_position][0]) + 1
        position = max(position + 1, next_position)
    return chunks
