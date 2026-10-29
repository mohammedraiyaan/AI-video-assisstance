import pytest

from src.chunking import TranscriptSegment, chunk_transcript


def test_chunks_preserve_timestamps_and_overlap():
    chunks = chunk_transcript(
        [
            TranscriptSegment(0, 4, "one two three four"),
            TranscriptSegment(4, 8, "five six seven eight"),
        ],
        chunk_size=12,
        overlap=4,
    )

    assert len(chunks) > 1
    assert chunks[0].start == 0
    assert chunks[-1].end == 8
    assert chunks[0].text.split()[-1] in chunks[1].text.split()


def test_chunking_rejects_invalid_size_and_overlap():
    with pytest.raises(ValueError):
        chunk_transcript([TranscriptSegment(0, 1, "text")], chunk_size=0)
    with pytest.raises(ValueError):
        chunk_transcript([TranscriptSegment(0, 1, "text")], chunk_size=10, overlap=10)
