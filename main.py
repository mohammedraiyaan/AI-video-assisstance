"""Command-line entry point for the AI Video Assistant."""

from dotenv import load_dotenv

from src.chunking import chunk_transcript
from src.config import get_settings
from src.generator import (
    answer_question,
    extract_action_items,
    extract_key_decisions,
    extract_questions,
    generate_title,
    summarize,
)
from src.ingest import ingest_video
from src.retrievers import create_retriever


def run_pipeline(source: str, language: str = "auto") -> dict:
    """Run the modular transcription, summary, extraction, and dense RAG pipeline."""
    load_dotenv()
    ingestion = ingest_video(source, language)
    transcript = ingestion.transcript
    chunks = chunk_transcript(
        ingestion.segments,
        chunk_size=get_settings().chunk_size,
        overlap=get_settings().chunk_overlap,
    )
    retriever = create_retriever(
        "dense",
        chunks,
        ingestion.video_id,
        cache_dir=str(get_settings().cache_dir),
    )
    return {
        "title": generate_title(transcript),
        "transcript": transcript,
        "summary": summarize(transcript),
        "action_items": extract_action_items(transcript),
        "key_decisions": extract_key_decisions(transcript),
        "open_questions": extract_questions(transcript),
        "retriever": retriever,
    }


def main() -> None:
    """Run the assistant and provide a timestamped CLI chat."""
    load_dotenv()
    source = input("Enter YouTube URL or local file path: ").strip()
    language = input("Language (auto/english/hindi/hinglish): ").strip() or "auto"
    result = run_pipeline(source, language)
    print(f"\nTitle: {result['title']}")
    print(f"\nSummary:\n{result['summary']}")
    print(f"\nAction items:\n{result['action_items']}")
    print(f"\nKey decisions:\n{result['key_decisions']}")
    print(f"\nOpen questions:\n{result['open_questions']}")

    print("\nChat with the transcript (type 'exit' to quit)")
    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit", "q"}:
            break
        if not question:
            continue
        retrieved = result["retriever"].retrieve(question, get_settings().retrieval_k)
        if (
            not retrieved
            or max(chunk.score for chunk in retrieved)
            < get_settings().retrieval_score_threshold
        ):
            print("Assistant: I could not find this in the video.")
            continue
        print(f"Assistant: {answer_question(question, retrieved)}")
        for chunk in retrieved:
            print(f"  Source: {chunk.start:.0f}s–{chunk.end:.0f}s")


if __name__ == "__main__":
    main()