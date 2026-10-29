"""Run the QA, retrieval, generation, and ablation evaluation suite."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import yaml
from dotenv import load_dotenv

from evaluation.metrics import bert_score, llm_judge, retrieval_metrics, rouge_l
from evaluation.build_presentation import build_presentation
from evaluation.error_analysis import generate_error_report
from evaluation.test_set_builder import build_test_set
from src.chunking import TranscriptSegment, chunk_transcript
from src.config import get_settings
from src.generator import answer_question, summarize
from src.ingest import identify_source, ingest_video
from src.retrievers import HybridRetriever, RetrievedChunk, create_retriever

APPROACHES = ("baseline", "sparse", "dense", "hybrid")


def _read_videos(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        data = yaml.safe_load(file)
    videos = data.get("videos") if isinstance(data, dict) else data
    if not isinstance(videos, list) or not videos:
        raise ValueError(f"{path} must contain a non-empty videos list")
    if any(not isinstance(video, dict) for video in videos):
        raise ValueError(f"{path} entries in videos must be mappings")
    return videos


def _estimate_tokens(text: str) -> int:
    try:
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except ImportError as exc:
        raise RuntimeError("Install tiktoken to measure query token counts") from exc


def _timestamp_context(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(
        f"[{chunk.start:.1f}s-{chunk.end:.1f}s] {chunk.text}" for chunk in chunks
    )


def _read_questions(qa_path: Path) -> list[dict[str, Any]]:
    with qa_path.open(encoding="utf-8") as file:
        questions = json.load(file)
    if not isinstance(questions, list) or not questions:
        raise ValueError(f"{qa_path} must contain a non-empty JSON array")
    if not 15 <= len(questions) <= 20:
        raise ValueError(f"{qa_path} must contain between 15 and 20 QA pairs")
    for index, item in enumerate(questions, 1):
        if (
            not isinstance(item, dict)
            or not all(field in item for field in ("question", "answer", "start", "end"))
            or not str(item.get("question", "")).strip()
            or not str(item.get("answer", "")).strip()
        ):
            raise ValueError(f"{qa_path} question {index} is missing required fields")
        try:
            start, end = float(item["start"]), float(item["end"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{qa_path} question {index} has non-numeric timestamps") from exc
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start:
            raise ValueError(f"{qa_path} question {index} has invalid timestamps")
    return questions


def _score_question(
    question: dict[str, Any], retriever: Any, approach: str, video_id: str
) -> dict[str, Any]:
    query = str(question["question"])
    reference = str(question["answer"])
    start, end = float(question["start"]), float(question["end"])
    started = time.perf_counter()
    retrieved = retriever.retrieve(query, 5)
    retrieval_seconds = time.perf_counter() - started
    scores = retrieval_metrics(retrieved, [(start, end)])
    answer_start = time.perf_counter()
    if (
        not retrieved
        or max(chunk.score for chunk in retrieved)
        < get_settings().retrieval_score_threshold
    ):
        answer = "I could not find this in the video."
    else:
        answer = answer_question(query, retrieved)
    latency = time.perf_counter() - answer_start
    context = _timestamp_context(retrieved)
    faithfulness, relevance, judge_prompt, judge_response = llm_judge(
        query, reference, answer, context
    )
    return {
        "video": video_id,
        "approach": approach,
        "question": query,
        "reference_answer": reference,
        "generated_answer": answer,
        "retrieved_chunks": [
            {
                "text": item.text,
                "score": item.score,
                "start": item.start,
                "end": item.end,
                "chunk_id": item.chunk_id,
            }
            for item in retrieved
        ],
        "retrieval_seconds": retrieval_seconds,
        "query_latency_seconds": latency,
        "end_to_end_latency_seconds": retrieval_seconds + latency,
        "input_tokens_estimate": _estimate_tokens(query + context)
        + _estimate_tokens(judge_prompt),
        "output_tokens_estimate": _estimate_tokens(answer + judge_response),
        "faithfulness": faithfulness,
        "answer_relevance": relevance,
        "rouge_l": rouge_l(reference, answer),
        "bert_score": bert_score(reference, answer),
        **scores,
    }


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _write_charts(rows: list[dict[str, Any]], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    metrics = (
        "recall@3",
        "recall@5",
        "mrr",
        "hit_rate",
        "faithfulness",
        "answer_relevance",
        "rouge_l",
        "bert_score",
        "summary_rouge_l",
        "summary_bert_score",
        "compression_ratio",
        "query_latency_seconds",
        "end_to_end_latency_seconds",
        "transcription_seconds",
        "estimated_cost_usd",
        "input_tokens_estimate",
        "output_tokens_estimate",
    )
    for metric in metrics:
        grouped = {
            approach: [
                float(row[metric])
                for row in rows
                if row["approach"] == approach and row.get(metric) is not None
            ]
            for approach in APPROACHES
        }
        if not any(grouped.values()):
            continue
        labels = list(grouped)
        means = [_mean(grouped[label]) or 0.0 for label in labels]
        figure, axis = plt.subplots(figsize=(8, 4))
        axis.bar(labels, means, color=["#64748b", "#0ea5e9", "#8b5cf6", "#f97316"])
        axis.set_title(metric.replace("_", " ").title())
        axis.set_ylabel("Seconds" if "seconds" in metric else "Score / value")
        figure.tight_layout()
        figure.savefig(output_dir / f"{metric.replace('@', '_at_')}.png", dpi=150)
        plt.close(figure)


def _run_ablation(
    video: dict[str, Any],
    segments: list[TranscriptSegment],
    questions: list[dict[str, Any]],
    video_id: str,
    cache_dir: str,
) -> list[dict[str, Any]]:
    ablation_rows = []
    for size in (256, 512, 1024):
        for overlap in (0, 64):
            if overlap >= size:
                continue
            chunks = chunk_transcript(segments, chunk_size=size, overlap=overlap)
            retriever = create_retriever(
                "dense",
                chunks,
                f"{video_id}-{size}-{overlap}",
                cache_dir=cache_dir,
            )
            per_question = [
                retrieval_metrics(
                    retriever.retrieve(item["question"], 5),
                    [(float(item["start"]), float(item["end"]))],
                )
                for item in questions
            ]
            ablation_rows.append(
                {
                    "video": video_id,
                    "approach": "dense_ablation",
                    "chunk_size": size,
                    "overlap": overlap,
                    **{
                        key: _mean([metric[key] for metric in per_question])
                        for key in ("recall@3", "recall@5", "mrr", "hit_rate")
                    },
                }
            )
    return ablation_rows


def run_evaluation(videos_file: Path) -> None:
    """Execute all configured approaches; never emit placeholder metric data."""
    load_dotenv()
    videos = _read_videos(videos_file)
    settings = get_settings()
    prepared_videos: list[tuple[dict[str, Any], str, Any]] = []
    generated_test_sets: list[Path] = []
    for index, video in enumerate(videos, 1):
        source = str(video.get("url") or video.get("path") or "").strip()
        if not source or "FILL" in source.upper():
            raise ValueError(
                f"videos.yaml entry {index} is a placeholder; provide a YouTube URL or file path."
            )
        video_id, _ = identify_source(source)
        ingestion = ingest_video(source, language=str(video.get("language", "auto")))
        qa_path = Path("evaluation/data") / f"{video_id}_qa.json"
        if not qa_path.is_file():
            timestamped = "\n".join(
                f"[{segment.start:.1f}s-{segment.end:.1f}s] {segment.text}"
                for segment in ingestion.segments
            )
            generated_test_sets.append(
                build_test_set(
                    video_id,
                    timestamped or ingestion.transcript,
                    qa_path.parent,
                )
            )
        prepared_videos.append((video, video_id, ingestion))
    if generated_test_sets:
        paths = "\n".join(str(path) for path in generated_test_sets)
        print(
            "Generated LLM QA references (editable, not human-verified):\n"
            f"{paths}"
        )

    settings.results_dir.mkdir(parents=True, exist_ok=True)
    all_questions: list[dict[str, Any]] = []
    per_video: list[dict[str, Any]] = []
    summaries_for_report: dict[str, Any] = {}
    cost_input_rate = os.getenv("LLM_INPUT_COST_PER_MILLION")
    cost_output_rate = os.getenv("LLM_OUTPUT_COST_PER_MILLION")
    for variable, value in (
        ("LLM_INPUT_COST_PER_MILLION", cost_input_rate),
        ("LLM_OUTPUT_COST_PER_MILLION", cost_output_rate),
    ):
        if value and (not math.isfinite(float(value)) or float(value) < 0):
            raise ValueError(f"{variable} must be non-negative")

    for index, (video, video_id, ingestion) in enumerate(prepared_videos, 1):
        questions = _read_questions(
            Path("evaluation/data") / f"{video_id}_qa.json"
        )
        summary = summarize(ingestion.transcript)
        summary_reference = str(video.get("reference_summary", "")).strip()
        summary_metrics = {
            "summary_rouge_l": rouge_l(summary_reference, summary)
            if summary_reference
            else None,
            "summary_bert_score": bert_score(summary_reference, summary)
            if summary_reference
            else None,
            "compression_ratio": (
                _estimate_tokens(summary) / _estimate_tokens(ingestion.transcript)
                if ingestion.transcript
                else None
            ),
        }
        summaries_for_report[video_id] = {
            "summary": summary,
            **summary_metrics,
            "reference_summary": summary_reference,
        }
        chunks = chunk_transcript(
            ingestion.segments,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )
        retrievers = {
            "baseline": create_retriever(
                "baseline", chunks, video_id, cache_dir=str(settings.cache_dir)
            ),
            "sparse": create_retriever(
                "sparse", chunks, video_id, cache_dir=str(settings.cache_dir)
            ),
            "dense": create_retriever(
                "dense", chunks, video_id, cache_dir=str(settings.cache_dir)
            ),
        }
        retrievers["hybrid"] = HybridRetriever(
            retrievers["sparse"],
            retrievers["dense"],
            settings.reranker_model,
            settings.rrf_constant,
        )
        question_rows: list[dict[str, Any]] = []
        for approach, retriever in retrievers.items():
            for question in questions:
                row = _score_question(question, retriever, approach, video_id)
                row["transcription_seconds"] = ingestion.transcription_seconds
                row.update(summary_metrics)
                if cost_input_rate and cost_output_rate:
                    row["estimated_cost_usd"] = (
                        row["input_tokens_estimate"] * float(cost_input_rate)
                        + row["output_tokens_estimate"] * float(cost_output_rate)
                    ) / 1_000_000
                else:
                    row["estimated_cost_usd"] = None
                question_rows.append(row)
                all_questions.append(row)

        metric_names = (
            "recall@3",
            "recall@5",
            "mrr",
            "hit_rate",
            "faithfulness",
            "answer_relevance",
            "rouge_l",
            "bert_score",
            "summary_rouge_l",
            "summary_bert_score",
            "compression_ratio",
            "query_latency_seconds",
            "end_to_end_latency_seconds",
            "transcription_seconds",
            "input_tokens_estimate",
            "output_tokens_estimate",
            "estimated_cost_usd",
        )
        for approach in APPROACHES:
            approach_rows = [row for row in question_rows if row["approach"] == approach]
            per_video.append(
                {
                    "video": video_id,
                    "approach": approach,
                    "summary": summary,
                    **summary_metrics,
                    **{
                        metric: _mean(
                            [
                                float(row[metric])
                                for row in approach_rows
                                if row.get(metric) is not None
                            ]
                        )
                        for metric in metric_names
                    },
                }
            )
        dense_rows = [row for row in question_rows if row["approach"] == "dense"]
        summaries_for_report[video_id]["sample_qas"] = [
            {
                "question": row["question"],
                "answer": row["generated_answer"],
                "retrieved_chunks": row["retrieved_chunks"],
            }
            for row in dense_rows[:3]
        ]
        if index == 1:
            ablation_rows = _run_ablation(
                video,
                ingestion.segments,
                questions,
                video_id,
                cache_dir=str(settings.cache_dir),
            )
            _write_csv(settings.results_dir / "ablation.csv", ablation_rows)

    result_path = settings.results_dir / "results.csv"
    per_video_path = settings.results_dir / "per_video.csv"
    _write_csv(result_path, _aggregate_by_approach(all_questions))
    _write_csv(per_video_path, per_video)
    (settings.results_dir / "question_details.json").write_text(
        json.dumps(all_questions, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    generate_error_report(
        settings.results_dir / "question_details.json",
        settings.results_dir / "errors.md",
    )
    (settings.results_dir / "test_case_summaries.json").write_text(
        json.dumps(summaries_for_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_charts(_aggregate_by_approach(all_questions), settings.results_dir)
    _write_results_summary(result_path, Path("docs/results_summary.md"))
    build_presentation(Path("docs/aat_presentation.pptx"), result_path)
    print(f"Evaluation complete: {result_path}, {per_video_path}")


def _aggregate_by_approach(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = (
        "recall@3",
        "recall@5",
        "mrr",
        "hit_rate",
        "faithfulness",
        "answer_relevance",
        "rouge_l",
        "bert_score",
        "summary_rouge_l",
        "summary_bert_score",
        "compression_ratio",
        "query_latency_seconds",
        "end_to_end_latency_seconds",
        "transcription_seconds",
        "input_tokens_estimate",
        "output_tokens_estimate",
        "estimated_cost_usd",
    )
    aggregate = []
    for approach in APPROACHES:
        approach_rows = [row for row in rows if row.get("approach") == approach]
        if approach_rows:
            aggregate.append(
                {
                    "approach": approach,
                    **{
                        metric: _mean(
                            [
                                float(row[metric])
                                for row in approach_rows
                                if row.get(metric) is not None
                            ]
                        )
                        for metric in metrics
                    },
                }
            )
    return aggregate


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty evaluation file: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_results_summary(csv_path: Path, markdown_path: Path) -> None:
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    columns = list(rows[0]) if rows else []
    lines = ["# Evaluation Results Summary", "", "Generated from `results/results.csv`.", ""]
    if rows:
        lines += [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
        ]
        lines.extend(
            "| " + " | ".join(str(row.get(column, "")) for column in columns) + " |"
            for row in rows
        )
    else:
        lines.append("No evaluation results are available yet. Run `python -m evaluation.run_all --videos videos.yaml` after configuring videos and API keys.")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, required=True, help="YAML test-video manifest")
    arguments = parser.parse_args()
    run_evaluation(arguments.videos)


if __name__ == "__main__":
    main()
