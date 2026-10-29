"""Create a concise report of the lowest-scoring evaluated questions."""

import argparse
import json
from pathlib import Path
from typing import Any


def _failure_reason(row: dict[str, Any]) -> str:
    chunks = row.get("retrieved_chunks", [])
    if not chunks or float(row.get("recall@5", 0.0)) == 0:
        return "Retrieval miss: no retrieved chunk overlaps the annotated answer timestamps."
    if any(
        "inaudible" in chunk.get("text", "").lower()
        or "unintelligible" in chunk.get("text", "").lower()
        for chunk in chunks
    ):
        return "Possible transcription error: retrieved text contains an audio uncertainty marker."
    if any(float(chunk.get("end", 0)) - float(chunk.get("start", 0)) > 180 for chunk in chunks):
        return "Possible chunk-granularity issue: retrieved context spans more than three minutes."
    faithfulness = float(row.get("faithfulness", 0.0))
    relevance = float(row.get("answer_relevance", 0.0))
    answer = str(row.get("generated_answer", "")).lower()
    if faithfulness < relevance - 0.2 and "could not find" not in answer:
        return "Possible hallucination: the answer was judged relevant but weakly supported by context."
    return "Likely answer-generation or annotation mismatch; inspect transcript, context, and reference."


def generate_error_report(
    details_path: Path = Path("results/question_details.json"),
    output_path: Path = Path("results/errors.md"),
) -> Path:
    """Write the ten lowest-scoring QA items for each retrieval approach."""
    if not details_path.is_file():
        raise FileNotFoundError(
            f"{details_path} is missing; run python -m evaluation.run_all first."
        )
    with details_path.open(encoding="utf-8") as file:
        rows = json.load(file)
    if not isinstance(rows, list):
        raise ValueError(f"{details_path} must contain a JSON array")
    rows = [row for row in rows if row.get("question") and row.get("approach")]
    approaches = sorted({str(row["approach"]) for row in rows})
    lines = [
        "# Evaluation Error Analysis",
        "",
        "Worst-scoring questions are ranked by the mean of faithfulness and answer relevance. "
        "Failure reasons are heuristic prompts for manual review, not ground-truth diagnoses.",
        "",
    ]
    for approach in approaches:
        applicable = [
            row for row in rows if row["approach"] == approach and row.get("generated_answer")
        ]
        applicable.sort(
            key=lambda row: (
                float(row.get("faithfulness", 0.0))
                + float(row.get("answer_relevance", 0.0))
            )
            / 2
        )
        lines += [f"## {approach}", ""]
        for index, row in enumerate(applicable[:10], 1):
            lines += [
                f"### {index}. {row['question']}",
                "",
                f"- **Video:** {row.get('video', 'unknown')}",
                f"- **Faithfulness / relevance:** {row.get('faithfulness')} / {row.get('answer_relevance')}",
                f"- **Likely failure reason:** {_failure_reason(row)}",
                f"- **Reference answer:** {row.get('reference_answer', '')}",
                f"- **Generated answer:** {row.get('generated_answer', '')}",
                "- **Retrieved chunks:**",
            ]
            for chunk in row.get("retrieved_chunks", []):
                lines.append(
                    f"  - [{chunk.get('start')}s–{chunk.get('end')}s, score "
                    f"{chunk.get('score')}]: {chunk.get('text', '')}"
                )
            lines.append("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/question_details.json"))
    parser.add_argument("--output", type=Path, default=Path("results/errors.md"))
    arguments = parser.parse_args()
    print(generate_error_report(arguments.input, arguments.output))


if __name__ == "__main__":
    main()
