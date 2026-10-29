"""LLM-assisted timestamped QA test-set generation."""

import json
import math
from pathlib import Path
import re
from typing import Any

from src.generator import generate_qa_pairs


def build_test_set(
    video_id: str, timestamped_transcript: str, output_dir: Path, count: int = 15
) -> Path:
    """Generate a 15-20 question test set for human review and editing."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{video_id}_qa.json"
    raw = generate_qa_pairs(timestamped_transcript, count=count)
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE)
    try:
        data: Any = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Gemini returned invalid JSON for the QA test set") from exc
    if not isinstance(data, list) or not 15 <= len(data) <= 20:
        raise RuntimeError("Generated test set must contain 15 to 20 QA pairs")
    for index, item in enumerate(data):
        if not isinstance(item, dict) or not all(
            key in item for key in ("question", "answer", "start", "end")
        ):
            raise RuntimeError(f"QA pair {index + 1} is missing required fields")
        start, end = float(item["start"]), float(item["end"])
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start:
            raise RuntimeError(f"QA pair {index + 1} has invalid timestamps")
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path
