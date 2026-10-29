"""RAG retrieval and answer-quality metrics."""

from collections.abc import Sequence
import re

from src.retrievers import RetrievedChunk


def retrieval_metrics(
    retrieved: Sequence[RetrievedChunk],
    relevant_ranges: Sequence[tuple[float, float]],
) -> dict[str, float]:
    """Compute recall@3, recall@5, reciprocal rank, and hit rate by time overlap."""
    if not relevant_ranges:
        raise ValueError("At least one ground-truth timestamp range is required")

    def is_relevant(chunk: RetrievedChunk) -> bool:
        return any(chunk.end >= start and chunk.start <= end for start, end in relevant_ranges)

    relevant_hits = [is_relevant(chunk) for chunk in retrieved]
    total_relevant = len(relevant_ranges)
    recall3 = min(
        1.0,
        sum(
            any(
                chunk.end >= start and chunk.start <= end
                for chunk in retrieved[:3]
            )
            for start, end in relevant_ranges
        )
        / total_relevant,
    )
    recall5 = min(
        1.0,
        sum(
            any(chunk.end >= start and chunk.start <= end for chunk in retrieved[:5])
            for start, end in relevant_ranges
        )
        / total_relevant,
    )
    rank = next((index for index, hit in enumerate(relevant_hits, 1) if hit), None)
    return {
        "recall@3": recall3,
        "recall@5": recall5,
        "mrr": 1.0 / rank if rank else 0.0,
        "hit_rate": 1.0 if any(relevant_hits) else 0.0,
    }


def rouge_l(reference: str, prediction: str) -> float:
    """Compute ROUGE-L F1 without external model downloads."""
    reference_tokens = re.findall(r"\w+", reference.lower())
    prediction_tokens = re.findall(r"\w+", prediction.lower())
    if not reference_tokens or not prediction_tokens:
        return 0.0
    previous = [0] * (len(prediction_tokens) + 1)
    for reference_token in reference_tokens:
        current = [0]
        for index, prediction_token in enumerate(prediction_tokens, 1):
            current.append(
                previous[index - 1] + 1
                if reference_token == prediction_token
                else max(previous[index], current[index - 1])
            )
        previous = current
    lcs = previous[-1]
    precision = lcs / len(prediction_tokens)
    recall = lcs / len(reference_tokens)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


_BERT_SCORER = None


def bert_score(reference: str, prediction: str) -> float:
    """Compute BERTScore F1; requires the pinned bert-score dependency."""
    global _BERT_SCORER
    try:
        from bert_score import BERTScorer
    except ImportError as exc:
        raise RuntimeError("Install bert-score to calculate BERTScore") from exc
    if _BERT_SCORER is None:
        _BERT_SCORER = BERTScorer(
            lang="en", device="cpu", rescale_with_baseline=False
        )
    _, _, f1 = _BERT_SCORER.score([prediction], [reference])
    return float(f1[0])


def llm_judge(
    question: str, reference: str, answer: str, context: str
) -> tuple[float, float, str, str]:
    """Ask the configured Gemini model to score faithfulness and relevance."""
    import json

    from src.generator import generate_text

    prompt = (
        "Evaluate the answer using only the supplied context. Return JSON with "
        'numeric "faithfulness" and "relevance" fields in [0,1]. Faithfulness '
        "measures whether every factual claim is supported. Relevance measures "
        "whether it answers the question. Do not include markdown.\n"
        f"Question: {question}\nReference answer: {reference}\n"
        f"Answer: {answer}\nContext: {context}"
    )
    response = generate_text(prompt)
    if response.startswith("```"):
        response = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            response.strip(),
            flags=re.IGNORECASE,
        )
    try:
        values = json.loads(response)
        faithfulness = float(values["faithfulness"])
        relevance = float(values["relevance"])
        if not 0 <= faithfulness <= 1 or not 0 <= relevance <= 1:
            raise ValueError("Judge scores must be within [0,1]")
        return faithfulness, relevance, prompt, response
    except (ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(f"Gemini judge returned invalid scores: {response}") from exc
