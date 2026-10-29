# AI Video Assistant — RAG evaluation project

Streamlit app for audio/video transcription, transcript-grounded summarization, and chat with source timestamps. Retrieval choices are baseline, BM25 sparse, Chroma/HuggingFace dense, and BM25+dense RRF with cross-encoder reranking.

## Setup

Use Python 3.10+ and install FFmpeg so `yt-dlp` and `pydub` can extract audio. Install Python dependencies:

```powershell
python -m pip install -r Requirements.txt
```

Copy `.env.example` to `.env`. Set `GOOGLE_API_KEY` with a Gemini API key from Google AI Studio for generation and evaluation; set `SARVAM_API_KEY` for Hindi/Hinglish transcription. The default generator is `gemini-2.5-flash`. Gemini free-tier availability and quotas depend on Google's current policies. Whisper is local and downloads its selected model on first use. Never put real keys in source files or commit `.env`.

Optional values in `.env` configure chunk size/overlap, retrieval threshold, models, cache paths, retry count, and explicit LLM per-million-token rates for cost estimates. If rates are not supplied, reports leave cost blank. Token counts are estimates using the configured tokenizer.

## Run the app

```powershell
streamlit run app.py
```

Enter a YouTube URL or local media path, choose automatic/English/Hindi/Hinglish handling, and select a retrieval strategy in the sidebar. Chat responses show timestamp citations. Analysis can be downloaded as Markdown. Transcripts and dense indexes are cached locally under `cache/`.

The optional interactive CLI is `python main.py`; `test.py` remains a compatibility launcher for it.

## Evaluation

1. Edit `videos.yaml` and replace all four placeholders with videos you may use for your coursework. Keep the requested characteristics (English lecture, Hindi/Hinglish talk, short video, long/noisy video). Add a `reference_summary` when you have one.
2. Copy `.env.example` to `.env` and set the API keys.
3. Run:

```powershell
python -m evaluation.run_all --videos videos.yaml
```

The evaluation creates editable 15–20 pair timestamped QA files under `evaluation/data/` when missing and continues with those LLM-generated references. Review and hand-edit the references before using scores as ground truth; rerun the same command after edits to refresh all results. Evaluation writes `results/results.csv`, `results/per_video.csv`, `results/ablation.csv`, per-metric PNG charts, test-case data, `docs/results_summary.md`, and the AAT PowerPoint. The small dense ablation varies chunk size `{256, 512, 1024}` and overlap `{0, 64}`.
The same run refreshes `docs/aat_presentation.pptx` from measured results. `docs/report.md` is the project report scaffold; complete its results discussion using generated CSVs, not planned values.

Answer faithfulness/relevance use Gemini as judge; ROUGE-L is calculated locally and BERTScore requires its model dependency/download. The report is based only on runs with configured videos—no example values are supplied.

To create a heuristic error report after evaluation:

```powershell
python -m evaluation.error_analysis
```

This writes `results/errors.md`; failure labels are suggestions to inspect, not adjudicated ground truth. The Streamlit **Test Cases** view displays the most recent generated summaries, three sample Q&As with retrieved timestamps, and per-video metrics.

## Tests

```powershell
python -m pytest
```

See [docs/architecture.md](docs/architecture.md) for the architecture diagram, numbered data flow, approach comparison, and metric definitions.
The evaluation helpers are exposed through `src/evaluate.py` as well as `evaluation/metrics.py`.
