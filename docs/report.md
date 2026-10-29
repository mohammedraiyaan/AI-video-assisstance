# RAG for an AI Video Summarizer

## Abstract

This project implements a transcript-grounded video assistant. It extracts audio from YouTube or local media, transcribes with timestamps, indexes transcript chunks, and uses a retrieval-augmented generation pipeline to summarize videos and answer questions with source citations. Four retrieval approaches are exposed for comparative evaluation.

## Problem and objectives

Long recordings are costly to review manually. The system is designed to make summaries and question answering practical while exposing retrieval evidence and measuring answer support, relevance, latency, and compression. The intended test set spans English, Hindi/Hinglish, short, and long/noisy videos.

## System design

The implementation has distinct ingestion, timestamped chunking, retrieval, generation, evaluation, and Streamlit presentation modules. See [architecture.md](./architecture.md) for the architecture diagram and numbered data flow. API credentials are loaded from `.env`; model and retrieval configuration is centralized in `src/config.py`.

## Retrieval approaches

The baseline passes a bounded full transcript as context. BM25 provides lexical sparse retrieval. Dense retrieval uses HuggingFace embeddings and a per-transcript Chroma index. Hybrid retrieval combines sparse and dense ranks with Reciprocal Rank Fusion and reranks candidates with a cross-encoder. Each approach uses the same timestamp-bearing retrieval interface.

## Evaluation methodology

The evaluation runner builds or reads editable question/answer references with timestamp ranges. Newly generated QA references are model-produced and should be manually verified before being described as ground truth. Retrieval is scored by Recall@3, Recall@5, MRR, and hit rate. Answers are compared with reference answers using ROUGE-L and BERTScore, and Gemini judges faithfulness and relevance. Summary metrics include reference-based ROUGE-L/BERTScore when a reference summary is supplied, plus a transcript-to-summary compression ratio. System measurements include transcription duration, end-to-end query latency, estimated tokens, and cost only when explicit per-token rates are provided.

The dense-retrieval ablation evaluates chunk sizes 256, 512, and 1024 with overlaps 0 and 64. Annotated QA files should be reviewed and corrected before drawing conclusions. Error-analysis labels are heuristic and require human confirmation.

## Results

No measured values are included in this report yet. Fill in `videos.yaml`, configure `.env`, and run `python -m evaluation.run_all --videos videos.yaml`. The generated `results/results.csv`, `results/per_video.csv`, `results/ablation.csv`, charts, and `results/errors.md` provide the measurements to discuss here. `docs/results_summary.md` and the PowerPoint deck are refreshed from actual result files by the runner.

## Limitations

Whisper/Sarvam transcription quality and timestamp granularity vary by recording. Sarvam's synchronous transcription is segmented at 25-second intervals, so those Hindi/Hinglish timestamp ranges are approximate. LLM-generated QA references and LLM-judge scores can be imperfect; hand-edit references and report that limitation. BERTScore requires its model weights, and model/API latency and cost depend on local hardware, provider availability, and configured rates.
