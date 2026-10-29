import streamlit as st
import time
import csv
import html
import json
import re
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from dotenv import load_dotenv
import yaml

from src.chunking import chunk_transcript
from src.config import get_settings
from src.generator import (
    answer_question,
    extract_action_items,
    extract_key_decisions,
    extract_questions,
    generate_qa_pairs,
    generate_title,
    summarize,
)
from src.ingest import identify_source, ingest_video
from src.retrievers import create_retriever

load_dotenv()

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

/* ── Root Variables ── */
:root {
    --bg: #0a0a0f;
    --surface: #111118;
    --surface-2: #1a1a25;
    --border: #2a2a3a;
    --accent: #7c3aed;
    --accent-glow: #9f67ff;
    --accent-2: #06b6d4;
    --text: #e8e8f0;
    --text-muted: #7070a0;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
}

/* ── Global Reset ── */
html, body, [class*="css"] {
    font-family: 'JetBrains Mono', monospace;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

.stApp {
    background: var(--bg) !important;
}

/* Animated grid background */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background-image:
        linear-gradient(rgba(124, 58, 237, 0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(124, 58, 237, 0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] * {
    color: var(--text) !important;
}

/* ── Headings ── */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Syne', sans-serif !important;
    color: var(--text) !important;
}

/* ── Hero Title ── */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2rem, 5vw, 3.5rem);
    font-weight: 800;
    line-height: 1.1;
    margin: 0;
    background: linear-gradient(135deg, #ffffff 0%, var(--accent-glow) 50%, var(--accent-2) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: var(--text-muted);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-top: 0.5rem;
}

/* ── Cards ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}

.card:hover {
    border-color: var(--accent);
}

.card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient(180deg, var(--accent), var(--accent-2));
}

.card-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.card-content {
    font-size: 0.875rem;
    line-height: 1.7;
    color: var(--text);
}

/* ── Accent Badge ── */
.badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.badge-purple { background: rgba(124,58,237,0.2); color: var(--accent-glow); border: 1px solid rgba(124,58,237,0.3); }
.badge-cyan   { background: rgba(6,182,212,0.15); color: var(--accent-2);    border: 1px solid rgba(6,182,212,0.3); }
.badge-green  { background: rgba(16,185,129,0.15); color: var(--success);    border: 1px solid rgba(16,185,129,0.3); }

/* ── Input & Buttons ── */
.stTextInput > div > div > input,
.stSelectbox > div > div {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
}

.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(124,58,237,0.2) !important;
}

.stButton > button {
    background: linear-gradient(135deg, var(--accent), #5b21b6) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.875rem !important;
    letter-spacing: 0.05em !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s !important;
    text-transform: uppercase !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px rgba(124,58,237,0.4) !important;
}

/* Secondary button */
.stButton > button[kind="secondary"] {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
}

/* ── Progress / Status ── */
.status-bar {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.75rem 1rem;
    background: var(--surface-2);
    border-radius: 8px;
    margin: 0.4rem 0;
    border: 1px solid var(--border);
    font-size: 0.8rem;
}

.status-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}

.dot-active   { background: var(--accent-glow); box-shadow: 0 0 8px var(--accent-glow); animation: pulse 1.5s infinite; }
.dot-done     { background: var(--success); }
.dot-pending  { background: var(--border); }

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.4; }
}

/* ── Chat ── */
.chat-container {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem;
    max-height: 420px;
    overflow-y: auto;
    margin-bottom: 1rem;
}

.chat-msg {
    margin-bottom: 1rem;
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
}

.chat-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
}

.chat-bubble {
    display: inline-block;
    padding: 0.6rem 1rem;
    border-radius: 10px;
    font-size: 0.85rem;
    line-height: 1.6;
    max-width: 90%;
}

.user-label  { color: var(--accent-glow); }
.bot-label   { color: var(--accent-2); }

.user-bubble { background: rgba(124,58,237,0.15); border: 1px solid rgba(124,58,237,0.25); align-self: flex-end; }
.bot-bubble  { background: rgba(6,182,212,0.1);  border: 1px solid rgba(6,182,212,0.2);   align-self: flex-start; }

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 1.5rem 0 !important;
}

/* ── Transcript box ── */
.transcript-box {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.25rem;
    font-size: 0.82rem;
    line-height: 1.8;
    max-height: 300px;
    overflow-y: auto;
    color: var(--text-muted);
    white-space: pre-wrap;
    word-break: break-word;
}

/* ── Stale Streamlit elements ── */
.stProgress > div > div > div { background: var(--accent) !important; }
.stSpinner > div { border-top-color: var(--accent) !important; }
[data-testid="stMarkdownContainer"] p { color: var(--text) !important; }
label { color: var(--text-muted) !important; font-size: 0.8rem !important; }

/* scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }
</style>
""", unsafe_allow_html=True)

# ─── Session State Init ──────────────────────────────────────────────────────────
for key, default in {
    "result": None,
    "chat_history": [],
    "processing": False,
    "pipeline_done": False,
    "pipeline_steps": {},
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─── Helpers ────────────────────────────────────────────────────────────────────
def step_status(steps: dict, key: str) -> str:
    s = steps.get(key, "pending")
    if s == "active":  return "dot-active"
    if s == "done":    return "dot-done"
    return "dot-pending"

def render_step_bar(label: str, key: str, icon: str):
    css = step_status(st.session_state.pipeline_steps, key)
    st.markdown(f"""
    <div class="status-bar">
        <div class="status-dot {css}"></div>
        <span>{icon} {label}</span>
    </div>""", unsafe_allow_html=True)

# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="hero-title" style="font-size:1.6rem">🎬 AI<br>Video</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Meeting Intelligence</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<span class="badge badge-purple">Input</span>', unsafe_allow_html=True)
    source = st.text_input("YouTube URL or File Path", placeholder="https://youtube.com/watch?v=... or /path/to/file.mp4")

    language = st.selectbox("Language", ["auto", "english", "hindi", "hinglish"], index=0)
    strategy = st.selectbox(
        "Retrieval strategy",
        ["baseline", "sparse", "dense", "hybrid"],
        index=2,
        format_func=lambda value: {
            "baseline": "Baseline (no retrieval)",
            "sparse": "Sparse (BM25)",
            "dense": "Dense (Chroma)",
            "hybrid": "Hybrid (RRF + reranker)",
        }[value],
        key="retrieval_strategy",
    )
    page = st.radio("View", ["Analysis", "Test Cases"], horizontal=True)

    run_btn = st.button("⚡  Analyse", use_container_width=True)

    if st.session_state.pipeline_done:
        st.markdown("---")
        st.markdown('<span class="badge badge-green">Pipeline Status</span>', unsafe_allow_html=True)
        for step, icon, label in [
            ("audio",      "🔊", "Audio Processing"),
            ("transcript", "📝", "Transcription"),
            ("title",      "🏷️", "Title Generation"),
            ("summary",    "📋", "Summarisation"),
            ("extract",    "🔍", "Extraction"),
            ("rag",        "🧠", "RAG Engine"),
        ]:
            render_step_bar(label, step, icon)

# ─── Main Area ──────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">AI Video Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Transcribe · Summarise · Chat with your meetings</div>', unsafe_allow_html=True)
st.markdown("---")

# ── Run Pipeline ────────────────────────────────────────────────────────────────
if run_btn:
    if not source.strip():
        st.error("Please enter a YouTube URL or file path.")
    else:
        st.session_state.pipeline_done = False
        st.session_state.result = None
        st.session_state.chat_history = []
        st.session_state.pipeline_steps = {}

        progress_placeholder = st.empty()

        def update_step(key, state):
            st.session_state.pipeline_steps[key] = state

        try:
            with progress_placeholder.container():
                st.info("⚙️ Pipeline running — see sidebar for live status…")

            update_step("audio", "active")
            ingestion = ingest_video(source, language)
            update_step("audio", "done")

            update_step("transcript", "active")
            transcript = ingestion.transcript
            update_step("transcript", "done")

            update_step("title", "active")
            title = generate_title(transcript)
            update_step("title", "done")

            update_step("summary", "active")
            summary = summarize(transcript)
            update_step("summary", "done")

            update_step("extract", "active")
            action_items  = extract_action_items(transcript)
            decisions     = extract_key_decisions(transcript)
            questions     = extract_questions(transcript)
            update_step("extract", "done")

            chunks = chunk_transcript(
                ingestion.segments,
                chunk_size=get_settings().chunk_size,
                overlap=get_settings().chunk_overlap,
            )
            update_step("rag", "active")
            retriever = create_retriever(
                strategy,
                chunks,
                ingestion.video_id,
                cache_dir=str(get_settings().cache_dir),
            )
            update_step("rag", "done")

            st.session_state.result = {
                "title": title,
                "transcript": transcript,
                "timestamped_transcript": "\n".join(
                    f"[{segment.start:.1f}s–{segment.end:.1f}s] {segment.text}"
                    for segment in ingestion.segments
                ),
                "summary": summary,
                "action_items": action_items,
                "key_decisions": decisions,
                "open_questions": questions,
                "chunks": chunks,
                "video_id": ingestion.video_id,
                "source_url": ingestion.source_url,
                "transcription_seconds": ingestion.transcription_seconds,
                "retrievers": {strategy: retriever},
            }
            st.session_state.pipeline_done = True
            progress_placeholder.success("✅ Analysis complete!")
            time.sleep(0.5)
            progress_placeholder.empty()
            st.rerun()

        except Exception as e:
            for k in ["audio","transcript","title","summary","extract","rag"]:
                if st.session_state.pipeline_steps.get(k) == "active":
                    st.session_state.pipeline_steps[k] = "pending"
            progress_placeholder.error(f"❌ Error: {e}")

# ── Results ──────────────────────────────────────────────────────────────────────
if st.session_state.result and page == "Analysis":
    r = st.session_state.result
    st.caption(f"Transcription completed in {r['transcription_seconds']:.2f} seconds.")

    # Title banner
    st.markdown(f"""
    <div class="card">
        <div class="card-title">📌 Session Title</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:700;color:var(--text)">
            {html.escape(r['title'])}
        </div>
    </div>""", unsafe_allow_html=True)

    # Top row: summary + transcript
    col1, col2 = st.columns([3, 2], gap="medium")

    with col1:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">📋 Summary</div>
            <div class="card-content">{html.escape(r['summary']).replace(chr(10), '<br>')}</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        with st.expander("📝 Full Transcript", expanded=False):
            st.markdown(
                f'<div class="transcript-box">{html.escape(r["timestamped_transcript"])}</div>',
                unsafe_allow_html=True,
            )

    # Second row: action items | decisions | questions
    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">✅ Action Items</div>
            <div class="card-content">{html.escape(r['action_items']).replace(chr(10), '<br>')}</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">🔑 Key Decisions</div>
            <div class="card-content">{html.escape(r['key_decisions']).replace(chr(10), '<br>')}</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">❓ Open Questions</div>
            <div class="card-content">{html.escape(r['open_questions']).replace(chr(10), '<br>')}</div>
        </div>""", unsafe_allow_html=True)

    if "export_qa" not in r and st.button("Generate timestamped Q&A for export"):
        try:
            qa_text = generate_qa_pairs(r["transcript"], count=15).strip()
            if qa_text.startswith("```"):
                qa_text = re.sub(
                    r"^```(?:json)?\s*|\s*```$",
                    "",
                    qa_text,
                    flags=re.IGNORECASE,
                )
            generated_qa = json.loads(qa_text)
            if not isinstance(generated_qa, list) or not 15 <= len(generated_qa) <= 20:
                raise ValueError("Gemini must return between 15 and 20 Q&A entries")
            for item in generated_qa:
                if (
                    not isinstance(item, dict)
                    or not all(key in item for key in ("question", "answer", "start", "end"))
                    or not str(item.get("question", "")).strip()
                    or not str(item.get("answer", "")).strip()
                ):
                    raise ValueError("Gemini returned a Q&A entry with missing fields")
                if float(item["start"]) < 0 or float(item["end"]) < float(item["start"]):
                    raise ValueError("Gemini returned an invalid Q&A timestamp")
            r["export_qa"] = generated_qa
        except (RuntimeError, TypeError, ValueError, json.JSONDecodeError) as error:
            st.error(f"Could not generate export Q&A: {error}")
    qa_export = "\n".join(
        f"### {index}. {item['question']}\n\n"
        f"**Answer:** {item['answer']}  \n"
        f"**Timestamp:** {float(item['start']):.1f}s–{float(item['end']):.1f}s"
        for index, item in enumerate(r.get("export_qa", []), 1)
    ) or "Generate timestamped Q&A in the app before exporting to include Q&A."
    markdown_export = (
        f"# {r['title']}\n\n## Summary\n{r['summary']}\n\n"
        f"## Key points\n### Action items\n{r['action_items']}\n\n"
        f"### Key decisions\n{r['key_decisions']}\n\n"
        f"### Open questions\n{r['open_questions']}\n\n## Q&A\n{qa_export}\n\n"
        f"## Timestamped transcript\n{r['timestamped_transcript']}\n"
    )
    st.download_button(
        "Download Markdown report",
        markdown_export,
        file_name=f"{r['video_id']}_summary.md",
        mime="text/markdown",
    )

    st.markdown("---")

    # ── RAG Chat ──────────────────────────────────────────────────────────────
    st.markdown('<div style="font-family:\'Syne\',sans-serif;font-size:1.2rem;font-weight:700;margin-bottom:1rem">💬 Chat with your Meeting</div>', unsafe_allow_html=True)

    # Chat history display
    if st.session_state.chat_history:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                for source_item in msg.get("sources", []):
                    if source_item.get("url"):
                        st.markdown(
                            f"[Source {source_item['start']:.0f}s–{source_item['end']:.0f}s]"
                            f"({source_item['url']})"
                        )
                    else:
                        st.caption(
                            f"Source: {source_item['start']:.0f}s–{source_item['end']:.0f}s"
                        )
    else:
        st.markdown("""
        <div class="card" style="text-align:center;padding:2rem">
            <div style="font-size:2rem;margin-bottom:0.5rem">💬</div>
            <div style="color:var(--text-muted);font-size:0.85rem">Ask anything about your meeting transcript</div>
        </div>""", unsafe_allow_html=True)

    # Chat input
    chat_col1, chat_col2 = st.columns([5, 1], gap="small")
    with chat_col1:
        user_input = st.text_input("Your question", placeholder="What were the main decisions made?", label_visibility="collapsed")
    with chat_col2:
        send_btn = st.button("Send →", use_container_width=True)

    if send_btn and user_input.strip():
        with st.spinner("Thinking…"):
            try:
                active_strategy = st.session_state.retrieval_strategy
                retriever = r["retrievers"].get(active_strategy)
                if retriever is None:
                    retriever = create_retriever(
                        active_strategy,
                        r["chunks"],
                        r["video_id"],
                        cache_dir=str(get_settings().cache_dir),
                    )
                    r["retrievers"][active_strategy] = retriever
                retrieved = retriever.retrieve(user_input.strip(), get_settings().retrieval_k)
                if (
                    not retrieved
                    or max(chunk.score for chunk in retrieved)
                    < get_settings().retrieval_score_threshold
                ):
                    answer = "I could not find this in the video."
                else:
                    answer = answer_question(user_input.strip(), retrieved)
                sources = []
                for chunk in retrieved:
                    timestamp_url = None
                    if r["source_url"]:
                        parts = urlsplit(r["source_url"])
                        query_parts = [
                            (key, value)
                            for key, value in parse_qsl(parts.query)
                            if key != "t"
                        ]
                        query_parts.append(("t", str(int(chunk.start))))
                        timestamp_url = urlunsplit(
                            (
                                parts.scheme,
                                parts.netloc,
                                parts.path,
                                urlencode(query_parts),
                                parts.fragment,
                            )
                        )
                    sources.append(
                        {
                            "start": chunk.start,
                            "end": chunk.end,
                            "url": timestamp_url,
                        }
                    )
            except Exception as error:
                st.error(f"Could not answer the question: {error}")
                sources = []
                answer = "The question could not be processed."
        st.session_state.chat_history.append({"role": "user",      "content": user_input.strip()})
        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
        st.rerun()

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()

elif page == "Analysis":
    # Empty state
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding:5rem 2rem;text-align:center">
        <div style="font-size:4rem;margin-bottom:1rem">🎬</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:700;color:var(--text);margin-bottom:0.5rem">
            Ready to Analyse
        </div>
        <div style="color:var(--text-muted);font-size:0.85rem;max-width:380px;line-height:1.7">
            Paste a YouTube URL or local file path in the sidebar, choose your language, and hit <strong>Analyse</strong> to get started.
        </div>
        <div style="margin-top:2rem;display:flex;gap:1rem;flex-wrap:wrap;justify-content:center">
            <span class="badge badge-purple">Transcription</span>
            <span class="badge badge-cyan">Summarisation</span>
            <span class="badge badge-green">RAG Chat</span>
        </div>
    </div>""", unsafe_allow_html=True)

if page == "Test Cases":
    st.header("Evaluation test cases")
    manifest_path = Path("videos.yaml")
    if not manifest_path.is_file():
        st.info("Create videos.yaml and add the four test-video sources to view test cases.")
    else:
        try:
            manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
            videos = manifest.get("videos", [])
            summary_path = Path("results/test_case_summaries.json")
            summaries = (
                json.loads(summary_path.read_text(encoding="utf-8"))
                if summary_path.is_file()
                else {}
            )
            metrics_path = Path("results/per_video.csv")
            metrics = (
                list(csv.DictReader(metrics_path.open(encoding="utf-8")))
                if metrics_path.is_file()
                else []
            )
            for video in videos:
                source = str(video.get("url") or video.get("path") or "")
                video_id = str(video.get("id") or "")
                try:
                    if source and "FILL" not in source.upper():
                        video_id, _ = identify_source(source)
                except (ValueError, FileNotFoundError):
                    pass
                st.subheader(video.get("name", video_id or "Unconfigured video"))
                record = summaries.get(video_id, {})
                if record:
                    st.markdown(record.get("summary", ""))
                    for index, sample in enumerate(record.get("sample_qas", [])[:3], 1):
                        with st.expander(f"Sample Q&A {index}: {sample['question']}"):
                            st.markdown(f"**Answer:** {sample['answer']}")
                            for chunk in sample.get("retrieved_chunks", []):
                                start = float(chunk.get("start", 0))
                                end = float(chunk.get("end", 0))
                                if source and source.startswith(("https://", "http://")):
                                    parts = urlsplit(source)
                                    query_items = [
                                        (key, value)
                                        for key, value in parse_qsl(parts.query)
                                        if key != "t"
                                    ]
                                    query_items.append(("t", str(int(start))))
                                    link = urlunsplit(
                                        (
                                            parts.scheme,
                                            parts.netloc,
                                            parts.path,
                                            urlencode(query_items),
                                            parts.fragment,
                                        )
                                    )
                                    st.markdown(f"[{start:.0f}s–{end:.0f}s]({link})")
                                else:
                                    st.caption(f"Retrieved at {start:.0f}s–{end:.0f}s")
                                st.write(chunk.get("text", ""))
                else:
                    st.info("Run the evaluation to populate summary and sample answers.")

                video_metrics = [row for row in metrics if row.get("video") == video_id]
                if video_metrics:
                    st.dataframe(video_metrics, use_container_width=True, hide_index=True)
                else:
                    st.caption("No measured metrics for this video yet.")
                st.markdown("---")
        except (OSError, yaml.YAMLError, json.JSONDecodeError, AttributeError) as error:
            st.error(f"Could not load evaluation test cases: {error}")