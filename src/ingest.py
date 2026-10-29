"""Audio acquisition and timestamped transcription."""

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
from urllib.parse import parse_qs, urlparse

from src.chunking import TranscriptSegment
from src.config import get_settings

_WHISPER_MODEL = None


@dataclass(frozen=True)
class IngestionResult:
    """Transcription output and timing details for one input video."""

    video_id: str
    segments: list[TranscriptSegment]
    transcript: str
    audio_path: str
    transcription_seconds: float
    source_url: str | None = None


def identify_source(source: str) -> tuple[str, str | None]:
    """Validate an input and derive a stable, filesystem-safe video identifier."""
    candidate = source.strip()
    if not candidate:
        raise ValueError("Provide a YouTube URL or a local audio/video file.")
    if re.match(r"^[A-Za-z]:[\\/]", candidate):
        parsed = urlparse("")
    else:
        parsed = urlparse(candidate)
    if parsed.scheme:
        if parsed.scheme not in {"http", "https"} or (
            parsed.hostname or ""
        ).lower() not in {
            "youtube.com",
            "www.youtube.com",
            "m.youtube.com",
            "youtu.be",
        }:
            raise ValueError("Only YouTube URLs are supported.")
        video_id = parse_qs(parsed.query).get("v", [""])[0]
        if not video_id and parsed.path.startswith(("/shorts/", "/embed/", "/live/")):
            video_id = parsed.path.split("/")[2]
        if not video_id and (parsed.hostname or "").lower() == "youtu.be":
            video_id = parsed.path.strip("/").split("/")[0]
        if not re.fullmatch(r"[A-Za-z0-9_-]{6,}", video_id):
            raise ValueError("The YouTube URL does not contain a valid video ID.")
        return video_id, candidate
    file_path = Path(candidate)
    if not file_path.is_file():
        raise FileNotFoundError(f"Audio/video file does not exist: {file_path}")
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "-", file_path.stem).strip("-")
    digest = hashlib.sha256(str(file_path.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"{safe_name[:40]}-{digest}", None


def _cache_path(video_id: str) -> Path:
    return get_settings().cache_dir / video_id


def _transcribe_whisper(audio_path: str) -> tuple[list[TranscriptSegment], str | None]:
    # Load Anaconda's MKL runtime before Torch's bundled OpenMP DLL.
    import numpy
    import whisper

    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        _WHISPER_MODEL = whisper.load_model(get_settings().whisper_model)
    model = _WHISPER_MODEL
    result = model.transcribe(audio_path, task="transcribe", verbose=False)
    detected = result.get("language")
    segments = [
        TranscriptSegment(float(item["start"]), float(item["end"]), item["text"].strip())
        for item in result.get("segments", [])
        if item.get("text", "").strip()
    ]
    return segments, detected


def _detect_whisper_language(audio_path: str) -> str:
    """Detect language from the first 30 seconds without decoding the full video."""
    import numpy
    import whisper

    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        _WHISPER_MODEL = whisper.load_model(get_settings().whisper_model)
    model = _WHISPER_MODEL
    audio = whisper.load_audio(audio_path)
    sample = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(sample, n_mels=model.dims.n_mels).to(model.device)
    _, probabilities = model.detect_language(mel)
    return max(probabilities, key=probabilities.get)


def _transcribe_sarvam(audio_path: str) -> list[TranscriptSegment]:
    import requests
    from pydub import AudioSegment

    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        raise RuntimeError("SARVAM_API_KEY is missing; configure it in .env")

    audio = AudioSegment.from_file(audio_path)
    piece_ms = get_settings().sarvam_piece_seconds * 1000
    endpoint = "https://api.sarvam.ai/speech-to-text-translate"
    segments: list[TranscriptSegment] = []
    for start_ms in range(0, len(audio), piece_ms):
        temp_handle, temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(temp_handle)
        piece_path = Path(temp_path)
        audio[start_ms : start_ms + piece_ms].export(piece_path, format="wav")
        try:
            last_error: Exception | None = None
            for attempt in range(get_settings().retry_attempts):
                try:
                    with piece_path.open("rb") as audio_file:
                        response = requests.post(
                            endpoint,
                            headers={"api-subscription-key": api_key},
                            files={"file": (piece_path.name, audio_file, "audio/wav")},
                            data={
                                "model": os.getenv("SARVAM_STT_MODEL", "saaras:v2.5"),
                                "with_diarization": "false",
                            },
                            timeout=120,
                        )
                    response.raise_for_status()
                    text = response.json().get("transcript", "").strip()
                    if text:
                        start_seconds = start_ms / 1000
                        end_seconds = min(len(audio), start_ms + piece_ms) / 1000
                        segments.append(
                            TranscriptSegment(start_seconds, end_seconds, text)
                        )
                    last_error = None
                    break
                except (requests.RequestException, ValueError) as exc:
                    last_error = exc
                    if attempt + 1 < get_settings().retry_attempts:
                        time.sleep(min(2**attempt, 8))
            if last_error:
                raise RuntimeError("Sarvam transcription failed after retries") from last_error
        finally:
            piece_path.unlink(missing_ok=True)
    return segments


def _load_cached(cache_file: Path) -> IngestionResult | None:
    if not cache_file.is_file():
        return None
    with cache_file.open(encoding="utf-8") as file:
        cached = json.load(file)
    return IngestionResult(
        video_id=cached["video_id"],
        segments=[TranscriptSegment(**segment) for segment in cached["segments"]],
        transcript=cached["transcript"],
        audio_path=cached["audio_path"],
        transcription_seconds=float(cached["transcription_seconds"]),
        source_url=cached.get("source_url"),
    )


def ingest_video(source: str, language: str = "auto") -> IngestionResult:
    """Acquire audio, detect language, transcribe with timestamps, and cache output."""
    video_id, source_url = identify_source(source)
    cache_dir = _cache_path(video_id)
    cache_file = cache_dir / f"transcript_{language.lower()}.json"
    cached = _load_cached(cache_file)
    if cached:
        return cached

    cache_dir.mkdir(parents=True, exist_ok=True)
    from utils.audio_processor import process_input

    audio_chunks = process_input(source)
    if not audio_chunks:
        raise RuntimeError("No audio could be extracted from the provided input.")
    started = time.perf_counter()
    segments: list[TranscriptSegment] = []
    elapsed_audio = 0.0
    effective_language = language.lower()
    audio_path = audio_chunks[0]
    for chunk_path in audio_chunks:
        if effective_language in {"hindi", "hinglish"}:
            local_segments = _transcribe_sarvam(chunk_path)
        elif effective_language == "auto" and _detect_whisper_language(chunk_path) == "hi":
            effective_language = "hindi"
            local_segments = _transcribe_sarvam(chunk_path)
        else:
            local_segments, _ = _transcribe_whisper(chunk_path)
        segments.extend(
            TranscriptSegment(
                start=item.start + elapsed_audio,
                end=item.end + elapsed_audio,
                text=item.text,
            )
            for item in local_segments
        )
        from pydub import AudioSegment

        elapsed_audio += len(AudioSegment.from_file(chunk_path)) / 1000

    transcript = " ".join(segment.text for segment in segments).strip()
    if not transcript:
        raise RuntimeError("Transcription produced an empty transcript.")
    result = IngestionResult(
        video_id=video_id,
        segments=segments,
        transcript=transcript,
        audio_path=audio_path,
        transcription_seconds=time.perf_counter() - started,
        source_url=source_url,
    )
    cache_file.write_text(
        json.dumps(
            {
                **asdict(result),
                "segments": [asdict(segment) for segment in segments],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return result
