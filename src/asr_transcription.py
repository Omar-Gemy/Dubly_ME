"""
asr_transcription.py — Phase C: ASR Layer
==========================================
Offline speech-to-text transcription using faster-whisper
(CTranslate2 backend).  No third-party APIs are used.

Pipeline:
  1. Load the normalised audio and the VAD-generated segments.json
  2. Load language registry and resolve source language config
  3. For each segment, slice the audio by start_time / end_time
  4. Pass each audio slice to faster-whisper with language forcing
  5. Populate the ``text`` field in every segment
  6. Write the enriched data to artifacts/transcripts.json

Usage:
  python src/asr_transcription.py
  python src/asr_transcription.py --model small --source-lang ar
"""

import argparse
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

# ──────────────────────────────────────────────
#  Project paths (relative to repo root)
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_AUDIO = PROJECT_ROOT / "data" / "audio_out" / "_temp_normalised.wav"
DEFAULT_SEGMENTS = ARTIFACTS_DIR / "segments.json"
DEFAULT_OUTPUT = ARTIFACTS_DIR / "transcripts.json"
LANGUAGE_REGISTRY_PATH = PROJECT_ROOT / "config" / "language_registry.json"


# ──────────────────────────────────────────────
#  Language Registry — dynamic config loader
# ──────────────────────────────────────────────
REQUIRED_LANG_KEYS = {"name", "whisper_language", "initial_prompt", "hallucination_patterns"}

# Fallback config used when the detected/forced language is not in the registry.
FALLBACK_LANGUAGE_CONFIG = {
    "name": "Unknown",
    "whisper_language": None,  # will be set to the detected code at runtime
    "initial_prompt": "",
    "hallucination_patterns": [],
}


def load_language_registry(registry_path: Path = LANGUAGE_REGISTRY_PATH) -> dict:
    """
    Load and validate the language registry JSON file.

    Each language entry must contain: name, whisper_language,
    initial_prompt, and hallucination_patterns.

    Returns the parsed registry dict keyed by language code.
    Raises SystemExit on I/O or validation errors.
    """
    if not registry_path.is_file():
        print(f"✖  Language registry not found: {registry_path}")
        sys.exit(1)

    with open(registry_path, "r", encoding="utf-8") as fh:
        registry = json.load(fh)

    # Validate every language entry
    for lang_code, config in registry.items():
        missing = REQUIRED_LANG_KEYS - set(config.keys())
        if missing:
            print(
                f"✖  Language registry: '{lang_code}' is missing "
                f"required key(s): {', '.join(sorted(missing))}"
            )
            sys.exit(1)

    return registry


def resolve_language_config(
    source_lang_cli: str,
    segments_data: dict,
    audio_data: np.ndarray,
    sample_rate: int,
    model: "WhisperModel",
    registry: dict,
) -> tuple[str, dict, list[str]]:
    """
    Determine the locked source language and return its registry config.

    Resolution cascade:
      1. If CLI --source-lang is explicit and not 'auto', use it.
      2. Else if segments.json contains source_language and it is not 'auto', use it.
      3. Else run Whisper language detection on the first 30 s.

    Returns:
      (language_code, language_config_dict, warnings_list)
    """
    warnings = []

    # ── Tier 1: explicit CLI ──────────────────
    if source_lang_cli and source_lang_cli != "auto":
        lang_code = source_lang_cli
        print(f"  ✔ Language forced via CLI: {lang_code}")

    # ── Tier 2: segments.json metadata ────────
    elif segments_data.get("source_language") and segments_data["source_language"] != "auto":
        lang_code = segments_data["source_language"]
        print(f"  ✔ Language from segments.json: {lang_code}")

    # ── Tier 3: auto-detection ────────────────
    else:
        print("  Detecting source language from first 30s …")
        lang_code = detect_language(audio_data, sample_rate, model)
        print(f"  ✔ Detected language: {lang_code}")

    # ── Look up in registry ───────────────────
    if lang_code in registry:
        lang_config = registry[lang_code]
        print(f"  ✔ Registry config loaded: {lang_config['name']}")
    else:
        fallback = dict(FALLBACK_LANGUAGE_CONFIG)
        fallback["whisper_language"] = lang_code
        lang_config = fallback
        warn_msg = (
            f"Language '{lang_code}' not found in registry — "
            f"using fallback config (no initial_prompt, no hallucination patterns)"
        )
        warnings.append(warn_msg)
        print(f"  ⚠ {warn_msg}")

    return lang_code, lang_config, warnings


# ──────────────────────────────────────────────
#  Step 1 — Load inputs
# ──────────────────────────────────────────────
def load_segments(segments_path: str) -> dict:
    """Read the VAD-generated segments.json data contract."""
    with open(segments_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_audio(audio_path: str) -> tuple[np.ndarray, int]:
    """
    Read the full audio file and return (samples, sample_rate).
    Uses soundfile for Windows compatibility.
    """
    data, sr = sf.read(audio_path, dtype="float32")
    # Ensure mono
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, sr


# ──────────────────────────────────────────────
#  Anti-hallucination: post-transcription filter
# ──────────────────────────────────────────────

def is_hallucination(text: str, hallucination_patterns: list[str]) -> bool:
    """
    Return True if *text* looks like a known Whisper hallucination.

    Checks for:
      1. Known hallucinated phrases (substring match) from the
         language-specific registry patterns.
      2. Excessive repetition (same trigram repeated ≥3 times)
    """
    stripped = text.strip()
    if not stripped:
        return False

    # Check against language-specific hallucination phrases
    for pattern in hallucination_patterns:
        if pattern in stripped:
            return True

    # Detect repetition: if any trigram appears ≥3 times
    words = stripped.split()
    if len(words) >= 6:
        trigrams = [" ".join(words[i:i + 3]) for i in range(len(words) - 2)]
        for tri in set(trigrams):
            if trigrams.count(tri) >= 3:
                return True

    return False


# ──────────────────────────────────────────────
#  Step 2 — Slice audio by VAD timestamps
# ──────────────────────────────────────────────
def slice_audio(
    audio_data: np.ndarray,
    sample_rate: int,
    start_time: float,
    end_time: float,
) -> np.ndarray:
    """
    Extract a segment from the audio array using sample-accurate
    start/end times (in seconds).
    """
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)
    # Clamp to valid range
    start_sample = max(0, start_sample)
    end_sample = min(len(audio_data), end_sample)
    return audio_data[start_sample:end_sample]


# ──────────────────────────────────────────────
#  Step 2b — Auto-detect source language
# ──────────────────────────────────────────────
LANGID_DURATION_SEC = 30  # seconds of audio used for language detection


def detect_language(
    audio_data: np.ndarray,
    sample_rate: int,
    model: WhisperModel,
) -> str:
    """
    Feed the first 30 seconds of the full audio to Whisper and
    return the detected language code (e.g. "ar", "en").

    This is a lightweight probe — we only need the ``info`` object,
    not the full transcription.
    """
    max_samples = int(LANGID_DURATION_SEC * sample_rate)
    probe_chunk = audio_data[:max_samples]

    # Run a minimal transcribe just to get the detected language
    _segments, info = model.transcribe(
        probe_chunk,
        vad_filter=False,
        without_timestamps=True,
    )
    # Consume the generator so CTranslate2 releases resources
    for _ in _segments:
        pass

    return info.language


# ──────────────────────────────────────────────
#  Step 3 — Transcribe each segment
# ──────────────────────────────────────────────
def transcribe_segments(
    audio_data: np.ndarray,
    sample_rate: int,
    segments_data: dict,
    model_size: str = "large-v3",
    device: str = "cuda",
    compute_type: str = "float16",
    source_lang: str = "auto",
    rms_gate_db: float = -35.0,
    min_segment_sec: float = 0.5,
) -> dict:
    """
    For each segment in *segments_data*, slice the audio and run
    faster-whisper transcription.  Whisper's internal VAD is disabled
    so we rely strictly on our Silero-based timestamps.

    Language config (initial_prompt, hallucination_patterns) is loaded
    dynamically from config/language_registry.json. If the resolved
    language is not in the registry, a fallback config is used with
    empty prompt/patterns and a warning is recorded in the output.

    Returns a new dict with populated ``text`` fields, suitable for
    writing to ``transcripts.json``.
    """
    print(f"  Loading faster-whisper model '{model_size}' …")
    model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
    )
    print("  ✔ Model loaded.\n")

    # ── Load language registry ────────────────
    registry = load_language_registry()

    # ── Resolve language config (3-tier cascade) ──
    detected_lang, lang_config, lang_warnings = resolve_language_config(
        source_lang_cli=source_lang,
        segments_data=segments_data,
        audio_data=audio_data,
        sample_rate=sample_rate,
        model=model,
        registry=registry,
    )
    print()

    # Extract language-specific settings from registry config
    initial_prompt = lang_config["initial_prompt"]
    hallucination_patterns = lang_config["hallucination_patterns"]
    whisper_language = lang_config["whisper_language"] or detected_lang

    # Store in the data contract for downstream stages
    segments_data["source_language"] = detected_lang
    segments_data["language_config"] = {
        "name": lang_config["name"],
        "registry_language": detected_lang,
        "used_fallback": detected_lang not in registry,
    }
    if lang_warnings:
        segments_data["language_config"]["warnings"] = lang_warnings

    segments = segments_data["segments"]
    total = len(segments)

    for idx, seg in enumerate(segments, start=1):
        start = seg["start_time"]
        end = seg["end_time"]
        duration = seg["duration"]

        print(
            f"  [{idx}/{total}]  Segment #{seg['segment_id']}  "
            f"({start:.3f}s → {end:.3f}s, {duration:.3f}s) … ",
            end="",
            flush=True,
        )

        # Slice the audio chunk for this VAD segment
        chunk = slice_audio(audio_data, sample_rate, start, end)

        if len(chunk) == 0:
            seg["text"] = ""
            print("⚠ empty slice, skipped.")
            continue

        # ── Duration guard ───────────────────────────
        # Sub-segments shorter than min_segment_sec (default 0.5s)
        # have a ~73% hallucination rate after diarization splitting.
        # Skip them entirely to prevent prompt-leakage and
        # training-data echo artefacts.
        if duration < min_segment_sec:
            seg["text"] = ""
            seg["_skipped_too_short"] = True
            print(f"⚠ skipped (duration {duration:.3f}s < {min_segment_sec}s)")
            continue

        # ── RMS energy gate ──────────────────────────
        # Skip ASR on segments with near-silence audio to prevent
        # Whisper from hallucinating fluent text on noise-floor input.
        rms = np.sqrt(np.mean(chunk ** 2))
        rms_db = 20 * np.log10(rms) if rms > 0 else -120.0
        seg["_rms_dbfs"] = round(rms_db, 1)

        if rms_db < rms_gate_db:
            seg["text"] = ""
            seg["_skipped_low_energy"] = True
            print(f"⚠ skipped (RMS {rms_db:.1f} dBFS < {rms_gate_db} dBFS)")
            continue

        # faster-whisper can accept a numpy array directly,
        # but it must be float32 mono at the model's expected rate (16 kHz).
        # If our audio is already 16 kHz mono (from ingestion), we can
        # pass it directly.  Otherwise, write a temp WAV.
        if sample_rate == 16000:
            audio_input = chunk
        else:
            # Write a temporary WAV file for non-16 kHz audio
            tmp = tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False
            )
            try:
                sf.write(tmp.name, chunk, sample_rate, subtype="PCM_16")
                audio_input = tmp.name
            finally:
                tmp.close()

        # ── Transcribe with full anti-hallucination guards ──
        transcribe_kwargs = dict(
            language=whisper_language,
            vad_filter=False,
            beam_size=8,
            temperature=[0.0, 0.2, 0.4],

            # ── Anti-hallucination parameters ──────────────
            condition_on_previous_text=False,       # prevent cascading hallucination chains
            no_speech_threshold=0.5,                # stricter silence detection (default 0.6)
            compression_ratio_threshold=1.6,        # reject repetitive outputs (default 2.4, prev 1.8)
            log_prob_threshold=-0.5,                # reject low-confidence text (default -1.0)
            repetition_penalty=1.2,                 # penalise repeated tokens in beam search

            # ── Word-level hallucination guard ─────────────
            word_timestamps=True,                   # required for hallucination_silence_threshold
            hallucination_silence_threshold=1.5,    # skip hallucinated text in silent gaps >1.5s
        )

        # ── Dynamic dialect-anchoring prompt ──────────────
        # Loaded from language registry instead of hardcoded.
        if initial_prompt:
            transcribe_kwargs["initial_prompt"] = initial_prompt

        whisper_segments, info = model.transcribe(audio_input, **transcribe_kwargs)

        # Collect the text from Whisper's output segments
        text_parts = []
        for ws in whisper_segments:
            text_parts.append(ws.text.strip())

        full_text = " ".join(text_parts).strip()

        # ── Post-transcription hallucination filter (energy-gated) ──
        if is_hallucination(full_text, hallucination_patterns):
            if rms_db < -30.0:
                # Low-energy audio + pattern match → confident hallucination
                seg["text"] = ""
                seg["_hallucination_filtered"] = full_text
                # Clean up temp file if we created one
                if sample_rate != 16000 and isinstance(audio_input, str):
                    os.unlink(audio_input)
                print(f"⚠ hallucination filtered: \"{full_text[:50]}…\"")
                continue
            else:
                # High-energy audio + pattern match → flag for review,
                # keep the text (may be a false positive like segment 25)
                seg["_hallucination_suspect"] = True
                seg["_hallucination_matched_pattern"] = full_text

        seg["text"] = full_text

        # Clean up temp file if we created one
        if sample_rate != 16000 and isinstance(audio_input, str):
            os.unlink(audio_input)

        # Truncate for display
        preview = full_text[:60] + "…" if len(full_text) > 60 else full_text
        suspect_tag = " ⚠suspect" if seg.get("_hallucination_suspect") else ""
        print(f"✔ \"{preview}\"{suspect_tag}")

    # ── Cross-segment duplicate detection ─────────
    # Catch prompt-leakage and repetition loops that span multiple segments.
    seen_texts = {}
    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        if text in seen_texts:
            seg["_duplicate_of_segment"] = seen_texts[text]
            seg["_hallucination_suspect"] = True
        else:
            seen_texts[text] = seg["segment_id"]

    return segments_data


# ──────────────────────────────────────────────
#  Step 4 — Save the enriched data contract
# ──────────────────────────────────────────────
def save_transcripts(data: dict, output_path: str) -> None:
    """Write the transcribed segments to a JSON file (UTF-8, pretty-printed)."""
    # Add ASR metadata
    data["asr_completed_at"] = datetime.now(timezone.utc).isoformat()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        def np_encoder(obj):
            if isinstance(obj, np.generic):
                return obj.item()
            raise TypeError

        json.dump(data, fh, indent=2, ensure_ascii=False, default=np_encoder)


# ──────────────────────────────────────────────
#  CLI entry-point
# ──────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dubly ME — Phase C: ASR Transcription (faster-whisper)",
    )
    parser.add_argument(
        "--input-audio",
        default=str(DEFAULT_AUDIO),
        help="Path to the normalised WAV file  (default: data/audio_out/_temp_normalised.wav)",
    )
    parser.add_argument(
        "--input-segments",
        default=str(DEFAULT_SEGMENTS),
        help="Path to the VAD segments JSON  (default: artifacts/segments.json)",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output JSON path  (default: artifacts/transcripts.json)",
    )
    parser.add_argument(
        "--model",
        default="large-v3",
        choices=["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo", "distil-large-v3"],
        help="Whisper model size  (default: large-v3)",
    )
    parser.add_argument(
        "--source-lang",
        default="auto",
        help="Source language code (e.g. 'ar', 'en') or 'auto' for detection  (default: auto)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Compute device: 'cpu', 'cuda', or 'auto'  (default: cuda)",
    )
    parser.add_argument(
        "--compute-type",
        default="float16",
        help="CTranslate2 compute type: 'int8', 'float16', 'float32', or 'auto'  (default: float16)",
    )
    parser.add_argument(
        "--rms-gate-db",
        type=float,
        default=-35.0,
        help="RMS energy gate in dBFS — segments below this are skipped  (default: -35.0)",
    )
    parser.add_argument(
        "--min-segment-sec",
        type=float,
        default=0.5,
        help="Minimum segment duration (seconds) — shorter segments are skipped  (default: 0.5)",
    )
    args = parser.parse_args()

    # ── Validate inputs ──────────────────────
    if not os.path.isfile(args.input_audio):
        print(f"✖  Audio file not found: {args.input_audio}")
        sys.exit(1)

    if not os.path.isfile(args.input_segments):
        print(f"✖  Segments JSON not found: {args.input_segments}")
        sys.exit(1)

    # ── Step 1: Load inputs ──────────────────
    print(f"[1/3]  Loading audio: {args.input_audio}")
    audio_data, sample_rate = load_audio(args.input_audio)
    audio_duration = len(audio_data) / sample_rate
    print(f"       ✔ {audio_duration:.1f}s of audio at {sample_rate} Hz")

    print(f"       Loading segments: {args.input_segments}")
    segments_data = load_segments(args.input_segments)
    n_segs = segments_data["total_segments"]
    print(f"       ✔ {n_segs} segment(s) loaded.")

    # ── Step 2–3: Transcribe ─────────────────
    print(f"\n[2/3]  Transcribing with faster-whisper (model={args.model})…\n")
    segments_data = transcribe_segments(
        audio_data,
        sample_rate,
        segments_data,
        model_size=args.model,
        device=args.device,
        compute_type=args.compute_type,
        source_lang=args.source_lang,
        rms_gate_db=args.rms_gate_db,
        min_segment_sec=args.min_segment_sec,
    )

    # ── Step 4: Save output ──────────────────
    print(f"\n[3/3]  Saving transcripts → {args.output}")
    # Record which model was used
    segments_data["asr_model"] = f"faster-whisper/{args.model}"
    segments_data["source_language"] = segments_data.get("source_language", args.source_lang)
    save_transcripts(segments_data, args.output)

    # ── Summary ──────────────────────────────
    filled = sum(1 for s in segments_data["segments"] if s.get("text"))
    empty = n_segs - filled

    print()
    print(f"{'═' * 55}")
    print(f"  ✅  transcripts.json saved → {args.output}")
    print(f"  Segments transcribed : {filled}/{n_segs}")
    if empty:
        print(f"  ⚠  Empty transcriptions: {empty}")
    print(f"{'═' * 55}")


if __name__ == "__main__":
    main()
