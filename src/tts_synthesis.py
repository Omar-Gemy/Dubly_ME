"""
tts_synthesis.py — Phase E: Voice Cloning & Text-to-Speech
============================================================
Synthesize English dubbed audio for each translated segment
using XTTS v2 voice cloning via the Coqui TTS API.

Hardware target: RTX 3050 Ti (4 GB VRAM).
XTTS v2 fits comfortably (~1.4 GB), leaving ample headroom.

Pipeline:
  1. Validate inputs (translation.json + voice reference WAV)
  2. Load the XTTS v2 model (GPU with auto fp16, or CPU fallback)
  3. Iterate over translated segments and synthesize speech
  4. Save audio clips + manifest to artifacts/

Inputs:
  - artifacts/translation.json   (Phase D data contract)
  - artifacts/voice_ref.wav      (pre-extracted speaker reference)

Outputs:
  - artifacts/audio_out/segment_XXX.wav   (one per valid segment)
  - artifacts/tts_manifest.json           (Phase E data contract)

Usage:
  python src/tts_synthesis.py
  python src/tts_synthesis.py --device cpu
  python src/tts_synthesis.py --input artifacts/translation.json
"""

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import soundfile as sf
import torch

import pipeline_core

# Coqui's XTTS v2 weights are gated behind a non-commercial licence prompt that
# blocks on stdin unless this is set. It is applied in accept_coqui_tos() right
# before the TTS import — NOT at module import time, so `import tts_synthesis`
# has no environment side-effect (5.10).
COQUI_TOS_ENV_VAR = "COQUI_TOS_AGREED"

# ──────────────────────────────────────────────
#  Project paths (relative to repo root)
# ──────────────────────────────────────────────
PROJECT_ROOT   = pipeline_core.PROJECT_ROOT
ARTIFACTS_DIR  = pipeline_core.ARTIFACTS_DIR
AUDIO_IN_DIR   = pipeline_core.DATA_AUDIO_IN
AUDIO_OUT_DIR  = ARTIFACTS_DIR / "audio_out"
VOICE_PROFILES = PROJECT_ROOT / "voice_profiles"

DEFAULT_INPUT      = ARTIFACTS_DIR / "translation.json"
DEFAULT_REF_AUDIO  = ARTIFACTS_DIR / "voice_ref.wav"
DEFAULT_SOURCE     = AUDIO_IN_DIR / "sample.mp4"


def accept_coqui_tos() -> None:
    """Set COQUI_TOS_AGREED for this process (idempotent, no-op if already set)."""
    os.environ.setdefault(COQUI_TOS_ENV_VAR, "1")


# ──────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────
XTTS_MODEL_NAME    = "tts_models/multilingual/multi-dataset/xtts_v2"
SAMPLE_RATE        = pipeline_core.XTTS_NATIVE_SAMPLE_RATE   # 22050 Hz (5.5)
CACHE_FLUSH_EVERY  = 5        # flush CUDA cache every N segments
MIN_VRAM_GB        = 3.5      # below this, force CPU rather than risk OOM

# XTTS v2's 17 built-in languages. The synthesis language is resolved from
# translation.json's `target_language` (or --target-lang), NOT hardcoded —
# so ar/en/es (and the rest) all work with no code change.
# Sourced from config/languages.json via pipeline_core (5.11).
XTTS_SUPPORTED_LANGS = pipeline_core.xtts_supported_languages()

# Phase 2 — multi-speaker voice references.
SPEAKER_UNKNOWN_LABEL = "SPEAKER_UNKNOWN"   # diarization catch-all bucket
MIN_SPEAKER_REF_SEC   = 3.0                 # min clean audio for a dedicated clone
MAX_SPEAKER_REF_SEC   = 8.0                 # cap the extracted reference window

# XTTS v2 generation controls (forwarded through tts_to_file → Xtts.inference).
# Verified against coqui-tts 0.27.5. Defaults tuned for dubbing: warmer than
# flat greedy output, but tightly controlled to avoid rambling/hallucination.
#   • repetition_penalty 5.0 (< the 10.0 XTTS default) → less clipped, more
#     natural prosody without runaway repeats.
#   • enable_text_splitting → long lines are chunked so intonation resets
#     naturally instead of degrading across a single long generation.
XTTS_GEN_DEFAULTS = {
    "temperature": 0.70,
    "length_penalty": 1.0,
    "repetition_penalty": 5.0,
    "top_k": 50,
    "top_p": 0.85,
    "speed": 1.0,
    "enable_text_splitting": True,
}

# Fallback reference extraction window (only used if voice_ref.wav
# is missing and --auto-extract is enabled)
FALLBACK_REF_START = 2.5      # seconds — start of segment #1
FALLBACK_REF_END   = 5.8      # seconds — end of segment #1


# ──────────────────────────────────────────────
#  Step 1 — Validate inputs
# ──────────────────────────────────────────────
def validate_inputs(
    translation_path: Path,
    ref_audio_path: Path,
    source_video: Path,
    auto_extract: bool,
) -> Path:
    """
    Validate that all required input files exist.

    If the voice reference WAV is missing and --auto-extract is set,
    extract a fallback clip from the source video using FFmpeg.

    Returns the confirmed path to the voice reference audio.
    """
    # Check translation.json
    if not translation_path.is_file():
        print(f"  ✖ Translation file not found: {translation_path}")
        sys.exit(1)
    print(f"  ✔ Translation file : {translation_path}")

    # Check voice reference
    if ref_audio_path.is_file():
        info = sf.info(str(ref_audio_path))
        print(f"  ✔ Voice reference  : {ref_audio_path}")
        print(f"    Duration : {info.duration:.2f}s  |  "
              f"Rate: {info.samplerate} Hz  |  "
              f"Channels: {info.channels}")
        return ref_audio_path

    # Reference not found — attempt fallback extraction
    print(f"  ⚠ Voice reference not found: {ref_audio_path}")

    if not auto_extract:
        print(f"  ✖ Cannot proceed without a voice reference.")
        print(f"    Either provide the file or re-run with --auto-extract")
        sys.exit(1)

    if not source_video.is_file():
        print(f"  ✖ Source video not found for extraction: {source_video}")
        sys.exit(1)

    print(f"  → Auto-extracting reference from {source_video}…")
    ref_audio_path = _extract_reference_clip(
        source_video, ref_audio_path,
        FALLBACK_REF_START, FALLBACK_REF_END,
    )
    return ref_audio_path


def _extract_reference_clip(
    source_video: Path,
    output_path: Path,
    start: float,
    end: float,
) -> Path:
    """
    Extract a short voice reference clip from the source video
    using FFmpeg.  Output is mono WAV at 22050 Hz.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration = end - start

    cmd = [
        "ffmpeg", "-y",
        "-i", str(source_video),
        "-ss", str(start),
        "-t", str(duration),
        "-vn",                    # discard video
        "-acodec", "pcm_s16le",   # 16-bit PCM
        "-ar", str(SAMPLE_RATE),  # resample to 22050 Hz
        "-ac", "1",               # mono
        str(output_path),
    ]

    print(f"    FFmpeg: {start:.1f}s → {end:.1f}s ({duration:.1f}s)")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

    if result.returncode != 0:
        print(f"  ✖ FFmpeg error:\n{result.stderr[:500]}")
        sys.exit(1)

    info = sf.info(str(output_path))
    print(f"  ✔ Fallback reference saved: {output_path} "
          f"({info.duration:.2f}s)")

    return output_path


# ──────────────────────────────────────────────
#  Step 1b — Build per-speaker reference map
# ──────────────────────────────────────────────
def _find_profile_file(profiles_dir: Path, speaker_id: str):
    """Return a hand-picked profile clip for this speaker, if one exists."""
    for ext in (".wav", ".flac", ".mp3", ".m4a"):
        cand = profiles_dir / f"{speaker_id}{ext}"
        if cand.is_file():
            return cand
    return None


def _best_ref_segment(segments: list[dict], speaker_id: str):
    """Longest clean (non-skipped) segment for this speaker, ≥ MIN_SPEAKER_REF_SEC."""
    cands = [
        s for s in segments
        if s.get("speaker_id") == speaker_id
        and not s.get("skip_translation", False)
        and (s.get("duration") or 0.0) >= MIN_SPEAKER_REF_SEC
    ]
    return max(cands, key=lambda s: s.get("duration") or 0.0) if cands else None


def build_speaker_ref_map(
    segments: list[dict],
    source_video: Path,
    profiles_dir: Path,
    global_ref: Path,
    auto_extract: bool,
    cache_dir: Path,
) -> dict:
    """
    Map each distinct speaker_id → a dedicated voice reference WAV.

    Resolution per speaker (best → fallback):
      1. Explicit profile  profiles_dir/<speaker_id>.{wav,flac,mp3,m4a}
      2. Auto-extracted longest clean segment → cache_dir/<speaker_id>.wav
      3. (implicit) no entry → caller uses the global reference

    SPEAKER_UNKNOWN is never given a dedicated voice (audit #6): it is a
    catch-all bucket that may mix speakers, so it routes to the global
    reference. Returns {speaker_id: Path} — speakers absent from the dict
    fall back to global_ref at the call site.
    """
    speaker_ids = sorted({s.get("speaker_id") for s in segments if s.get("speaker_id")})
    ref_map: dict = {}
    print(f"  Speakers present : {len(speaker_ids)}  "
          f"({', '.join(speaker_ids) or 'none'})")

    for sid in speaker_ids:
        # 1. Hand-picked profile
        prof = _find_profile_file(profiles_dir, sid)
        if prof is not None:
            ref_map[sid] = prof
            print(f"    {sid:<16} → profile  {prof.name}")
            continue

        # SPEAKER_UNKNOWN → global, never a dedicated clone
        if sid == SPEAKER_UNKNOWN_LABEL:
            print(f"    {sid:<16} → global   (unknown bucket → shared voice)")
            continue

        # 2. Auto-extract the longest clean segment for this speaker
        if auto_extract and source_video.is_file():
            best = _best_ref_segment(segments, sid)
            if best is not None:
                start = best["start_time"]
                end = start + min(best.get("duration", 0.0), MAX_SPEAKER_REF_SEC)
                out = cache_dir / f"{sid}.wav"
                try:
                    if not out.is_file():
                        _extract_reference_clip(source_video, out, start, end)
                    ref_map[sid] = out
                    print(f"    {sid:<16} → extract  {out.name} "
                          f"({end - start:.1f}s @ {start:.1f}s)")
                    continue
                except SystemExit:
                    # _extract_reference_clip exits on FFmpeg failure; downgrade
                    # to a soft fallback rather than killing the whole run.
                    print(f"    {sid:<16} → global   (extraction failed)")
                    continue

        # 3. Fallback to global reference
        print(f"    {sid:<16} → global   (no profile / no clean ref segment)")

    return ref_map


# ──────────────────────────────────────────────
#  Step 2 — Load XTTS v2 model
# ──────────────────────────────────────────────
def load_tts_model(device: str = "auto"):
    """
    Load XTTS v2 via the Coqui TTS API.

    Memory strategy for 4 GB VRAM (RTX 3050 Ti):
      - XTTS v2 uses ~1.4 GB VRAM — fits comfortably
      - Load once, reuse for all segments (no repeated init)
      - Fall back to CPU if CUDA is unavailable
      - All inference runs under torch.no_grad() context
    """
    # Must precede the TTS import: the licence prompt is evaluated there (5.10).
    accept_coqui_tos()
    from TTS.api import TTS

    # ── Resolve device ──────────────────────────
    device = pipeline_core.resolve_device_str(device)

    if device == "cuda":
        gpu_name, vram_gb = pipeline_core.cuda_device_summary()
        print(f"  GPU detected : {gpu_name} ({vram_gb:.1f} GB VRAM)")

        if vram_gb < MIN_VRAM_GB:
            print(f"  ⚠ VRAM below {MIN_VRAM_GB} GB — forcing CPU to avoid OOM")
            device = "cpu"
    else:
        print(f"  Running on CPU (slower but safe)")

    print(f"  Loading XTTS v2 on '{device}'…")
    print(f"  Model: {XTTS_MODEL_NAME}")

    # ── Load model ──────────────────────────────
    # Coqui TTS handles model download & caching automatically.
    # First run will download ~1.8 GB to the local cache.
    tts = TTS(
        model_name=XTTS_MODEL_NAME,
        progress_bar=True,
    ).to(device)

    print(f"  ✔ XTTS v2 loaded successfully on {device}.")

    # ── Report VRAM usage ───────────────────────
    if device == "cuda":
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved  = torch.cuda.memory_reserved() / (1024 ** 3)
        print(f"    VRAM allocated : {allocated:.2f} GB")
        print(f"    VRAM reserved  : {reserved:.2f} GB")

    return tts, device


# ──────────────────────────────────────────────
#  Step 3 — Synthesize segments
# ──────────────────────────────────────────────
def synthesize_segments(
    tts,
    device: str,
    segments: list[dict],
    speaker_ref_map: dict,
    global_ref: Path,
    output_dir: Path,
    language: str,
    gen_params: dict | None = None,
) -> list[dict]:
    """
    Iterate over translated segments from translation.json,
    synthesize each using XTTS v2 with the speaker reference,
    and save individual WAV files.

    Skips segments where:
      - skip_translation is True
      - translated_text is None or empty

    Memory safety:
      - torch.cuda.empty_cache() every CACHE_FLUSH_EVERY segments
      - gc.collect() alongside cache flush
      - Per-segment try/except so one failure won't crash the run

    Returns:
      A manifest list of dicts describing each segment's outcome.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    total = len(segments)
    synthesized_count = 0
    skipped_count = 0
    failed_count = 0

    # XTTS generation controls (temperature, repetition_penalty, …). Forwarded
    # to tts_to_file → Xtts.inference. Falls back to the tuned defaults.
    gen = dict(XTTS_GEN_DEFAULTS)
    if gen_params:
        gen.update({k: v for k, v in gen_params.items() if v is not None})
    print(f"  XTTS gen params : temp={gen['temperature']} "
          f"rep_pen={gen['repetition_penalty']} top_k={gen['top_k']} "
          f"top_p={gen['top_p']} speed={gen['speed']} "
          f"split={gen['enable_text_splitting']}")

    t_start_all = time.perf_counter()

    for idx, seg in enumerate(segments, start=1):
        seg_id = seg["segment_id"]
        text = (seg.get("translated_text") or "").strip()

        # ── Skip invalid segments ───────────────
        if seg.get("skip_translation", False) or not text:
            reason_parts = []
            if seg.get("transcription_failed"):
                reason_parts.append("transcription-failed")
            if seg.get("low_confidence"):
                reason_parts.append("low-confidence")
            if not text:
                reason_parts.append("no-translated-text")
            reason_str = ", ".join(reason_parts) or "flagged"

            print(f"  [{idx:>2}/{total}]  Seg #{seg_id:<3}  "
                  f"⏭ SKIPPED ({reason_str})")

            manifest.append({
                "segment_id": seg_id,
                "status": "skipped",
                "speaker_id": seg.get("speaker_id"),
                "reason": reason_str,
                "output_file": None,
                "duration_s": None,
                "original_duration_s": seg.get("duration"),
            })
            skipped_count += 1
            continue

        # ── Synthesize this segment ─────────────
        speaker_id = seg.get("speaker_id")
        ref_path = speaker_ref_map.get(speaker_id, global_ref)
        ref_rel = pipeline_core.rel_or_abs(ref_path)

        out_filename = f"segment_{seg_id:03d}.wav"
        out_path = output_dir / out_filename

        preview = text[:55] + "…" if len(text) > 55 else text
        print(f"  [{idx:>2}/{total}]  Seg #{seg_id:<3}  \"{preview}\"")

        t_seg_start = time.perf_counter()

        try:
            with torch.no_grad():
                tts.tts_to_file(
                    text=text,
                    file_path=str(out_path),
                    speaker_wav=str(ref_path),
                    language=language,
                    **gen,
                )

            # Verify the output file was actually created
            if not out_path.is_file():
                raise FileNotFoundError(
                    f"TTS produced no output file at {out_path}"
                )

            info = sf.info(str(out_path))
            t_seg_elapsed = time.perf_counter() - t_seg_start

            manifest.append({
                "segment_id": seg_id,
                "status": "success",
                "speaker_id": speaker_id,
                "reference_audio": ref_rel,
                "output_file": pipeline_core.rel_or_abs(out_path),
                "duration_s": round(info.duration, 3),
                "original_duration_s": seg.get("duration"),
                "text": text,
            })
            synthesized_count += 1

            print(f"           ✔ saved → {out_filename}  "
                  f"({info.duration:.2f}s, took {t_seg_elapsed:.1f}s)")

        except Exception as e:
            t_seg_elapsed = time.perf_counter() - t_seg_start
            print(f"           ✖ FAILED ({t_seg_elapsed:.1f}s): {e}")

            manifest.append({
                "segment_id": seg_id,
                "status": "error",
                "speaker_id": speaker_id,
                "error": str(e),
                "output_file": None,
                "duration_s": None,
                "original_duration_s": seg.get("duration"),
            })
            failed_count += 1

        # ── Periodic VRAM cleanup ───────────────
        if device == "cuda" and idx % CACHE_FLUSH_EVERY == 0:
            torch.cuda.empty_cache()
            gc.collect()

    # ── Final cleanup ───────────────────────────
    if device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()

    t_total = time.perf_counter() - t_start_all
    print(f"\n  ─── Synthesis Complete ───")
    print(f"  Synthesized : {synthesized_count}/{total}")
    print(f"  Skipped     : {skipped_count}/{total}")
    if failed_count > 0:
        print(f"  Failed      : {failed_count}/{total}")
    print(f"  Total time  : {t_total:.1f}s")

    return manifest


# ──────────────────────────────────────────────
#  Step 4 — Save TTS manifest (data contract)
# ──────────────────────────────────────────────
def save_manifest(
    manifest: list[dict],
    ref_audio_path: Path,
    speaker_ref_map: dict,
    output_path: Path,
    model_name: str,
) -> None:
    """
    Save the TTS run manifest as a JSON data contract.

    This mirrors the convention from Phase D's translation.json:
    a top-level dict with metadata + a segments array. The speaker_profiles
    map records which reference clip cloned each speaker (Phase 2).
    """
    _rel = pipeline_core.rel_or_abs

    data = {
        "phase": "E",
        "description": "Voice Cloning & TTS Synthesis",
        "tts_model": model_name,
        "reference_audio": _rel(ref_audio_path),
        "speaker_profiles": {sid: _rel(p) for sid, p in speaker_ref_map.items()},
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_segments": len(manifest),
        "synthesized": sum(1 for m in manifest if m["status"] == "success"),
        "skipped": sum(1 for m in manifest if m["status"] == "skipped"),
        "failed": sum(1 for m in manifest if m["status"] == "error"),
        "segments": manifest,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)

    print(f"  ✔ Manifest saved → {output_path}")


# ──────────────────────────────────────────────
#  Summary — Duration comparison for QA
# ──────────────────────────────────────────────
def print_duration_comparison(manifest: list[dict]) -> None:
    """
    Print a table comparing original segment durations with
    synthesized audio durations.  Flags overflows > 0.5s so
    Phase F can prioritise which segments need timing work.
    """
    success_entries = [m for m in manifest if m["status"] == "success"]
    if not success_entries:
        print("  No successfully synthesized segments to compare.")
        return

    print(f"\n  {'Seg':<6} {'Original':>9} {'Synth':>9} {'Delta':>9}  Notes")
    print(f"  {'─' * 50}")

    overflow_count = 0
    for entry in success_entries:
        seg_id = entry["segment_id"]
        orig   = entry.get("original_duration_s") or 0.0
        synth  = entry.get("duration_s") or 0.0
        delta  = synth - orig

        flag = ""
        if delta > 0.5:
            flag = "⚠ OVERFLOW"
            overflow_count += 1
        elif delta < -0.5:
            flag = "← short"

        print(f"  #{seg_id:<5} {orig:>8.1f}s {synth:>8.2f}s {delta:>+8.2f}s  {flag}")

    if overflow_count > 0:
        print(f"\n  ⚠ {overflow_count} segment(s) exceed the original "
              f"duration by > 0.5s — address in Phase F.")


# ──────────────────────────────────────────────
#  CLI entry-point
# ──────────────────────────────────────────────
def main() -> None:
    # UTF-8 stdio before the first banner: the box-drawing glyphs below die on
    # a cp1252 fallback when stdout is piped or redirected (Windows).
    pipeline_core.enable_utf8_stdio()
    parser = argparse.ArgumentParser(
        description=f"Dubly ME — {pipeline_core.phase_title('E')}",
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Path to translation.json  "
             "(default: artifacts/translation.json)",
    )
    parser.add_argument(
        "--ref-audio",
        default=str(DEFAULT_REF_AUDIO),
        help="Path to the voice reference WAV  "
             "(default: artifacts/voice_ref.wav)",
    )
    parser.add_argument(
        "--source",
        default=str(DEFAULT_SOURCE),
        help="Path to source video for fallback reference extraction  "
             "(default: data/audio_in/sample.mp4)",
    )
    parser.add_argument(
        "--auto-extract",
        action="store_true",
        default=False,
        help="If set, automatically extract a reference clip from "
             "the source video when voice_ref.wav is missing",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device: 'auto', 'cpu', or 'cuda'  (default: auto)",
    )
    parser.add_argument(
        "--target-lang",
        default=None,
        help="XTTS dub language tag (ar/en/es/...). "
             "Default: target_language from translation.json, else 'en'.",
    )
    parser.add_argument(
        "--speaker-profiles-dir",
        default=str(VOICE_PROFILES),
        help="Dir of hand-picked per-speaker reference clips "
             "(<speaker_id>.wav). Default: voice_profiles/",
    )
    parser.add_argument(
        "--single-voice",
        action="store_true",
        help="Force one shared voice for all speakers (pre-Phase-2 behavior).",
    )
    # XTTS generation controls (override the tuned defaults by ear).
    parser.add_argument(
        "--temperature", type=float, default=None,
        help=f"XTTS sampling temperature (default: {XTTS_GEN_DEFAULTS['temperature']}).",
    )
    parser.add_argument(
        "--repetition-penalty", type=float, default=None,
        help=f"XTTS repetition penalty (default: {XTTS_GEN_DEFAULTS['repetition_penalty']}).",
    )
    parser.add_argument(
        "--top-k", type=int, default=None,
        help=f"XTTS top-k (default: {XTTS_GEN_DEFAULTS['top_k']}).",
    )
    parser.add_argument(
        "--top-p", type=float, default=None,
        help=f"XTTS top-p (default: {XTTS_GEN_DEFAULTS['top_p']}).",
    )
    parser.add_argument(
        "--length-penalty", type=float, default=None,
        help=f"XTTS length penalty (default: {XTTS_GEN_DEFAULTS['length_penalty']}).",
    )
    parser.add_argument(
        "--speed", type=float, default=None,
        help=f"XTTS speaking rate multiplier (default: {XTTS_GEN_DEFAULTS['speed']}).",
    )
    args = parser.parse_args()

    print()
    print(f"{'═' * 60}")
    print(f"  Dubly ME — {pipeline_core.phase_title('E')}")
    print(f"{'═' * 60}")

    # ── Step 1: Validate inputs ─────────────────
    print(f"\n[1/4]  Validating inputs…\n")
    ref_audio = validate_inputs(
        translation_path=Path(args.input),
        ref_audio_path=Path(args.ref_audio),
        source_video=Path(args.source),
        auto_extract=args.auto_extract,
    )

    # ── Step 2: Load TTS model ──────────────────
    print(f"\n[2/4]  Loading TTS model…\n")
    tts, device = load_tts_model(device=args.device)

    # ── Step 3: Load translations & synthesize ──
    print(f"\n[3/4]  Synthesizing translated segments…\n")
    with open(args.input, "r", encoding="utf-8") as fh:
        translation_data = json.load(fh)

    # Resolve the synthesis language: CLI override → translation.json → "en".
    tts_language = (
        args.target_lang or translation_data.get("target_language") or "en"
    ).strip().lower()
    if tts_language not in XTTS_SUPPORTED_LANGS:
        print(f"  ✖ Unsupported TTS language '{tts_language}'. "
              f"XTTS v2 supports: {', '.join(sorted(XTTS_SUPPORTED_LANGS))}")
        sys.exit(1)
    print(f"  Target TTS language : {tts_language}")

    segments = translation_data["segments"]
    total_segs = len(segments)
    valid_segs = sum(
        1 for s in segments
        if not s.get("skip_translation", False)
        and (s.get("translated_text") or "").strip()
    )
    print(f"  Total segments  : {total_segs}")
    print(f"  To synthesize   : {valid_segs}")
    print(f"  To skip         : {total_segs - valid_segs}")
    print()

    # ── Resolve per-speaker voice references (Phase 2) ──
    if args.single_voice:
        print("  Voice mode      : single (--single-voice)")
        speaker_ref_map = {}
    else:
        print("  Voice mode      : multi-speaker")
        speaker_ref_map = build_speaker_ref_map(
            segments=segments,
            source_video=Path(args.source),
            profiles_dir=Path(args.speaker_profiles_dir),
            global_ref=ref_audio,
            auto_extract=args.auto_extract,
            cache_dir=AUDIO_OUT_DIR / "speaker_refs",
        )
    print()

    # CLI overrides for XTTS generation (None → keep tuned default).
    gen_params = {
        "temperature": args.temperature,
        "repetition_penalty": args.repetition_penalty,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "length_penalty": args.length_penalty,
        "speed": args.speed,
    }

    manifest = synthesize_segments(
        tts=tts,
        device=device,
        segments=segments,
        speaker_ref_map=speaker_ref_map,
        global_ref=ref_audio,
        output_dir=AUDIO_OUT_DIR,
        language=tts_language,
        gen_params=gen_params,
    )

    # ── Step 4: Save manifest ───────────────────
    manifest_path = ARTIFACTS_DIR / "tts_manifest.json"
    print(f"\n[4/4]  Saving TTS manifest…\n")
    save_manifest(manifest, ref_audio, speaker_ref_map, manifest_path, XTTS_MODEL_NAME)

    # ── Final Summary ───────────────────────────
    success = sum(1 for m in manifest if m["status"] == "success")
    skipped = sum(1 for m in manifest if m["status"] == "skipped")
    failed  = sum(1 for m in manifest if m["status"] == "error")

    print()
    print(f"{'═' * 60}")
    print(f"  ✅  Phase E complete — {pipeline_core.PHASES['E']['label']}")
    print(f"{'─' * 60}")
    print(f"  Segments synthesized : {success}/{total_segs}")
    print(f"  Segments skipped     : {skipped}/{total_segs}")
    if failed:
        print(f"  ⚠  Segments failed   : {failed}/{total_segs}")
    print(f"{'─' * 60}")
    print(f"  Audio output dir     : {AUDIO_OUT_DIR}")
    print(f"  Manifest             : {manifest_path}")
    print(f"  Voice reference      : {ref_audio}")
    print(f"{'═' * 60}")

    # Duration comparison for QA review
    print_duration_comparison(manifest)

    # Exit with error code if any segments failed
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
