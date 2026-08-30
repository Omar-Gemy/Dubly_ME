"""
ingestion_vad.py — Phase A: Ingestion & VAD
===========================================
Audio ingestion and Voice Activity Detection (VAD) using the
self-hosted Silero VAD model.  No third-party APIs are used.

Pipeline:
  1. Extract audio from any media file via FFmpeg → mono 16 kHz WAV
  2. Apply EBU R128 loudness normalisation
  3. Run Silero VAD to detect speech segments
  4. Write results to  artifacts/segments.json

Usage:
  python src/ingestion_vad.py data/audio_in/sample.mp4
  python src/ingestion_vad.py data/audio_in/sample.wav --threshold 0.4
"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone

import torch
import torchaudio

import pipeline_core

# ──────────────────────────────────────────────
#  Project paths (relative to repo root)
# ──────────────────────────────────────────────
PROJECT_ROOT = pipeline_core.PROJECT_ROOT
ARTIFACTS_DIR = pipeline_core.ARTIFACTS_DIR
SEGMENTS_FILE = ARTIFACTS_DIR / "segments.json"
AUDIO_OUT_DIR = pipeline_core.DATA_AUDIO_OUT

# Silero VAD / WhisperX operating rate — declared once in pipeline_core (5.5).
SAMPLE_RATE = pipeline_core.ASR_SAMPLE_RATE

# ──────────────────────────────────────────────
#  Custom read_audio  (Windows-safe replacement
#  for Silero's utils_vad.read_audio which uses
#  sox_effects — unsupported on Windows)
# ──────────────────────────────────────────────
def read_audio(path: str, sampling_rate: int = SAMPLE_RATE) -> torch.Tensor:
    """
    Read an audio file and return a 1-D float32 torch.Tensor,
    resampled to *sampling_rate* if necessary.

    Uses the ``soundfile`` backend so it works on Windows without
    the sox extension that Silero's default helper requires.
    """
    wav, sr = torchaudio.load(path, backend="soundfile")

    # Convert to mono if needed
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # Resample if the file's rate differs from target
    if sr != sampling_rate:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sr, new_freq=sampling_rate
        )
        wav = resampler(wav)

    # Squeeze to 1-D (Silero expects shape [samples])
    return wav.squeeze(0)


# ──────────────────────────────────────────────
#  Step 1 — Audio extraction & normalisation
#  (two-pass LINEAR EBU R128 — preserves SNR)
# ──────────────────────────────────────────────
def extract_and_normalize_audio(
    input_path: str,
    output_path: str,
    sample_rate: int = SAMPLE_RATE,
    target_i: float = -16.0,
    target_tp: float = -1.5,
    target_lra: float = 11.0,
) -> str:
    """
    Use FFmpeg to:
      • strip any video track
      • convert to 16-bit PCM mono WAV at *sample_rate* Hz
      • apply EBU R128 loudness normalisation (**two-pass LINEAR mode**)

    Two-pass linear mode applies a single constant gain offset rather
    than dynamically pumping the volume — this preserves the original
    signal-to-noise ratio and avoids amplifying the background noise floor.

    Returns the *output_path* on success; raises RuntimeError otherwise.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # ── Pass 1: Analyse loudness (no output file) ─────────
    loudnorm_filter = (
        f"loudnorm=I={target_i}:TP={target_tp}:LRA={target_lra}"
        f":print_format=json"
    )

    analyse_cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-vn",
        "-af", loudnorm_filter,
        "-f", "null",
        "-",
    ]

    pass1 = subprocess.run(analyse_cmd, capture_output=True, text=True)
    if pass1.returncode != 0:
        raise RuntimeError(
            f"FFmpeg loudnorm analysis (pass 1) failed.\n"
            f"stderr:\n{pass1.stderr}"
        )

    # Extract the JSON stats block from stderr
    json_match = re.search(r"\{[^{}]+\}", pass1.stderr, re.DOTALL)
    if not json_match:
        raise RuntimeError(
            "Could not extract loudnorm measurement JSON from FFmpeg.\n"
            f"stderr:\n{pass1.stderr}"
        )
    stats = json.loads(json_match.group())

    # ── Pass 2: Apply linear normalisation ────────────────
    linear_filter = (
        f"loudnorm=I={target_i}:TP={target_tp}:LRA={target_lra}"
        f":measured_I={stats['input_i']}"
        f":measured_TP={stats['input_tp']}"
        f":measured_LRA={stats['input_lra']}"
        f":measured_thresh={stats['input_thresh']}"
        f":offset={stats['target_offset']}"
        f":linear=true"
    )

    apply_cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",
        "-af", linear_filter,
        output_path,
    ]

    pass2 = subprocess.run(apply_cmd, capture_output=True, text=True)
    if pass2.returncode != 0:
        raise RuntimeError(
            f"FFmpeg loudnorm application (pass 2) failed.\n"
            f"stderr:\n{pass2.stderr}"
        )

    return output_path


# ──────────────────────────────────────────────
#  Step 2 — Load Silero VAD (self-hosted model)
# ──────────────────────────────────────────────
def load_silero_vad():
    """
    Download / cache the Silero VAD model via torch.hub.
    Returns (model, utils_tuple).
    """
    # Single-thread inference is faster for this lightweight model
    torch.set_num_threads(1)

    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        trust_repo=True,
    )
    return model, utils


# ──────────────────────────────────────────────
#  Step 3 — Run VAD
# ──────────────────────────────────────────────
def run_vad(
    audio_path: str,
    model,
    utils,
    threshold: float = 0.20,
    min_speech_ms: int = 200,
    min_silence_ms: int = 400,
    speech_pad_ms: int = 250,
):
    """
    Run Silero VAD on a 16 kHz mono WAV file.

    Returns:
      speech_timestamps – list of dicts with 'start' / 'end' keys (in seconds)
      sampling_rate     – the rate used (always 16 000)
    """
    # Unpack only get_speech_timestamps from Silero utils;
    # we intentionally skip Silero's read_audio (sox-dependent)
    # and use our own Windows-safe implementation instead.
    (get_speech_timestamps, *_rest) = utils
    SAMPLING_RATE = SAMPLE_RATE

    wav = read_audio(audio_path, sampling_rate=SAMPLING_RATE)

    speech_timestamps = get_speech_timestamps(
        wav,
        model,
        sampling_rate=SAMPLING_RATE,
        threshold=threshold,
        min_speech_duration_ms=min_speech_ms,
        min_silence_duration_ms=min_silence_ms,
        speech_pad_ms=speech_pad_ms,   # protect against clipped onsets/offsets
        return_seconds=True,           # gives float seconds directly
    )

    return speech_timestamps, SAMPLING_RATE


# ──────────────────────────────────────────────
#  Step 4 — Format & persist the data contract
# ──────────────────────────────────────────────
def format_segments(
    speech_timestamps,
    source_file: str,
    source_lang: str,
    target_lang: str,
) -> dict:
    """
    Build the ``segments.json`` data-contract object.

    Each segment carries placeholder fields (speaker_id, text) that
    downstream pipeline stages will populate.
    """
    segments = []
    for idx, ts in enumerate(speech_timestamps, start=1):
        start_sec = round(ts["start"], 3)
        end_sec   = round(ts["end"], 3)
        duration  = round(end_sec - start_sec, 3)

        segments.append(
            {
                "segment_id": idx,
                "start_time": start_sec,
                "end_time": end_sec,
                "duration": duration,
                "speaker_id": None,   # filled by diarisation stage
                "text": None,         # filled by transcription stage
            }
        )

    return {
        "source_file": os.path.basename(source_file),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_language": source_lang,
        "target_language": target_lang,
        "vad_model": "silero_vad_v5",
        "vad_threshold": None,        # will be set by caller
        "total_segments": len(segments),
        "segments": segments,
    }


def save_segments(data: dict, output_path: str) -> None:
    """Write the segments dict to a JSON file (UTF-8, pretty-printed)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


# ──────────────────────────────────────────────
#  CLI entry-point
# ──────────────────────────────────────────────
def main() -> None:
    # UTF-8 stdio before the first banner: the box-drawing glyphs below die on
    # a cp1252 fallback when stdout is piped or redirected (Windows).
    pipeline_core.enable_utf8_stdio()
    parser = argparse.ArgumentParser(
        description=f"Dubly ME — {pipeline_core.phase_title('A')}",
    )
    parser.add_argument(
        "input",
        help="Path to the input audio or video file  (e.g. data/audio_in/sample.mp4)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.20,
        help="VAD confidence threshold  (0.0–1.0, default 0.20)",
    )
    parser.add_argument(
        "--min-speech-ms",
        type=int,
        default=200,
        help="Minimum speech duration in ms to keep  (default 200)",
    )
    parser.add_argument(
        "--min-silence-ms",
        type=int,
        default=400,
        help="Minimum silence duration in ms to split on  (default 400)",
    )
    parser.add_argument(
        "--speech-pad-ms",
        type=int,
        default=250,
        help="Padding added to each side of detected speech  (default 250)",
    )
    parser.add_argument(
        "--output",
        default=str(SEGMENTS_FILE),
        help="Output JSON path  (default: artifacts/segments.json)",
    )
    parser.add_argument(
        "--source-lang",
        default="auto",
        help="Source language code (e.g. 'es', 'ar') or 'auto' for detection  (default: auto)",
    )
    parser.add_argument(
        "--target-lang",
        default="en",
        help="Target language code for dubbing  (default: en)",
    )
    args = parser.parse_args()

    # ── Validate input ────────────────────────
    input_path = args.input
    if not os.path.isfile(input_path):
        print(f"✖  Input file not found: {input_path}")
        sys.exit(1)

    # ── Step 1: Extract & normalise audio ─────
    print(f"[1/3]  Extracting & normalising audio from: {input_path}")
    normalised_wav = str(AUDIO_OUT_DIR / "_temp_normalised.wav")
    extract_and_normalize_audio(input_path, normalised_wav)
    print(f"       ✔ Saved normalised WAV → {normalised_wav}")

    # ── Step 2: Load Silero VAD ───────────────
    print("[2/3]  Loading Silero VAD model (torch.hub — cached locally)…")
    model, utils = load_silero_vad()
    print("       ✔ Model loaded.")

    # ── Step 3: Run VAD ───────────────────────
    print(
        f"[3/3]  Running VAD  "
        f"(threshold={args.threshold}, "
        f"min_speech={args.min_speech_ms}ms, "
        f"min_silence={args.min_silence_ms}ms, "
        f"speech_pad={args.speech_pad_ms}ms)…"
    )
    speech_ts, _sr = run_vad(
        normalised_wav,
        model,
        utils,
        threshold=args.threshold,
        min_speech_ms=args.min_speech_ms,
        min_silence_ms=args.min_silence_ms,
        speech_pad_ms=args.speech_pad_ms,
    )
    print(f"       ✔ Detected {len(speech_ts)} speech segment(s).")

    # ── Format & save data contract ───────────
    segments_data = format_segments(
        speech_ts,
        input_path,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
    )
    segments_data["vad_threshold"] = args.threshold
    save_segments(segments_data, args.output)

    # ── Summary ───────────────────────────────
    print()
    print(f"{'═' * 50}")
    print(f"  ✅  segments.json saved → {args.output}")
    print(f"  Total segments : {segments_data['total_segments']}")
    print(f"{'═' * 50}")

    # Preview up to 5 segments
    for seg in segments_data["segments"][:5]:
        print(
            f"  #{seg['segment_id']:>3d}  "
            f"{seg['start_time']:>8.3f}s → {seg['end_time']:>8.3f}s  "
            f"({seg['duration']:.3f}s)"
        )
    if segments_data["total_segments"] > 5:
        remaining = segments_data["total_segments"] - 5
        print(f"  … and {remaining} more segment(s).")


if __name__ == "__main__":
    main()
