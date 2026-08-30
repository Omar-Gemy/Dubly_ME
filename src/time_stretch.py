"""
time_stretch.py — Phase F1: Duration Fitting via Time-Stretching
================================================================
Fit synthesised TTS segments into the original timing budget using
high-quality WSOLA time-stretching (pyrubberband / FFmpeg atempo).

Strategy (approved by Tech Lead):
  - Tier 1 (ratio ≤ 1.15×): segment already fits or needs trivial stretch
  - Tier 2 (1.15× < ratio ≤ 2.0×): apply WSOLA time-stretching
  - Over-cap (ratio > 2.0×): cap at 2.0× compression, allow overflow
    → Do NOT compress beyond 2.0× to preserve intelligibility (Q1: Option A)

Inputs:
  - artifacts/tts_manifest.json   (Phase E data contract)
  - artifacts/audio_out/*.wav     (synthesised segment WAVs)

Outputs:
  - artifacts/audio_out/stretched/segment_XXX.wav  (time-fitted WAVs)
  - artifacts/stretch_manifest.json                (Phase F1 data contract)

Usage:
  python src/time_stretch.py
  python src/time_stretch.py --max-ratio 2.0
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

import pipeline_core

# ──────────────────────────────────────────────
#  Project paths
# ──────────────────────────────────────────────
PROJECT_ROOT    = pipeline_core.PROJECT_ROOT
ARTIFACTS_DIR   = pipeline_core.ARTIFACTS_DIR
AUDIO_OUT_DIR   = ARTIFACTS_DIR / "audio_out"
STRETCHED_DIR   = AUDIO_OUT_DIR / "stretched"
TTS_MANIFEST    = ARTIFACTS_DIR / "tts_manifest.json"
STRETCH_MANIFEST = ARTIFACTS_DIR / "stretch_manifest.json"

# Phase F1 is rate-PRESERVING by design: it never declares a sample rate of its
# own, so Phase E's rate passes straight through to Phase F2, which owns the
# single documented conversion to pipeline_core.PIPELINE_SAMPLE_RATE (5.5).

# ──────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────
MAX_STRETCH_RATIO   = 2.0     # Never compress beyond 2.0× (Q1 decision)
TIER1_THRESHOLD     = 1.15    # Below this ratio → trivial / no-stretch
# RELATIVE tolerance (5%), NOT seconds — despite the "±50ms" banner below and
# the `fit_tolerance_s` manifest key, both of which are wrong. Renaming the key
# and making the metric two-sided is audit 6.2, scheduled for Chunk 6; Chunk 1
# only records the truth here so nobody tunes it on the wrong assumption.
FIT_TOLERANCE       = 0.05

# Over-cap segments get a short fade-out on the trimmed tail to de-click the
# overflow edge (audit #27).
OVERCAP_FADE_MS     = 10


# ──────────────────────────────────────────────
#  Time-stretch engine selection (Rubber Band → atempo)
# ──────────────────────────────────────────────
def _rubberband_available() -> bool:
    """True if both the pyrubberband wrapper and the rubberband CLI are present."""
    try:
        import pyrubberband  # noqa: F401
    except Exception:
        return False
    return shutil.which("rubberband") is not None


def _stretch_with_rubberband(
    input_path: Path,
    output_path: Path,
    speed_factor: float,
) -> Path:
    """
    Time-stretch using Rubber Band (formant-preserving WSOLA) via pyrubberband.

    speed_factor > 1.0 → compress (speak faster).  pyrubberband's `rate`
    argument uses the same convention (rate > 1.0 shortens the clip), so the
    factor passes through unchanged. Preserves the input sample rate.
    """
    import pyrubberband as pyrb

    output_path.parent.mkdir(parents=True, exist_ok=True)
    y, sr = sf.read(str(input_path), dtype="float32")
    # Formant preservation keeps the cloned timbre natural under compression.
    stretched = pyrb.time_stretch(y, sr, speed_factor, rbargs={"-F": ""})
    sf.write(str(output_path), stretched, sr, subtype="PCM_16")
    return output_path


# ──────────────────────────────────────────────
#  Time-stretch via FFmpeg atempo filter (fallback)
# ──────────────────────────────────────────────
def _stretch_with_ffmpeg(
    input_path: Path,
    output_path: Path,
    speed_factor: float,
) -> Path:
    """
    Apply time-stretching using FFmpeg's atempo filter.

    The atempo filter supports values between 0.5 and 100.0.
    For values outside [0.5, 2.0], FFmpeg requires chaining
    multiple atempo filters. We handle this automatically.

    speed_factor > 1.0 means speed UP (compress duration).
    speed_factor < 1.0 means slow DOWN (expand duration).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # FFmpeg atempo only supports [0.5, 100.0] per filter instance.
    # For values > 2.0, chain multiple filters.
    # For our use case, speed_factor will be between 1.0 and 2.0.
    atempo_filters = []
    remaining = speed_factor

    while remaining > 2.0:
        atempo_filters.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        atempo_filters.append("atempo=0.5")
        remaining /= 0.5

    atempo_filters.append(f"atempo={remaining:.6f}")
    filter_chain = ",".join(atempo_filters)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-af", filter_chain,
        "-acodec", "pcm_s16le",
        str(output_path),
    ]

    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=60
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg atempo failed for {input_path.name}:\n"
            f"{result.stderr[:500]}"
        )

    return output_path


# ──────────────────────────────────────────────
#  Engine dispatch + tail de-click
# ──────────────────────────────────────────────
_STRETCH_ENGINE = None  # resolved once, lazily: "rubberband" | "ffmpeg"


def _stretch_segment(
    input_path: Path,
    output_path: Path,
    speed_factor: float,
) -> Path:
    """
    Stretch via Rubber Band when available (best speech quality), else fall
    back to FFmpeg atempo. Engine is resolved once and reported on first use.
    """
    global _STRETCH_ENGINE
    if _STRETCH_ENGINE is None:
        _STRETCH_ENGINE = "rubberband" if _rubberband_available() else "ffmpeg"
        label = ("Rubber Band (formant-preserving)"
                 if _STRETCH_ENGINE == "rubberband"
                 else "FFmpeg atempo (rubberband-cli not found)")
        print(f"  Time-stretch engine: {label}\n")

    if _STRETCH_ENGINE == "rubberband":
        try:
            return _stretch_with_rubberband(input_path, output_path, speed_factor)
        except Exception as e:
            print(f"    ⚠ rubberband failed ({e}) — falling back to atempo")
    return _stretch_with_ffmpeg(input_path, output_path, speed_factor)


def _apply_tail_fade(path: Path, fade_ms: float) -> None:
    """Fade the trailing *fade_ms* to zero in-place, de-clicking a hard tail."""
    y, sr = sf.read(str(path), dtype="float32")
    n = int(fade_ms / 1000.0 * sr)
    if 0 < n < len(y):
        y[-n:] *= np.linspace(1.0, 0.0, n, dtype=np.float32)
        sf.write(str(path), y, sr, subtype="PCM_16")


# ──────────────────────────────────────────────
#  Classify and process a single segment
# ──────────────────────────────────────────────
def classify_segment(
    tts_duration: float,
    original_duration: float,
    max_ratio: float = MAX_STRETCH_RATIO,
) -> dict:
    """
    Classify a segment into a stretching tier.

    Returns a dict with:
      - tier: "no-stretch" | "tier1" | "tier2" | "over-cap"
      - ratio: raw ratio (tts_duration / original_duration)
      - speed_factor: the actual atempo speed to apply
      - target_duration: what the output duration will be
      - overflow_s: how much the output exceeds the original (0 if fits)
    """
    if original_duration <= 0:
        return {
            "tier": "invalid",
            "ratio": 0.0,
            "speed_factor": 1.0,
            "target_duration": tts_duration,
            "overflow_s": 0.0,
        }

    ratio = tts_duration / original_duration

    # Already fits (within tolerance)
    if ratio <= (1.0 + FIT_TOLERANCE):
        return {
            "tier": "no-stretch",
            "ratio": round(ratio, 4),
            "speed_factor": 1.0,
            "target_duration": tts_duration,
            "overflow_s": 0.0,
        }

    # Tier 1: mild stretch (up to TIER1_THRESHOLD)
    if ratio <= TIER1_THRESHOLD:
        return {
            "tier": "tier1",
            "ratio": round(ratio, 4),
            "speed_factor": round(ratio, 6),
            "target_duration": original_duration,
            "overflow_s": 0.0,
        }

    # Tier 2: moderate stretch (up to max_ratio)
    if ratio <= max_ratio:
        return {
            "tier": "tier2",
            "ratio": round(ratio, 4),
            "speed_factor": round(ratio, 6),
            "target_duration": original_duration,
            "overflow_s": 0.0,
        }

    # Over-cap: would need > max_ratio compression
    # Cap at max_ratio, allow overflow
    capped_duration = tts_duration / max_ratio
    overflow = capped_duration - original_duration

    return {
        "tier": "over-cap",
        "ratio": round(ratio, 4),
        "speed_factor": round(max_ratio, 6),
        "target_duration": round(capped_duration, 3),
        "overflow_s": round(overflow, 3),
    }


def process_segment(
    seg_entry: dict,
    max_ratio: float,
    input_dir: Path,
    output_dir: Path,
) -> dict:
    """
    Process a single segment: classify, stretch if needed, save output.

    Returns a manifest entry dict.
    """
    seg_id = seg_entry["segment_id"]
    status = seg_entry["status"]

    # ── Skip non-success segments ────────────
    if status != "success":
        return {
            "segment_id": seg_id,
            "status": "skipped",
            "reason": seg_entry.get("reason", "not-synthesized"),
            "input_file": None,
            "output_file": None,
            "original_duration_s": seg_entry.get("original_duration_s"),
            "tts_duration_s": None,
            "output_duration_s": None,
            "tier": "skipped",
            "stretch_ratio": None,
            "speed_factor": None,
            "overflow_s": 0.0,
        }

    tts_duration = seg_entry["duration_s"]
    orig_duration = seg_entry["original_duration_s"]
    input_file = PROJECT_ROOT / seg_entry["output_file"]

    if not input_file.is_file():
        return {
            "segment_id": seg_id,
            "status": "error",
            "error": f"Input WAV not found: {input_file}",
            "input_file": str(input_file),
            "output_file": None,
            "original_duration_s": orig_duration,
            "tts_duration_s": tts_duration,
            "output_duration_s": None,
            "tier": "error",
            "stretch_ratio": None,
            "speed_factor": None,
            "overflow_s": 0.0,
        }

    # ── Classify ─────────────────────────────
    classification = classify_segment(tts_duration, orig_duration, max_ratio)
    tier = classification["tier"]
    speed_factor = classification["speed_factor"]

    out_filename = f"segment_{seg_id:03d}.wav"
    out_path = output_dir / out_filename

    # ── No stretch needed ────────────────────
    if tier == "no-stretch":
        # Just copy the file as-is
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(input_file), str(out_path))
        output_duration = tts_duration

        print(f"  #{seg_id:<3}  ✔ NO-STRETCH  "
              f"({tts_duration:.2f}s → {output_duration:.2f}s, "
              f"ratio {classification['ratio']:.2f}×)")

    else:
        # ── Apply time-stretch ───────────────
        try:
            _stretch_segment(input_file, out_path, speed_factor)
            # De-click the trimmed tail of over-cap (overflowing) segments.
            if tier == "over-cap":
                _apply_tail_fade(out_path, OVERCAP_FADE_MS)
            info = sf.info(str(out_path))
            output_duration = round(info.duration, 3)

            tier_label = tier.upper()
            flag = ""
            if tier == "over-cap":
                flag = f"  ⚠ OVERFLOW +{classification['overflow_s']:.2f}s"

            print(f"  #{seg_id:<3}  ✔ {tier_label:<9}  "
                  f"({tts_duration:.2f}s → {output_duration:.2f}s, "
                  f"speed {speed_factor:.2f}×){flag}")

        except Exception as e:
            print(f"  #{seg_id:<3}  ✖ FAILED: {e}")
            return {
                "segment_id": seg_id,
                "status": "error",
                "error": str(e),
                "input_file": pipeline_core.rel_or_abs(input_file),
                "output_file": None,
                "original_duration_s": orig_duration,
                "tts_duration_s": tts_duration,
                "output_duration_s": None,
                "tier": tier,
                "stretch_ratio": classification["ratio"],
                "speed_factor": speed_factor,
                "overflow_s": classification["overflow_s"],
            }

    return {
        "segment_id": seg_id,
        "status": "success",
        "input_file": pipeline_core.rel_or_abs(input_file),
        "output_file": pipeline_core.rel_or_abs(out_path),
        "original_duration_s": orig_duration,
        "tts_duration_s": tts_duration,
        "output_duration_s": output_duration,
        "tier": tier,
        "stretch_ratio": classification["ratio"],
        "speed_factor": speed_factor,
        "overflow_s": classification["overflow_s"],
    }


# ──────────────────────────────────────────────
#  Main orchestration
# ──────────────────────────────────────────────
def run_time_stretch(
    manifest_path: Path,
    max_ratio: float = MAX_STRETCH_RATIO,
) -> list[dict]:
    """
    Load TTS manifest, process all segments, return stretch manifest.
    """
    # ── Load TTS manifest ────────────────────
    with open(manifest_path, "r", encoding="utf-8") as fh:
        tts_data = json.load(fh)

    segments = tts_data["segments"]
    total = len(segments)

    print(f"\n  Total segments    : {total}")
    print(f"  Max stretch ratio : {max_ratio}×")
    print(f"  Tier 1 threshold  : ≤{TIER1_THRESHOLD}×")
    print(f"  Fit tolerance     : ±{FIT_TOLERANCE * 1000:.0f}ms")
    print()

    # ── Process each segment ─────────────────
    STRETCHED_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    t_start = time.perf_counter()

    for seg_entry in segments:
        result = process_segment(
            seg_entry, max_ratio, AUDIO_OUT_DIR, STRETCHED_DIR
        )
        results.append(result)

    t_elapsed = time.perf_counter() - t_start

    # ── Summary ──────────────────────────────
    success = sum(1 for r in results if r["status"] == "success")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    errors  = sum(1 for r in results if r["status"] == "error")
    overflows = [r for r in results if r.get("tier") == "over-cap"]
    no_stretch = sum(1 for r in results if r.get("tier") == "no-stretch")
    tier1 = sum(1 for r in results if r.get("tier") == "tier1")
    tier2 = sum(1 for r in results if r.get("tier") == "tier2")

    print(f"\n  ─── Stretch Summary ───")
    print(f"  Success     : {success}/{total}")
    print(f"  Skipped     : {skipped}/{total}")
    if errors:
        print(f"  Errors      : {errors}/{total}")
    print(f"  No-stretch  : {no_stretch}")
    print(f"  Tier 1      : {tier1}")
    print(f"  Tier 2      : {tier2}")
    if overflows:
        print(f"  Over-cap ⚠  : {len(overflows)} segment(s)")
        for ov in overflows:
            print(f"    → Seg #{ov['segment_id']}: "
                  f"ratio {ov['stretch_ratio']}×, "
                  f"overflow +{ov['overflow_s']:.2f}s")
    print(f"  Total time  : {t_elapsed:.1f}s")

    return results


def save_stretch_manifest(
    results: list[dict],
    output_path: Path,
    max_ratio: float,
) -> None:
    """Save the stretch manifest as a JSON data contract."""
    data = {
        "phase": "F",
        "step": "time-stretch",
        "description": "Duration fitting via WSOLA time-stretching",
        "max_stretch_ratio": max_ratio,
        "tier1_threshold": TIER1_THRESHOLD,
        "fit_tolerance_s": FIT_TOLERANCE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_segments": len(results),
        "success": sum(1 for r in results if r["status"] == "success"),
        "skipped": sum(1 for r in results if r["status"] == "skipped"),
        "errors": sum(1 for r in results if r["status"] == "error"),
        "overflow_count": sum(
            1 for r in results if r.get("tier") == "over-cap"
        ),
        "segments": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)

    print(f"  ✔ Stretch manifest saved → {output_path}")


# ──────────────────────────────────────────────
#  CLI entry-point
# ──────────────────────────────────────────────
def main() -> None:
    # UTF-8 stdio before the first banner: the box-drawing glyphs below die on
    # a cp1252 fallback when stdout is piped or redirected (Windows).
    pipeline_core.enable_utf8_stdio()
    parser = argparse.ArgumentParser(
        description=f"Dubly ME — {pipeline_core.phase_title('F1')}",
    )
    parser.add_argument(
        "--input",
        default=str(TTS_MANIFEST),
        help="Path to tts_manifest.json  "
             "(default: artifacts/tts_manifest.json)",
    )
    parser.add_argument(
        "--max-ratio",
        type=float,
        default=MAX_STRETCH_RATIO,
        help=f"Maximum compression ratio  (default: {MAX_STRETCH_RATIO})",
    )
    parser.add_argument(
        "--output",
        default=str(STRETCH_MANIFEST),
        help="Output stretch manifest path  "
             "(default: artifacts/stretch_manifest.json)",
    )
    args = parser.parse_args()

    print()
    print(f"{'═' * 60}")
    print(f"  Dubly ME — {pipeline_core.phase_title('F1')}")
    print(f"{'═' * 60}")

    # ── Validate input ───────────────────────
    manifest_path = Path(args.input)
    if not manifest_path.is_file():
        print(f"\n  ✖ TTS manifest not found: {manifest_path}")
        sys.exit(1)

    print(f"\n[1/2]  Processing segments…")
    results = run_time_stretch(manifest_path, args.max_ratio)

    print(f"\n[2/2]  Saving stretch manifest…\n")
    save_stretch_manifest(results, Path(args.output), args.max_ratio)

    # ── Final banner ─────────────────────────
    success = sum(1 for r in results if r["status"] == "success")
    errors  = sum(1 for r in results if r["status"] == "error")

    print()
    print(f"{'═' * 60}")
    print(f"  ✅  Phase F1 complete — {pipeline_core.PHASES['F1']['label']}")
    print(f"{'─' * 60}")
    print(f"  Stretched segments : {success}/{len(results)}")
    print(f"  Output directory   : {STRETCHED_DIR}")
    print(f"  Manifest           : {args.output}")
    print(f"{'═' * 60}")

    if errors > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
