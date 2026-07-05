"""
diarization.py — Phase B: Speaker Diarization
===============================================
Assign speaker identities to VAD segments using pyannote.audio 4.x
(local inference, no external APIs).

Strategy — Global-to-Local Intersection:
  1. Load the full normalised audio and the VAD segments from segments.json
  2. Run pyannote.audio on the FULL audio to get globally consistent
     speaker labels (required for cross-segment identity consistency)
  3. Intersect each VAD segment with the diarization turns:
       • Single-speaker segments → assign speaker_id directly
       • Multi-speaker segments  → split at speaker-change boundaries
  4. Renumber segment IDs sequentially and save back to segments.json

VRAM Management (4 GB budget):
  pyannote.audio 4.x loads three lightweight models SEQUENTIALLY,
  never concurrently:
    1. Segmentation model  (~80 MB)  — frame-level speaker activity
    2. Embedding model     (~20 MB)  — ECAPA-TDNN speaker vectors
    3. Clustering          (CPU)     — agglomerative grouping
  Peak VRAM usage is ~1.5–2.5 GB, well within a 4 GB GPU.

Usage:
  python src/diarization.py
  python src/diarization.py --hf-token hf_xxx --max-speakers 3
  python src/diarization.py --device cpu
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch
import torchaudio

# WhisperX 3.3.1 forces pyannote.audio 3.1.1, which breaks on recent torchaudio
# (missing torchaudio.AudioMetaData). Patch it before pyannote is imported.
import torchaudio_compat
torchaudio_compat.apply()

# ──────────────────────────────────────────────
#  Logging
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("diarization")

# ──────────────────────────────────────────────
#  Project paths (relative to repo root)
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
SEGMENTS_FILE = ARTIFACTS_DIR / "segments.json"
DEFAULT_AUDIO = PROJECT_ROOT / "data" / "audio_out" / "_temp_normalised.wav"

# ──────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────
MIN_SUB_SEGMENT_SEC = 0.3   # sub-segments shorter than this are merged
PYANNOTE_PIPELINE_ID = "pyannote/speaker-diarization-3.1"

# A VAD segment is only split across speakers when a SECONDARY speaker's
# coverage is substantial — otherwise a brief diarization boundary error would
# sever a single continuous sentence (observed: seg 17 "…لازم تلاقيه في" | "كل
# بيت" split onto two speakers). Both conditions must hold to justify a split.
SPLIT_MIN_SECONDARY_FRAC = 0.25   # ≥25% of the segment's duration, AND
SPLIT_MIN_SECONDARY_SEC = 0.7     # ≥0.7s of absolute coverage


# ══════════════════════════════════════════════
#  Step 1 — Load inputs
# ══════════════════════════════════════════════
def load_segments(segments_path: Path) -> dict:
    """
    Read the VAD-generated segments.json data contract.

    Validates that the file exists and contains a 'segments' array.
    Raises SystemExit on I/O or structural errors.
    """
    if not segments_path.is_file():
        log.error("Segments file not found: %s", segments_path)
        sys.exit(1)

    with open(segments_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    if "segments" not in data or not isinstance(data["segments"], list):
        log.error("Invalid segments.json — missing or malformed 'segments' array.")
        sys.exit(1)

    log.info("Loaded %d segment(s) from %s", len(data["segments"]), segments_path.name)
    return data


def load_audio(audio_path: Path) -> tuple:
    """
    Read the full normalised WAV file.

    Returns (waveform_tensor, sample_rate).
    Uses the soundfile backend for Windows compatibility.
    """
    if not audio_path.is_file():
        log.error("Audio file not found: %s", audio_path)
        sys.exit(1)

    waveform, sample_rate = torchaudio.load(str(audio_path), backend="soundfile")

    # Ensure mono — average channels if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    duration_sec = waveform.shape[1] / sample_rate
    log.info(
        "Loaded audio: %.1fs at %d Hz (%s)",
        duration_sec, sample_rate, audio_path.name,
    )
    return waveform, sample_rate


# ══════════════════════════════════════════════
#  Step 2 — Load the pyannote diarization pipeline
# ══════════════════════════════════════════════
def resolve_hf_token(cli_token: Optional[str] = None) -> str:
    """
    Resolve the Hugging Face access token from (in priority order):
      1. Explicit CLI flag  --hf-token
      2. Environment variable  HF_TOKEN

    Raises SystemExit if neither source provides a token.
    """
    token = cli_token or os.environ.get("HF_TOKEN")

    if not token:
        log.error(
            "Hugging Face token required but not found.\n"
            "  Option A: set environment variable  HF_TOKEN\n"
            "            $env:HF_TOKEN = 'hf_your_token_here'   (PowerShell)\n"
            "  Option B: pass  --hf-token hf_your_token_here\n\n"
            "  You must also accept the model licenses at:\n"
            "    https://huggingface.co/pyannote/speaker-diarization-3.1\n"
            "    https://huggingface.co/pyannote/segmentation-3.0"
        )
        sys.exit(1)

    log.info("HF token resolved (%s…%s)", token[:5], token[-4:])
    return token


def resolve_device(requested: str) -> torch.device:
    """
    Determine the compute device with a VRAM safety check.

    If CUDA is requested but unavailable or has < 1.5 GB free,
    falls back to CPU with a warning.
    """
    if requested == "cpu":
        log.info("Device: CPU (explicitly requested)")
        return torch.device("cpu")

    if not torch.cuda.is_available():
        log.warning("CUDA not available — falling back to CPU.")
        return torch.device("cpu")

    # VRAM safety check
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_gb = free_bytes / (1024 ** 3)
    total_gb = total_bytes / (1024 ** 3)
    log.info("GPU VRAM: %.2f GB free / %.2f GB total", free_gb, total_gb)

    if free_gb < 1.5:
        log.warning(
            "Only %.2f GB VRAM free — pyannote needs ~1.5–2.5 GB. "
            "Falling back to CPU. Close other GPU applications to use CUDA.",
            free_gb,
        )
        return torch.device("cpu")

    log.info("Device: CUDA (%.2f GB free — sufficient for pyannote)", free_gb)
    return torch.device("cuda")


def load_diarization_pipeline(
    hf_token: str,
    device: torch.device,
):
    """
    Initialise the pyannote.audio speaker diarization pipeline.

    The pipeline is loaded onto the resolved device (CUDA or CPU).
    Model weights are downloaded/cached on first run via Hugging Face Hub.

    Returns the instantiated Pipeline object.
    """
    # Late import — avoid import errors if pyannote is not installed
    try:
        from pyannote.audio import Pipeline
    except ImportError:
        log.error(
            "pyannote.audio is not installed.\n"
            "  Install it with:  pip install 'pyannote.audio>=4.0'\n"
            "  Or run:           pip install -r requirements.txt"
        )
        sys.exit(1)

    log.info("Loading pyannote pipeline: %s", PYANNOTE_PIPELINE_ID)
    log.info("  (First run will download ~300 MB of model weights)")

    try:
        try:
            pipeline = Pipeline.from_pretrained(
                PYANNOTE_PIPELINE_ID,
                token=hf_token,
            )
        except TypeError:
            # pyannote 3.1.1 (pinned transitively by whisperx) uses the
            # older `use_auth_token` kwarg instead of `token` (pyannote 4.x).
            pipeline = Pipeline.from_pretrained(
                PYANNOTE_PIPELINE_ID,
                use_auth_token=hf_token,
            )
    except Exception as exc:
        # Common failure: token lacks access or licenses not accepted
        error_str = str(exc).lower()
        if "401" in error_str or "403" in error_str or "access" in error_str:
            log.error(
                "Authentication failed. Ensure your HF token is valid AND "
                "you have accepted the model licenses at:\n"
                "  https://huggingface.co/pyannote/speaker-diarization-3.1\n"
                "  https://huggingface.co/pyannote/segmentation-3.0"
            )
        else:
            log.error("Failed to load pyannote pipeline: %s", exc)
        sys.exit(1)

    # Move pipeline to the target device
    pipeline.to(device)
    log.info("Pipeline loaded and moved to %s", device)

    return pipeline


# ══════════════════════════════════════════════
#  Step 3 — Run diarization on the full audio
# ══════════════════════════════════════════════
def run_diarization(
    pipeline,
    waveform: torch.Tensor,
    sample_rate: int,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
) -> list[dict]:
    """
    Execute the pyannote diarization pipeline on the full audio waveform.

    Parameters
    ----------
    pipeline : pyannote.audio.Pipeline
        The loaded diarization pipeline.
    waveform : torch.Tensor
        Shape (1, num_samples) — mono audio at *sample_rate* Hz.
    sample_rate : int
        Sampling rate of the waveform.
    min_speakers : int, optional
        Minimum expected number of speakers (helps clustering accuracy).
    max_speakers : int, optional
        Maximum expected number of speakers.

    Returns
    -------
    List of diarization turns, each a dict with keys:
        - start   : float  (seconds)
        - end     : float  (seconds)
        - speaker : str    (e.g. "SPEAKER_00")
    """
    log.info("Running diarization on full audio …")

    # Build the kwargs for the pipeline call
    pipeline_kwargs = {}
    if min_speakers is not None:
        pipeline_kwargs["min_speakers"] = min_speakers
        log.info("  min_speakers = %d", min_speakers)
    if max_speakers is not None:
        pipeline_kwargs["max_speakers"] = max_speakers
        log.info("  max_speakers = %d", max_speakers)

    # pyannote expects a dict with "waveform" and "sample_rate" keys
    audio_input = {
        "waveform": waveform,
        "sample_rate": sample_rate,
    }

    diarization = pipeline(audio_input, **pipeline_kwargs)

    # pyannote 4.x returns a DiarizeOutput dataclass (Annotation under
    # .speaker_diarization); pyannote 3.1.x returns the Annotation directly.
    # WhisperX 3.3.1 pins 3.1.1, so support both shapes.
    annotation = getattr(diarization, "speaker_diarization", diarization)

    turns = []
    for turn, _track, speaker in annotation.itertracks(yield_label=True):
        turns.append({
            "start": round(turn.start, 3),
            "end": round(turn.end, 3),
            "speaker": speaker,
        })

    # Sort by start time for deterministic processing
    turns.sort(key=lambda t: (t["start"], t["end"]))

    # Collect unique speakers
    unique_speakers = sorted(set(t["speaker"] for t in turns))
    log.info(
        "Diarization complete: %d turn(s), %d unique speaker(s): %s",
        len(turns), len(unique_speakers), ", ".join(unique_speakers),
    )

    return turns


# ══════════════════════════════════════════════
#  Step 4 — Normalise speaker labels
# ══════════════════════════════════════════════
def normalise_speaker_labels(turns: list[dict]) -> dict:
    """
    Map pyannote's raw speaker labels (e.g., "SPEAKER_00", "SPEAKER_01")
    to our pipeline's canonical format ("SPEAKER_01", "SPEAKER_02", …).

    pyannote uses 0-indexed labels; our pipeline uses 1-indexed for
    human readability and consistency with segment_id numbering.

    Returns a mapping dict: { raw_label: canonical_label }.
    """
    raw_labels = sorted(set(t["speaker"] for t in turns))
    mapping = {}
    for idx, raw in enumerate(raw_labels, start=1):
        mapping[raw] = f"SPEAKER_{idx:02d}"

    log.info("Speaker label mapping: %s", mapping)
    return mapping


# ══════════════════════════════════════════════
#  Step 5 — Intersect diarization turns with
#           VAD segments and split multi-speaker
#           segments
# ══════════════════════════════════════════════
def _compute_overlap(seg_start: float, seg_end: float,
                     turn_start: float, turn_end: float) -> float:
    """Return the overlap duration (seconds) between a segment and a turn."""
    overlap_start = max(seg_start, turn_start)
    overlap_end = min(seg_end, turn_end)
    return max(0.0, overlap_end - overlap_start)


def intersect_and_split(
    segments: list[dict],
    turns: list[dict],
    label_map: dict,
    min_sub_segment_sec: float = MIN_SUB_SEGMENT_SEC,
) -> list[dict]:
    """
    Cross-reference pyannote diarization turns with VAD segments.

    For each VAD segment:
      - Find all diarization turns that overlap with it
      - If only one speaker overlaps → assign that speaker_id
      - If multiple speakers overlap → split the segment at speaker
        change boundaries, creating sub-segments

    Sub-segments shorter than *min_sub_segment_sec* are merged into
    the adjacent speaker's turn to avoid fragments too short for
    downstream TTS.

    Parameters
    ----------
    segments : list[dict]
        The VAD segments from segments.json.
    turns : list[dict]
        Diarization turns from pyannote (start, end, speaker).
    label_map : dict
        Raw → canonical speaker label mapping.
    min_sub_segment_sec : float
        Minimum duration for a sub-segment after splitting.

    Returns
    -------
    New list of segments with speaker_id populated. May contain more
    segments than the input if multi-speaker segments were split.
    """
    new_segments = []
    split_count = 0
    dominant_kept_count = 0   # multi-speaker segments NOT split (minor secondary)

    for seg in segments:
        seg_start = seg["start_time"]
        seg_end = seg["end_time"]
        original_id = seg["segment_id"]

        # ── Find all overlapping diarization turns ────────────
        overlapping_turns = []
        for turn in turns:
            overlap = _compute_overlap(seg_start, seg_end, turn["start"], turn["end"])
            if overlap > 0:
                overlapping_turns.append({
                    "start": max(seg_start, turn["start"]),
                    "end": min(seg_end, turn["end"]),
                    "speaker": label_map[turn["speaker"]],
                    "overlap": overlap,
                })

        # ── Case 1: No diarization coverage ───────────────────
        # (Rare — segment falls in a gap between diarization turns.
        #  Assign "SPEAKER_UNKNOWN" and preserve the segment.)
        if not overlapping_turns:
            log.warning(
                "Segment #%d (%.1fs–%.1fs): no diarization coverage — "
                "assigning SPEAKER_UNKNOWN",
                original_id, seg_start, seg_end,
            )
            result_seg = dict(seg)
            result_seg["speaker_id"] = "SPEAKER_UNKNOWN"
            new_segments.append(result_seg)
            continue

        # ── Aggregate per-speaker coverage within this segment ──
        coverage: dict[str, float] = {}
        for t in overlapping_turns:
            coverage[t["speaker"]] = coverage.get(t["speaker"], 0.0) + t["overlap"]
        unique_speakers = set(coverage)
        dominant_speaker = max(coverage, key=coverage.get)
        seg_duration = seg_end - seg_start

        # A secondary speaker justifies a split only if its coverage is BOTH
        # ≥ SPLIT_MIN_SECONDARY_FRAC of the segment AND ≥ SPLIT_MIN_SECONDARY_SEC.
        def _justifies_split(spk: str) -> bool:
            cov = coverage[spk]
            frac = cov / seg_duration if seg_duration > 0 else 0.0
            return cov >= SPLIT_MIN_SECONDARY_SEC and frac >= SPLIT_MIN_SECONDARY_FRAC

        secondary_justifies = any(
            _justifies_split(spk) for spk in unique_speakers if spk != dominant_speaker
        )

        # ── Case 2: Single speaker, or no secondary speaker meets the
        #            split threshold → assign whole segment to the dominant. ──
        if len(unique_speakers) == 1 or not secondary_justifies:
            if len(unique_speakers) > 1:
                dominant_kept_count += 1
                log.info(
                    "Segment #%d (%.1fs–%.1fs, %.1fs): %d speakers overlap but "
                    "secondary coverage below threshold — assigning dominant %s",
                    original_id, seg_start, seg_end, seg["duration"],
                    len(unique_speakers), dominant_speaker,
                )
            result_seg = dict(seg)
            result_seg["speaker_id"] = dominant_speaker
            new_segments.append(result_seg)
            continue

        # ── Case 3: Genuine multi-speaker — split at boundaries ──
        split_count += 1
        log.info(
            "Segment #%d (%.1fs–%.1fs, %.1fs): splitting across %d speakers: %s",
            original_id, seg_start, seg_end, seg["duration"],
            len(unique_speakers), ", ".join(sorted(unique_speakers)),
        )

        # Sort overlapping turns by start time
        overlapping_turns.sort(key=lambda t: t["start"])

        # Merge consecutive turns from the same speaker
        merged_turns = [overlapping_turns[0].copy()]
        for turn in overlapping_turns[1:]:
            prev = merged_turns[-1]
            if turn["speaker"] == prev["speaker"]:
                # Extend the previous turn
                prev["end"] = turn["end"]
                prev["overlap"] = round(prev["end"] - prev["start"], 3)
            else:
                merged_turns.append(turn.copy())

        # Filter out sub-segments that are too short — merge them
        # into the nearest neighbour
        final_turns = _merge_short_subsegments(merged_turns, min_sub_segment_sec)

        # Create sub-segments
        for sub_turn in final_turns:
            sub_start = round(sub_turn["start"], 3)
            sub_end = round(sub_turn["end"], 3)
            sub_duration = round(sub_end - sub_start, 3)

            sub_seg = {
                "segment_id": None,  # will be renumbered later
                "start_time": sub_start,
                "end_time": sub_end,
                "duration": sub_duration,
                "speaker_id": sub_turn["speaker"],
                "text": seg.get("text"),  # preserve if already transcribed
                "_original_segment_id": original_id,
            }
            new_segments.append(sub_seg)

    log.info(
        "Intersection complete: %d input segment(s) → %d output segment(s) "
        "(%d split, %d multi-speaker kept whole)",
        len(segments), len(new_segments), split_count, dominant_kept_count,
    )

    return new_segments


# ══════════════════════════════════════════════
#  Step 5b — Re-join over-split fragments
# ══════════════════════════════════════════════
CONTIGUITY_TOLERANCE_SEC = 0.05   # fragments within this gap count as contiguous


def rejoin_same_origin_fragments(segments: list[dict]) -> list[dict]:
    """
    Merge adjacent fragments that came from the SAME original VAD segment and
    resolve to the SAME speaker — undoing over-splitting that would otherwise
    sever one continuous utterance into independently-translated pieces.

    Conservative by design (same-origin only):
      - both fragments must carry the same non-null ``_original_segment_id``
      - both must have the same ``speaker_id``
      - they must be time-contiguous (gap ≤ CONTIGUITY_TOLERANCE_SEC)

    Fragments without an ``_original_segment_id`` (i.e. never split) and
    boundaries between different origins or speakers are left untouched. The
    merged segment spans [first.start, last.end]; its ``_merged_from`` records
    each contributing fragment's origin id and time range for later review.

    NOTE: because Case 3 already coalesces consecutive same-speaker turns
    within a split, this pass is primarily a safety net and fires rarely on
    well-behaved diarization output.
    """
    if not segments:
        return segments

    merged: list[dict] = []
    for seg in segments:
        prev = merged[-1] if merged else None
        origin = seg.get("_original_segment_id")

        can_merge = (
            prev is not None
            and origin is not None
            and prev.get("_original_segment_id") == origin
            and prev.get("speaker_id") == seg.get("speaker_id")
            and (seg["start_time"] - prev["end_time"]) <= CONTIGUITY_TOLERANCE_SEC
        )

        if not can_merge:
            merged.append(dict(seg))
            continue

        # ── Merge seg into prev ────────────────────────────────
        # Seed _merged_from with prev's own span on first merge.
        if "_merged_from" not in prev:
            prev["_merged_from"] = [{
                "original_segment_id": prev.get("_original_segment_id"),
                "start_time": prev["start_time"],
                "end_time": prev["end_time"],
            }]
        prev["_merged_from"].append({
            "original_segment_id": origin,
            "start_time": seg["start_time"],
            "end_time": seg["end_time"],
        })

        prev["end_time"] = seg["end_time"]
        prev["duration"] = round(prev["end_time"] - prev["start_time"], 3)

        # Preserve any transcribed text (usually None at Phase B).
        prev_text = (prev.get("text") or "").strip()
        seg_text = (seg.get("text") or "").strip()
        if seg_text and seg_text != prev_text:
            prev["text"] = (prev_text + " " + seg_text).strip() if prev_text else seg_text

    n_rejoined = sum(1 for s in merged if "_merged_from" in s)
    if n_rejoined:
        log.info(
            "Re-join pass: %d segment(s) → %d after merging %d over-split group(s)",
            len(segments), len(merged), n_rejoined,
        )
    return merged


def _merge_short_subsegments(
    turns: list[dict],
    min_duration: float,
) -> list[dict]:
    """
    Merge sub-segments shorter than *min_duration* into adjacent turns.

    Strategy:
      - If a short sub-segment has a neighbour, merge into the
        longer neighbour (preserves the dominant speaker)
      - If it is the only sub-segment, keep it regardless of duration
    """
    if len(turns) <= 1:
        return turns

    # Iteratively merge until no short segments remain
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(turns):
            duration = turns[i]["end"] - turns[i]["start"]
            if duration < min_duration and len(turns) > 1:
                # Determine merge target: prefer the longer neighbour
                if i == 0:
                    merge_idx = 1
                elif i == len(turns) - 1:
                    merge_idx = i - 1
                else:
                    # Pick the longer neighbour
                    prev_dur = turns[i - 1]["end"] - turns[i - 1]["start"]
                    next_dur = turns[i + 1]["end"] - turns[i + 1]["start"]
                    merge_idx = (i - 1) if prev_dur >= next_dur else (i + 1)

                # Merge: expand the target and remove the short segment
                target = turns[merge_idx]
                short = turns[i]
                target["start"] = min(target["start"], short["start"])
                target["end"] = max(target["end"], short["end"])
                target["overlap"] = round(target["end"] - target["start"], 3)

                turns.pop(i)
                changed = True
                # Don't increment i — re-check at the same index
            else:
                i += 1

    return turns


# ══════════════════════════════════════════════
#  Step 6 — Renumber segment IDs
# ══════════════════════════════════════════════
def renumber_segments(segments: list[dict]) -> list[dict]:
    """
    Assign sequential segment_id values (1, 2, 3, …) and recalculate
    durations for consistency after splitting.
    """
    for idx, seg in enumerate(segments, start=1):
        seg["segment_id"] = idx
        seg["duration"] = round(seg["end_time"] - seg["start_time"], 3)
    return segments


# ══════════════════════════════════════════════
#  Step 7 — Save updated segments
# ══════════════════════════════════════════════
def save_segments(data: dict, output_path: Path) -> None:
    """
    Write the diarised segments back to segments.json.

    Preserves all existing metadata and adds diarization-specific fields.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)

    log.info("Saved %d segment(s) → %s", data["total_segments"], output_path)


# ══════════════════════════════════════════════
#  Orchestration — run the full Phase B pipeline
# ══════════════════════════════════════════════
def run_phase_b(
    audio_path: Path,
    segments_path: Path,
    output_path: Path,
    hf_token: str,
    device: torch.device,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    min_sub_segment_sec: float = MIN_SUB_SEGMENT_SEC,
) -> dict:
    """
    End-to-end Phase B: load → diarize → intersect → save.

    This function is designed to be called both from the CLI
    and programmatically from a master pipeline orchestrator.

    Returns the updated segments data dict.
    """
    # ── 1. Load inputs ────────────────────────────────────────
    print(f"\n[1/4]  Loading inputs")
    segments_data = load_segments(segments_path)
    waveform, sample_rate = load_audio(audio_path)

    # ── 2. Load diarization pipeline ──────────────────────────
    print(f"\n[2/4]  Loading pyannote diarization pipeline")
    pipeline = load_diarization_pipeline(hf_token, device)

    # ── 3. Run diarization ────────────────────────────────────
    print(f"\n[3/4]  Running speaker diarization")
    turns = run_diarization(
        pipeline,
        waveform,
        sample_rate,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )

    # Free GPU memory after diarization is complete
    del pipeline
    if device.type == "cuda":
        torch.cuda.empty_cache()
        log.info("GPU memory released after diarization.")

    # ── 4. Intersect & split ──────────────────────────────────
    print(f"\n[4/4]  Intersecting diarization with VAD segments")
    label_map = normalise_speaker_labels(turns)

    original_count = len(segments_data["segments"])
    new_segments = intersect_and_split(
        segments_data["segments"],
        turns,
        label_map,
        min_sub_segment_sec=min_sub_segment_sec,
    )

    # Re-join fragments over-split from the same original VAD segment onto the
    # same speaker (prevents one utterance from being translated in pieces).
    post_split_count = len(new_segments)
    new_segments = rejoin_same_origin_fragments(new_segments)
    rejoined_count = post_split_count - len(new_segments)

    # Renumber and update the data contract
    new_segments = renumber_segments(new_segments)

    segments_data["segments"] = new_segments
    segments_data["total_segments"] = len(new_segments)

    # ── Add diarization metadata ──────────────────────────────
    unique_speakers = sorted(set(
        s["speaker_id"] for s in new_segments
        if s["speaker_id"] != "SPEAKER_UNKNOWN"
    ))
    segments_data["diarization_model"] = PYANNOTE_PIPELINE_ID
    segments_data["diarization_completed_at"] = datetime.now(timezone.utc).isoformat()
    segments_data["diarization_stats"] = {
        "original_segment_count": original_count,
        "final_segment_count": len(new_segments),
        "segments_split": post_split_count - original_count,
        "segments_rejoined": rejoined_count,
        "unique_speakers": len(unique_speakers),
        "speaker_labels": unique_speakers,
    }

    # ── Save ──────────────────────────────────────────────────
    save_segments(segments_data, output_path)

    return segments_data


# ══════════════════════════════════════════════
#  CLI entry-point
# ══════════════════════════════════════════════
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dubly ME — Phase B: Speaker Diarization (pyannote.audio 4.x)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python src/diarization.py\n"
            "  python src/diarization.py --hf-token hf_xxx --max-speakers 3\n"
            "  python src/diarization.py --device cpu\n"
            "\n"
            "Hugging Face token:\n"
            "  Set via env var:   $env:HF_TOKEN = 'hf_your_token'  (PowerShell)\n"
            "  Or via CLI flag:   --hf-token hf_your_token\n"
            "\n"
            "  You MUST accept the model licenses before first use:\n"
            "    https://huggingface.co/pyannote/speaker-diarization-3.1\n"
            "    https://huggingface.co/pyannote/segmentation-3.0"
        ),
    )
    parser.add_argument(
        "--input-audio",
        default=str(DEFAULT_AUDIO),
        help="Path to the normalised WAV file  (default: data/audio_out/_temp_normalised.wav)",
    )
    parser.add_argument(
        "--input-segments",
        default=str(SEGMENTS_FILE),
        help="Path to the VAD segments JSON  (default: artifacts/segments.json)",
    )
    parser.add_argument(
        "--output",
        default=str(SEGMENTS_FILE),
        help="Output JSON path  (default: artifacts/segments.json — overwrites in-place)",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Hugging Face access token  (default: reads from HF_TOKEN env var)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Compute device  (default: cuda — auto-falls back to CPU if needed)",
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        default=None,
        help="Minimum expected number of speakers  (default: auto-detect)",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=None,
        help="Maximum expected number of speakers  (default: auto-detect)",
    )
    parser.add_argument(
        "--min-sub-segment",
        type=float,
        default=MIN_SUB_SEGMENT_SEC,
        help=(
            f"Minimum sub-segment duration (seconds) after splitting  "
            f"(default: {MIN_SUB_SEGMENT_SEC})"
        ),
    )
    args = parser.parse_args()

    # ── Validate speaker constraints ──────────────────────────
    if args.min_speakers is not None and args.max_speakers is not None:
        if args.min_speakers > args.max_speakers:
            log.error(
                "--min-speakers (%d) cannot exceed --max-speakers (%d)",
                args.min_speakers, args.max_speakers,
            )
            sys.exit(1)

    if args.min_speakers is not None and args.min_speakers < 1:
        log.error("--min-speakers must be at least 1")
        sys.exit(1)

    # ── Resolve HF token ─────────────────────────────────────
    hf_token = resolve_hf_token(args.hf_token)

    # ── Resolve device ───────────────────────────────────────
    device = resolve_device(args.device)

    # ── Run Phase B ──────────────────────────────────────────
    segments_data = run_phase_b(
        audio_path=Path(args.input_audio),
        segments_path=Path(args.input_segments),
        output_path=Path(args.output),
        hf_token=hf_token,
        device=device,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        min_sub_segment_sec=args.min_sub_segment,
    )

    # ── Summary ──────────────────────────────────────────────
    stats = segments_data["diarization_stats"]
    print()
    print(f"{'═' * 58}")
    print(f"  ✅  Phase B: Speaker Diarization Complete")
    print(f"{'─' * 58}")
    print(f"  Model             : {PYANNOTE_PIPELINE_ID}")
    print(f"  Device            : {device}")
    print(f"  Segments (before) : {stats['original_segment_count']}")
    print(f"  Segments (after)  : {stats['final_segment_count']}")
    print(f"  Segments split    : {stats['segments_split']}")
    print(f"  Unique speakers   : {stats['unique_speakers']}")
    print(f"  Speaker labels    : {', '.join(stats['speaker_labels'])}")
    print(f"  Output            : {args.output}")
    print(f"{'═' * 58}")

    # ── Preview first 8 segments ──────────────────────────────
    print()
    for seg in segments_data["segments"][:8]:
        split_tag = ""
        if "_original_segment_id" in seg:
            split_tag = f"  ← split from #{seg['_original_segment_id']}"
        print(
            f"  #{seg['segment_id']:>3d}  "
            f"{seg['start_time']:>8.3f}s → {seg['end_time']:>8.3f}s  "
            f"({seg['duration']:.3f}s)  "
            f"[{seg['speaker_id']}]"
            f"{split_tag}"
        )
    remaining = len(segments_data["segments"]) - 8
    if remaining > 0:
        print(f"  … and {remaining} more segment(s).")


if __name__ == "__main__":
    main()
