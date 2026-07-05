"""
test_asr_gating.py — unit tests for the Phase C post-transcription gate
=======================================================================
Covers the pure, GPU-free functions that implement the 3-tier gate and the
robust word→segment mapping in src/asr_transcription.py:

  - build_speech_intervals        (trusted-speech mask merge)
  - assign_words_to_segments_by_overlap   (max-overlap mapping)
  - apply_segment_gates           (energy + duration gate)

WhisperX / torch are imported lazily inside transcribe_full_file, so importing
the module here does NOT require the heavy ASR stack.

Run standalone (no pytest needed):
    python tests/test_asr_gating.py
Or with pytest:
    pytest tests/test_asr_gating.py
"""

import sys
from pathlib import Path

import numpy as np

# Make src/ importable when run directly.
SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import asr_transcription as asr  # noqa: E402


def _seg(seg_id, start, end, speaker="SPEAKER_01"):
    return {
        "segment_id": seg_id,
        "start_time": start,
        "end_time": end,
        "duration": round(end - start, 3),
        "speaker_id": speaker,
        "text": None,
    }


def _word(token, start, end, score=0.9):
    return {"word": token, "start": start, "end": end, "score": score}


# ──────────────────────────────────────────────
#  build_speech_intervals
# ──────────────────────────────────────────────
def test_build_speech_intervals_merges_overlaps_and_adjacency():
    segs = [
        _seg(1, 0.0, 2.0),
        _seg(2, 1.5, 3.0),   # overlaps #1 → merges to [0, 3]
        _seg(3, 5.0, 6.0),   # disjoint
        _seg(4, 6.0, 6.5),   # touches #3 → merges to [5, 6.5]
    ]
    assert asr.build_speech_intervals(segs) == [(0.0, 3.0), (5.0, 6.5)]


def test_build_speech_intervals_skips_zero_width():
    segs = [_seg(1, 1.0, 1.0), _seg(2, 2.0, 3.0)]
    assert asr.build_speech_intervals(segs) == [(2.0, 3.0)]


# ──────────────────────────────────────────────
#  assign_words_to_segments_by_overlap
# ──────────────────────────────────────────────
def test_boundary_word_goes_to_dominant_speaker():
    p1, p2 = _seg(1, 0.0, 2.0, "S1"), _seg(2, 2.0, 4.0, "S2")
    # Word straddles the boundary but sits mostly in S2 (0.1s vs 0.4s).
    dropped = asr.assign_words_to_segments_by_overlap(
        [p1, p2], [_word("خلاص", 1.9, 2.4)], min_overlap_frac=0.5
    )
    assert dropped == 0
    assert p1["text"] == ""
    assert p2["text"] == "خلاص"


def test_short_turn_is_rescued():
    # Rapid P1 → P2(short) → P1 exchange; P2 is a 0.5s interjection.
    p1a = _seg(1, 0.0, 3.0, "S1")
    p2 = _seg(2, 3.0, 3.5, "S2")
    p1b = _seg(3, 3.5, 6.0, "S1")
    words = [
        _word("فتحة", 0.4, 0.9),
        _word("حلو", 3.08, 3.26),
        _word("ده", 3.28, 3.46),
        _word("حاضر", 3.7, 4.1),
    ]
    dropped = asr.assign_words_to_segments_by_overlap([p1a, p2, p1b], words)
    assert dropped == 0
    assert p2["text"] == "حلو ده"      # short turn kept its own words
    assert p1a["text"] == "فتحة"
    assert p1b["text"] == "حاضر"


def test_gap_background_word_is_dropped():
    p1, p2 = _seg(1, 0.0, 2.0), _seg(2, 4.0, 6.0)
    # A background word (radio/Quran) in the 2s→4s gap overlaps no segment.
    dropped = asr.assign_words_to_segments_by_overlap(
        [p1, p2], [_word("قرآن", 2.6, 2.9)]
    )
    assert dropped == 1
    assert p1["text"] == "" and p2["text"] == ""


def test_word_below_overlap_floor_is_dropped():
    seg = _seg(1, 0.0, 2.0)
    # 2.0s word only 0.1s inside the segment → 5% < 50% floor → dropped.
    dropped = asr.assign_words_to_segments_by_overlap(
        [seg], [_word("noise", 1.9, 3.9)], min_overlap_frac=0.5
    )
    assert dropped == 1
    assert seg["text"] == ""


def test_disabling_overlap_floor_keeps_touching_words():
    seg = _seg(1, 0.0, 2.0)
    dropped = asr.assign_words_to_segments_by_overlap(
        [seg], [_word("edge", 1.9, 3.9)], min_overlap_frac=0.0
    )
    assert dropped == 0
    assert seg["text"] == "edge"


# ──────────────────────────────────────────────
#  apply_segment_gates
# ──────────────────────────────────────────────
def _audio_with(sr, loud_spans):
    """3s of silence with 0.5-amplitude tone in the given [start,end] spans."""
    audio = np.zeros(3 * sr, dtype=np.float32)
    for s, e in loud_spans:
        audio[int(s * sr):int(e * sr)] = 0.5
    return audio


def test_energy_gate_clears_silent_segment():
    sr = 16000
    audio = _audio_with(sr, loud_spans=[(1.0, 2.0)])  # only [1,2] has energy
    silent = _seg(1, 0.0, 1.0)
    silent["text"] = "hallucinated on silence"
    loud = _seg(2, 1.0, 2.0)
    loud["text"] = "real speech"

    stats = asr.apply_segment_gates([silent, loud], audio, sample_rate=sr)

    assert silent["text"] == "" and silent["_skipped_low_energy"] is True
    assert loud["text"] == "real speech"
    assert stats["gated_low_energy"] == 1


def test_duration_gate_clears_short_segment():
    sr = 16000
    audio = _audio_with(sr, loud_spans=[(2.0, 2.3)])
    short = _seg(1, 2.0, 2.3)          # 0.3s < 0.5s floor
    short["text"] = "too short"

    stats = asr.apply_segment_gates([short], audio, sample_rate=sr)

    assert short["text"] == "" and short["_skipped_too_short"] is True
    assert stats["gated_too_short"] == 1


def test_gates_ignore_already_empty_segments():
    sr = 16000
    audio = _audio_with(sr, loud_spans=[])
    empty = _seg(1, 0.0, 1.0)
    empty["text"] = ""
    stats = asr.apply_segment_gates([empty], audio, sample_rate=sr)
    assert stats == {"gated_low_energy": 0, "gated_too_short": 0}
    assert "_skipped_low_energy" not in empty


# ──────────────────────────────────────────────
#  Standalone runner (no pytest dependency)
# ──────────────────────────────────────────────
def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for fn in tests:
        try:
            fn()
            print(f"  [PASS] {fn.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"  [FAIL] {fn.__name__}  — {exc or 'assertion failed'}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  [ERROR] {fn.__name__}  — {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return failures


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
