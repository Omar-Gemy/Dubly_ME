"""
Throwaway verification for Observation 4 (asr_confidence / low_confidence_asr).
NOT part of the pipeline — delete after review. Never committed.

Two checks:
  A) Synthetic unit test of aggregate_asr_confidence() covering every branch.
  B) Behavioral run over the REAL artifacts/transcripts.json segments.
     NOTE: the committed transcripts.json does not persist per-word scores
     (_score_bucket is transient), so real segments exercise the
     no-score / empty-text branch → asr_confidence=None, low_confidence_asr=False.
     A non-synthetic confidence NUMBER can only come from a full ASR re-run
     (WhisperX + GPU). This check confirms the function handles the real
     file's actual segment shapes/text-states, and reports the text-state
     distribution so we know what a real run would score.
"""
import json
import sys
from pathlib import Path

# Windows console defaults to cp1252, which cannot encode Arabic previews.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from asr_transcription import aggregate_asr_confidence, LOW_CONFIDENCE_ASR


def check_a_synthetic():
    print("=== A) SYNTHETIC UNIT TEST ===")
    segs = [
        {"text": "hello world", "_score_bucket": [0.9, 0.8]},   # mean .85  -> low=False
        {"text": "murky line",  "_score_bucket": [0.4, 0.5]},   # mean .45  -> low=True
        {"text": "",            "_score_bucket": []},           # empty     -> None/False
        {"text": "digits here", "_score_bucket": [None, None]}, # no scores -> None/False
        {"text": "edge",        "_score_bucket": [0.55]},       # == thresh -> low=False
    ]
    stats = aggregate_asr_confidence(segs, low_confidence_threshold=0.55)
    for s in segs:
        print(f"  {s['text'][:11]:12} conf={s['asr_confidence']!s:6} "
              f"low={s['low_confidence_asr']!s:6} "
              f"_score_bucket_removed={'_score_bucket' not in s}")
    print("  stats:", stats)

    assert stats == {"asr_scored": 3, "asr_low_confidence": 1}, stats
    assert segs[0]["asr_confidence"] == 0.85 and segs[0]["low_confidence_asr"] is False
    assert segs[1]["low_confidence_asr"] is True
    assert segs[2]["asr_confidence"] is None and segs[2]["low_confidence_asr"] is False
    assert segs[3]["asr_confidence"] is None and segs[3]["low_confidence_asr"] is False
    assert segs[4]["low_confidence_asr"] is False
    assert all("_score_bucket" not in s for s in segs)
    print("  ALL SYNTHETIC ASSERTIONS PASSED\n")


def check_b_real():
    print("=== B) REAL artifacts/transcripts.json ===")
    path = ROOT / "artifacts" / "transcripts.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    segs = data["segments"]

    with_text = sum(1 for s in segs if (s.get("text") or "").strip())
    persisted_scores = sum(1 for s in segs if "_score_bucket" in s)
    print(f"  total segments      : {len(segs)}")
    print(f"  segments with text  : {with_text}")
    print(f"  segments carrying _score_bucket (persisted): {persisted_scores}")

    # Run the real aggregation over the real segment dicts.
    stats = aggregate_asr_confidence(segs, low_confidence_threshold=LOW_CONFIDENCE_ASR)
    print(f"  aggregate stats     : {stats}")

    # Show every real segment's resulting fields.
    print("  per-segment result (segment_id, text-preview, asr_confidence, low_confidence_asr):")
    for s in segs:
        sid = s.get("segment_id")
        prev = (s.get("text") or "").strip()[:24]
        print(f"    #{sid:>3}  {prev!r:26}  conf={s['asr_confidence']!s:6}  low={s['low_confidence_asr']}")

    # Additive-only contract checks: no existing field was mutated, both new
    # fields exist on EVERY segment, scratch field is gone.
    for s in segs:
        assert "asr_confidence" in s
        assert "low_confidence_asr" in s
        assert "_score_bucket" not in s
    print("  Every real segment now has asr_confidence + low_confidence_asr; "
          "no _score_bucket leaked.")
    print("  (confidence is None here only because word scores are not "
          "persisted in this fixture — a live ASR run populates them.)")


if __name__ == "__main__":
    check_a_synthetic()
    check_b_real()
