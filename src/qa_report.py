"""
qa_report.py — Phase F3: Timing QA Report
=========================================
Generate a human-readable QA report summarising the dubbing
pipeline results, with focus on timing accuracy, stretch ratios,
and quality flags.

Strategy (approved by Tech Lead):
  - Manual visual review only for lip-sync (Q3: Option B)
  - No automated lip-sync analysis
  - Report flags segments that overflowed their timing budget
  - Report includes per-segment stretch ratios and tier classification

Inputs:
  - artifacts/tts_manifest.json       (Phase E data contract)
  - artifacts/stretch_manifest.json   (Phase F1 data contract)
  - artifacts/mix_manifest.json       (Phase F2 data contract)
  - artifacts/segments.json           (Phase B data contract)

Outputs:
  - artifacts/qa_report.json          (machine-readable QA data)
  - artifacts/qa_report.md            (human-readable QA report)

Usage:
  python src/qa_report.py
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pipeline_core

# ──────────────────────────────────────────────
#  Project paths
# ──────────────────────────────────────────────
PROJECT_ROOT     = pipeline_core.PROJECT_ROOT
ARTIFACTS_DIR    = pipeline_core.ARTIFACTS_DIR

SEGMENTS_FILE    = ARTIFACTS_DIR / "segments.json"
TTS_MANIFEST     = ARTIFACTS_DIR / "tts_manifest.json"
STRETCH_MANIFEST = ARTIFACTS_DIR / "stretch_manifest.json"
MIX_MANIFEST     = ARTIFACTS_DIR / "mix_manifest.json"

QA_REPORT_JSON   = ARTIFACTS_DIR / "qa_report.json"
QA_REPORT_MD     = ARTIFACTS_DIR / "qa_report.md"


# ──────────────────────────────────────────────
#  Load all manifests
# ──────────────────────────────────────────────
def load_manifests() -> dict:
    """Load all pipeline manifests. Returns a dict of loaded data."""
    manifests = {}

    for name, path in [
        ("segments", SEGMENTS_FILE),
        ("tts", TTS_MANIFEST),
        ("stretch", STRETCH_MANIFEST),
        ("mix", MIX_MANIFEST),
    ]:
        if path.is_file():
            with open(path, "r", encoding="utf-8") as fh:
                manifests[name] = json.load(fh)
            print(f"  ✔ Loaded {name}: {path}")
        else:
            manifests[name] = None
            print(f"  ⚠ Not found: {path}")

    return manifests


# ──────────────────────────────────────────────
#  Analyse timing quality
# ──────────────────────────────────────────────
def analyse_timing(manifests: dict) -> dict:
    """
    Analyse timing quality across the pipeline.
    Returns a structured QA analysis dict.
    """
    stretch_data = manifests.get("stretch")
    tts_data = manifests.get("tts")
    mix_data = manifests.get("mix")
    segments_data = manifests.get("segments")

    analysis = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_phases_present": {
            "segments": segments_data is not None,
            "tts": tts_data is not None,
            "stretch": stretch_data is not None,
            "mix": mix_data is not None,
        },
        "design_decisions": {
            "Q1_overflow_policy": "Option A — Prioritise quality, allow slight overflow. Max 2.0× compression.",
            "Q2_skipped_segments": "Option B — Original Arabic audio passthrough.",
            "Q3_lipsync_qa": "Option B — Manual visual review only.",
            "Q4_audio_codec": "Option A — AAC 192k.",
        },
        "segment_details": [],
        "summary": {},
    }

    if stretch_data is None:
        analysis["summary"]["error"] = "stretch_manifest.json not found"
        return analysis

    # Build lookup tables
    tts_lookup = {}
    if tts_data:
        for seg in tts_data["segments"]:
            tts_lookup[seg["segment_id"]] = seg

    mix_lookup = {}
    if mix_data:
        for entry in mix_data.get("placement_log", []):
            mix_lookup[entry["segment_id"]] = entry

    timing_lookup = {}
    if segments_data:
        for seg in segments_data["segments"]:
            timing_lookup[seg["segment_id"]] = seg

    # Analyse each segment
    total = 0
    fits_count = 0
    overflow_count = 0
    over_cap_count = 0
    skipped_count = 0
    error_count = 0
    stretch_ratios = []

    for seg in stretch_data["segments"]:
        seg_id = seg["segment_id"]
        total += 1

        detail = {
            "segment_id": seg_id,
            "status": seg["status"],
            "tier": seg.get("tier", "unknown"),
        }

        # Get timing info
        timing = timing_lookup.get(seg_id, {})
        detail["start_time"] = timing.get("start_time")
        detail["original_duration_s"] = seg.get("original_duration_s")

        if seg["status"] == "success":
            tts_dur = seg.get("tts_duration_s", 0)
            output_dur = seg.get("output_duration_s", 0)
            orig_dur = seg.get("original_duration_s", 0)
            ratio = seg.get("stretch_ratio", 1.0)
            overflow = seg.get("overflow_s", 0)

            detail["tts_duration_s"] = tts_dur
            detail["output_duration_s"] = output_dur
            detail["stretch_ratio"] = ratio
            detail["speed_factor"] = seg.get("speed_factor")
            detail["overflow_s"] = overflow

            stretch_ratios.append(ratio)

            # Classify fit quality
            if overflow > 0:
                detail["fit_quality"] = "OVERFLOW"
                overflow_count += 1
                over_cap_count += 1
            elif output_dur <= orig_dur * 1.05:
                detail["fit_quality"] = "GOOD"
                fits_count += 1
            else:
                detail["fit_quality"] = "SLIGHT_OVERFLOW"
                overflow_count += 1

            # Mix placement info
            mix_entry = mix_lookup.get(seg_id, {})
            detail["mix_type"] = mix_entry.get("type", "unknown")

        elif seg["status"] == "skipped":
            detail["reason"] = seg.get("reason", "unknown")
            skipped_count += 1

            mix_entry = mix_lookup.get(seg_id, {})
            detail["mix_type"] = mix_entry.get("type", "unknown")

        else:
            detail["error"] = seg.get("error", "unknown")
            error_count += 1

        analysis["segment_details"].append(detail)

    # Summary statistics
    avg_ratio = (
        sum(stretch_ratios) / len(stretch_ratios)
        if stretch_ratios else 0
    )
    max_ratio = max(stretch_ratios) if stretch_ratios else 0

    analysis["summary"] = {
        "total_segments": total,
        "dubbed_segments": total - skipped_count - error_count,
        "skipped_segments": skipped_count,
        "error_segments": error_count,
        "segments_fit_budget": fits_count,
        "segments_overflow": overflow_count,
        "segments_over_cap": over_cap_count,
        "timing_fit_rate": (
            f"{fits_count / (total - skipped_count - error_count) * 100:.1f}%"
            if (total - skipped_count - error_count) > 0 else "N/A"
        ),
        "avg_stretch_ratio": round(avg_ratio, 3),
        "max_stretch_ratio": round(max_ratio, 3),
        "max_allowed_ratio": stretch_data.get("max_stretch_ratio", 2.0),
    }

    # Mix stats
    if mix_data:
        analysis["summary"]["dubbed_placed"] = mix_data.get(
            "dubbed_segments", 0
        )
        analysis["summary"]["arabic_passthrough_placed"] = mix_data.get(
            "arabic_passthrough", 0
        )
        analysis["summary"]["output_audio"] = mix_data.get("output_audio")
        analysis["summary"]["output_video"] = mix_data.get("output_video")
        analysis["summary"]["audio_codec"] = mix_data.get("audio_codec")

    return analysis


# ──────────────────────────────────────────────
#  Generate markdown report
# ──────────────────────────────────────────────
def generate_markdown_report(analysis: dict) -> str:
    """Generate a human-readable markdown QA report."""
    lines = []
    lines.append("# Dubly ME — Phase F: QA Report")
    lines.append("")
    lines.append(f"**Generated:** {analysis['generated_at']}")
    lines.append("")

    # Design decisions
    lines.append("## Design Decisions Applied")
    lines.append("")
    dd = analysis["design_decisions"]
    lines.append(f"| Question | Decision |")
    lines.append(f"|----------|----------|")
    lines.append(f"| Q1: Overflow | {dd['Q1_overflow_policy']} |")
    lines.append(f"| Q2: Skipped  | {dd['Q2_skipped_segments']} |")
    lines.append(f"| Q3: Lip-sync | {dd['Q3_lipsync_qa']} |")
    lines.append(f"| Q4: Codec    | {dd['Q4_audio_codec']} |")
    lines.append("")

    # Summary
    summary = analysis["summary"]
    lines.append("## Summary")
    lines.append("")

    if "error" in summary:
        lines.append(f"> ⚠ **Error:** {summary['error']}")
        return "\n".join(lines)

    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(
        f"| Total segments | {summary['total_segments']} |"
    )
    lines.append(
        f"| Dubbed segments | {summary['dubbed_segments']} |"
    )
    lines.append(
        f"| Skipped (Arabic passthrough) | {summary['skipped_segments']} |"
    )
    if summary['error_segments'] > 0:
        lines.append(
            f"| ⚠ Error segments | {summary['error_segments']} |"
        )
    lines.append(
        f"| Segments within timing budget | {summary['segments_fit_budget']} |"
    )
    lines.append(
        f"| Segments with overflow | {summary['segments_overflow']} |"
    )
    lines.append(
        f"| Timing fit rate | {summary['timing_fit_rate']} |"
    )
    lines.append(
        f"| Average stretch ratio | {summary['avg_stretch_ratio']}× |"
    )
    lines.append(
        f"| Max stretch ratio applied | {summary['max_stretch_ratio']}× |"
    )
    lines.append(
        f"| Max allowed ratio (cap) | {summary['max_allowed_ratio']}× |"
    )

    if summary.get("output_video"):
        lines.append(
            f"| Output video | `{summary['output_video']}` |"
        )
    if summary.get("output_audio"):
        lines.append(
            f"| Output audio | `{summary['output_audio']}` |"
        )
    if summary.get("audio_codec"):
        lines.append(
            f"| Audio codec | {summary['audio_codec']} |"
        )
    lines.append("")

    # Per-segment table
    lines.append("## Per-Segment Details")
    lines.append("")
    lines.append(
        "| Seg | Start | Original | TTS | Output | "
        "Ratio | Tier | Fit | Mix Type |"
    )
    lines.append(
        "|-----|-------|----------|-----|--------|"
        "-------|------|-----|----------|"
    )

    for detail in analysis["segment_details"]:
        seg_id = detail["segment_id"]
        start = detail.get("start_time")
        start_str = f"{start:.1f}s" if start is not None else "—"
        orig = detail.get("original_duration_s")
        orig_str = f"{orig:.1f}s" if orig is not None else "—"

        if detail["status"] == "success":
            tts = detail.get("tts_duration_s", 0)
            out = detail.get("output_duration_s", 0)
            ratio = detail.get("stretch_ratio", 0)
            tier = detail.get("tier", "—")
            fit = detail.get("fit_quality", "—")
            mix_type = detail.get("mix_type", "—")

            fit_icon = {
                "GOOD": "✅",
                "SLIGHT_OVERFLOW": "⚠️",
                "OVERFLOW": "🔴",
            }.get(fit, "—")

            lines.append(
                f"| {seg_id} | {start_str} | {orig_str} | "
                f"{tts:.2f}s | {out:.2f}s | "
                f"{ratio:.2f}× | {tier} | {fit_icon} {fit} | {mix_type} |"
            )
        elif detail["status"] == "skipped":
            reason = detail.get("reason", "—")
            mix_type = detail.get("mix_type", "—")
            lines.append(
                f"| {seg_id} | {start_str} | {orig_str} | "
                f"— | — | — | skipped | ⏭ | {mix_type} |"
            )
        else:
            lines.append(
                f"| {seg_id} | {start_str} | {orig_str} | "
                f"— | — | — | error | ❌ | — |"
            )

    lines.append("")

    # Overflow details
    overflows = [
        d for d in analysis["segment_details"]
        if d.get("fit_quality") in ("OVERFLOW", "SLIGHT_OVERFLOW")
    ]

    if overflows:
        lines.append("## ⚠ Overflow Segments (Requires Visual Review)")
        lines.append("")
        lines.append(
            "These segments exceeded their timing budget. Per Q1 decision, "
            "compression was capped at 2.0× to preserve intelligibility."
        )
        lines.append("")

        for d in overflows:
            overflow_s = d.get("overflow_s", 0)
            lines.append(
                f"- **Segment #{d['segment_id']}** "
                f"@ {d.get('start_time', 0):.1f}s: "
                f"ratio {d.get('stretch_ratio', 0):.2f}×, "
                f"overflow +{overflow_s:.2f}s"
            )

        lines.append("")
        lines.append(
            "> **Action required:** Manual visual review of these segments "
            "to confirm the slight timing overflow is acceptable in context. "
            "(Q3: Manual review only)"
        )
        lines.append("")

    # Manual review checklist
    lines.append("## Manual Review Checklist (Q3: Visual Review)")
    lines.append("")
    lines.append(
        "Since automated lip-sync QA is not enabled (Q3: Option B), "
        "please review the following manually:"
    )
    lines.append("")
    lines.append("- [ ] Play `final_dubbed.mp4` end-to-end")
    lines.append("- [ ] Check dubbed segments sound natural and intelligible")
    lines.append(
        "- [ ] Verify Arabic passthrough segments transition smoothly"
    )
    lines.append("- [ ] Check overflow segments for visual sync issues")
    lines.append("- [ ] Verify crossfade transitions (no clicks or pops)")
    lines.append("- [ ] Confirm overall audio loudness is consistent")
    lines.append("")

    return "\n".join(lines)


# ──────────────────────────────────────────────
#  Save reports
# ──────────────────────────────────────────────
def save_reports(
    analysis: dict,
    json_path: Path,
    md_path: Path,
) -> None:
    """Save both JSON and markdown QA reports."""
    # JSON
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(analysis, fh, indent=2, ensure_ascii=False)
    print(f"  ✔ QA report (JSON) → {json_path}")

    # Markdown
    md_content = generate_markdown_report(analysis)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(md_content)
    print(f"  ✔ QA report (MD)   → {md_path}")


# ──────────────────────────────────────────────
#  CLI entry-point
# ──────────────────────────────────────────────
def main() -> None:
    # UTF-8 stdio before the first banner: the box-drawing glyphs below die on
    # a cp1252 fallback when stdout is piped or redirected (Windows).
    pipeline_core.enable_utf8_stdio()
    parser = argparse.ArgumentParser(
        description=f"Dubly ME — {pipeline_core.phase_title('F3')}",
    )
    parser.add_argument(
        "--output-json",
        default=str(QA_REPORT_JSON),
        help="Output JSON report path",
    )
    parser.add_argument(
        "--output-md",
        default=str(QA_REPORT_MD),
        help="Output markdown report path",
    )
    args = parser.parse_args()

    print()
    print(f"{'═' * 60}")
    print(f"  Dubly ME — {pipeline_core.phase_title('F3')}")
    print(f"{'═' * 60}")

    # ── Load manifests ───────────────────────
    print(f"\n[1/3]  Loading pipeline manifests…\n")
    manifests = load_manifests()

    # ── Analyse ──────────────────────────────
    print(f"\n[2/3]  Analysing timing quality…\n")
    analysis = analyse_timing(manifests)

    summary = analysis["summary"]
    if "error" not in summary:
        print(f"  Timing fit rate     : {summary['timing_fit_rate']}")
        print(f"  Avg stretch ratio   : {summary['avg_stretch_ratio']}×")
        print(f"  Max stretch ratio   : {summary['max_stretch_ratio']}×")
        print(f"  Overflow segments   : {summary['segments_overflow']}")
        print(f"  Arabic passthrough  : {summary['skipped_segments']}")

    # ── Save reports ─────────────────────────
    print(f"\n[3/3]  Saving QA reports…\n")
    save_reports(analysis, Path(args.output_json), Path(args.output_md))

    # ── Final banner ─────────────────────────
    print()
    print(f"{'═' * 60}")
    print(f"  ✅  Phase F3 complete — {pipeline_core.PHASES['F3']['label']}")
    print(f"{'─' * 60}")
    print(f"  JSON report  : {args.output_json}")
    print(f"  MD report    : {args.output_md}")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
