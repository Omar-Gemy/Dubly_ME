"""
run_pipeline.py — Dubly ME end-to-end orchestrator
==================================================
Runs Phases A→F3 in order, each as an ISOLATED subprocess so the three-venv
boundary (Rule 3) and the no-shared-memory rule (Rule 2) are preserved: phases
still communicate only through the JSON contracts on disk.

Interpreter resolution per phase, highest priority first:
  1. CLI flag        --python-asr / --python-llm / --python-audio / --python-demucs
  2. Environment     DUBLY_PY_ASR / DUBLY_PY_LLM / DUBLY_PY_AUDIO / DUBLY_PY_DEMUCS
  3. sys.executable  (single-venv local development)

Usage:
  python src/run_pipeline.py --source data/audio_in/sample.mp4
  python src/run_pipeline.py --source ... --from E --to F3
  python src/run_pipeline.py --only F3
  python src/run_pipeline.py --source ... --dry-run
  python src/run_pipeline.py --source ... --archive baselines/pre-refactor

  # Colab, one interpreter per venv:
  python src/run_pipeline.py --source ... \
      --python-audio /content/.venv_tts/bin/python \
      --python-demucs /content/.venv_demucs/bin/python

  # Forward extra flags to one phase (repeatable):
  python src/run_pipeline.py --source ... --arg "C=--model large-v3 --device cuda"
"""

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pipeline_core

SRC_DIR = pipeline_core.SRC_DIR

# F0 is order-independent (it only needs the source media); it is placed here so
# its stems exist before F2 wants them.
PHASE_ORDER = ["A", "B", "C", "D", "E", "F0", "F1", "F2", "F3"]

VENV_FLAGS = {
    "asr": "python_asr",
    "llm": "python_llm",
    "audio": "python_audio",
    "demucs": "python_demucs",
}

# Phases that need to be told where the source media is. Everything else reads
# its inputs from the JSON contracts written by the phase before it.
SOURCE_POSITIONAL = {"A"}
SOURCE_FLAGGED = {"E", "F0", "F2"}


def resolve_interpreter(venv_key: str, args) -> str:
    """CLI flag → DUBLY_PY_<KEY> env var → this interpreter."""
    flag_value = getattr(args, VENV_FLAGS[venv_key], None)
    if flag_value:
        return flag_value
    env_value = os.environ.get(f"DUBLY_PY_{venv_key.upper()}")
    if env_value:
        return env_value
    return sys.executable


def build_command(phase_id: str, args, extra: dict) -> list[str]:
    """Assemble the argv for one phase."""
    entry = pipeline_core.PHASES[phase_id]
    script = SRC_DIR / f"{entry['module']}.py"
    cmd = [resolve_interpreter(entry["venv"], args), str(script)]

    if phase_id in SOURCE_POSITIONAL:
        cmd.append(str(args.source))
    elif phase_id in SOURCE_FLAGGED:
        cmd += ["--source", str(args.source)]

    cmd += extra.get(phase_id, [])
    return cmd


def selected_phases(args) -> list[str]:
    """Apply --only / --from / --to / --skip to PHASE_ORDER."""
    if args.only:
        chosen = [p for p in PHASE_ORDER if p in set(args.only)]
    else:
        start = PHASE_ORDER.index(args.from_phase) if args.from_phase else 0
        end = PHASE_ORDER.index(args.to_phase) if args.to_phase else len(PHASE_ORDER) - 1
        if start > end:
            raise SystemExit(f"--from {args.from_phase} comes after --to {args.to_phase}")
        chosen = PHASE_ORDER[start:end + 1]
    return [p for p in chosen if p not in set(args.skip or [])]


def parse_extra(raw_list) -> dict:
    """Turn ['C=--model large-v3', ...] into {'C': ['--model', 'large-v3']}."""
    extra: dict = {}
    for item in raw_list or []:
        if "=" not in item:
            raise SystemExit(f"--arg must look like PHASE=flags, got: {item!r}")
        phase_id, _, flags = item.partition("=")
        phase_id = phase_id.strip()
        if phase_id not in pipeline_core.PHASES:
            raise SystemExit(
                f"--arg names unknown phase {phase_id!r}. "
                f"Known: {', '.join(PHASE_ORDER)}"
            )
        extra.setdefault(phase_id, []).extend(shlex.split(flags))
    return extra


def archive_outputs(dest: Path) -> None:
    """
    Snapshot every artifact + final deliverable into *dest* — the golden
    baseline the later quality chunks are measured against.
    """
    dest.mkdir(parents=True, exist_ok=True)
    if pipeline_core.ARTIFACTS_DIR.is_dir():
        shutil.copytree(
            pipeline_core.ARTIFACTS_DIR,
            dest / "artifacts",
            dirs_exist_ok=True,
        )
    copied = 0
    for name in ("final_dubbed.wav", "final_dubbed.mp4"):
        src = pipeline_core.DATA_AUDIO_OUT / name
        if src.is_file():
            shutil.copy2(src, dest / name)
            copied += 1
    print(f"\n  ✔ Baseline archived → {dest}  ({copied} deliverable file(s))")


def main() -> None:
    pipeline_core.enable_utf8_stdio()
    parser = argparse.ArgumentParser(
        description="Dubly ME — end-to-end pipeline orchestrator (Phases A to F3)",
    )
    parser.add_argument("--source", default=None,
                        help="Source media path (required for phases A, E, F0, F2)")
    parser.add_argument("--from", dest="from_phase", choices=PHASE_ORDER,
                        default=None, help="First phase to run")
    parser.add_argument("--to", dest="to_phase", choices=PHASE_ORDER,
                        default=None, help="Last phase to run")
    parser.add_argument("--only", nargs="+", choices=PHASE_ORDER, default=None,
                        help="Run exactly these phases")
    parser.add_argument("--skip", nargs="+", choices=PHASE_ORDER, default=None,
                        help="Exclude these phases")
    parser.add_argument("--arg", action="append", default=None, metavar="PHASE=FLAGS",
                        help="Forward extra flags to one phase (repeatable)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the commands without running them")
    parser.add_argument("--keep-going", action="store_true",
                        help="Continue after a phase fails (default: stop)")
    parser.add_argument("--archive", default=None, metavar="DIR",
                        help="Copy artifacts + deliverables into DIR when done")
    for key, dest in VENV_FLAGS.items():
        parser.add_argument(f"--python-{key}", dest=dest, default=None,
                            help=f"Interpreter for the '{key}' venv "
                                 f"(env: DUBLY_PY_{key.upper()})")
    args = parser.parse_args()

    phases = selected_phases(args)
    extra = parse_extra(args.arg)

    needs_source = [p for p in phases if p in SOURCE_POSITIONAL | SOURCE_FLAGGED]
    if needs_source and not args.source:
        raise SystemExit(
            f"--source is required for phase(s): {', '.join(needs_source)}"
        )
    if args.source and not Path(args.source).is_file():
        raise SystemExit(f"Source media not found: {args.source}")

    print()
    print("═" * 64)
    print("  Dubly ME — Pipeline Orchestrator")
    print("═" * 64)
    print(f"  Phases : {' → '.join(phases)}")
    print(f"  Source : {args.source or '(not needed)'}")
    if args.dry_run:
        print("  Mode   : DRY RUN (nothing will be executed)")

    results: list[tuple[str, int, float]] = []
    t_all = time.perf_counter()

    for phase_id in phases:
        cmd = build_command(phase_id, args, extra)
        title = pipeline_core.phase_title(phase_id)

        print()
        print("─" * 64)
        print(f"  {title}")
        print(f"  $ {' '.join(cmd)}")
        print("─" * 64)

        if args.dry_run:
            results.append((phase_id, 0, 0.0))
            continue

        t0 = time.perf_counter()
        # UTF-8 in the child too: the phase banners are box-drawing glyphs, and a
        # piped/redirected run on Windows would otherwise die on the cp1252
        # fallback partway through a phase.
        child_env = dict(os.environ, PYTHONIOENCODING="utf-8")
        code = subprocess.run(
            cmd, cwd=str(pipeline_core.PROJECT_ROOT), env=child_env
        ).returncode
        elapsed = time.perf_counter() - t0
        results.append((phase_id, code, elapsed))

        if code != 0:
            print(f"\n  ✖ {title} exited with code {code}")
            if not args.keep_going:
                _summarise(results, time.perf_counter() - t_all)
                sys.exit(code)

    _summarise(results, time.perf_counter() - t_all)

    if args.archive and not args.dry_run:
        archive_outputs(Path(args.archive))

    if any(code != 0 for _, code, _ in results):
        sys.exit(1)


def _summarise(results: list[tuple[str, int, float]], total_s: float) -> None:
    print()
    print("═" * 64)
    print("  Pipeline summary")
    print("─" * 64)
    for phase_id, code, elapsed in results:
        status = "✔ ok  " if code == 0 else f"✖ {code:<4}"
        print(f"  {status}  Phase {phase_id:<3}  "
              f"{pipeline_core.PHASES[phase_id]['label']:<32} {elapsed:6.1f}s")
    print("─" * 64)
    print(f"  Total: {total_s:.1f}s")
    print("═" * 64)


if __name__ == "__main__":
    main()
