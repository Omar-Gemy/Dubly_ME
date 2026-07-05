# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**Dubly ME** is a self-hosted AI video dubbing pipeline (Arabic → English is the primary path, driven by `config/language_registry.json`). It ingests a video, detects who speaks when, transcribes, translates with timing-aware ("isochronous") rewriting, clones the voice, fits each line into its original time window, and renders a final dubbed video. The design goal is output that sounds like a real dubbing team, not an AI overlay — timing fit, emotional fidelity, and mix quality come before flashy features. See `docs/professional_ai_dubbing_project_plan_en.md` for the full vision.

## Hard Constraints (from `skills/dubbing_rules.md` — do not violate)

1. **Zero third-party APIs.** No commercial/cloud API calls (no DeepL, Google, OpenAI, etc.). Everything runs on self-hosted open models: faster-whisper, Silero VAD, pyannote.audio, Qwen, XTTS v2.
2. **Every stage emits a JSON data contract** into `artifacts/`. Never skip writing these — they are how stages hand off and how runs are debugged.
3. **Git workflow:** never push to `main`; branch per feature (current branch: `feature/architecture-refactor`).
4. **Keep it modular.** No end-to-end "magic" models — each stage must be independently runnable, testable, and replaceable.

## Architecture — a linear chain of CLI stages

Each `src/*.py` is a standalone CLI stage that reads the previous stage's JSON artifact and writes its own. Stages communicate **only** through files in `artifacts/` (plus audio under `data/audio_out/` and `artifacts/audio_out/`). There is no orchestrator module in `src/` — the notebook (`notebooks/colab_pipeline.py` / `.ipynb`) is the orchestrator, invoking each script via `subprocess` in order.

Run order, script, and the artifact each produces:

| # | Script | Reads | Writes |
|---|--------|-------|--------|
| 1 | `src/ingestion_vad.py` | source video/audio | `data/audio_out/_temp_normalised.wav`, `artifacts/segments.json` |
| 2 | `src/diarization.py` | `_temp_normalised.wav` + `segments.json` | rewrites `segments.json` (adds `speaker_id`) |
| 3 | `src/asr_transcription.py` | `_temp_normalised.wav` + `segments.json` | `artifacts/transcripts.json` |
| 4 | `src/translation.py` | `transcripts.json` | `artifacts/translation.json` |
| 5 | `src/tts_synthesis.py` | `translation.json` + `artifacts/voice_ref.wav` | `artifacts/audio_out/segment_XXX.wav`, `artifacts/tts_manifest.json` |
| 6 | `src/time_stretch.py` | `tts_manifest.json` + segment WAVs | `artifacts/audio_out/stretched/*.wav`, `artifacts/stretch_manifest.json` |
| 7 | `src/mix_render.py` | `stretch_manifest.json` + `segments.json` + source video | `data/audio_out/final_dubbed.{wav,mp4}`, `artifacts/mix_manifest.json` |
| 8 | `src/qa_report.py` | tts/stretch/mix manifests + `segments.json` | `artifacts/qa_report.{json,md}` |

The `segments.json` schema is the backbone contract: top-level run metadata (`source_file`, `source_language`, `target_language`, `vad_threshold`, `total_segments`) plus a `segments` array where each entry has `segment_id`, `start_time`, `end_time`, `duration`, `speaker_id`, and `text` (populated by ASR). Later stages enrich segments in place rather than reshaping them.

### Key stage behaviors worth knowing before editing

- **Ingestion** normalises to mono 16 kHz WAV via FFmpeg with two-pass **linear** EBU R128 (preserves SNR), then runs Silero VAD. It ships a Windows-safe `read_audio` (soundfile backend) because Silero's default uses `sox_effects`, which is unsupported on Windows.
- **Diarization** uses a *global-to-local intersection* strategy: run pyannote on the **full** audio for globally consistent speaker labels, then intersect with VAD segments, splitting multi-speaker segments at turn boundaries. Models load **sequentially** to stay within a 4 GB VRAM budget. Requires an HF token (`HF_TOKEN`) and accepted licenses (see requirements.txt).
- **ASR** transcribes the **full file in one pass** with WhisperX (`large-v3` backend) + wav2vec2 forced alignment, then intersects the aligned **word timestamps** with the diarization segments (same global-to-local pattern as diarization). This preserves linguistic context that per-slice transcription destroyed. Language config comes from `config/language_registry.json`: each entry supplies a short stylistic Whisper `initial_prompt` (kept content-free to avoid prompt-echo) and a `hallucination_patterns` list. A prompt-echo guard discards any transcript that regurgitates the prompt. Add new languages here, not in code.
- **Translation** adapts (not literally translates) with a self-hosted Qwen (`Qwen/Qwen2.5-14B-Instruct-AWQ` default, pre-quantized 4-bit). Each line is given an explicit **isochrony budget** — a target English syllable count derived from the segment `duration` (~4 syll/s ±15%) — so the dub fits its window. Short conversational fillers are translated, not skipped; only empty/loop/echo segments are hard-skipped.
- **Time-stretch** tiers: ≤1.15× → trivial/no stretch; 1.15–2.0× → WSOLA stretch; >2.0× → **capped at 2.0×**, overflow allowed (never sacrifice intelligibility). 50 ms fit tolerance.
- **Mix/render**: skipped segments fall back to original-audio passthrough; segments placed at original `start_time` with crossfades; final video is AAC 192k. Lip-sync QA is manual only.

## Running the pipeline

There are two execution environments; scripts are identical, only deps differ.

**Local (Windows, `requirements.txt`)** — run each stage in order from repo root:
```powershell
python src/ingestion_vad.py data/audio_in/sample.mp4   # --threshold 0.4
python src/diarization.py                               # --hf-token hf_xxx --max-speakers 3 --device cpu
python src/asr_transcription.py                        # --model large-v3 --source-lang ar --device cuda
python src/translation.py                              # --model Qwen/Qwen2.5-14B-Instruct-AWQ
python src/tts_synthesis.py                             # --device cpu
python src/time_stretch.py                              # --max-ratio 2.0
python src/mix_render.py                                # --source data/audio_in/sample.mp4
python src/qa_report.py
```
Every script accepts `--input`/`--output`/path overrides and defaults to the `artifacts/` locations above, so stages chain with no arguments once the input video is at `data/audio_in/sample.mp4`.

**Colab GPU (`requirements-colab.txt`)** — the intended full-run path. Open `notebooks/colab_pipeline.ipynb` (source of truth is `notebooks/colab_pipeline.py`, kept in sync). It mounts Drive, points `HF_HOME`/`HF_HUB_CACHE`/`TORCH_HOME` at Drive so large models are cached once, clones this repo, installs deps, and runs Phases A–D. **`colab_pipeline.py` and the `.ipynb` must be edited together** — recent commits repeatedly re-sync them.

> Phase naming differs between the notebook and the source docstrings: the notebook labels ingestion "Phase A", while `ingestion_vad.py`/`diarization.py` docstrings call the speaker layer "Phase B". Follow the table above for actual run order.

## Setup

- **FFmpeg is a system dependency** (not pip). `winget install FFmpeg` (Win) / `brew install ffmpeg` (mac) / `apt install ffmpeg` (Linux); must be on PATH.
- **Hugging Face token** required for pyannote diarization: accept licenses for `pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0`, then `$env:HF_TOKEN = "hf_..."` (PowerShell) or pass `--hf-token`.
- Local dev targets an RTX 3050 Ti / 4 GB VRAM budget; all stages are written to fit that (8-bit LLM, sequential model loading, CPU fallbacks via `--device cpu`).

## Conventions

- No test suite exists yet; verification is via the QA report (`artifacts/qa_report.md`) and manual review of `artifacts/audio_out/` and `data/audio_out/final_dubbed.mp4`.
- `artifacts/`, `.venv/`, and `data/audio_out/` are gitignored — the JSON contracts are runtime output, not committed source (the two committed `segments.json`/`transcripts.json` are sample fixtures).
- Stage scripts use `PROJECT_ROOT = Path(__file__).resolve().parent.parent`-relative paths; run them from anywhere but keep the `src/ … artifacts/ … data/` layout intact.
- User-facing console output uses ✔/✖/▶/✅ status glyphs and `[n/N]` step counters — match this style when adding stage output.

## Current Session State (2026-07-05) — QA remediation on feature/architecture-refactor

A QA pass over a real 53-segment translation.json test run found 4 issues.
Status:

- ✅ **Observation 1** (commit `e866115`): fixed `detect_whisper_loop` false-positives
  in `src/translation.py` — was dropping ~19% of valid segments as false "loops".
  Now token-based, punctuation-normalized, decoupled advisory-flag-vs-hard-skip.
- ✅ **Observation 2** (commit `482e31b`): added `config/name_glossary.json` +
  injection into the translation prompt in `src/translation.py`, for stable
  character-name transliteration.
- ✅ **Observation 3 Part A** (commit `424dd18`): raised the diarization
  split threshold in `src/diarization.py` (`SPLIT_MIN_SECONDARY_FRAC=0.25`,
  `SPLIT_MIN_SECONDARY_SEC=0.7`) to stop spurious speaker-change splits.
- ✅ **Observation 3 Part B** (just committed — confirm commit hash with
  `git log -1 --oneline`): added `rejoin_same_origin_fragments()` in
  `src/diarization.py` — merges adjacent same-origin/same-speaker fragments
  post-split, with a `_merged_from` provenance field and a new
  `segments_rejoined` stat in `diarization_stats`.

### Next task — Observation 4 (not started)

Surface ASR confidence in `src/asr_transcription.py`:
- Add `asr_confidence` (aggregated from faster-whisper/whisperx's existing
  word-level alignment scores) and a `low_confidence_asr` flag to the
  `transcripts.json` output.
- Additive only — do not change existing fields or the output path.

### Standing process rules for this remediation work
- One fix per commit. Show the full diff and wait for explicit "approved"
  before moving to the next task.
- Never run git yourself (add/commit/push) — the human handles this after
  reviewing each diff.
- No `git add .` — explicit files only.
- Don't touch files outside the current task's stated scope.
