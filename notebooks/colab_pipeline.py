# -*- coding: utf-8 -*-
"""
colab_pipeline.py — Dubly ME: Sequential Colab Pipeline (Phases A–D)
=====================================================================
Run this script cell-by-cell in Google Colab (GPU runtime, T4 recommended).
Each section is delimited with  # %%  for easy copy-paste into .ipynb.

Architecture:
  - Google Drive is mounted for persistent storage of models and artifacts.
  - HF_HOME is pointed at Drive so heavy LLMs are never re-downloaded.
  - Each phase invokes the corresponding CLI script from  src/ .
  - All source code and dependencies are pulled from the GitHub repo —
    no manual patches needed.
"""

# %% [Cell 0] Environment Setup — Mount Drive, Cache Models, Install Deps
"""
Mount Google Drive, configure persistent HuggingFace model cache,
clone/sync the repo, and install Python dependencies.
"""
import os
import subprocess

# ── 0.1  Mount Google Drive ──────────────────────────────────────────
from google.colab import drive
drive.mount("/content/drive")

# ── 0.2  Persistent model cache on Drive (prevents re-downloading) ──
#    This MUST be set BEFORE any import of transformers / pyannote / etc.
STORAGE_ROOT = "/content/drive/MyDrive/Dubly_ME_Storage"
os.environ["HF_HOME"]       = f"{STORAGE_ROOT}/models"
os.environ["HF_HUB_CACHE"]  = f"{STORAGE_ROOT}/models/hub"
os.environ["TORCH_HOME"]    = f"{STORAGE_ROOT}/models/torch"
os.makedirs(os.environ["HF_HOME"], exist_ok=True)
os.makedirs(os.environ["TORCH_HOME"], exist_ok=True)

print(f"✔ HF_HOME       = {os.environ['HF_HOME']}")
print(f"✔ HF_HUB_CACHE  = {os.environ['HF_HUB_CACHE']}")
print(f"✔ TORCH_HOME    = {os.environ['TORCH_HOME']}")

# ── 0.3  Set HF_TOKEN from Colab Secrets (required for pyannote) ────
try:
    from google.colab import userdata
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
    print("✔ HF_TOKEN loaded from Colab Secrets.")
except Exception:
    print("⚠ Could not load HF_TOKEN from Colab Secrets.")
    print("  Set it manually:  os.environ['HF_TOKEN'] = 'hf_your_token'")

# ── 0.4  Sync the project repo to the Colab VM (via Git) ────────────
REPO_DIR = "/content/Dubly_ME"
GIT_URL = "https://github.com/Omar-Gemy/Dubly_ME.git"
BRANCH = "feature/architecture-refactor"

if not os.path.exists(REPO_DIR):
    print(f"▶ Cloning repo from {GIT_URL} (branch: {BRANCH})...")
    subprocess.run(["git", "clone", "-b", BRANCH, GIT_URL, REPO_DIR], check=True)
    print("✔ Repo cloned successfully.")
else:
    print("▶ Pulling latest changes...")
    subprocess.run(["git", "pull", "origin", BRANCH], cwd=REPO_DIR, check=True)
    print("✔ Repo updated successfully.")

# ── 0.5  Install dependencies ──────────────────────────────────────
subprocess.run(
    ["pip", "install", "-q", "-r", f"{REPO_DIR}/requirements-colab.txt"],
    check=True,
)
print("\n✔ All dependencies installed.")

# ── 0.6  Verify FFmpeg (pre-installed on Colab GPU runtimes) ───────
subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
print("✔ FFmpeg is available.")

print("\n" + "═" * 60)
print("  ✅  Cell 0: Environment Setup Complete")
print("═" * 60)


# %% [Cell 1] Phase A — Audio Ingestion & Voice Activity Detection
"""
Extract audio from the source media file, normalise loudness (EBU R128),
and run Silero VAD to detect speech segments.

Script:   src/ingestion_vad.py
Input:    source media file (video or audio)
Output:   artifacts/segments.json  +  data/audio_out/_temp_normalised.wav
"""
import subprocess, os

REPO_DIR = "/content/Dubly_ME"
INPUT_MEDIA = "/content/drive/MyDrive/Dubly_ME_Storage/data/audio_in/source_media.mp4"

# Validate input exists
if not os.path.isfile(INPUT_MEDIA):
    raise FileNotFoundError(
        f"Source media not found: {INPUT_MEDIA}\n"
        f"Upload your video/audio file to this Google Drive path."
    )

print(f"▶ Phase A: Ingesting {INPUT_MEDIA}")
subprocess.run(
    [
        "python", f"{REPO_DIR}/src/ingestion_vad.py",
        INPUT_MEDIA,
        "--source-lang", "ar",
        "--target-lang", "en",
    ],
    check=True,
    cwd=REPO_DIR,
)

print("\n" + "═" * 60)
print("  ✅  Phase A: Audio Ingestion & VAD Complete")
print("═" * 60)


# %% [Cell 2] Phase B — Speaker Diarization
"""
Assign speaker identities to VAD segments using the pyannote.audio 4.x
ensemble (Segmentation → Embedding → Clustering).

Script:   src/diarization.py
Input:    data/audio_out/_temp_normalised.wav  +  artifacts/segments.json
Output:   artifacts/segments.json  (enriched with speaker_id)

Requires: HF_TOKEN set (for gated pyannote models).
"""
import subprocess

REPO_DIR = "/content/Dubly_ME"

print("▶ Phase B: Speaker Diarization")
subprocess.run(
    [
        "python", f"{REPO_DIR}/src/diarization.py",
        "--device", "cuda",
    ],
    check=True,
    cwd=REPO_DIR,
)

print("\n" + "═" * 60)
print("  ✅  Phase B: Speaker Diarization Complete")
print("═" * 60)


# %% [Cell 3] Phase C — ASR Transcription
"""
Transcribe the full audio in one pass with WhisperX (large-v3 backend),
run wav2vec2 forced alignment for precise word-level timestamps, then
intersect words with the diarization segments (global-to-local mapping).

Script:   src/asr_transcription.py
Input:    data/audio_out/_temp_normalised.wav  +  artifacts/segments.json
Output:   artifacts/transcripts.json
"""
import subprocess

REPO_DIR = "/content/Dubly_ME"

print("▶ Phase C: ASR Transcription")
subprocess.run(
    [
        "python", f"{REPO_DIR}/src/asr_transcription.py",
        "--model", "large-v3",
        "--source-lang", "ar",
        "--device", "cuda",
        "--compute-type", "float16",
    ],
    check=True,
    cwd=REPO_DIR,
)

print("\n" + "═" * 60)
print("  ✅  Phase C: ASR Transcription Complete")
print("═" * 60)


# %% [Cell 4] Phase D — Contextual Translation (Isochronous)
"""
Adapt the Arabic transcripts into natural spoken English for dubbing using
Qwen2.5-14B-Instruct-AWQ. The model is a pre-quantized 4-bit AWQ checkpoint
(~9–10 GB VRAM, fits the 16 GB T4) and is cached on Google Drive via HF_HOME
to prevent re-downloading on subsequent sessions.

Each line is translated against an explicit isochrony budget (target syllable
count derived from the segment duration) so the dub fits its time window.

Script:   src/translation.py
Input:    artifacts/transcripts.json
Output:   artifacts/translation.json
"""
import subprocess

REPO_DIR = "/content/Dubly_ME"

print("▶ Phase D: Contextual Translation (Isochronous)")
subprocess.run(
    [
        "python", f"{REPO_DIR}/src/translation.py",
        "--model", "Qwen/Qwen2.5-14B-Instruct-AWQ",
        "--device", "auto",
    ],
    check=True,
    cwd=REPO_DIR,
)

print("\n" + "═" * 60)
print("  ✅  Phase D: Translation Complete")
print("═" * 60)


# %% [Cell 5] Pipeline Summary — Verify Artifacts
"""
Quick sanity check: verify that all expected output artifacts exist
and print a summary of what was produced.
"""
import json, os

REPO_DIR = "/content/Dubly_ME"
ARTIFACTS = f"{REPO_DIR}/artifacts"

expected_files = {
    "Phase A (VAD)":         f"{ARTIFACTS}/segments.json",
    "Phase C (ASR)":         f"{ARTIFACTS}/transcripts.json",
    "Phase D (Translation)": f"{ARTIFACTS}/translation.json",
}

print("Pipeline Artifact Verification")
print("─" * 50)
all_ok = True
for phase, path in expected_files.items():
    if os.path.isfile(path):
        size_kb = os.path.getsize(path) / 1024
        # Load to count segments
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        n_segs = data.get("total_segments", len(data.get("segments", [])))
        print(f"  ✔ {phase:<25s}  {n_segs:>3d} segments  ({size_kb:.1f} KB)")
    else:
        print(f"  ✖ {phase:<25s}  FILE MISSING: {path}")
        all_ok = False

print("─" * 50)
if all_ok:
    print("  ✅  All pipeline artifacts verified successfully!")
else:
    print("  ⚠  Some artifacts are missing — check the logs above.")
print("═" * 50)
