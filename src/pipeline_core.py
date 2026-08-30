"""
pipeline_core.py — Shared foundations for every Dubly ME phase
==============================================================
The ONE place that owns cross-phase constants and helpers. Phases A–F stay
decoupled at the *data* level (JSON contracts only, per project Rule 2); this
module is a shared *library*, not shared state — importing it never creates a
runtime dependency between phases and never carries data across a phase
boundary.

Contents
  1. Project paths
  2. Phase registry           — canonical phase ids / labels  (fixes 5.8)
  3. Sample-rate registry     — one declared rate per stage   (fixes 5.5)
  4. Path helpers             — rel_or_abs                    (fixes 5.6)
  5. Audio helpers            — resample                      (fixes 5.6)
  6. Device helpers           — CUDA probing / resolution      (fixes 5.6)
  7. Determinism helpers      — seeding                       (for Chunk 4)
  8. Provenance helpers       — hashing / run identity        (for Chunk 2)
  9. Language configuration   — unified matrix                (fixes 5.11)

Heavy third-party imports (torch / numpy / scipy) are LAZY: this module is
imported by stdlib-only phases (qa_report) and by three different virtualenvs
that do not all contain the same packages.
"""

import hashlib
import json
import os
from pathlib import Path

# ──────────────────────────────────────────────
#  1. Project paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
CONFIG_DIR = PROJECT_ROOT / "config"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DATA_DIR = PROJECT_ROOT / "data"
DATA_AUDIO_IN = DATA_DIR / "audio_in"
DATA_AUDIO_OUT = DATA_DIR / "audio_out"

LANGUAGES_FILE = CONFIG_DIR / "languages.json"
LEGACY_LANGUAGE_REGISTRY = CONFIG_DIR / "language_registry.json"


# ──────────────────────────────────────────────
#  2. Phase registry  (fixes 5.8 — label drift)
# ──────────────────────────────────────────────
# Canonical phase identity for every pipeline module. Docstrings, CLI banners
# and (from Chunk 2) the `phase` field of each JSON contract all read from here
# instead of hand-writing a label per file.
PHASES = {
    "A":  {"module": "ingestion_vad",      "label": "Ingestion & VAD",            "venv": "asr"},
    "B":  {"module": "diarization",        "label": "Speaker Diarization",        "venv": "asr"},
    "C":  {"module": "asr_transcription",  "label": "ASR Transcription",          "venv": "asr"},
    "D":  {"module": "translation",        "label": "Translation & Adaptation",   "venv": "llm"},
    "E":  {"module": "tts_synthesis",      "label": "Voice Cloning & TTS",        "venv": "audio"},
    "F0": {"module": "source_separation",  "label": "Source Separation",          "venv": "demucs"},
    "F1": {"module": "time_stretch",       "label": "Duration Fitting",           "venv": "audio"},
    "F2": {"module": "mix_render",         "label": "Final Mix & Video Render",   "venv": "audio"},
    "F3": {"module": "qa_report",          "label": "QA Report",                  "venv": "audio"},
}


def phase_title(phase_id: str) -> str:
    """Return e.g. 'Phase C — ASR Transcription' for banners and docstrings."""
    entry = PHASES[phase_id]
    return f"Phase {phase_id} — {entry['label']}"


# ──────────────────────────────────────────────
#  3. Sample-rate registry  (fixes 5.5 — silent rate drift)
# ──────────────────────────────────────────────
# Every rate in the pipeline is DECLARED here with the reason it exists. Before
# this, four modules each defined a private constant and one of them documented
# the wrong value (mix_render claimed 24 kHz was "XTTS v2 native"; XTTS v2 is
# 22050 Hz, so dubbed segments ARE resampled at mix time).
#
#   ASR_SAMPLE_RATE        16000  Silero VAD + WhisperX/wav2vec2 operating rate.
#   XTTS_NATIVE_SAMPLE_RATE 22050 XTTS v2 decoder output rate (Phase E WAVs).
#   SEPARATION_WORK_RATE   44100  Demucs native working rate (Phase F0 input).
#   PIPELINE_SAMPLE_RATE   24000  Timeline assembly rate: the rate the final mix
#                                 and the finalised F0 stems live at. Phase E
#                                 output is resampled 22050 → 24000 here.
#
# NOTE (scope): Chunk 1 only makes the rates truthful and single-sourced. The
# resample itself is unchanged; whether to move the timeline to 22050 (removing
# the conversion) is a Chunk 10 decision, taken with the loudness chain.
ASR_SAMPLE_RATE = 16000
XTTS_NATIVE_SAMPLE_RATE = 22050
SEPARATION_WORK_RATE = 44100
PIPELINE_SAMPLE_RATE = 24000

# Phase F1 (time_stretch) is deliberately absent: it is rate-PRESERVING and
# must never impose a rate of its own — it hands Phase E's rate straight
# through to Phase F2, which performs the single documented conversion.

SAMPLE_RATES = {
    "asr": ASR_SAMPLE_RATE,
    "xtts_native": XTTS_NATIVE_SAMPLE_RATE,
    "separation_work": SEPARATION_WORK_RATE,
    "pipeline": PIPELINE_SAMPLE_RATE,
}


# ──────────────────────────────────────────────
#  4. Path helpers  (fixes 5.6 — _rel_or_abs ×3)
# ──────────────────────────────────────────────
def rel_or_abs(path) -> str:
    """
    POSIX-style path relative to PROJECT_ROOT when it lives inside the repo,
    otherwise the absolute path. Source media may sit outside the repo (e.g. on
    Google Drive) where ``Path.relative_to`` would raise.
    """
    p = Path(path)
    try:
        return str(p.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(p).replace("\\", "/")


def enable_utf8_stdio() -> bool:
    """
    Force UTF-8 on stdout/stderr when the current encoding cannot represent the
    box-drawing and status glyphs every phase banner prints.

    On Windows, a redirected or piped stream falls back to the locale codepage
    (cp1252), where a single '→' raises UnicodeEncodeError and kills the run
    mid-phase. Call this at the top of main() — never at import time, so
    importing a phase still has no side effect (cf. 5.10).

    Returns True when a stream was reconfigured.
    """
    import sys

    changed = False
    for stream in (sys.stdout, sys.stderr):
        encoding = (getattr(stream, "encoding", None) or "").lower()
        if encoding.replace("-", "") in {"utf8", "utf8sig"}:
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
            changed = True
        except Exception:
            pass
    return changed


# ──────────────────────────────────────────────
#  5. Audio helpers  (fixes 5.6 — _resample ×2)
# ──────────────────────────────────────────────
def resample(audio, src_sr: int, dst_sr: int):
    """
    Anti-aliased resample of a 1-D float32 array.

    Uses scipy's polyphase ``resample_poly`` when available and falls back to
    linear interpolation (aliasing-prone, kept only so an absent scipy degrades
    instead of crashing). A no-op when the rates already match.
    """
    import numpy as np

    if src_sr == dst_sr:
        return audio.astype(np.float32, copy=False)

    try:
        from scipy.signal import resample_poly
    except Exception:
        old_len = len(audio)
        new_len = int(old_len * dst_sr / src_sr)
        return np.interp(
            np.linspace(0, old_len - 1, new_len),
            np.arange(old_len),
            audio,
        ).astype(np.float32)

    from math import gcd

    g = gcd(src_sr, dst_sr)
    return resample_poly(audio, dst_sr // g, src_sr // g).astype(np.float32)


def have_scipy() -> bool:
    """True when scipy's polyphase resampler is importable."""
    try:
        from scipy.signal import resample_poly  # noqa: F401
    except Exception:
        return False
    return True


# ──────────────────────────────────────────────
#  6. Device helpers  (fixes 5.6 — resolve_device ×3)
# ──────────────────────────────────────────────
# The three former implementations differed in what they RETURNED (str vs
# torch.device) and in whether they enforced a free-VRAM floor. Rather than
# collapse them into one signature and change behaviour, the primitives live
# here and each phase keeps its own reporting.
def cuda_available() -> bool:
    """True when torch is importable AND reports a usable CUDA device."""
    try:
        import torch
    except Exception:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def cuda_memory_gb() -> tuple[float, float]:
    """(free_gb, total_gb) for CUDA device 0. (0.0, 0.0) when unavailable."""
    if not cuda_available():
        return (0.0, 0.0)
    import torch

    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return (free_bytes / (1024 ** 3), total_bytes / (1024 ** 3))


def cuda_device_summary() -> tuple[str, float]:
    """(device_name, total_vram_gb) for CUDA device 0. ('', 0.0) when absent."""
    if not cuda_available():
        return ("", 0.0)
    import torch

    props = torch.cuda.get_device_properties(0)
    return (torch.cuda.get_device_name(0), props.total_memory / (1024 ** 3))


def resolve_device_str(requested: str = "auto") -> str:
    """
    Resolve ``'auto'`` to ``'cuda'`` when CUDA is usable, else ``'cpu'``.
    Any other value passes through untouched (explicit user intent wins).
    """
    if requested != "auto":
        return requested
    return "cuda" if cuda_available() else "cpu"


# ──────────────────────────────────────────────
#  7. Determinism helpers
# ──────────────────────────────────────────────
# Provided now, WIRED IN CHUNK 4. Phases D and E currently sample without a
# seed, so no run is reproducible (audit 6.9); these are the primitives that
# fix it once the QA baseline exists to measure against.
def seed_everything(seed: int) -> int:
    """Seed python / numpy / torch (incl. CUDA) from one integer. Returns it."""
    import random

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2 ** 32))
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    return seed


def derive_segment_seed(base_seed: int, segment_id: int) -> int:
    """
    Per-segment seed derived from the run seed, so re-running ONE segment
    reproduces exactly what the full run produced for it.
    """
    digest = hashlib.sha256(f"{base_seed}:{segment_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


# ──────────────────────────────────────────────
#  8. Provenance helpers
# ──────────────────────────────────────────────
# Provided now, WIRED IN CHUNK 2 (run identity / lineage validation, audit 6.4).
def sha256_file(path, chunk_bytes: int = 1024 * 1024) -> str:
    """Streaming SHA-256 of a file — safe on multi-GB media."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(payload) -> str:
    """Stable SHA-256 of a JSON-serialisable payload (sorted keys)."""
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


# ──────────────────────────────────────────────
#  9. Language configuration  (fixes 5.11)
# ──────────────────────────────────────────────
# Before this there were THREE language sources of truth: a JSON registry read
# by Phase C, a hardcoded `LANGUAGES` dict in Phase D, and a hardcoded set in
# Phase E. config/languages.json now owns all three views:
#
#   <code>.name                  display / ASR label
#   <code>.asr.*                 whisper_language, initial_prompt,
#                                hallucination_patterns   (Phase C)
#   <code>.translation.*         name, output_desc, unit, leakage  (Phase D)
#   <code>.tts.xtts_language     XTTS v2 language tag              (Phase E)
#   xtts_supported[]             XTTS v2's 17 built-in languages   (Phase E)
ASR_LANGUAGE_KEYS = {
    "name",
    "whisper_language",
    "initial_prompt",
    "hallucination_patterns",
}
TRANSLATION_LANGUAGE_KEYS = {"name", "output_desc", "unit", "leakage"}

_LANGUAGES_CACHE: dict | None = None


class LanguageConfigError(RuntimeError):
    """Raised when config/languages.json is missing or structurally invalid."""


def load_languages(path=None, *, force_reload: bool = False) -> dict:
    """
    Load and validate config/languages.json (cached).

    Returns the raw document: ``{"languages": {...}, "xtts_supported": [...]}``.
    Raises LanguageConfigError with an actionable message on any problem.
    """
    global _LANGUAGES_CACHE
    if path is None and _LANGUAGES_CACHE is not None and not force_reload:
        return _LANGUAGES_CACHE

    target = Path(path) if path is not None else LANGUAGES_FILE
    if not target.is_file():
        raise LanguageConfigError(
            f"Language configuration not found: {target}\n"
            f"  This file is the single source of truth for ASR prompts, "
            f"translation prompt wording and TTS language tags."
        )

    with open(target, "r", encoding="utf-8") as fh:
        doc = json.load(fh)

    _validate_languages(doc, target)

    if path is None:
        _LANGUAGES_CACHE = doc
    return doc


def _validate_languages(doc: dict, source) -> None:
    """Structural validation — fail loudly at load, never mid-run."""
    if not isinstance(doc.get("languages"), dict) or not doc["languages"]:
        raise LanguageConfigError(
            f"{source}: missing or empty top-level 'languages' object."
        )
    if not isinstance(doc.get("xtts_supported"), list) or not doc["xtts_supported"]:
        raise LanguageConfigError(
            f"{source}: missing or empty top-level 'xtts_supported' array."
        )

    for code, entry in doc["languages"].items():
        if not isinstance(entry, dict):
            raise LanguageConfigError(f"{source}: language '{code}' is not an object.")
        if not entry.get("name"):
            raise LanguageConfigError(f"{source}: language '{code}' is missing 'name'.")

        asr = entry.get("asr")
        if not isinstance(asr, dict):
            raise LanguageConfigError(f"{source}: language '{code}' is missing 'asr'.")
        missing = {"whisper_language", "initial_prompt", "hallucination_patterns"} - set(asr)
        if missing:
            raise LanguageConfigError(
                f"{source}: language '{code}'.asr is missing key(s): "
                f"{', '.join(sorted(missing))}"
            )

        tr = entry.get("translation")
        if not isinstance(tr, dict):
            raise LanguageConfigError(
                f"{source}: language '{code}' is missing 'translation'."
            )
        missing = TRANSLATION_LANGUAGE_KEYS - set(tr)
        if missing:
            raise LanguageConfigError(
                f"{source}: language '{code}'.translation is missing key(s): "
                f"{', '.join(sorted(missing))}"
            )


def asr_language_registry(path=None) -> dict:
    """
    Phase C view: ``{code: {name, whisper_language, initial_prompt,
    hallucination_patterns}}`` — byte-compatible with the legacy
    config/language_registry.json shape Phase C already consumes.

    A legacy registry file may still be passed explicitly (it is accepted
    as-is), so an operator can pin an old prompt set without editing code.
    """
    if path is not None and Path(path).name == LEGACY_LANGUAGE_REGISTRY.name:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    doc = load_languages(path)
    registry = {}
    for code, entry in doc["languages"].items():
        asr = entry["asr"]
        registry[code] = {
            "name": entry["name"],
            "whisper_language": asr["whisper_language"],
            "initial_prompt": asr["initial_prompt"],
            "hallucination_patterns": list(asr["hallucination_patterns"]),
        }
    return registry


def translation_language_matrix(path=None) -> dict:
    """
    Phase D view: ``{code: {name, output_desc, unit, leakage}}`` — the matrix
    that drives prompt wording and post-generation leakage stripping.
    """
    doc = load_languages(path)
    matrix = {}
    for code, entry in doc["languages"].items():
        tr = entry["translation"]
        matrix[code] = {
            "name": tr["name"],
            "output_desc": tr["output_desc"],
            "unit": tr["unit"],
            "leakage": list(tr["leakage"]),
        }
    return matrix


def xtts_supported_languages(path=None) -> set:
    """Phase E view: the set of XTTS v2 language tags the model can synthesise."""
    return set(load_languages(path)["xtts_supported"])
