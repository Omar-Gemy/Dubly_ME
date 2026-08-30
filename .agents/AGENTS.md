# Dubly ME Project Rules

1. **ZERO EXTERNAL APIs:**
   - NEVER suggest or write code for commercial APIs (OpenAI, Anthropic, Google Cloud, ElevenLabs, etc.).
   - All machine learning inference (ASR, LLM, TTS, Separation) must execute LOCALLY using open-source models (Whisper, Qwen, XTTS v2, Demucs).

2. **DECOUPLED ARCHITECTURE MANDATE:**
   - Phases (A through F) MUST NOT share state in memory.
   - All data transfer between phases happens STRICTLY via JSON Data Contracts (segments.json -> transcripts.json -> etc.).
   - Do NOT merge scripts or bypass the JSON file I/O.

3. **VIRTUAL ENVIRONMENT ISOLATION:**
   - Assume code runs across 3 isolated venvs (ASR, LLM, Audio), plus a 4th for Demucs (Phase F0).
   - Do not attempt to unify dependencies into a single requirements file that causes conflicts.
   - Per-venv pins live in `requirements/{asr,llm,audio,demucs}.txt`. The root
     `requirements.txt` is an index only and installs nothing.

4. **CONTRACT HYGIENE — MINIMAL OWNED CONTRACTS:**
   - Each phase OWNS its output contract and writes it via an EXPLICIT projection
     (e.g. `build_transcript_contract`, `build_translation_contract`), declaring
     exactly which fields it emits.
   - A contract carries only (a) the routing/timing context the next phase needs
     and (b) that phase's own payload. Upstream run metadata — VAD thresholds,
     diarization stats, ASR gate stats — MUST NOT be propagated downstream.
   - Do NOT deep-copy an upstream contract and append to it. That is the "JSON
     snowball" anti-pattern removed in Issue 5.1; re-introducing it is a
     regression, not enrichment.
   - Never silently overwrite a field a phase does not own. Adding a field means
     adding it to that phase's declared projection.

5. **SHARED LIBRARY ≠ SHARED STATE:**
   - `src/pipeline_core.py` holds cross-phase CONSTANTS and PURE HELPERS only
     (paths, sample rates, phase registry, resample, device probing, seeding,
     hashing, language config). Importing it is allowed from every phase and
     does not violate Rule 2.
   - It must never hold pipeline data, mutable module state, or anything that
     carries information from one phase to another. Data crosses phases only
     through the JSON contracts.
   - Its heavy imports (torch / numpy / scipy) stay LAZY: the module is imported
     by stdlib-only phases and by four environments that do not share packages.

6. **SINGLE SOURCE OF TRUTH FOR CONFIG:**
   - Sample rates: `pipeline_core` only. Never redeclare a rate in a phase.
   - Phase ids and labels: `pipeline_core.PHASES` only.
   - Languages: `config/languages.json` only — it feeds Phase C's ASR prompts,
     Phase D's prompt matrix and Phase E's XTTS tag set. Adding a language is a
     config edit, never a code edit.
