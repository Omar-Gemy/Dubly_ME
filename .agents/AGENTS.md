# Dubly ME Project Rules

1. **ZERO EXTERNAL APIs:**
   - NEVER suggest or write code for commercial APIs (OpenAI, Anthropic, Google Cloud, ElevenLabs, etc.).
   - All machine learning inference (ASR, LLM, TTS, Separation) must execute LOCALLY using open-source models (Whisper, Qwen, XTTS v2, Demucs).

2. **DECOUPLED ARCHITECTURE MANDATE:**
   - Phases (A through F) MUST NOT share state in memory.
   - All data transfer between phases happens STRICTLY via JSON Data Contracts (segments.json -> transcripts.json -> etc.).
   - Do NOT merge scripts or bypass the JSON file I/O.

3. **VIRTUAL ENVIRONMENT ISOLATION:**
   - Assume code runs across 3 isolated venvs (ASR, LLM, Audio).
   - Do not attempt to unify dependencies into a single requirements file that causes conflicts.

4. **CONTRACT HYGIENE:**
   - Do NOT overwrite existing fields in the JSON payload. Phases must "enrich" the JSON by adding their specific payload tier while preserving previous data.
