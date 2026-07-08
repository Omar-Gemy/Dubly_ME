"""
translation.py — Phase D: Contextual Translation & Adaptation
==============================================================
Offline Arabic → English dubbing translation using a self-hosted
Qwen LLM.  No third-party APIs are used.

Pipeline:
  1. Load transcripts.json from the ASR stage
  2. Pre-process: detect and flag Whisper loops and short hallucinations
  3. For each valid segment, build a context-aware prompt and translate
  4. Save enriched data to artifacts/translation.json

Usage:
  python src/translation.py
  python src/translation.py --model Qwen/Qwen2-7B-Instruct
  python src/translation.py --input artifacts/transcripts.json --output artifacts/translation.json
"""

import argparse
import copy
import re
import json
import os
import string
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ──────────────────────────────────────────────
#  Project paths (relative to repo root)
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_INPUT = ARTIFACTS_DIR / "transcripts.json"
DEFAULT_OUTPUT = ARTIFACTS_DIR / "translation.json"
DEFAULT_MODEL = "Qwen/Qwen2.5-14B-Instruct-AWQ"
DEFAULT_GLOSSARY = PROJECT_ROOT / "config" / "name_glossary.json"


# ──────────────────────────────────────────────
#  Name / terminology glossary
# ──────────────────────────────────────────────
def load_name_glossary(path: str | Path = DEFAULT_GLOSSARY) -> list[dict]:
    """
    Load the proper-noun glossary (canonical Arabic→English spellings).

    Non-fatal: a missing or malformed file returns an empty list with a
    warning, so translation still runs — the glossary is an enhancement, not
    a hard dependency. Each returned entry has at least 'ar' and 'en'; an
    optional 'variants' list holds alternate Arabic surface forms.
    """
    path = Path(path)
    if not path.is_file():
        print(f"       ⚠ No name glossary at {path} — proceeding without one.")
        return []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        entries = data.get("names", [])
        # Keep only well-formed entries.
        clean = [e for e in entries if e.get("ar") and e.get("en")]
        print(f"       ✔ Name glossary loaded: {len(clean)} entries.")
        return clean
    except (json.JSONDecodeError, OSError) as exc:
        print(f"       ⚠ Could not read glossary {path}: {exc} — proceeding without one.")
        return []


def _entry_surface_forms(entry: dict) -> list[str]:
    """All Arabic surface forms for a glossary entry (canonical + variants)."""
    forms = [entry["ar"]] + list(entry.get("variants") or [])
    # Longer forms first so 'عم شوقي' is matched before bare 'شوقي'.
    return sorted({f for f in forms if f}, key=len, reverse=True)


def relevant_glossary_entries(
    entries: list[dict],
    *texts: str,
) -> list[dict]:
    """
    Return the glossary entries whose canonical form or any variant appears
    in any of *texts* (the current line, plus scene context). Restricting the
    injected glossary to names actually present keeps the prompt tight.
    """
    haystack = " ".join(t for t in texts if t)
    hits = []
    for entry in entries:
        if any(form in haystack for form in _entry_surface_forms(entry)):
            hits.append(entry)
    return hits


def format_glossary_directive(entries: list[dict]) -> str:
    """
    Render glossary *entries* as an instruction block for the system prompt.
    Returns "" when there are no entries so the prompt is unchanged.
    """
    if not entries:
        return ""
    lines = [
        "\nTERMINOLOGY — use these EXACT English spellings for proper nouns; "
        "never vary or re-transliterate them:",
    ]
    for e in entries:
        forms = " / ".join(_entry_surface_forms(e))
        line = f"- {forms} → {e['en']}"
        if e.get("note"):
            line += f"  ({e['note']})"
        lines.append(line)
    return "\n".join(lines) + "\n"

# ──────────────────────────────────────────────
#  Heuristic thresholds
# ──────────────────────────────────────────────
SHORT_SEGMENT_DURATION = 1.0   # seconds — segments shorter than this are suspect
SHORT_SEGMENT_MAX_WORDS = 2    # if ≤ this many words AND short duration → flag
# Loop detection operates on WHOLE tokens (never sub-word substrings). Natural
# Egyptian reduplication repeats a word only 2–3× ("حاضر حاضر", "زن زن زن"); a
# real Whisper decode loop repeats many times. So we only *suspect* a loop at
# ≥ LOOP_MIN_REPEATS and only *skip* an egregious one at ≥ LOOP_SKIP_REPEATS.
LOOP_MIN_REPEATS = 4           # ≥ this many consecutive repeats → suspected loop (advisory)
LOOP_SKIP_REPEATS = 6          # ≥ this many → egregious loop, safe to hard-skip
LOOP_MIN_TOKEN_LEN = 2         # ignore single-character tokens (stray ا/ه/و) as loop units
# Punctuation stripped from token edges before loop comparison, so that a
# repeated word does not read as distinct just because one instance carries a
# trailing mark ("زن زن زن." → three equal "زن", not two + "زن."). Covers ASCII
# punctuation plus common Arabic marks.
LOOP_STRIP_CHARS = string.punctuation + "،؛؟…«»ـ" + "‏‎"

# ──────────────────────────────────────────────
#  Isochrony budget
# ──────────────────────────────────────────────
# Conversational English is spoken at ~4 syllables/sec. We convert each
# segment's duration into a target syllable count (with an accept band) and
# hand it to the model so the dubbed line fits its original time window —
# preventing the downstream TTS / time-stretch stages from over-compressing.
SYLLABLES_PER_SEC = 4.0
SYLL_BAND_LOW = 0.85
SYLL_BAND_HIGH = 1.15


def syllable_budget(duration: float) -> tuple[int, int, int]:
    """Return (target, low, high) English syllable counts for *duration* (s)."""
    base = max(1.0, duration) * SYLLABLES_PER_SEC
    target = max(1, round(base))
    lo = max(1, round(base * SYLL_BAND_LOW))
    hi = max(target, round(base * SYLL_BAND_HIGH))
    return target, lo, hi


# ──────────────────────────────────────────────
#  Step 1 — Load inputs
# ──────────────────────────────────────────────
def load_transcripts(path: str) -> dict:
    """Read the ASR-generated transcripts.json data contract."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ──────────────────────────────────────────────
#  Step 2 — Pre-processing heuristics
# ──────────────────────────────────────────────
def max_consecutive_repeat(text: str) -> int:
    """
    Return the length of the longest run of a WHOLE token (or 2-/3-token
    phrase) repeated consecutively.

    Operates strictly on whitespace-split tokens — never on sub-word
    substrings — so it cannot false-fire on a word that merely ends with the
    same letter the next word begins with (e.g. "ده هيقعد", which the old regex
    misread as a "ه ه" loop). Edge punctuation is stripped before comparison so
    a mark on one instance ("زن زن زن.") does not hide the repeat, and
    single-character tokens are ignored as repeat units, since stray one-letter
    ASR artefacts (ا/ه/و) are not decode loops.
    """
    raw = text.strip().split() if text else []
    # Normalise: strip edge punctuation, drop tokens that become empty.
    words = [tok for tok in (w.strip(LOOP_STRIP_CHARS) for w in raw) if tok]
    if len(words) < 2:
        return 0

    best = 1

    # Unigram, bigram, trigram phrase repeats.
    for n in (1, 2, 3):
        if len(words) < 2 * n:
            continue
        ngrams = [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]
        i = 0
        while i < len(ngrams):
            # Skip degenerate single-character unigram units.
            if n == 1 and len(ngrams[i]) < LOOP_MIN_TOKEN_LEN:
                i += 1
                continue
            run = 1
            j = i + n  # step by n so phrase repeats don't overlap
            while j < len(ngrams) and ngrams[j] == ngrams[i]:
                run += 1
                j += n
            best = max(best, run)
            i += 1

    return best


def detect_whisper_loop(text: str, min_repeats: int = LOOP_MIN_REPEATS) -> bool:
    """
    Return True if *text* contains a token/phrase repeated ≥ *min_repeats*
    times consecutively — the signature of a Whisper decode loop
    (e.g. "والأداء والأداء والأداء والأداء").

    Natural Egyptian reduplication ("حاضر حاضر", "كده كده", "زن زن زن") repeats
    only 2–3×, so the default threshold of 4 leaves it untouched.
    """
    if not text or not text.strip():
        return False
    return max_consecutive_repeat(text) >= min_repeats


def detect_short_hallucination(text: str, duration: float) -> bool:
    """
    Flag segments that are very short in duration AND have suspiciously
    brief text — likely ASR hallucinations on silence / noise.
    """
    if duration >= SHORT_SEGMENT_DURATION:
        return False

    if not text or not text.strip():
        return True  # empty text on a short segment is always suspect

    word_count = len(text.strip().split())
    return word_count <= SHORT_SEGMENT_MAX_WORDS


def preprocess_segments(segments: list[dict]) -> list[dict]:
    """
    Run heuristic checks on each segment BEFORE translation.
    Adds QA flags but does NOT modify the original text.

    Flags added:
      - transcription_failed (bool): Whisper loop detected
      - low_confidence (bool): short-duration hallucination suspect
      - _asr_hallucination (bool): ASR flagged as hallucination suspect
      - _asr_skipped (bool): ASR skipped due to low energy or too short
      - skip_translation (bool): True if any skip condition is met
    """
    flagged_loops = 0
    flagged_short = 0
    flagged_asr_hallucination = 0
    flagged_asr_skipped = 0
    flagged_empty = 0

    for seg in segments:
        text = seg.get("text", "") or ""
        duration = seg.get("duration", 0.0)

        is_loop = detect_whisper_loop(text)
        # Only an EGREGIOUS loop (many repeats) is safe to hard-skip; a merely
        # suspected loop stays translatable and is flagged for QA instead.
        is_egregious_loop = detect_whisper_loop(text, min_repeats=LOOP_SKIP_REPEATS)
        is_short_hallucination = detect_short_hallucination(text, duration)

        # ── ASR-flagged hallucination suspects ────────
        # Segments flagged by ASR's pattern matcher + duplicate detector
        is_asr_hallucination = seg.get("_hallucination_suspect", False)

        # ── ASR-skipped segments (low energy / too short) ──
        # These had empty text set by ASR — no point sending to LLM
        is_skipped_by_asr = (
            seg.get("_skipped_low_energy", False)
            or seg.get("_skipped_too_short", False)
        )

        # ── Empty text guard ──────────────────────────
        # Catch any segment with blank text regardless of flags
        is_empty = not text.strip()

        seg["transcription_failed"] = is_loop
        seg["low_confidence"] = is_short_hallucination
        seg["_asr_hallucination"] = is_asr_hallucination
        seg["_asr_skipped"] = is_skipped_by_asr
        # NOTE: a SUSPECTED loop (transcription_failed) is advisory only — it no
        # longer forces a skip, because the old detector false-fired on ~19% of
        # valid lines. is_short_hallucination is likewise advisory. Only an
        # egregious loop, ASR-confirmed hallucination, ASR-skip, or empty text
        # is dropped from translation.
        seg["skip_translation"] = (
            is_egregious_loop or is_asr_hallucination or is_skipped_by_asr or is_empty
        )

        if is_loop:
            flagged_loops += 1
        if is_short_hallucination:
            flagged_short += 1
        if is_asr_hallucination:
            flagged_asr_hallucination += 1
        if is_skipped_by_asr:
            flagged_asr_skipped += 1
        if is_empty:
            flagged_empty += 1

    print(f"       Suspected loops (advisory): {flagged_loops}")
    print(f"       Short hallucinations    : {flagged_short}")
    print(f"       ASR hallucination flags : {flagged_asr_hallucination}")
    print(f"       ASR skipped (energy/dur): {flagged_asr_skipped}")
    print(f"       Empty text segments     : {flagged_empty}")
    skipped = sum(1 for s in segments if s["skip_translation"])
    print(f"       Segments to skip        : {skipped}")
    translatable = len(segments) - skipped
    print(f"       Segments to translate   : {translatable}")

    return segments


# ──────────────────────────────────────────────
#  Step 3 — LLM-based contextual translation
# ──────────────────────────────────────────────
# ── Dynamic Language Matrix (Phase 1b) ───────────────────────────
# Drives prompt wording + post-generation leakage stripping per language.
# Each key is an XTTS v2 language tag, so translation and TTS stay in lockstep.
LANGUAGES = {
    "ar": {
        "name": "Egyptian Arabic",
        "output_desc": "natural, idiomatic spoken Egyptian Arabic dialogue",
        "unit": "Arabic",
        "leakage": ["بالتأكيد", "إليك", "الترجمة هي", "طبعاً", "بالطبع",
                    "يمكنني", "هذه هي", "إليكم"],
    },
    "en": {
        "name": "English",
        "output_desc": "natural, idiomatic spoken English",
        "unit": "English",
        "leakage": ["sure,", "here is", "i can help", "please provide",
                    "i'd be happy", "of course", "certainly",
                    "let me", "i'll translate", "the translation is"],
    },
    "es": {
        "name": "Spanish",
        "output_desc": "natural, idiomatic spoken Spanish (neutral Latin American)",
        "unit": "Spanish",
        "leakage": ["claro,", "aquí está", "aquí tienes", "por supuesto",
                    "puedo ayudar", "la traducción es", "desde luego"],
    },
}


def build_system_prompt(source_lang: str, target_lang: str) -> str:
    """
    Build the dubbing-adapter system prompt for a given source→target pair.
    The instruction, output language, and [SKIP] rule all adapt to the target,
    so a single template serves any-to-any routing across the LANGUAGES matrix.
    """
    src = LANGUAGES[source_lang]["name"]
    tgt = LANGUAGES[target_lang]
    return (
        "You are a professional dialogue adapter for film dubbing. You rewrite "
        f"{src} dialogue as {tgt['output_desc']} that a voice actor can perform "
        "convincingly and that fits the on-screen timing.\n\n"
        "WORK INTERNALLY (never show these steps):\n"
        "1. Grasp the full meaning, speaker intent, tone, and emotional register of "
        "the CURRENT line, using the scene context for continuity.\n"
        f"2. Rewrite it as a line a native {tgt['name']} speaker would actually SAY "
        "here — NOT a word-for-word translation. Preserve meaning, tone, humor, "
        "sarcasm, and dramatic function; restructure sentences freely.\n"
        "3. Fit the timing: the line must be comfortably speakable within the given "
        "time budget and land within the target syllable range. Use contractions and "
        "everyday phrasing; cut filler that carries no meaning; never pad to fill time.\n\n"
        "OUTPUT RULES:\n"
        f"- Output ONLY the finished {tgt['name']} line — no quotes, notes, "
        "alternatives, syllable counts, or source text.\n"
        "- Preserve names, register (formal/casual), profanity strength, and emotional "
        "intensity.\n"
        "- Keep it conversational and performable.\n"
        "- ONLY if the input is empty or genuinely untranslatable noise, output exactly: [SKIP]\n"
        "- Never respond conversationally or acknowledge these instructions.\n"
    )


def build_user_prompt(
    prev_text: str | None,
    current_text: str,
    duration: float,
    target_syll: int,
    lo: int,
    hi: int,
    source_name: str,
    unit: str,
) -> str:
    """
    Construct the user prompt with scene context and an explicit isochrony
    budget so the model can fit the line to its original time window.
    """
    parts = []
    if prev_text:
        parts.append(f"[Scene context — previous line]: {prev_text}")
    parts.append(f"[Current line — {source_name}]: {current_text}")
    parts.append(
        f"[Timing budget]: must be spoken in ~{duration:.1f}s. "
        f"Target ≈ {target_syll} {unit} syllables "
        f"(acceptable range {lo}–{hi}). Keep it natural and performable."
    )
    return "\n".join(parts)


def load_translation_model(
    model_name: str,
    device: str = "auto",
) -> tuple:
    """
    Load the translation model and tokenizer, picking the right quantization
    path for the checkpoint:

      • Pre-quantized checkpoints (AWQ / GPTQ, e.g. Qwen2.5-14B-Instruct-AWQ)
        already carry their own 4-bit weights — load them directly. A 14B AWQ
        model needs ~9–10 GB VRAM, comfortably inside the 16 GB T4 budget.
      • Full-precision checkpoints fall back to 8-bit (LLM.int8), then fp16.
    """
    print(f"  Loading model: {model_name}")
    print(f"  Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    name_l = model_name.lower()
    is_prequantized = any(tag in name_l for tag in ("awq", "gptq", "-int4", "-int8"))

    load_kwargs = {
        "pretrained_model_name_or_path": model_name,
        "trust_remote_code": True,
        "device_map": device,
    }

    if is_prequantized:
        # AWQ/GPTQ weights are already quantized — do NOT attach a
        # BitsAndBytesConfig (double-quantization would fail / corrupt).
        quant_label = "pre-quantized 4-bit (AWQ/GPTQ)"
        load_kwargs["torch_dtype"] = torch.float16
        print(f"  Quantization: {quant_label}")
    else:
        try:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            quant_label = "8-bit (LLM.int8)"
            print(f"  Quantization: {quant_label}")
        except Exception:
            print("  ⚠ 8-bit quantization unavailable, falling back to float16 …")
            load_kwargs["torch_dtype"] = torch.float16
            quant_label = "float16"

    model = AutoModelForCausalLM.from_pretrained(**load_kwargs)

    print(f"  ✔ Model loaded ({quant_label}).\n")
    return model, tokenizer


# ── Post-generation chatbot leakage patterns ─────────
# Now per-language: each entry in LANGUAGES carries its own "leakage" prefix
# list, applied against the TARGET language's output in translate_single().


def translate_single(
    model,
    tokenizer,
    prev_text: str | None,
    current_text: str,
    duration: float,
    source_lang: str,
    target_lang: str,
    max_new_tokens: int = 256,
    glossary_directive: str = "",
) -> str:
    """
    Adapt a single segment into dub-ready target-language text using the loaded
    LLM, passing the isochrony budget derived from *duration*.

    *glossary_directive* (optional) is appended to the system prompt to force
    consistent proper-noun spellings.

    Includes conservative post-generation sanitization: known chatbot
    prefixes are stripped ONLY when a separator makes the real line
    recoverable — a valid translation is never discarded on a prefix match
    alone.
    """
    target_syll, lo, hi = syllable_budget(duration)

    system_content = build_system_prompt(source_lang, target_lang) + glossary_directive
    messages = [
        {"role": "system", "content": system_content},
        {
            "role": "user",
            "content": build_user_prompt(
                prev_text, current_text, duration, target_syll, lo, hi,
                LANGUAGES[source_lang]["name"], LANGUAGES[target_lang]["unit"],
            ),
        },
    ]

    # Use the model's chat template for proper formatting
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            # Light sampling → more natural, less flat-literal phrasing than
            # pure greedy, while staying tightly controlled for dubbing.
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    input_len = inputs["input_ids"].shape[1]
    generated = outputs[0][input_len:]
    result = tokenizer.decode(generated, skip_special_tokens=True).strip()

    # ── Conservative post-generation sanitization ────
    # If (and only if) the output opens with a known conversational prefix
    # AND a separator lets us recover the line after it, strip the prefix.
    # Otherwise keep the text untouched — never convert a real line to [SKIP].
    result_lower = result.lower()
    for pattern in LANGUAGES[target_lang]["leakage"]:
        if result_lower.startswith(pattern):
            for sep in [":\n", ":\r\n", "\n", ": ", " — ", " - "]:
                if sep in result:
                    result = result.split(sep, 1)[1].strip()
                    break
            break

    return result


# ──────────────────────────────────────────────
#  QW-1 — Post-translation length gate
# ──────────────────────────────────────────────
# Catches egregious translation bloat that would cause >2× time-stretch.
# Fires only when the output exceeds the generous syllable ceiling by ≥50%.

def estimate_syllables(text: str) -> int:
    """
    Estimate syllable count for length-gate purposes.

    English: counts vowel clusters (a-e-i-o-u-y groups). ~85% accurate —
    sufficient for a ±15% tolerance band.
    Arabic:  counts Arabic vowel letters (ا و ي ى ة) + explicit diacritics
    (fatHa, kasra, Damma, tanwiin, shadda, sukuun). This is a rough proxy;
    Arabic syllable structure is complex, but for the purpose of detecting
    *gross* overruns (≥50% over budget) it is adequate.
    Other:   falls back to the English heuristic.
    """
    if not text or not text.strip():
        return 0

    # Detect whether the text is predominantly Arabic (Unicode block 0600–06FF).
    arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    total_alpha = sum(1 for c in text if c.isalpha()) or 1

    if arabic_chars / total_alpha > 0.5:
        # Arabic heuristic: vowel letters + diacritics as syllable nuclei.
        vowel_letters = len(re.findall(r'[اوييىةآأإؤئ]', text))
        diacritics = len(re.findall(r'[\u064B-\u0652]', text))  # fatHa → sukuun
        count = max(1, vowel_letters + diacritics)
        # Arabic words without diacritics: estimate ~2 syllables per whitespace token.
        if diacritics == 0:
            count = max(count, len(text.split()) * 2)
        return count

    # English / Latin heuristic: count vowel clusters.
    text_lower = text.lower()
    # Split on non-alpha to get words.
    words = re.findall(r"[a-z']+", text_lower)
    count = 0
    for word in words:
        # Count vowel groups.
        vowel_groups = re.findall(r'[aeiouy]+', word)
        n = len(vowel_groups)
        # Silent trailing 'e' heuristic.
        if word.endswith('e') and n > 1:
            n -= 1
        count += max(1, n)  # every word has at least 1 syllable
    return max(1, count)


def build_shortening_system_prompt(target_lang: str) -> str:
    """
    Dedicated system prompt for the isochrony shortening pass.
    Separate from the main translation prompt — this is an *editing* task,
    not a re-translation.
    """
    dialect_rule = ""
    if target_lang == "ar":
        dialect_rule = (
            "7. You MUST output the shortened text strictly in Egyptian Arabic "
            "(Masri) dialect. Do not use Modern Standard Arabic (MSA) under "
            "any circumstances.\n"
        )

    return (
        "You are a dubbing timing editor. Your ONLY job is to shorten a dubbed "
        "line that is too long to fit its on-screen time window.\n\n"
        "RULES:\n"
        "1. Output ONLY the shortened line — no explanation, no alternatives, "
        "no syllable counts, no quotes.\n"
        "2. You must preserve: the speaker's meaning, emotional tone, register "
        "(casual/formal), sarcasm, humor, and names.\n"
        "3. Use everyday, conversational phrasing. Cut filler words, redundant "
        "modifiers, and unnecessary clauses. Restructure freely.\n"
        "4. The shortened line MUST be speakable in the given time budget. "
        "Treat the syllable ceiling as a HARD LIMIT — never exceed it.\n"
        "5. Do NOT add new information, pad the line, or explain what you "
        "changed.\n"
        "6. If the line is already at minimum viable meaning and cannot be "
        "shortened further, output the shortest version you can.\n"
        + dialect_rule
    )


def build_shortening_user_prompt(
    overlong_text: str,
    actual_syllables: int,
    max_syllables: int,
) -> str:
    """User prompt with quantitative feedback for the shortening pass."""
    return (
        "The following dubbed line is too long for its time window.\n\n"
        f"[OVERLONG LINE]: {overlong_text}\n"
        f"[PROBLEM]: This line is approximately {actual_syllables} syllables, "
        f"but the time window only allows at most {max_syllables} syllables.\n"
        f"[REQUIRED]: Rewrite it in at most {max_syllables} syllables. "
        "Keep the same meaning, tone, and speaker register."
    )


def shorten_single(
    model,
    tokenizer,
    overlong_text: str,
    actual_syllables: int,
    max_syllables: int,
    target_lang: str,
    max_new_tokens: int = 256,
) -> str:
    """
    Re-prompt the LLM with the overlong translation to compress it.
    Uses a dedicated shortening prompt (not the main translation prompt).
    Reuses the already-loaded model and tokenizer — no extra model loading.

    Returns the shortened text, or the original if shortening fails.
    """
    system_content = build_shortening_system_prompt(target_lang)
    user_content = build_shortening_user_prompt(
        overlong_text, actual_syllables, max_syllables,
    )

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    generated = outputs[0][input_len:]
    result = tokenizer.decode(generated, skip_special_tokens=True).strip()

    # Strip chatbot leakage (same logic as translate_single).
    result_lower = result.lower()
    for pattern in LANGUAGES[target_lang]["leakage"]:
        if result_lower.startswith(pattern):
            for sep in [":\n", ":\r\n", "\n", ": ", " — ", " - "]:
                if sep in result:
                    result = result.split(sep, 1)[1].strip()
                    break
            break

    # If the model returned garbage or [SKIP], keep the original.
    if not result.strip() or result.strip() == "[SKIP]":
        return overlong_text

    return result


def translate_segments(
    segments_data: dict,
    model_name: str,
    source_lang: str,
    target_lang: str,
    device: str = "auto",
    glossary: list[dict] | None = None,
) -> dict:
    """
    Translate all valid (non-skipped) segments using the LLM.
    Each segment receives the previous segment's text as context, plus any
    glossary entries whose names appear in the current or previous line.
    """
    model, tokenizer = load_translation_model(model_name, device)
    glossary = glossary or []

    segments = segments_data["segments"]
    total = len(segments)
    translated_count = 0

    prev_text = None  # rolling context window

    for idx, seg in enumerate(segments, start=1):
        seg_id = seg["segment_id"]
        text = seg.get("text", "") or ""

        # Skip flagged segments
        if seg.get("skip_translation", False):
            reason = []
            if seg.get("transcription_failed"):
                reason.append("whisper-loop")
            if seg.get("low_confidence"):
                reason.append("short-hallucination")
            if seg.get("_asr_hallucination"):
                reason.append("asr-hallucination")
            if seg.get("_asr_skipped"):
                reason.append("asr-skipped")
            if not (seg.get("text", "") or "").strip():
                reason.append("empty-text")
            reason_str = ", ".join(reason) or "flagged"

            print(
                f"  [{idx}/{total}]  Segment #{seg_id}  "
                f"⏭ SKIPPED ({reason_str})"
            )
            seg["translated_text"] = None
            # Don't update prev_text with bad data
            continue

        print(
            f"  [{idx}/{total}]  Segment #{seg_id}  "
            f"\"{text[:50]}{'…' if len(text) > 50 else ''}\"  → ",
            end="",
            flush=True,
        )

        # Inject only the glossary names present in this line or its context,
        # so recurring characters keep a stable English spelling.
        directive = format_glossary_directive(
            relevant_glossary_entries(glossary, text, prev_text or "")
        )
        translated = translate_single(
            model, tokenizer, prev_text, text, seg.get("duration", 0.0),
            source_lang, target_lang,
            glossary_directive=directive,
        )

        # Handle untranslatable output from the model
        if translated == "[SKIP]" or not translated.strip():
            seg["translated_text"] = None
            seg["_translation_skipped"] = True
            print(f"⏭ SKIPPED (model returned [SKIP])")
            continue

        # ── QW-1: Post-translation length gate ────
        # Catch egregious bloat that would cause >2× time-stretch.
        target_syll, lo, hi = syllable_budget(seg.get("duration", 0.0))
        actual_syll = estimate_syllables(translated)
        length_gate_ceiling = int(hi * 1.5)

        if actual_syll > length_gate_ceiling:
            print(
                f"✔ (overlong: ~{actual_syll} syll, budget ≤{hi}) → shortening… ",
                end="", flush=True,
            )
            shortened = shorten_single(
                model, tokenizer, translated,
                actual_syll, hi, target_lang,
            )
            new_syll = estimate_syllables(shortened)
            translated = shortened
            seg["_length_gate"] = {
                "original_syllables": actual_syll,
                "shortened_syllables": new_syll,
                "ceiling": hi,
                "triggered": True,
            }
            print(f"✔ (~{new_syll} syll)")
        else:
            seg["_length_gate"] = {"triggered": False}

        seg["translated_text"] = translated
        # Record the isochrony budget this line was adapted against, for the
        # downstream time-stretch / QA stages.
        seg["syllable_budget"] = {"target": target_syll, "low": lo, "high": hi}
        translated_count += 1

        # Update rolling context
        prev_text = text

        preview = translated[:55] + "…" if len(translated) > 55 else translated
        if not seg["_length_gate"]["triggered"]:
            print(f"✔ \"{preview}\"")

    print(f"\n  Translated {translated_count}/{total} segments.")
    return segments_data


# ──────────────────────────────────────────────
#  Step 4 — Save the enriched data contract
# ──────────────────────────────────────────────
def save_translation(data: dict, output_path: str) -> None:
    """Write the translated segments to a JSON file (UTF-8, pretty-printed)."""
    data["translation_completed_at"] = datetime.now(timezone.utc).isoformat()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


# ──────────────────────────────────────────────
#  CLI entry-point
# ──────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dubly ME — Phase D: Contextual Translation & Adaptation",
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Path to the transcripts JSON  (default: artifacts/transcripts.json)",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output JSON path  (default: artifacts/translation.json)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"HuggingFace model ID  (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device map: 'auto', 'cpu', or 'cuda:0'  (default: auto)",
    )
    parser.add_argument(
        "--source-lang",
        default=None,
        help="Source language code (ar/en/es). "
             "Default: source_language from transcripts.json, else 'ar'.",
    )
    parser.add_argument(
        "--target-lang",
        default=None,
        help="Target/dub language code (ar/en/es). "
             "Default: target_language from transcripts.json, else 'en'.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Max tokens to generate per segment  (default: 256)",
    )
    parser.add_argument(
        "--glossary",
        default=str(DEFAULT_GLOSSARY),
        help="Path to the name/terminology glossary JSON  (default: config/name_glossary.json)",
    )
    parser.add_argument(
        "--no-glossary",
        action="store_true",
        help="Disable proper-noun glossary injection.",
    )
    args = parser.parse_args()

    # ── Validate input ───────────────────────
    if not os.path.isfile(args.input):
        print(f"✖  Input file not found: {args.input}")
        sys.exit(1)

    # ── Step 1: Load transcripts ─────────────
    print(f"[1/4]  Loading transcripts: {args.input}")
    data = load_transcripts(args.input)
    n_segs = data["total_segments"]
    print(f"       ✔ {n_segs} segment(s) loaded.")

    # ── Resolve the language pair (CLI → transcripts.json → ar→en) ──
    source_lang = (args.source_lang or data.get("source_language") or "ar").strip().lower()
    target_lang = (args.target_lang or data.get("target_language") or "en").strip().lower()
    for role, code in (("source", source_lang), ("target", target_lang)):
        if code not in LANGUAGES:
            print(f"✖  Unsupported {role} language '{code}'. "
                  f"Supported: {', '.join(sorted(LANGUAGES))}")
            sys.exit(1)
    # Persist the actual pair used so downstream TTS reads a correct target.
    data["source_language"] = source_lang
    data["target_language"] = target_lang
    print(f"       Language matrix: {source_lang} → {target_lang}")

    # ── Load name glossary (non-fatal) ───────
    glossary = [] if args.no_glossary else load_name_glossary(args.glossary)

    # ── Step 2: Pre-processing heuristics ────
    print(f"\n[2/4]  Running pre-processing heuristics…")
    data["segments"] = preprocess_segments(data["segments"])

    # ── Step 3: Contextual translation ───────
    print(f"\n[3/4]  Translating with LLM (model={args.model})…\n")
    data = translate_segments(
        data, model_name=args.model, source_lang=source_lang,
        target_lang=target_lang, device=args.device, glossary=glossary,
    )

    # ── Step 4: Save output ──────────────────
    print(f"\n[4/4]  Saving translation → {args.output}")
    data["translation_model"] = args.model
    # Additive contract field — records glossary provenance for this run.
    data["name_glossary"] = {
        "path": None if args.no_glossary else str(args.glossary),
        "entries": len(glossary),
        "applied": [e["en"] for e in glossary],
    }
    save_translation(data, args.output)

    # ── Summary ──────────────────────────────
    translated = sum(1 for s in data["segments"] if s.get("translated_text"))
    skipped = sum(1 for s in data["segments"] if s.get("skip_translation"))

    print()
    print(f"{'═' * 60}")
    print(f"  ✅  translation.json saved → {args.output}")
    print(f"  Segments translated : {translated}/{n_segs}")
    print(f"  Segments skipped    : {skipped}/{n_segs}")
    if skipped:
        print(f"  ⚠  Skipped segments need manual review")
    print(f"{'═' * 60}")

    # Preview skipped segments for quick QA
    skipped_segs = [s for s in data["segments"] if s.get("skip_translation")]
    if skipped_segs:
        print(f"\n  Skipped segments detail:")
        for seg in skipped_segs:
            flags = []
            if seg.get("transcription_failed"):
                flags.append("LOOP")
            if seg.get("low_confidence"):
                flags.append("SHORT")
            print(
                f"    #{seg['segment_id']:>3d}  "
                f"{seg.get('duration', 0):.1f}s  "
                f"[{','.join(flags)}]  "
                f"\"{seg.get('text', '')}\""
            )


if __name__ == "__main__":
    main()
