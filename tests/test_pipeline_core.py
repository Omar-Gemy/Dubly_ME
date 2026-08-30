"""Unit tests for pipeline_core — the Chunk 1 shared foundation.

Focus: the invariants that make the later chunks safe. Nothing here needs a GPU,
a model download, or the source media.
"""

import json
import sys
import unittest
from pathlib import Path

# Reproduce the import environment the CLIs run in (src/ on sys.path). conftest
# does this for pytest; repeated here so the file also runs under plain unittest.
SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pipeline_core as pc  # noqa: E402


class PathHelperTests(unittest.TestCase):
    def test_inside_repo_becomes_posix_relative(self):
        target = pc.ARTIFACTS_DIR / "audio_out" / "segment_001.wav"
        self.assertEqual(
            pc.rel_or_abs(target), "artifacts/audio_out/segment_001.wav"
        )

    def test_outside_repo_falls_back_to_absolute(self):
        # The source video legitimately lives outside the repo (Google Drive).
        outside = Path("/content/drive/MyDrive/clip.mp4")
        self.assertEqual(pc.rel_or_abs(outside), "/content/drive/MyDrive/clip.mp4")

    def test_accepts_str_and_path(self):
        target = pc.ARTIFACTS_DIR / "segments.json"
        self.assertEqual(pc.rel_or_abs(str(target)), pc.rel_or_abs(target))


class SampleRateRegistryTests(unittest.TestCase):
    def test_declared_rates_are_the_documented_values(self):
        self.assertEqual(pc.ASR_SAMPLE_RATE, 16000)
        self.assertEqual(pc.XTTS_NATIVE_SAMPLE_RATE, 22050)
        self.assertEqual(pc.SEPARATION_WORK_RATE, 44100)
        self.assertEqual(pc.PIPELINE_SAMPLE_RATE, 24000)

    def test_registry_mirrors_the_constants(self):
        self.assertEqual(
            pc.SAMPLE_RATES,
            {
                "asr": pc.ASR_SAMPLE_RATE,
                "xtts_native": pc.XTTS_NATIVE_SAMPLE_RATE,
                "separation_work": pc.SEPARATION_WORK_RATE,
                "pipeline": pc.PIPELINE_SAMPLE_RATE,
            },
        )

    def test_xtts_rate_differs_from_timeline_rate(self):
        # Guards the 5.5 misconception: mix_render's comment used to claim the
        # 24 kHz timeline WAS XTTS's native rate, implying no resample happens.
        self.assertNotEqual(pc.XTTS_NATIVE_SAMPLE_RATE, pc.PIPELINE_SAMPLE_RATE)


class PhaseRegistryTests(unittest.TestCase):
    def test_every_phase_maps_to_a_module_that_exists(self):
        for phase_id, entry in pc.PHASES.items():
            script = pc.SRC_DIR / f"{entry['module']}.py"
            self.assertTrue(script.is_file(), f"Phase {phase_id}: missing {script}")

    def test_every_phase_declares_a_known_venv(self):
        known = {"asr", "llm", "audio", "demucs"}
        for phase_id, entry in pc.PHASES.items():
            self.assertIn(entry["venv"], known, f"Phase {phase_id}")

    def test_phase_title_format(self):
        self.assertEqual(pc.phase_title("C"), "Phase C — ASR Transcription")


class ResampleTests(unittest.TestCase):
    def setUp(self):
        import numpy as np

        self.np = np
        self.tone = np.sin(
            2 * np.pi * 220 * np.arange(pc.XTTS_NATIVE_SAMPLE_RATE) /
            pc.XTTS_NATIVE_SAMPLE_RATE
        ).astype(np.float32)

    def test_matching_rates_are_a_no_op(self):
        out = pc.resample(self.tone, 24000, 24000)
        self.assertIs(out.dtype.type, self.np.float32)
        self.np.testing.assert_array_equal(out, self.tone)

    def test_length_scales_with_the_rate_ratio(self):
        out = pc.resample(
            self.tone, pc.XTTS_NATIVE_SAMPLE_RATE, pc.PIPELINE_SAMPLE_RATE
        )
        expected = len(self.tone) * pc.PIPELINE_SAMPLE_RATE // pc.XTTS_NATIVE_SAMPLE_RATE
        self.assertAlmostEqual(len(out), expected, delta=2)
        self.assertIs(out.dtype.type, self.np.float32)

    def test_amplitude_is_preserved_within_tolerance(self):
        out = pc.resample(
            self.tone, pc.XTTS_NATIVE_SAMPLE_RATE, pc.PIPELINE_SAMPLE_RATE
        )
        self.assertLess(abs(float(self.np.abs(out).max()) - 1.0), 0.05)


class LanguageConfigTests(unittest.TestCase):
    """config/languages.json must reproduce all three legacy sources exactly."""

    def test_asr_view_matches_the_legacy_registry_byte_for_byte(self):
        legacy_path = pc.LEGACY_LANGUAGE_REGISTRY
        if not legacy_path.is_file():
            self.skipTest("legacy language_registry.json no longer present")
        with open(legacy_path, "r", encoding="utf-8") as fh:
            legacy = json.load(fh)
        self.assertEqual(pc.asr_language_registry(), legacy)

    def test_asr_view_has_every_required_key(self):
        for code, entry in pc.asr_language_registry().items():
            missing = pc.ASR_LANGUAGE_KEYS - set(entry)
            self.assertFalse(missing, f"{code} missing {missing}")

    def test_translation_view_has_every_required_key(self):
        for code, entry in pc.translation_language_matrix().items():
            missing = pc.TRANSLATION_LANGUAGE_KEYS - set(entry)
            self.assertFalse(missing, f"{code} missing {missing}")

    def test_arabic_keeps_its_two_distinct_display_names(self):
        # Phase C prints the registry name; Phase D injects the prompt name.
        # Collapsing them would silently change the LLM's system prompt.
        self.assertEqual(pc.asr_language_registry()["ar"]["name"], "Arabic (Egyptian)")
        self.assertEqual(pc.translation_language_matrix()["ar"]["name"], "Egyptian Arabic")

    def test_xtts_set_is_the_17_builtin_languages(self):
        langs = pc.xtts_supported_languages()
        self.assertEqual(len(langs), 17)
        self.assertIn("zh-cn", langs)

    def test_every_translatable_language_is_synthesisable(self):
        # Phase D and Phase E must stay in lockstep: a language we can translate
        # into but not synthesise would fail at Phase E after paying for D.
        for code in pc.translation_language_matrix():
            self.assertIn(code, pc.xtts_supported_languages())

    def test_missing_file_raises_an_actionable_error(self):
        with self.assertRaises(pc.LanguageConfigError):
            pc.load_languages(pc.CONFIG_DIR / "does_not_exist.json")


class DeterminismHelperTests(unittest.TestCase):
    def test_segment_seeds_are_stable_and_distinct(self):
        a = pc.derive_segment_seed(1234, 7)
        self.assertEqual(a, pc.derive_segment_seed(1234, 7))
        self.assertNotEqual(a, pc.derive_segment_seed(1234, 8))
        self.assertNotEqual(a, pc.derive_segment_seed(1235, 7))

    def test_segment_seed_fits_in_a_32_bit_range(self):
        # numpy's legacy seeder rejects anything wider.
        for seg_id in (1, 42, 999, 10_000):
            seed = pc.derive_segment_seed(0, seg_id)
            self.assertGreaterEqual(seed, 0)
            self.assertLess(seed, 2 ** 32)

    def test_seed_everything_is_reproducible(self):
        import random

        pc.seed_everything(99)
        first = [random.random() for _ in range(5)]
        pc.seed_everything(99)
        self.assertEqual(first, [random.random() for _ in range(5)])


class ProvenanceHelperTests(unittest.TestCase):
    def test_json_hash_ignores_key_order(self):
        self.assertEqual(
            pc.sha256_json({"a": 1, "b": [2, 3]}),
            pc.sha256_json({"b": [2, 3], "a": 1}),
        )

    def test_json_hash_detects_a_value_change(self):
        self.assertNotEqual(pc.sha256_json({"a": 1}), pc.sha256_json({"a": 2}))

    def test_file_hash_matches_hashlib(self):
        import hashlib
        import tempfile

        payload = b"dubly" * 1000
        with tempfile.NamedTemporaryFile(delete=False) as fh:
            fh.write(payload)
            path = fh.name
        try:
            self.assertEqual(pc.sha256_file(path), hashlib.sha256(payload).hexdigest())
        finally:
            Path(path).unlink()


if __name__ == "__main__":
    unittest.main()
