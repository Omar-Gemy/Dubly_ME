"""Unit tests for the Phase C/D JSON contract boundaries."""

import sys
import types
import unittest
from pathlib import Path


SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import asr_transcription as asr  # noqa: E402


def _load_translation_module():
    """Import Phase D without requiring the isolated LLM venv in ASR tests."""
    torch = types.ModuleType("torch")
    transformers = types.ModuleType("transformers")
    transformers.AutoModelForCausalLM = object
    transformers.AutoTokenizer = object
    transformers.BitsAndBytesConfig = object
    sys.modules["torch"] = torch
    sys.modules["transformers"] = transformers

    import translation  # noqa: E402

    return translation


class ContractProjectionTests(unittest.TestCase):
    def setUp(self):
        self.input_contract = {
            "source_file": "source.mp4",
            "vad_threshold": 0.2,
            "diarization_stats": {"speakers": 2},
            "source_language": "ar",
            "target_language": "en",
            "segments": [
                {
                    "segment_id": 7,
                    "start_time": 3.0,
                    "end_time": 5.5,
                    "duration": 2.5,
                    "speaker_id": "SPEAKER_01",
                    "text": "hello",
                    "asr_confidence": 0.98,
                    "low_confidence_asr": False,
                    "_rms_dbfs": -20.0,
                    "_skipped_low_energy": False,
                    "_skipped_too_short": False,
                    "_hallucination_suspect": False,
                    "_prompt_echo_filtered": None,
                    "_original_segment_id": 6,
                    "_merged_from": [6, 7],
                }
            ],
            "language_config": {"name": "Arabic"},
            "gate_stats": {"words_raw": 1},
        }

    def test_transcript_contract_excludes_upstream_run_metadata(self):
        contract = asr.build_transcript_contract(
            self.input_contract,
            asr_model="whisperx/large-v3",
            align_model="whisperx-default",
        )

        self.assertNotIn("vad_threshold", contract)
        self.assertNotIn("diarization_stats", contract)
        self.assertNotIn("source_file", contract)
        self.assertNotIn("_original_segment_id", contract["segments"][0])
        self.assertNotIn("_merged_from", contract["segments"][0])
        self.assertEqual(contract["segments"][0]["text"], "hello")
        self.assertEqual(contract["total_segments"], 1)

    def test_translation_contract_excludes_asr_payload(self):
        translation = _load_translation_module()
        segment = self.input_contract["segments"][0]
        segment.update(
            {
                "translated_text": "Hello",
                "skip_translation": False,
                "transcription_failed": False,
                "low_confidence": False,
                "syllable_budget": {"target": 5, "low": 4, "high": 6},
            }
        )

        contract = translation.build_translation_contract(
            self.input_contract,
            source_language="ar",
            target_language="en",
            translation_model="local-qwen",
            name_glossary={"entries": 0},
        )

        output_segment = contract["segments"][0]
        self.assertNotIn("text", output_segment)
        self.assertNotIn("asr_confidence", output_segment)
        self.assertNotIn("_hallucination_suspect", output_segment)
        self.assertEqual(output_segment["translated_text"], "Hello")
        self.assertEqual(output_segment["speaker_id"], "SPEAKER_01")


if __name__ == "__main__":
    unittest.main()
