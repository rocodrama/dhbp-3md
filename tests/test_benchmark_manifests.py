from pathlib import Path
import tempfile
import unittest

from threemd.benchmark_manifests import (
    MMLU_MEDICAL_SUBJECTS,
    convert_medmcqa_row,
    convert_mmlu_row,
    convert_pmc_vqa_row,
    convert_pubmedqa_row,
    convert_vqa_rad_row,
    write_manifest,
)
from threemd.manifests import load_jsonl


class BenchmarkManifestConversionTests(unittest.TestCase):
    def test_medmcqa_rejects_unlabeled_test_rows(self) -> None:
        with self.assertRaisesRegex(ValueError, "unlabeled"):
            convert_medmcqa_row(
                {
                    "id": "hidden",
                    "question": "Which marker is correct?",
                    "opa": "A marker",
                    "opb": "B marker",
                    "opc": "C marker",
                    "opd": "D marker",
                    "cop": -1,
                },
                split="test",
            )

    def test_pubmedqa_nested_data_becomes_multiple_choice_record(self) -> None:
        record = convert_pubmedqa_row(
            {
                "id": "pm-1",
                "data": {
                    "Question": "Is the intervention useful?",
                    "Context": ["First abstract sentence.", "Second abstract sentence."],
                    "Options": {"A": "yes", "B": "no", "C": "maybe"},
                    "Correct Option": "C",
                    "Correct Answer": "maybe",
                },
            },
            split="test",
        )

        self.assertEqual(record.benchmark, "PubMedQA")
        self.assertEqual(record.modality, "text")
        self.assertIn("Context:", record.question)
        self.assertEqual(record.options, {"A": "yes", "B": "no", "C": "maybe"})
        self.assertEqual(record.answer, "C")
        self.assertEqual(record.meta["answer_text"], "maybe")

    def test_mmlu_record_keeps_medical_subject_in_metadata(self) -> None:
        record = convert_mmlu_row(
            {
                "question": "A lesion causes which finding?",
                "subject": "anatomy",
                "choices": ["A finding", "B finding", "C finding", "D finding"],
                "answer": 0,
            },
            split="test",
            row_index=4,
        )

        self.assertIn("anatomy", MMLU_MEDICAL_SUBJECTS)
        self.assertEqual(record.item_id, "mmlu:anatomy:test:4")
        self.assertEqual(record.answer, "A")
        self.assertEqual(record.meta["subject"], "anatomy")

    def test_vqa_rad_uses_materialized_image_path(self) -> None:
        record = convert_vqa_rad_row(
            {
                "qid": 10,
                "image_name": "synpic42202.jpg",
                "question": "Is there evidence of an aortic aneurysm?",
                "answer_normalized": "yes",
                "answer_type": "CLOSED",
            },
            split="test",
            image_path=Path("03_DATA/processed/images/vqa_rad/test/synpic42202.jpg"),
        )

        self.assertEqual(record.item_id, "vqa_rad:test:10")
        self.assertEqual(record.modality, "image")
        self.assertEqual(record.image_path, "03_DATA/processed/images/vqa_rad/test/synpic42202.jpg")
        self.assertEqual(record.answer, "yes")
        self.assertEqual(record.meta["answer_type"], "CLOSED")

    def test_pmc_vqa_strips_choice_prefixes(self) -> None:
        record = convert_pmc_vqa_row(
            {
                "index": "62",
                "Figure_path": "PMC8253867_Fig2_41.jpg",
                "Question": " What artery is displaced? ",
                "Choice A": " A: Right Coronary Artery ",
                "Choice B": " B: Left Anterior Descending Coronary Artery ",
                "Choice C": " C: Circumflex Coronary Artery ",
                "Choice D": " D: Superior Mesenteric Artery ",
                "Answer": "B",
                "split": "test",
            },
            split="test",
            image_path=Path("03_DATA/processed/images/pmc_vqa/test/PMC8253867_Fig2_41.jpg"),
        )

        self.assertEqual(record.answer, "B")
        self.assertEqual(record.options["B"], "Left Anterior Descending Coronary Artery")
        self.assertEqual(record.meta["figure_path"], "PMC8253867_Fig2_41.jpg")

    def test_write_manifest_round_trips_through_validator(self) -> None:
        record = convert_pubmedqa_row(
            {
                "id": "pm-2",
                "data": {
                    "Question": "Is the result supported?",
                    "Context": ["Context."],
                    "Options": {"A": "yes", "B": "no", "C": "maybe"},
                    "Correct Option": "A",
                    "Correct Answer": "yes",
                },
            },
            split="validation",
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pubmedqa.jsonl"
            write_manifest([record], path)
            loaded = list(load_jsonl(path))

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].item_id, "pm-2")


if __name__ == "__main__":
    unittest.main()
