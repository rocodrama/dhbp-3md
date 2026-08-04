import json
from pathlib import Path
import tempfile
import unittest

from threemd.manifests import ManifestRecord, load_jsonl


class ManifestTests(unittest.TestCase):
    def test_text_record_accepts_minimal_required_fields(self) -> None:
        record = ManifestRecord.from_dict(
            {
                "benchmark": "MedQA",
                "item_id": "medqa-1",
                "modality": "text",
                "question": "Which option is best?",
                "answer": "A",
                "options": {"A": "Alpha", "B": "Beta"},
            }
        )

        self.assertEqual(record.benchmark, "MedQA")
        self.assertIsNone(record.image_path)

    def test_image_record_requires_image_path(self) -> None:
        with self.assertRaisesRegex(ValueError, "image_path"):
            ManifestRecord.from_dict(
                {
                    "benchmark": "VQA-RAD",
                    "item_id": "vqa-1",
                    "modality": "image",
                    "question": "What is shown?",
                    "answer": "mass",
                }
            )

    def test_load_jsonl_reports_line_number_for_bad_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.jsonl"
            manifest.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "benchmark": "PubMedQA",
                                "item_id": "pm-1",
                                "modality": "text",
                                "question": "Is the statement supported?",
                                "answer": "yes",
                            }
                        ),
                        json.dumps({"benchmark": "broken"}),
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "line 2"):
                list(load_jsonl(manifest))


if __name__ == "__main__":
    unittest.main()
