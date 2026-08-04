from pathlib import Path
import unittest


class ScriptDefaultsTests(unittest.TestCase):
    def test_smoke_vllm_defaults_to_active_prompt_root(self) -> None:
        script = Path("02_CODE/scripts/smoke_vllm.py").read_text(encoding="utf-8")

        self.assertIn('--prompt-root", default="02_CODE/configs/prompts"', script)

    def test_smoke_vllm_uses_closed_option_question(self) -> None:
        script = Path("02_CODE/scripts/smoke_vllm.py").read_text(encoding="utf-8")

        self.assertIn("Options:", script)
        self.assertIn('answer_spec="A, B, C, or D"', script)

    def test_smoke_vllm_fails_when_answer_line_is_missing(self) -> None:
        script = Path("02_CODE/scripts/smoke_vllm.py").read_text(encoding="utf-8")

        self.assertIn("raise SystemExit(2)", script)
        self.assertIn("if not has_answer_line(content):", script)
        self.assertIn("strip_gemma_thought_channel(raw_content)", script)

    def test_build_manifests_defaults_to_local_layout(self) -> None:
        script = Path("02_CODE/scripts/build_manifests.py").read_text(encoding="utf-8")

        self.assertIn('--raw-dir", default="03_DATA/raw"', script)
        self.assertIn('--out-dir", default="03_DATA/metadata"', script)
        self.assertIn('--processed-dir", default="03_DATA/processed"', script)
        self.assertIn('"medmcqa": ("validation",)', script)


if __name__ == "__main__":
    unittest.main()
