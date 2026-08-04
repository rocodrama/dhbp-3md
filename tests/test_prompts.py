from pathlib import Path
import re
import tempfile
import unittest

from threemd.prompts import PromptBuilder


def write_prompt_fixture(root: Path) -> None:
    prompts = root / "prompts"
    personas = root / "personas"
    prompts.mkdir()
    personas.mkdir()
    (prompts / "system.md").write_text("System text.", encoding="utf-8")
    (prompts / "medical.md").write_text("Medical framing.", encoding="utf-8")
    (prompts / "output_format.md").write_text("ANSWER: {answer_spec}", encoding="utf-8")
    (prompts / "probe_format.md").write_text("DIFFERENTIAL:", encoding="utf-8")
    (personas / "common.md").write_text("Common persona.", encoding="utf-8")
    (personas / "placebo.md").write_text("Neutral profile.", encoding="utf-8")
    (personas / "INTJ.md").write_text("Structured profile.", encoding="utf-8")


class PromptBuilderTests(unittest.TestCase):
    def test_base_arm_omits_persona_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_prompt_fixture(root)

            messages = PromptBuilder(root).build(
                arm="BASE",
                question="What is the diagnosis?",
                answer_spec="A, B, C, or D",
            )

        self.assertEqual(
            messages,
            [
                {"role": "system", "content": "System text."},
                {
                    "role": "user",
                    "content": "Medical framing.\n\nWhat is the diagnosis?\n\nANSWER: A, B, C, or D",
                },
            ],
        )

    def test_persona_arm_includes_common_and_selected_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_prompt_fixture(root)

            messages = PromptBuilder(root).build(
                arm="INTJ",
                question="What is the diagnosis?",
                answer_spec="A-D",
            )

        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(
            messages[0]["content"],
            "System text.\n\nCommon persona.\n\nStructured profile.",
        )

    def test_system_preamble_can_be_folded_into_user_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_prompt_fixture(root)

            messages = PromptBuilder(root, use_system_role=False).build(
                arm="PLACEBO",
                question="Open vignette",
                answer_spec=None,
                probe=True,
            )

        self.assertEqual(
            messages,
            [
                {
                    "role": "user",
                    "content": (
                        "System text.\n\nCommon persona.\n\nNeutral profile.\n\n"
                        "Medical framing.\n\nOpen vignette\n\nDIFFERENTIAL:"
                    ),
                }
            ],
        )

    def test_unknown_arm_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_prompt_fixture(root)

            with self.assertRaisesRegex(ValueError, "unknown arm"):
                PromptBuilder(root).build("NOPE", "Question", "A-D")

    def test_active_prompt_root_builds_all_experimental_arms(self) -> None:
        root = Path("02_CODE/configs/prompts")
        arms = [
            "BASE",
            "PLACEBO",
            "INTJ",
            "INTP",
            "ENTJ",
            "ENTP",
            "INFJ",
            "INFP",
            "ENFJ",
            "ENFP",
            "ISTJ",
            "ISTP",
            "ESTJ",
            "ESTP",
            "ISFJ",
            "ISFP",
            "ESFJ",
            "ESFP",
        ]

        for arm in arms:
            with self.subTest(arm=arm):
                messages = PromptBuilder(root).build(
                    arm=arm,
                    question="Which option is best?",
                    answer_spec="A, B, C, or D",
                )
                combined = "\n\n".join(
                    message["content"]
                    if isinstance(message["content"], str)
                    else message["content"][0]["text"]
                    for message in messages
                )
                self.assertIn("ANSWER:", combined)
                self.assertNotIn("TODO", combined)

    def test_active_personas_are_length_balanced_around_placebo(self) -> None:
        persona_root = Path("02_CODE/configs/prompts/personas")
        files = [
            "placebo.md",
            "INTJ.md",
            "INTP.md",
            "ENTJ.md",
            "ENTP.md",
            "INFJ.md",
            "INFP.md",
            "ENFJ.md",
            "ENFP.md",
            "ISTJ.md",
            "ISTP.md",
            "ESTJ.md",
            "ESTP.md",
            "ISFJ.md",
            "ISFP.md",
            "ESFJ.md",
            "ESFP.md",
        ]

        counts = {}
        for name in files:
            text = (persona_root / name).read_text(encoding="utf-8")
            counts[name] = len(re.findall(r"[A-Za-z0-9']+", text))

        placebo_count = counts["placebo.md"]
        for name, count in counts.items():
            with self.subTest(profile=name):
                self.assertLessEqual(abs(count - placebo_count) / placebo_count, 0.15)

    def test_active_personas_do_not_name_their_labels(self) -> None:
        persona_root = Path("02_CODE/configs/prompts/personas")
        for path in persona_root.glob("*.md"):
            if path.name == "common.md":
                continue
            text = path.read_text(encoding="utf-8")
            with self.subTest(profile=path.name):
                self.assertNotRegex(text, r"\b(MBTI|persona|profile|INTJ|INTP|ENTJ|ENTP|INFJ|INFP|ENFJ|ENFP|ISTJ|ISTP|ESTJ|ESTP|ISFJ|ISFP|ESFJ|ESFP)\b")

    def test_active_medical_prompt_does_not_invite_visible_checklist(self) -> None:
        text = Path("02_CODE/configs/prompts/prompts/medical.md").read_text(encoding="utf-8")

        self.assertNotIn("- Identify exactly", text)
        self.assertIn("Use these checks silently", text)


if __name__ == "__main__":
    unittest.main()
