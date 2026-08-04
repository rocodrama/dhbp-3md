import unittest

from threemd.responses import has_answer_line, strip_gemma_thought_channel


class ResponseTests(unittest.TestCase):
    def test_strip_gemma_unused_thought_channel(self) -> None:
        raw = "<unused94>thought\nhidden reasoning<unused95>REASONING:\nvisible\nANSWER: B"

        self.assertEqual(
            strip_gemma_thought_channel(raw),
            "REASONING:\nvisible\nANSWER: B",
        )

    def test_strip_gemma_text_thought_channel(self) -> None:
        raw = "<|channel|>thought\nhidden reasoning<|channel|>REASONING:\nvisible\nANSWER: B"

        self.assertEqual(
            strip_gemma_thought_channel(raw),
            "REASONING:\nvisible\nANSWER: B",
        )

    def test_text_without_thought_channel_is_unchanged(self) -> None:
        raw = "REASONING:\nvisible\nANSWER: B"

        self.assertEqual(strip_gemma_thought_channel(raw), raw)

    def test_has_answer_line_matches_line_start(self) -> None:
        self.assertTrue(has_answer_line("REASONING:\ntext\nANSWER: B"))
        self.assertTrue(has_answer_line("ANSWER: yes"))
        self.assertFalse(has_answer_line("The ANSWER: B is likely."))


if __name__ == "__main__":
    unittest.main()
