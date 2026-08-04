from __future__ import annotations

from pathlib import Path
from typing import Any


class PromptBuilder:
    def __init__(self, prompt_root: str | Path, use_system_role: bool = True) -> None:
        self.prompt_root = Path(prompt_root)
        self.use_system_role = use_system_role

    def build(
        self,
        arm: str,
        question: str,
        answer_spec: str | None,
        *,
        probe: bool = False,
        image_url: str | None = None,
    ) -> list[dict[str, Any]]:
        arm = arm.upper()
        system_text = self._system_text(arm)
        user_text = self._user_text(question, answer_spec, probe)

        if image_url:
            user_content: str | list[dict[str, Any]] = [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]
        else:
            user_content = user_text

        if self.use_system_role:
            return [
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_content},
            ]

        if image_url:
            assert isinstance(user_content, list)
            user_content[0]["text"] = f"{system_text}\n\n{user_text}"
            return [{"role": "user", "content": user_content}]
        return [{"role": "user", "content": f"{system_text}\n\n{user_text}"}]

    def _system_text(self, arm: str) -> str:
        parts = [self._read("prompts", "system.md")]
        if arm == "BASE":
            return "\n\n".join(parts)

        persona_file = "placebo.md" if arm == "PLACEBO" else f"{arm}.md"
        persona_path = self.prompt_root / "personas" / persona_file
        if not persona_path.exists():
            raise ValueError(f"unknown arm: {arm}")
        parts.extend(
            [
                self._read("personas", "common.md"),
                persona_path.read_text(encoding="utf-8").strip(),
            ]
        )
        return "\n\n".join(parts)

    def _user_text(self, question: str, answer_spec: str | None, probe: bool) -> str:
        format_name = "probe_format.md" if probe else "output_format.md"
        output_format = self._read("prompts", format_name)
        if not probe:
            if answer_spec is None:
                raise ValueError("answer_spec is required unless probe=True")
            output_format = output_format.format(answer_spec=answer_spec)
        return "\n\n".join([self._read("prompts", "medical.md"), question.strip(), output_format])

    def _read(self, folder: str, name: str) -> str:
        return (self.prompt_root / folder / name).read_text(encoding="utf-8").strip()
