from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterator


@dataclass(frozen=True)
class ManifestRecord:
    benchmark: str
    item_id: str
    modality: str
    question: str
    answer: str
    options: dict[str, str] | None = None
    image_path: str | None = None
    split: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "ManifestRecord":
        for key in ["benchmark", "item_id", "modality", "question", "answer"]:
            if not row.get(key):
                raise ValueError(f"missing required field: {key}")

        modality = str(row["modality"])
        if modality not in {"text", "image"}:
            raise ValueError("modality must be 'text' or 'image'")
        if modality == "image" and not row.get("image_path"):
            raise ValueError("image records require image_path")

        options = row.get("options")
        if options is not None and not isinstance(options, dict):
            raise ValueError("options must be an object when provided")

        return cls(
            benchmark=str(row["benchmark"]),
            item_id=str(row["item_id"]),
            modality=modality,
            question=str(row["question"]),
            answer=str(row["answer"]),
            options={str(k): str(v) for k, v in options.items()} if options else None,
            image_path=str(row["image_path"]) if row.get("image_path") else None,
            split=str(row["split"]) if row.get("split") else None,
            meta=dict(row.get("meta", {})),
        )


def load_jsonl(path: str | Path) -> Iterator[ManifestRecord]:
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                yield ManifestRecord.from_dict(row)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}: line {line_number}: {exc}") from exc
