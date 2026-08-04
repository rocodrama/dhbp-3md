from __future__ import annotations

from dataclasses import asdict
import json
import re
from pathlib import Path
from typing import Any, Iterable

from .manifests import ManifestRecord


MMLU_MEDICAL_SUBJECTS = (
    "anatomy",
    "clinical_knowledge",
    "college_medicine",
    "medical_genetics",
    "professional_medicine",
)


def convert_medqa_row(row: dict[str, Any], split: str, row_index: int = 0) -> ManifestRecord:
    options = {
        "A": _required_text(row, "ending0"),
        "B": _required_text(row, "ending1"),
        "C": _required_text(row, "ending2"),
        "D": _required_text(row, "ending3"),
    }
    sent2 = _clean(row.get("sent2"))
    question = _required_text(row, "sent1")
    if sent2:
        question = f"{question}\n\n{sent2}"

    return ManifestRecord.from_dict(
        {
            "benchmark": "MedQA",
            "item_id": _item_id(row, "medqa", split, row_index),
            "modality": "text",
            "question": question,
            "options": options,
            "answer": _label(row.get("label"), len(options)),
            "split": split,
        }
    )


def convert_medmcqa_row(row: dict[str, Any], split: str, row_index: int = 0) -> ManifestRecord:
    options = {
        "A": _required_text(row, "opa"),
        "B": _required_text(row, "opb"),
        "C": _required_text(row, "opc"),
        "D": _required_text(row, "opd"),
    }
    return ManifestRecord.from_dict(
        {
            "benchmark": "MedMCQA",
            "item_id": _item_id(row, "medmcqa", split, row_index),
            "modality": "text",
            "question": _required_text(row, "question"),
            "options": options,
            "answer": _label(row.get("cop"), len(options)),
            "split": split,
            "meta": {
                "choice_type": row.get("choice_type"),
                "subject_name": row.get("subject_name"),
                "topic_name": row.get("topic_name"),
            },
        }
    )


def convert_pubmedqa_row(row: dict[str, Any], split: str, row_index: int = 0) -> ManifestRecord:
    data = row.get("data") or row
    options = {str(k).upper(): _clean(v) for k, v in (data.get("Options") or {}).items()}
    if not options:
        options = {"A": "yes", "B": "no", "C": "maybe"}

    correct_option = data.get("Correct Option") or data.get("correct_option")
    answer_text = _clean(data.get("Correct Answer") or data.get("final_decision") or data.get("label"))
    answer = _label(correct_option, len(options)) if correct_option else _match_option(answer_text, options)
    contexts = data.get("Context") or data.get("CONTEXTS") or data.get("contexts") or []
    if isinstance(contexts, str):
        contexts = [contexts]
    context_text = "\n".join(_clean(value) for value in contexts if _clean(value))
    question = _required_text(data, "Question")
    if context_text:
        question = f"{question}\n\nContext:\n{context_text}"

    return ManifestRecord.from_dict(
        {
            "benchmark": "PubMedQA",
            "item_id": _item_id(row, "pubmedqa", split, row_index),
            "modality": "text",
            "question": question,
            "options": options,
            "answer": answer,
            "split": split,
            "meta": {"answer_text": answer_text},
        }
    )


def convert_mmlu_row(row: dict[str, Any], split: str, row_index: int = 0) -> ManifestRecord:
    subject = _required_text(row, "subject")
    choices = list(row.get("choices") or [])
    options = {chr(ord("A") + index): _clean(choice) for index, choice in enumerate(choices)}
    return ManifestRecord.from_dict(
        {
            "benchmark": "MMLU-medical",
            "item_id": f"mmlu:{subject}:{split}:{row_index}",
            "modality": "text",
            "question": _required_text(row, "question"),
            "options": options,
            "answer": _label(row.get("answer"), len(options)),
            "split": split,
            "meta": {"subject": subject},
        }
    )


def convert_vqa_rad_row(
    row: dict[str, Any],
    split: str,
    image_path: str | Path,
    row_index: int = 0,
) -> ManifestRecord:
    return ManifestRecord.from_dict(
        {
            "benchmark": "VQA-RAD",
            "item_id": f"vqa_rad:{split}:{_clean(row.get('qid')) or row_index}",
            "modality": "image",
            "image_path": _path_text(image_path),
            "question": _required_text(row, "question"),
            "answer": _clean(row.get("answer_normalized") or row.get("answer")),
            "split": split,
            "meta": {
                "image_name": row.get("image_name"),
                "image_organ": row.get("image_organ"),
                "answer_type": row.get("answer_type"),
                "question_type": row.get("question_type_primary"),
            },
        }
    )


def convert_slake_row(
    row: dict[str, Any],
    split: str,
    image_path: str | Path,
    row_index: int = 0,
) -> ManifestRecord:
    return ManifestRecord.from_dict(
        {
            "benchmark": "SLAKE",
            "item_id": f"slake:{split}:{_clean(row.get('qid')) or row_index}",
            "modality": "image",
            "image_path": _path_text(image_path),
            "question": _required_text(row, "question"),
            "answer": _required_text(row, "answer"),
            "split": split,
            "meta": {
                "img_id": row.get("img_id"),
                "img_name": row.get("img_name"),
                "q_lang": row.get("q_lang"),
                "answer_type": row.get("answer_type"),
                "modality": row.get("modality"),
                "location": row.get("location"),
            },
        }
    )


def convert_pathvqa_row(
    row: dict[str, Any],
    split: str,
    image_path: str | Path,
    row_index: int = 0,
) -> ManifestRecord:
    image = row.get("image") if isinstance(row.get("image"), dict) else {}
    return ManifestRecord.from_dict(
        {
            "benchmark": "PathVQA",
            "item_id": f"pathvqa:{split}:{row_index}",
            "modality": "image",
            "image_path": _path_text(image_path),
            "question": _required_text(row, "question"),
            "answer": _required_text(row, "answer"),
            "split": split,
            "meta": {"image_source_path": image.get("path")},
        }
    )


def convert_pmc_vqa_row(
    row: dict[str, Any],
    split: str,
    image_path: str | Path,
    row_index: int = 0,
) -> ManifestRecord:
    options = {
        "A": _strip_choice_prefix(row.get("Choice A"), "A"),
        "B": _strip_choice_prefix(row.get("Choice B"), "B"),
        "C": _strip_choice_prefix(row.get("Choice C"), "C"),
        "D": _strip_choice_prefix(row.get("Choice D"), "D"),
    }
    return ManifestRecord.from_dict(
        {
            "benchmark": "PMC-VQA",
            "item_id": f"pmc_vqa:{split}:{_clean(row.get('index')) or row_index}",
            "modality": "image",
            "image_path": _path_text(image_path),
            "question": _required_text(row, "Question"),
            "options": options,
            "answer": _label(row.get("Answer"), len(options)),
            "split": split,
            "meta": {
                "figure_path": row.get("Figure_path"),
                "caption": row.get("Caption"),
            },
        }
    )


def write_manifest(records: Iterable[ManifestRecord], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(_record_dict(record), ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def _record_dict(record: ManifestRecord) -> dict[str, Any]:
    row = asdict(record)
    return {key: value for key, value in row.items() if value is not None}


def _clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _path_text(path: str | Path) -> str:
    return Path(path).as_posix()


def _required_text(row: dict[str, Any], key: str) -> str:
    value = _clean(row.get(key))
    if not value:
        raise ValueError(f"missing required field: {key}")
    return value


def _item_id(row: dict[str, Any], prefix: str, split: str, row_index: int) -> str:
    return _clean(row.get("id")) or f"{prefix}:{split}:{row_index}"


def _label(value: Any, option_count: int) -> str:
    if isinstance(value, str):
        text = value.strip().upper()
        if not text or text in {"-1", "NONE", "NULL"}:
            raise ValueError("unlabeled answer")
        if len(text) == 1 and "A" <= text <= chr(ord("A") + option_count - 1):
            return text
        if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
            value = int(text)
        else:
            raise ValueError(f"invalid answer label: {value!r}")

    index = int(value)
    if index < 0:
        raise ValueError("unlabeled answer")
    if index >= option_count:
        raise ValueError(f"answer index {index} outside {option_count} options")
    return chr(ord("A") + index)


def _match_option(answer_text: str, options: dict[str, str]) -> str:
    normalized = answer_text.strip().lower()
    for label, option in options.items():
        if option.strip().lower() == normalized:
            return label
    raise ValueError("answer text does not match any option")


def _strip_choice_prefix(value: Any, label: str) -> str:
    text = _clean(value)
    text = re.sub(rf"^\s*{re.escape(label)}\s*[:.)-]\s*", "", text, flags=re.IGNORECASE)
    if not text:
        raise ValueError(f"missing required field: Choice {label}")
    return text
