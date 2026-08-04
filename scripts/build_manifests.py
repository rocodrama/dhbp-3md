from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import zipfile
from typing import Any, Callable, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from threemd.benchmark_manifests import (
    MMLU_MEDICAL_SUBJECTS,
    convert_medmcqa_row,
    convert_medqa_row,
    convert_mmlu_row,
    convert_pathvqa_row,
    convert_pmc_vqa_row,
    convert_pubmedqa_row,
    convert_slake_row,
    convert_vqa_rad_row,
    write_manifest,
)
from threemd.manifests import ManifestRecord


DEFAULT_EVAL_SPLITS = {
    "medqa": ("test",),
    "medmcqa": ("validation",),
    "pubmedqa": ("test",),
    "mmlu_medical": ("test",),
    "vqa_rad": ("test",),
    "slake": ("test",),
    "pathvqa": ("test",),
    "pmc_vqa": ("test",),
}
BENCHMARKS = tuple(DEFAULT_EVAL_SPLITS)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build 3MD canonical JSONL manifests from local raw benchmarks.")
    parser.add_argument("--raw-dir", default="03_DATA/raw")
    parser.add_argument("--out-dir", default="03_DATA/metadata")
    parser.add_argument("--processed-dir", default="03_DATA/processed")
    parser.add_argument("--benchmarks", default="all", help="Comma-separated benchmark ids or 'all'.")
    parser.add_argument("--splits", default="", help="Optional comma-separated split override.")
    parser.add_argument("--limit", type=int, default=0, help="Stop after N records per benchmark; useful for smoke checks.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--include-non-english-slake", action="store_true")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    processed_dir = Path(args.processed_dir)
    selected = _selected_benchmarks(args.benchmarks)
    split_override = _csv_values(args.splits)
    limit = args.limit or None

    split_rows: list[dict[str, str]] = []
    for benchmark in selected:
        splits = tuple(split_override or DEFAULT_EVAL_SPLITS[benchmark])
        records, skipped = BUILDERS[benchmark](
            raw_dir,
            processed_dir,
            splits,
            limit,
            include_non_english_slake=args.include_non_english_slake,
        )
        path = out_dir / f"{benchmark}.jsonl"
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
        count = write_manifest(records, path)
        split_rows.extend(
            {
                "benchmark": record.benchmark,
                "item_id": record.item_id,
                "split": record.split or "",
                "manifest": path.name,
            }
            for record in records
        )
        suffix = f", skipped {skipped}" if skipped else ""
        print(f"{benchmark}: wrote {count} records to {path}{suffix}")

    splits_path = out_dir / "splits.csv"
    if splits_path.exists() and not args.overwrite:
        raise FileExistsError(f"{splits_path} exists; pass --overwrite to replace it")
    _write_splits(split_rows, splits_path)
    print(f"splits: wrote {len(split_rows)} rows to {splits_path}")
    return 0


def build_medqa(
    raw_dir: Path,
    processed_dir: Path,
    splits: tuple[str, ...],
    limit: int | None,
    **_: Any,
) -> tuple[list[ManifestRecord], int]:
    return _collect_parquet(
        raw_dir / "medqa_usmle_4options" / "data",
        splits,
        limit,
        lambda row, split, index: convert_medqa_row(row, split, index),
    )


def build_medmcqa(
    raw_dir: Path,
    processed_dir: Path,
    splits: tuple[str, ...],
    limit: int | None,
    **_: Any,
) -> tuple[list[ManifestRecord], int]:
    return _collect_parquet(
        raw_dir / "medmcqa" / "data",
        splits,
        limit,
        lambda row, split, index: convert_medmcqa_row(row, split, index),
    )


def build_pubmedqa(
    raw_dir: Path,
    processed_dir: Path,
    splits: tuple[str, ...],
    limit: int | None,
    **_: Any,
) -> tuple[list[ManifestRecord], int]:
    return _collect_parquet(
        raw_dir / "pubmedqa" / "data",
        splits,
        limit,
        lambda row, split, index: convert_pubmedqa_row(row, split, index),
    )


def build_mmlu_medical(
    raw_dir: Path,
    processed_dir: Path,
    splits: tuple[str, ...],
    limit: int | None,
    **_: Any,
) -> tuple[list[ManifestRecord], int]:
    records: list[ManifestRecord] = []
    skipped = 0
    for split in splits:
        for subject in MMLU_MEDICAL_SUBJECTS:
            files = _parquet_files(raw_dir / "mmlu" / subject, split)
            for row_index, row in _iter_parquet(files):
                if limit and len(records) >= limit:
                    return records, skipped
                records.append(convert_mmlu_row(row, split, row_index))
    return records, skipped


def build_vqa_rad(
    raw_dir: Path,
    processed_dir: Path,
    splits: tuple[str, ...],
    limit: int | None,
    **_: Any,
) -> tuple[list[ManifestRecord], int]:
    records: list[ManifestRecord] = []
    for split in splits:
        files = _parquet_files(raw_dir / "vqa_rad" / "data", split)
        for row_index, row in _iter_parquet(files):
            if limit and len(records) >= limit:
                return records, 0
            image_name = _clean(row.get("image_name")) or _image_source_name(row, row_index)
            image_path = processed_dir / "images" / "vqa_rad" / split / _safe_relative(image_name)
            _write_embedded_image(row, image_path)
            records.append(convert_vqa_rad_row(row, split, image_path, row_index))
    return records, 0


def build_slake(
    raw_dir: Path,
    processed_dir: Path,
    splits: tuple[str, ...],
    limit: int | None,
    include_non_english_slake: bool = False,
    **_: Any,
) -> tuple[list[ManifestRecord], int]:
    records: list[ManifestRecord] = []
    skipped = 0
    zip_path = raw_dir / "slake" / "imgs.zip"
    with zipfile.ZipFile(zip_path) as image_zip:
        lookup = _zip_lookup(image_zip)
        for split in splits:
            rows = json.loads((raw_dir / "slake" / f"{split}.json").read_text(encoding="utf-8"))
            for row_index, row in enumerate(rows):
                if not include_non_english_slake and _clean(row.get("q_lang")).lower() != "en":
                    skipped += 1
                    continue
                if limit and len(records) >= limit:
                    return records, skipped
                source_name = _required_value(row, "img_name")
                image_path = processed_dir / "images" / "slake" / split / _safe_relative(source_name)
                _extract_zip_image(image_zip, lookup, source_name, image_path)
                records.append(convert_slake_row(row, split, image_path, row_index))
    return records, skipped


def build_pathvqa(
    raw_dir: Path,
    processed_dir: Path,
    splits: tuple[str, ...],
    limit: int | None,
    **_: Any,
) -> tuple[list[ManifestRecord], int]:
    records: list[ManifestRecord] = []
    for split in splits:
        files = _parquet_files(raw_dir / "pathvqa" / "data", split)
        for row_index, row in _iter_parquet(files):
            if limit and len(records) >= limit:
                return records, 0
            image_path = processed_dir / "images" / "pathvqa" / split / f"{row_index:06d}{_image_suffix(row)}"
            _write_embedded_image(row, image_path)
            records.append(convert_pathvqa_row(row, split, image_path, row_index))
    return records, 0


def build_pmc_vqa(
    raw_dir: Path,
    processed_dir: Path,
    splits: tuple[str, ...],
    limit: int | None,
    **_: Any,
) -> tuple[list[ManifestRecord], int]:
    records: list[ManifestRecord] = []
    zip_path = raw_dir / "pmc_vqa" / "images_2.zip"
    with zipfile.ZipFile(zip_path) as image_zip:
        lookup = _zip_lookup(image_zip)
        for split in splits:
            csv_path = raw_dir / "pmc_vqa" / f"{split}_2.csv"
            with csv_path.open(encoding="utf-8-sig", newline="") as handle:
                for row_index, row in enumerate(csv.DictReader(handle)):
                    if limit and len(records) >= limit:
                        return records, 0
                    source_name = _required_value(row, "Figure_path")
                    image_path = processed_dir / "images" / "pmc_vqa" / split / _safe_relative(source_name)
                    _extract_zip_image(image_zip, lookup, source_name, image_path)
                    records.append(convert_pmc_vqa_row(row, split, image_path, row_index))
    return records, 0


BUILDERS: dict[str, Callable[..., tuple[list[ManifestRecord], int]]] = {
    "medqa": build_medqa,
    "medmcqa": build_medmcqa,
    "pubmedqa": build_pubmedqa,
    "mmlu_medical": build_mmlu_medical,
    "vqa_rad": build_vqa_rad,
    "slake": build_slake,
    "pathvqa": build_pathvqa,
    "pmc_vqa": build_pmc_vqa,
}


def _collect_parquet(
    data_dir: Path,
    splits: tuple[str, ...],
    limit: int | None,
    convert: Callable[[dict[str, Any], str, int], ManifestRecord],
) -> tuple[list[ManifestRecord], int]:
    records: list[ManifestRecord] = []
    skipped = 0
    for split in splits:
        for row_index, row in _iter_parquet(_parquet_files(data_dir, split)):
            if limit and len(records) >= limit:
                return records, skipped
            try:
                records.append(convert(row, split, row_index))
            except ValueError as exc:
                if "unlabeled" not in str(exc):
                    raise
                skipped += 1
    return records, skipped


def _iter_parquet(files: Iterable[Path]) -> Iterable[tuple[int, dict[str, Any]]]:
    import pyarrow.parquet as pq

    row_index = 0
    for path in files:
        table = pq.read_table(path)
        for row in table.to_pylist():
            yield row_index, row
            row_index += 1


def _parquet_files(data_dir: Path, split: str) -> list[Path]:
    files = sorted(data_dir.glob(f"{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files for split {split!r} under {data_dir}")
    return files


def _write_embedded_image(row: dict[str, Any], path: Path) -> None:
    data = _image_bytes(row)
    if not data:
        raise ValueError(f"missing embedded image bytes for {path}")
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _image_bytes(row: dict[str, Any]) -> bytes | None:
    image = row.get("image")
    if not isinstance(image, dict):
        return None
    data = image.get("bytes")
    return bytes(data) if isinstance(data, (bytes, bytearray, memoryview)) else None


def _image_source_name(row: dict[str, Any], row_index: int) -> str:
    image = row.get("image") if isinstance(row.get("image"), dict) else {}
    return _clean(image.get("path")) or f"{row_index:06d}{_image_suffix(row)}"


def _image_suffix(row: dict[str, Any]) -> str:
    image = row.get("image") if isinstance(row.get("image"), dict) else {}
    suffix = Path(_clean(image.get("path"))).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        return suffix
    data = _image_bytes(row) or b""
    if data.startswith(b"\x89PNG"):
        return ".png"
    return ".jpg"


def _zip_lookup(image_zip: zipfile.ZipFile) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for member in image_zip.namelist():
        if member.endswith("/"):
            continue
        normalized = member.replace("\\", "/")
        parts = normalized.split("/")
        keys = {
            normalized.lower(),
            parts[-1].lower(),
            "/".join(parts[-2:]).lower() if len(parts) >= 2 else parts[-1].lower(),
        }
        for key in keys:
            lookup.setdefault(key, member)
    return lookup


def _extract_zip_image(image_zip: zipfile.ZipFile, lookup: dict[str, str], source_name: str, path: Path) -> None:
    if path.exists():
        return
    member = lookup.get(source_name.replace("\\", "/").lower())
    if not member:
        member = lookup.get(Path(source_name).name.lower())
    if not member:
        raise FileNotFoundError(f"{source_name!r} not found in {image_zip.filename}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with image_zip.open(member) as source, path.open("wb") as target:
        shutil.copyfileobj(source, target)


def _safe_relative(value: str) -> Path:
    parts = [
        part
        for part in PurePosixPath(value.replace("\\", "/")).parts
        if part not in {"", ".", ".."} and not part.endswith(":")
    ]
    if not parts:
        raise ValueError(f"invalid relative path: {value!r}")
    return Path(*parts)


def _required_value(row: dict[str, Any], key: str) -> str:
    value = _clean(row.get(key))
    if not value:
        raise ValueError(f"missing required field: {key}")
    return value


def _clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _selected_benchmarks(value: str) -> tuple[str, ...]:
    if value.strip().lower() == "all":
        return BENCHMARKS
    selected = tuple(_csv_values(value))
    unknown = sorted(set(selected) - set(BENCHMARKS))
    if unknown:
        raise ValueError(f"unknown benchmark(s): {', '.join(unknown)}")
    return selected


def _csv_values(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _write_splits(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["benchmark", "item_id", "split", "manifest"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
