from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REQUIRED_RECORD_SNIPPETS = {
    "cifar10": [
        "cifar10",
        "Source:",
        "cifar-10-python.tar.gz",
        "sha256",
        "Train: 50,000 images",
        "Test: 10,000 images",
        "Resolution: 32x32 RGB",
        "Labels: class labels are available",
        "Raw preprocessing: none",
        "Server dataset directory size",
        "/root/autodl-tmp/CoFiTok/datasets/cifar10",
        "/root/autodl-tmp/CoFiTok/checkpoints",
    ],
    "tiny_imagenet_200": [
        "tiny_imagenet_200",
        "Source:",
        "tiny-imagenet-200.zip",
        "sha256",
        "Train: 100,000 images",
        "Validation: 10,000 images",
        "Test: 10,000 images",
        "Resolution: 64x64 RGB",
        "Labels: class labels are available",
        "Raw preprocessing: none",
        "Server dataset directory size",
        "/root/autodl-tmp/CoFiTok/datasets/tiny_imagenet_200",
        "/root/autodl-tmp/CoFiTok/checkpoints",
    ],
}

HF_PLAN_SNIPPETS = [
    "dataset alias: imagenet_1k_64x64_hf",
    "Hugging Face repo: benjamin-paine/imagenet-1k-64x64",
    "repo URL: https://huggingface.co/datasets/benjamin-paine/imagenet-1k-64x64",
    "columns: image, label",
    "class labels: ImageNet-1K labels",
    "train_manifest rows: 1,281,167",
    "validation_manifest rows: 50,000",
    "dataset dir size: 14G",
    "Keep its source and preprocessing distinct from `downsampled_imagenet_64`",
]

DOWNSAMPLED_PLAN_SNIPPETS = [
    "alias: downsampled_imagenet_64",
    "preferred public loader: TensorFlow Datasets downsampled_imagenet/64x64",
    "result:",
    "failed before downloading data",
    "HTTP code: 404",
    "Academic Torrents fallback attempt",
    "imagenet_1k_64x64_hf",
    "This fallback is ImageNet-1K 64x64 repack/resized data, not the exact TFDS",
]

DOWNSAMPLED_STRICT_RECORD_SNIPPETS = [
    "# downsampled_imagenet_64 Dataset Record",
    "Completed and verified from the strict Academic Torrents source.",
    "https://academictorrents.com/details/96816a530ee002254d29bf7a61c0c158d3dedc3b",
    "96816a530ee002254d29bf7a61c0c158d3dedc3b",
    "aria2c magnet+DHT on pro6000",
    "/root/autodl-tmp/CoFiTok/datasets/downsampled_imagenet_64/raw/small/train_64x64.tar",
    "/root/autodl-tmp/CoFiTok/datasets/downsampled_imagenet_64/raw/small/valid_64x64.tar",
    "e2a2c1947a748d0d256e98e6d82800d855441dc9e4de2e71a773f1324c4524d7",
    "eeac03ec4f585baf871d2b4cb9c974bfe345e74878a74e5820fe6eb1041629cd",
    "1,281,149",
    "49,999",
    "No crop, resize, normalization",
]

EXPECTED_DOWNSAMPLED_HASH = "96816a530ee002254d29bf7a61c0c158d3dedc3b"
EXPECTED_DOWNSAMPLED_SHA256 = {
    "96816a530ee002254d29bf7a61c0c158d3dedc3b.torrent": "1104077da8c88db149d74ffb6d1f9fe17b32abf535880966ee1684559820cdfa",
    "train_64x64.tar": "e2a2c1947a748d0d256e98e6d82800d855441dc9e4de2e71a773f1324c4524d7",
    "valid_64x64.tar": "eeac03ec4f585baf871d2b4cb9c974bfe345e74878a74e5820fe6eb1041629cd",
}


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    evidence: dict[str, Any]
    missing: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate CoFiTok dataset condition records and manifests.")
    parser.add_argument("--conditions-dir", default="docs/experiment_conditions")
    parser.add_argument("--require-ok", action="store_true")
    return parser.parse_args()


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _snippets_missing(text: str, snippets: list[str]) -> list[str]:
    return [snippet for snippet in snippets if snippet not in text]


def _check_snippet_group(name: str, text: str, snippets: list[str]) -> CheckResult:
    missing = _snippets_missing(text, snippets)
    return CheckResult(
        name=name,
        status="ok" if not missing else "missing",
        evidence={"required_snippet_count": len(snippets), "present_count": len(snippets) - len(missing)},
        missing=missing,
    )


def check_cifar_tiny_records(conditions_dir: Path) -> CheckResult:
    path = conditions_dir / "datasets_2026-07-07.md"
    text = _read_text(path)
    missing = []
    evidence: dict[str, Any] = {"path": path.as_posix(), "datasets": {}}
    for alias, snippets in REQUIRED_RECORD_SNIPPETS.items():
        alias_missing = _snippets_missing(text, snippets)
        evidence["datasets"][alias] = {
            "required_snippet_count": len(snippets),
            "present_count": len(snippets) - len(alias_missing),
        }
        missing.extend(f"{alias}: {snippet}" for snippet in alias_missing)
    return CheckResult(
        name="cifar_tiny_records",
        status="ok" if not missing else "missing",
        evidence=evidence,
        missing=missing,
    )


def check_imagenet_hf_records(conditions_dir: Path) -> CheckResult:
    plan_path = conditions_dir / "imagenet_1k_64x64_hf_plan_2026-07-08.md"
    manifest_path = conditions_dir / "imagenet_1k_64x64_hf_manifest_summary_2026-07-08.json"
    plan = _read_text(plan_path)
    manifest = _read_json(manifest_path)
    missing = _snippets_missing(plan, HF_PLAN_SNIPPETS)

    split_rows = {
        str(split.get("split")): int(split.get("examples", 0))
        for split in manifest.get("splits", [])
        if isinstance(split, dict)
    }
    if manifest.get("dataset_alias") != "imagenet_1k_64x64_hf":
        missing.append("manifest dataset_alias == imagenet_1k_64x64_hf")
    if manifest.get("source_loader") != "benjamin-paine/imagenet-1k-64x64":
        missing.append("manifest source_loader == benjamin-paine/imagenet-1k-64x64")
    if split_rows.get("train") != 1_281_167:
        missing.append("manifest train examples == 1,281,167")
    if split_rows.get("validation") != 50_000:
        missing.append("manifest validation examples == 50,000")
    return CheckResult(
        name="imagenet_hf_records",
        status="ok" if not missing else "missing",
        evidence={
            "plan": plan_path.as_posix(),
            "manifest": manifest_path.as_posix(),
            "manifest_dataset_alias": manifest.get("dataset_alias"),
            "manifest_source_loader": manifest.get("source_loader"),
            "split_rows": split_rows,
        },
        missing=missing,
    )


def check_downsampled_fallback_record(conditions_dir: Path) -> CheckResult:
    plan_path = conditions_dir / "downsampled_imagenet_64_plan_2026-07-08.md"
    inspect_path = conditions_dir / "downsampled_imagenet_64_tfds_inspect_2026-07-08.json"
    plan = _read_text(plan_path)
    inspect = _read_json(inspect_path)
    missing = _snippets_missing(plan, DOWNSAMPLED_PLAN_SNIPPETS)
    if inspect.get("dataset_alias") != "downsampled_imagenet_64":
        missing.append("inspect dataset_alias == downsampled_imagenet_64")
    if inspect.get("source_loader") != "downsampled_imagenet":
        missing.append("inspect source_loader == downsampled_imagenet")
    if inspect.get("tfds_version") != "4.9.10":
        missing.append("inspect tfds_version == 4.9.10")
    return CheckResult(
        name="downsampled_fallback_record",
        status="ok" if not missing else "missing",
        evidence={
            "plan": plan_path.as_posix(),
            "inspect": inspect_path.as_posix(),
            "inspect_dataset_alias": inspect.get("dataset_alias"),
            "inspect_source_loader": inspect.get("source_loader"),
            "tfds_version": inspect.get("tfds_version"),
            "fallback": "imagenet_1k_64x64_hf",
        },
        missing=missing,
    )


def check_downsampled_strict_record(conditions_dir: Path) -> CheckResult:
    record_path = conditions_dir / "downsampled_imagenet_64_2026-07-09.md"
    manifest_path = conditions_dir / "downsampled_imagenet_64_manifest_summary_2026-07-09.json"
    record = _read_text(record_path)
    manifest = _read_json(manifest_path)
    missing = _snippets_missing(record, DOWNSAMPLED_STRICT_RECORD_SNIPPETS)

    raw_files = {
        str(item.get("name")): item
        for item in manifest.get("raw_files", [])
        if isinstance(item, dict)
    }
    splits = manifest.get("splits", {})
    sample_verification = manifest.get("sample_png_verification", {})

    if manifest.get("alias") != "downsampled_imagenet_64":
        missing.append("manifest alias == downsampled_imagenet_64")
    if manifest.get("status") != "completed":
        missing.append("manifest status == completed")
    if manifest.get("source", {}).get("info_hash") != EXPECTED_DOWNSAMPLED_HASH:
        missing.append("manifest source.info_hash matches Academic Torrents hash")
    if splits.get("train", {}).get("png_files") != 1_281_149:
        missing.append("manifest train png_files == 1,281,149")
    if splits.get("valid", {}).get("png_files") != 49_999:
        missing.append("manifest valid png_files == 49,999")

    for name, expected_sha in EXPECTED_DOWNSAMPLED_SHA256.items():
        raw = raw_files.get(name)
        if raw is None:
            missing.append(f"manifest raw_files contains {name}")
            continue
        if raw.get("sha256") != expected_sha:
            missing.append(f"manifest sha256 for {name}")
        if not isinstance(raw.get("bytes"), int) or int(raw["bytes"]) <= 0:
            missing.append(f"manifest positive byte size for {name}")

    for split in ["train", "valid"]:
        samples = sample_verification.get(split, [])
        if not samples:
            missing.append(f"sample verification includes {split}")
            continue
        bad_sizes = [sample for sample in samples if sample.get("size") != [64, 64]]
        if bad_sizes:
            missing.append(f"sample verification {split} images are 64x64")

    return CheckResult(
        name="downsampled_strict_record",
        status="ok" if not missing else "missing",
        evidence={
            "record": record_path.as_posix(),
            "manifest": manifest_path.as_posix(),
            "manifest_alias": manifest.get("alias"),
            "source_info_hash": manifest.get("source", {}).get("info_hash"),
            "split_png_files": {
                "train": splits.get("train", {}).get("png_files"),
                "valid": splits.get("valid", {}).get("png_files"),
            },
            "raw_file_count": len(raw_files),
        },
        missing=missing,
    )


def validate_dataset_conditions(conditions_dir: Path) -> dict[str, Any]:
    conditions_dir = conditions_dir.resolve()
    checks = [
        check_cifar_tiny_records(conditions_dir),
        check_imagenet_hf_records(conditions_dir),
        check_downsampled_fallback_record(conditions_dir),
        check_downsampled_strict_record(conditions_dir),
    ]
    missing = [check for check in checks if check.status != "ok"]
    return {
        "conditions_dir": conditions_dir.as_posix(),
        "status": "ok" if not missing else "missing",
        "check_count": len(checks),
        "ok_count": len(checks) - len(missing),
        "missing_count": len(missing),
        "checks": [asdict(check) for check in checks],
    }


def main() -> None:
    args = parse_args()
    result = validate_dataset_conditions(Path(args.conditions_dir))
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.require_ok and result["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
