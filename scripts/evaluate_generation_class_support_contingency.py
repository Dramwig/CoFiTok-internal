"""Evaluate a separately authorized frozen class-support contingency stage.

The evaluator never trains or samples. It consumes only the two frozen
10K-sample trees from the 100K checkpoints named by an immutable preparation
and refuses to run without a separate, exact execution authorization.
"""

# Parsed artifact type mismatches are schema-value failures, not API misuse.
# ruff: noqa: TRY004

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, read_image
from torchvision.models import ResNet50_Weights, resnet50

from cofitok.generation_class_support_contingency import (
    CLASSIFIER,
    METHODS,
    build_result,
    physical_sample_tree_identity,
    stable_file_identity,
    validate_execution_authorization,
    validate_git_identity,
    validate_preparation,
    validate_result,
)


class _IndexedDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, root: Path) -> None:
        self.paths = [root / f"{index:06d}.png" for index in range(10_000)]
        self.transform = ResNet50_Weights.IMAGENET1K_V2.transforms()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = read_image(self.paths[index].as_posix(), mode=ImageReadMode.RGB)
        return self.transform(image), index


def _read_json_bound(
    path: Path, *, expected_sha256: str, name: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    before = stable_file_identity(path)
    if before["sha256"] != expected_sha256:
        raise ValueError(f"{name} SHA256 differs")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must contain an object")
    after = stable_file_identity(path)
    if after != before:
        raise RuntimeError(f"{name} changed while it was read")
    return payload, before


def _checkout_identity(project_root: Path, script: Path) -> dict[str, Any]:
    root = project_root.resolve(strict=True)
    status = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all"],
        text=True,
    )
    if status:
        raise ValueError("evaluator checkout must be completely clean")
    resolved_script = script.resolve(strict=True)
    relative = resolved_script.relative_to(root).as_posix()
    subprocess.run(
        ["git", "-C", str(root), "ls-files", "--error-unmatch", "--", relative],
        check=True,
        capture_output=True,
    )
    if (
        subprocess.check_output(["git", "-C", str(root), "show", f"HEAD:{relative}"])
        != resolved_script.read_bytes()
    ):
        raise ValueError("evaluator script differs from the exact HEAD blob")
    return validate_git_identity(
        {
            "revision": subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD^{commit}"], text=True
            ).strip(),
            "tree": subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD^{tree}"], text=True
            ).strip(),
            "branch": subprocess.check_output(
                ["git", "-C", str(root), "branch", "--show-current"], text=True
            ).strip(),
            "tracked_dirty": False,
        },
        "evaluator Git",
    )


def _load_classifier(path: Path, device: torch.device) -> torch.nn.Module:
    identity = stable_file_identity(path)
    if (
        identity["bytes"] != CLASSIFIER["weights_bytes"]
        or identity["sha256"] != CLASSIFIER["weights_sha256"]
    ):
        raise ValueError("classifier bytes or SHA256 differ")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError("classifier checkpoint is malformed")
    model = resnet50(weights=None)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model.to(device)


def _image_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _predict(
    *,
    root: Path,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> list[dict[str, Any]]:
    dataset = _IndexedDataset(root)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for images, indices in loader:
            logits = model(images.to(device, non_blocking=device.type == "cuda"))
            if (
                logits.shape != (len(indices), 1_000)
                or not torch.isfinite(logits).all()
            ):
                raise ValueError("classifier logits are malformed or non-finite")
            probabilities = torch.softmax(logits.float(), dim=1)
            top_probabilities, top_classes = torch.topk(probabilities, k=5, dim=1)
            probabilities = probabilities.cpu()
            top_probabilities = top_probabilities.cpu()
            top_classes = top_classes.cpu()
            for offset, raw_index in enumerate(indices.tolist()):
                index = int(raw_index)
                requested = index % 1_000
                path = root / f"{index:06d}.png"
                rows.append(
                    {
                        "sample_index": index,
                        "filename": path.name,
                        "requested_class": requested,
                        "requested_probability": float(
                            probabilities[offset, requested]
                        ),
                        "predicted_top1_class": int(top_classes[offset, 0]),
                        "predicted_top1_probability": float(
                            top_probabilities[offset, 0]
                        ),
                        "predicted_top5_classes": [
                            int(value) for value in top_classes[offset].tolist()
                        ],
                        "predicted_top5_probabilities": [
                            float(value) for value in top_probabilities[offset].tolist()
                        ],
                        "image_sha256": _image_sha256(path),
                    }
                )
    if [row["sample_index"] for row in rows] != list(range(10_000)):
        raise RuntimeError("classifier did not consume the exact ordered sample set")
    return rows


def _exclusive_jsonl(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii", newline="\n") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        row,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=True,
                        allow_nan=False,
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return stable_file_identity(path)


def _exclusive_json(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    encoded = (
        json.dumps(
            payload, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False
        )
        + "\n"
    ).encode("ascii")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return stable_file_identity(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--preparation-sha256", required=True)
    parser.add_argument("--execution-authorization", type=Path, required=True)
    parser.add_argument("--execution-authorization-sha256", required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch size or worker count is invalid")
    if not args.output_dir.is_absolute():
        raise ValueError("diagnostic output directory must be absolute")
    try:
        import scipy
        from scipy.optimize import linear_sum_assignment  # noqa: F401
    except ImportError as error:
        raise RuntimeError("authorized evaluator requires scipy") from error
    preparation, preparation_id = _read_json_bound(
        args.preparation,
        expected_sha256=args.preparation_sha256,
        name="class-support contingency preparation",
    )
    prepared = validate_preparation(preparation)
    authorization, authorization_id = _read_json_bound(
        args.execution_authorization,
        expected_sha256=args.execution_authorization_sha256,
        name="class-support contingency execution authorization",
    )
    evaluator_git = _checkout_identity(args.project_root, args.script)
    authorized = validate_execution_authorization(
        authorization,
        preparation=prepared,
        preparation_identity=preparation_id,
    )
    if authorized["evaluator_git"] != evaluator_git:
        raise ValueError("current evaluator Git is not the authorized evaluator")
    stage_approval, stage_approval_id = _read_json_bound(
        Path(authorized["stage_approval"]["path"]),
        expected_sha256=authorized["stage_approval"]["sha256"],
        name="class-support stage approval",
    )
    if (
        stage_approval_id != authorized["stage_approval"]
        or stage_approval != authorized["stage_approval_record"]
    ):
        raise ValueError("stage approval changed after authorization")
    expected_output = Path(prepared["output_contract"]["output_root"])
    if args.output_dir.resolve() != expected_output.resolve():
        raise ValueError("output directory differs from the preparation")
    if args.output_dir.exists():
        raise FileExistsError(f"diagnostic output already exists: {args.output_dir}")
    classifier_path = Path(prepared["source_evidence"]["fixed_classifier"]["path"])
    if (
        stable_file_identity(classifier_path)
        != prepared["source_evidence"]["fixed_classifier"]
    ):
        raise ValueError("fixed classifier changed after preparation")
    roots = {
        method: Path(
            prepared["source_evidence"]["frozen_methods"][method]["sample_tree"]["path"]
        )
        for method in METHODS
    }
    before = {
        method: physical_sample_tree_identity(roots[method]) for method in METHODS
    }
    for method in METHODS:
        if (
            before[method]
            != prepared["source_evidence"]["frozen_methods"][method]["sample_tree"]
        ):
            raise ValueError(f"{method} sample tree changed after preparation")
    start = time.perf_counter()
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    model = _load_classifier(classifier_path, device)
    rows_by_method = {
        method: _predict(
            root=roots[method],
            model=model,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        for method in METHODS
    }
    after = {
        method: physical_sample_tree_identity(
            roots[method],
            expected_image_sha256=[
                row["image_sha256"] for row in rows_by_method[method]
            ],
        )
        for method in METHODS
    }
    if after != before:
        raise RuntimeError("a frozen sample tree changed during classifier inference")
    classifier_after = stable_file_identity(classifier_path)
    if classifier_after != prepared["source_evidence"]["fixed_classifier"]:
        raise RuntimeError("fixed classifier changed during classifier inference")
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    os.mkdir(args.output_dir, 0o755)
    prediction_ids = {
        method: _exclusive_jsonl(
            args.output_dir / prepared["output_contract"]["per_sample_files"][method],
            rows_by_method[method],
        )
        for method in METHODS
    }
    recall = {
        method: float(
            prepared["source_evidence"]["frozen_methods"][method][
                "existing_generation_metrics"
            ]["recall"]
        )
        for method in METHODS
    }
    runtime = {
        "device": device.type,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "torch_version": torch.__version__,
        "torchvision_version": __import__("torchvision").__version__,
        "scipy_version": scipy.__version__,
    }
    result = build_result(
        preparation=prepared,
        preparation_identity=preparation_id,
        execution_authorization=authorized,
        execution_authorization_identity=authorization_id,
        evaluator_git=evaluator_git,
        prediction_identities=prediction_ids,
        rows_by_method=rows_by_method,
        existing_recall=recall,
        runtime=runtime,
        elapsed_started_at=start,
    )
    if stable_file_identity(args.preparation) != preparation_id:
        raise RuntimeError("preparation changed during evaluation")
    if stable_file_identity(args.execution_authorization) != authorization_id:
        raise RuntimeError("execution authorization changed during evaluation")
    if (
        stable_file_identity(Path(authorized["stage_approval"]["path"]))
        != stage_approval_id
    ):
        raise RuntimeError("stage approval changed during evaluation")
    if stable_file_identity(classifier_path) != classifier_after:
        raise RuntimeError("fixed classifier changed after analysis")
    for method in METHODS:
        prediction_path = (
            args.output_dir / prepared["output_contract"]["per_sample_files"][method]
        )
        if stable_file_identity(prediction_path) != prediction_ids[method]:
            raise RuntimeError(f"{method} predictions changed during analysis")
    validate_result(
        result,
        preparation=prepared,
        preparation_identity=preparation_id,
    )
    result_path = args.output_dir / prepared["output_contract"]["result_file"]
    result_id = _exclusive_json(result_path, result)
    os.chmod(args.output_dir, 0o555)
    print(
        json.dumps(
            {
                "status": "completed",
                "scientific_status": result["scientific_status"],
                "primary_interpretation": result["analysis"]["interpretation"][
                    "primary_interpretation"
                ],
                "result": result_id,
                "confirmation_preparation_allowed": False,
                "full_training_launch_allowed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
