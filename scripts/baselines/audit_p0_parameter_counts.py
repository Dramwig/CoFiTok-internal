from __future__ import annotations

import argparse
import gc
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import torch


SUPPORTED = {"edm", "improved_diffusion"}
DATASETS = {
    "cifar10",
    "tiny_imagenet_200",
    "imagenet_1k_64x64_hf",
    "downsampled_imagenet_64",
    "ffhq_64",
    "afhqv2_64",
    "imagenet_256_10pct",
    "imagenet_256",
}
PINNED_COMMITS = {
    "edm": "008a4e5316c8e3bfe61a62f874bddba254295afb",
    "improved_diffusion": "1bc7bbbdc414d83d4abf2ad8cc1446dc36c4e4d5",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit parameter counts from trusted pinned P0 baseline checkpoints."
    )
    parser.add_argument("--reports-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _checkpoint_path(report: dict[str, Any]) -> Path:
    checkpoints = report.get("checkpoints", {})
    baseline = report.get("baseline")
    if baseline == "edm":
        value = checkpoints.get("snapshot")
    else:
        value = checkpoints.get("sample_checkpoint") or checkpoints.get("ema")
    if not value:
        raise ValueError(f"no parameter-audit checkpoint for {baseline}")
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _count_state_dict_tensors(state: Any) -> int:
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"expected state dict, got {type(state)!r}")
    tensors = [value for value in state.values() if torch.is_tensor(value)]
    if not tensors:
        raise ValueError("checkpoint has no tensors")
    return sum(tensor.numel() for tensor in tensors)


def audit_report(report_path: Path) -> dict[str, Any]:
    report = _read_json(report_path)
    baseline = str(report.get("baseline"))
    if baseline not in SUPPORTED:
        raise ValueError(f"unsupported baseline: {baseline}")
    checkpoint = _checkpoint_path(report)

    if baseline == "edm":
        repo = Path(str(report.get("repo", "")))
        if not repo.is_dir():
            raise FileNotFoundError(repo)
        repo_text = str(repo)
        sys.path.insert(0, repo_text)
        try:
            with checkpoint.open("rb") as handle:
                snapshot = pickle.load(handle)
            network = snapshot.get("ema") if isinstance(snapshot, dict) else None
            if network is None or not hasattr(network, "parameters"):
                raise ValueError("EDM snapshot has no EMA module")
            parameter_count = sum(parameter.numel() for parameter in network.parameters())
            count_method = "pickled_ema_module_parameters"
        finally:
            if sys.path and sys.path[0] == repo_text:
                sys.path.pop(0)
            snapshot = None
            network = None
            gc.collect()
    else:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        parameter_count = _count_state_dict_tensors(state)
        count_method = "ema_state_dict_tensor_elements"
        del state
        gc.collect()

    return {
        "baseline": baseline,
        "dataset": report.get("dataset"),
        "run_name": report.get("run_name"),
        "repo_commit": report.get("repo_commit"),
        "checkpoint": str(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "parameter_count": int(parameter_count),
        "count_method": count_method,
        "train_report": report_path.as_posix(),
        "status": "completed",
    }


def contract_violations(rows: list[dict[str, Any]]) -> list[str]:
    violations: list[str] = []
    keys = [(str(row["baseline"]), str(row["dataset"])) for row in rows]
    expected = {(baseline, dataset) for baseline in SUPPORTED for dataset in DATASETS}
    observed = set(keys)
    if len(keys) != len(observed):
        violations.append("duplicate baseline-dataset parameter rows")
    for key in sorted(expected - observed):
        violations.append(f"missing row: {key[0]}/{key[1]}")
    for key in sorted(observed - expected):
        violations.append(f"unexpected row: {key[0]}/{key[1]}")
    for row in rows:
        baseline = str(row["baseline"])
        if row.get("repo_commit") != PINNED_COMMITS.get(baseline):
            violations.append(f"commit mismatch: {baseline}/{row.get('dataset')}")
        if int(row.get("parameter_count", 0)) <= 0:
            violations.append(f"invalid parameter count: {baseline}/{row.get('dataset')}")
    return violations


def main() -> None:
    args = parse_args()
    rows = []
    failures = []
    for path in sorted(args.reports_root.rglob("baseline_train_report.json")):
        report = _read_json(path)
        if report.get("baseline") not in SUPPORTED or report.get("status") != "completed":
            continue
        try:
            rows.append(audit_report(path))
        except Exception as error:
            failures.append(
                {
                    "baseline": report.get("baseline"),
                    "dataset": report.get("dataset"),
                    "run_name": report.get("run_name"),
                    "train_report": path.as_posix(),
                    "error": repr(error),
                }
            )
    payload = {
        "schema_version": 1,
        "supported_baselines": sorted(SUPPORTED),
        "row_count": len(rows),
        "failure_count": len(failures),
        "rows": rows,
        "failures": failures,
        "contract_violations": contract_violations(rows),
        "notes": (
            "EDM counts nn.Module parameters from the pinned EMA pickle. Improved-DDPM "
            "counts tensor elements in the pinned EMA state dict; its architecture uses no "
            "running-stat buffers, so this equals the model parameter count."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)
    if args.strict and (failures or payload["contract_violations"]):
        raise RuntimeError(
            "parameter audit failed: "
            f"{len(failures)} load failures, "
            f"{len(payload['contract_violations'])} contract violations"
        )


if __name__ == "__main__":
    main()
