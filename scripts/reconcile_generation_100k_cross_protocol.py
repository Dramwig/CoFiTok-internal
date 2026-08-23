from __future__ import annotations

import argparse
import json
import math
import os
import socket
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image

from cofitok.environment import capture_runtime_environment, runtime_environment_sha256
from cofitok.generation.cross_protocol_reconciliation import (
    CLAIM_BOUNDARY,
    INTERMEDIATE_METRIC_ROLE,
    INTERMEDIATE_METRIC_SCHEMA_VERSION,
    REPORT_ROLE,
    REPORT_SCHEMA_VERSION,
    build_reconciliation_comparison,
    metric_summary,
    validate_decision_route,
    validate_quality_hold,
)
from cofitok.image_integrity import IMAGE_TREE_DIGEST_SCHEMA, image_tree_sha256, sample_set_sha256
from cofitok.inference_replay import file_identity, read_json_object, reject_symlink_chain
from cofitok.output_lock import exclusive_output_lock
from cofitok.reporting import git_provenance, write_json_report

try:
    from scripts.evaluate_generation_metrics import (
        calculate_metrics,
        content_addressed_real_cache_name,
        find_images,
        torch_fidelity_version,
        validate_sampling_provenance,
    )
except ModuleNotFoundError:
    from evaluate_generation_metrics import (
        calculate_metrics,
        content_addressed_real_cache_name,
        find_images,
        torch_fidelity_version,
        validate_sampling_provenance,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_FILENAME = "cross_protocol_reconciliation.json"
STATUS_FILENAME = "status.json"
METHOD_LAYOUTS = {
    "cofitok": {
        "run": "cofitok_rgbtail3_rollout_x0_u2_ema_teacher",
        "prefix_budget": 8,
    },
    "dense_identity": {
        "run": "dense_rollout_x0_u2_ema_teacher",
        "prefix_budget": 1,
    },
}
TRAINING_REVISION = "cf0e5faa94bf4ab38d947b921935b3b765b5537a"
TRAINING_TREE = "6cef27723196fd363379bca2e7b85b1678ebd777"
TRAINING_BRANCH = "scale/generation-stability-quality-bridge-100k"
TORCH_FIDELITY_VERSION = "0.4.0"
REAL_CACHE_BASE = "imagenet256_val_50k_torch_fidelity_v04"
INDEX_COUNT = 2_048


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_identity(project: Path) -> dict[str, Any]:
    identity = git_provenance(project)
    tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=project,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {**identity, "tree": tree, "path": project.resolve().as_posix()}


def _require_git_identity(
    project: Path,
    *,
    revision: str,
    tree: str,
    branch: str,
    label: str,
) -> dict[str, Any]:
    identity = _git_identity(project)
    expected = {
        "revision": revision,
        "tree": tree,
        "branch": branch,
        "tracked_dirty": False,
        "path": project.resolve().as_posix(),
    }
    if identity != expected:
        raise ValueError(f"{label} Git identity differs")
    return identity


def _bound_json(
    path: Path,
    *,
    expected_sha256: str | None,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    identity = file_identity(path)
    if expected_sha256 is not None and identity["sha256"] != expected_sha256:
        raise ValueError(f"{label} SHA256 differs")
    return read_json_object(path, name=label), identity


def _require_claimed_identity(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} identity is missing")
    identity = file_identity(Path(str(value.get("path", ""))))
    if identity != value:
        raise ValueError(f"{label} identity differs")
    return identity


def _numbered_pngs(directory: Path, count: int) -> list[Path]:
    root = reject_symlink_chain(directory, name="generated PNG directory")
    if not root.is_dir():
        raise FileNotFoundError(f"generated PNG directory is missing: {root}")
    images = sorted(root.glob("*.png"))
    expected = [f"{index:06d}.png" for index in range(count)]
    if len(images) != count or [path.name for path in images] != expected:
        raise ValueError(f"generated PNG set is not exactly 0..{count - 1}: {root}")
    if any(path.is_symlink() or not path.is_file() for path in images):
        raise ValueError(f"generated PNG set contains a non-regular file: {root}")
    return images


def _metric_report(
    path: Path,
    *,
    generated_dir: Path,
    expected_count: int,
    expected_prc: bool,
    sampling_provenance: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    report, identity = _bound_json(
        path,
        expected_sha256=None,
        label="generation metrics report",
    )
    if (
        report.get("status") != "completed"
        or report.get("protocol") != "torch_fidelity_directory_metrics"
        or report.get("implementation")
        != {"package": "torch_fidelity", "version": TORCH_FIDELITY_VERSION}
        or report.get("counts", {}).get("generated_image_count") != expected_count
        or Path(str(report.get("paths", {}).get("generated_dir", ""))).resolve()
        != generated_dir.resolve()
        or report.get("parameters", {}).get("precision_recall_enabled")
        is not expected_prc
        or report.get("sample_provenance", {}).get("report_identity")
        != sampling_provenance["report_identity"]
        or report.get("sample_provenance", {}).get("manifest_identity")
        != sampling_provenance["manifest_identity"]
        or report.get("sample_provenance", {}).get("sample_set_sha256")
        != sampling_provenance["sample_set_sha256"]
        or report.get("sample_provenance", {}).get("checkpoint_sha256")
        != sampling_provenance["checkpoint_sha256"]
    ):
        raise ValueError(f"generation metrics report binding differs: {path}")
    metric_summary(report.get("metrics", {}), label=path.as_posix())
    return report, identity


def _validate_checkpoint(
    run_dir: Path,
    *,
    training_git: dict[str, Any],
) -> dict[str, Any]:
    latest_path = run_dir / "latest.json"
    latest, latest_identity = _bound_json(
        latest_path,
        expected_sha256=None,
        label="100K latest checkpoint pointer",
    )
    if (
        latest.get("step") != 100_000
        or latest.get("git_revision") != TRAINING_REVISION
        or latest.get("git_branch") != TRAINING_BRANCH
        or latest.get("git_dirty") is not False
    ):
        raise ValueError("100K latest checkpoint pointer differs")
    payload = reject_symlink_chain(
        run_dir / str(latest.get("checkpoint", "")),
        name="100K checkpoint payload",
    )
    sidecar = reject_symlink_chain(
        run_dir / str(latest.get("integrity_manifest", "")),
        name="100K checkpoint integrity sidecar",
    )
    payload_identity = file_identity(payload)
    sidecar_payload, sidecar_identity = _bound_json(
        sidecar,
        expected_sha256=None,
        label="100K checkpoint integrity sidecar",
    )
    expected_payload = {
        "path": payload_identity["path"],
        "bytes": latest["checkpoint_bytes"],
        "sha256": latest["checkpoint_sha256"],
    }
    if payload_identity != expected_payload:
        raise ValueError("100K checkpoint physical payload differs from latest.json")
    for field, expected in (
        ("checkpoint", payload.name),
        ("checkpoint_bytes", payload_identity["bytes"]),
        ("checkpoint_sha256", payload_identity["sha256"]),
        ("step", 100_000),
        ("git_revision", TRAINING_REVISION),
        ("git_branch", TRAINING_BRANCH),
        ("git_dirty", False),
        ("dataset_identity_sha256", latest["dataset_identity_sha256"]),
        ("runtime_environment_sha256", latest["runtime_environment_sha256"]),
    ):
        if sidecar_payload.get(field) != expected:
            raise ValueError(f"100K checkpoint sidecar {field} differs")
    return {
        "status": "physically_verified",
        "training_git": training_git,
        "latest": latest_identity,
        "integrity_sidecar": sidecar_identity,
        "payload": payload_identity,
        "dataset_identity_sha256": latest["dataset_identity_sha256"],
        "runtime_environment_sha256": latest["runtime_environment_sha256"],
        "checkpoint_step": 100_000,
    }


def _sampling_paths(run_dir: Path, *, budget: int) -> dict[str, dict[str, Path]]:
    milestone = run_dir / "milestones/step_00100000/samples_2048_ddim50_cfg15"
    terminal = run_dir / "terminal_100k/samples_10000_ddim100_cfg15"
    return {
        "ddim50_2048": {
            "root": milestone,
            "generated": milestone / f"prefix_{budget}",
            "sampling_report": milestone / "sampling_report.json",
            "metrics_report": milestone / "metrics/generation_metrics_report.json",
        },
        "ddim100_10000": {
            "root": terminal,
            "generated": terminal / f"prefix_{budget}",
            "sampling_report": terminal / "sampling_report.json",
            "metrics_report": terminal / "metrics/generation_metrics_report.json",
        },
    }


def _protocol_common(sampling: dict[str, Any]) -> dict[str, Any]:
    excluded = {"actual_timesteps", "num_samples", "sample_steps", "prefix_budgets"}
    return {key: value for key, value in sampling.items() if key not in excluded}


def _validate_method_sources(
    quality_root: Path,
    *,
    method: str,
    layout: dict[str, Any],
    training_git: dict[str, Any],
) -> dict[str, Any]:
    run_dir = reject_symlink_chain(
        quality_root / str(layout["run"]),
        name=f"{method} run directory",
    )
    if not run_dir.is_dir():
        raise FileNotFoundError(f"{method} run directory is missing: {run_dir}")
    checkpoint = _validate_checkpoint(run_dir, training_git=training_git)
    paths = _sampling_paths(run_dir, budget=int(layout["prefix_budget"]))
    protocols: dict[str, Any] = {}
    for name, expected_count, expected_steps, expected_prc in (
        ("ddim50_2048", 2_048, 50, False),
        ("ddim100_10000", 10_000, 100, True),
    ):
        row = paths[name]
        images = _numbered_pngs(row["generated"], expected_count)
        provenance = validate_sampling_provenance(
            row["sampling_report"], row["generated"], images
        )
        sampling = provenance["sampling"]
        if (
            sampling.get("sampler") != "ddim"
            or sampling.get("sample_steps") != expected_steps
            or sampling.get("num_samples") != expected_count
            or sampling.get("seed") != 0
            or sampling.get("start_index") != 0
            or sampling.get("guidance_scale") != 1.5
            or sampling.get("guidance_rescale") != 0.0
            or sampling.get("eta") != 0.0
            or sampling.get("precision") != "bf16"
            or sampling.get("class_schedule") != "balanced_modulo"
            or sampling.get("prefix_budgets") != [int(layout["prefix_budget"])]
            or provenance.get("checkpoint_step") != 100_000
            or provenance.get("checkpoint_sha256") != checkpoint["payload"]["sha256"]
        ):
            raise ValueError(f"{method} {name} sampling protocol differs")
        metric_report, metric_identity = _metric_report(
            row["metrics_report"],
            generated_dir=row["generated"],
            expected_count=expected_count,
            expected_prc=expected_prc,
            sampling_provenance=provenance,
        )
        protocols[name] = {
            "paths": {key: value.resolve().as_posix() for key, value in row.items()},
            "images": images,
            "sample_set_sha256": provenance["sample_set_sha256"],
            "sampling_provenance": provenance,
            "source_metrics": {
                "identity": metric_identity,
                "metrics": metric_summary(
                    metric_report["metrics"], label=f"{method} {name}"
                ),
                "implementation": metric_report["implementation"],
                "parameters": metric_report["parameters"],
                "real_set": metric_report["real_set"],
                "runtime_environment_sha256": metric_report[
                    "runtime_environment_sha256"
                ],
            },
        }
    if _protocol_common(
        protocols["ddim50_2048"]["sampling_provenance"]["sampling"]
    ) != _protocol_common(
        protocols["ddim100_10000"]["sampling_provenance"]["sampling"]
    ):
        raise ValueError(f"{method} cross-protocol shared sampling contract differs")
    first_2048 = protocols["ddim100_10000"]["images"][:INDEX_COUNT]
    protocols["ddim100_first_2048"] = {
        "source_images": first_2048,
        "source_sample_set_sha256": sample_set_sha256(first_2048),
        "selected_index_interval": [0, INDEX_COUNT - 1],
        "selection_rule": "existing_ddim100_zero_based_indices_000000_through_002047",
        "new_sampling_performed": False,
    }
    return {
        "method": method,
        "run_dir": run_dir.resolve().as_posix(),
        "prefix_budget": int(layout["prefix_budget"]),
        "checkpoint": checkpoint,
        "protocols": protocols,
    }


def _real_set_evidence(methods: dict[str, Any]) -> dict[str, Any]:
    declared = []
    for method in methods.values():
        for protocol_name in ("ddim50_2048", "ddim100_10000"):
            declared.append(method["protocols"][protocol_name]["source_metrics"]["real_set"])
    if not declared or any(value != declared[0] for value in declared[1:]):
        raise ValueError("source metric reports use different real sets")
    real_dir = reject_symlink_chain(declared[0]["root"], name="real image directory")
    real_images = find_images(real_dir)
    if len(real_images) != int(declared[0]["image_count"]):
        raise ValueError("real image count differs from source metric reports")
    physical_sha = image_tree_sha256(real_images, root=real_dir)
    if physical_sha != declared[0]["sha256"]:
        raise ValueError("real image tree SHA256 differs from source metric reports")
    return {
        "status": "physically_verified",
        "root": real_dir.resolve().as_posix(),
        "image_count": len(real_images),
        "digest_schema": IMAGE_TREE_DIGEST_SCHEMA,
        "sha256": physical_sha,
        "images": real_images,
    }


def collect_source_evidence(args: argparse.Namespace) -> dict[str, Any]:
    evaluator_git = _require_git_identity(
        PROJECT_ROOT,
        revision=args.expected_evaluator_revision,
        tree=args.expected_evaluator_tree,
        branch=args.expected_evaluator_branch,
        label="cross-protocol evaluator checkout",
    )
    training_checkout = reject_symlink_chain(
        args.training_checkout, name="quality bridge training checkout"
    )
    training_git = _require_git_identity(
        training_checkout,
        revision=TRAINING_REVISION,
        tree=TRAINING_TREE,
        branch=TRAINING_BRANCH,
        label="quality bridge training checkout",
    )
    quality_root = reject_symlink_chain(
        args.quality_bridge_root, name="quality bridge output root"
    )
    decision, decision_identity = _bound_json(
        args.decision,
        expected_sha256=args.expected_decision_sha256,
        label="authoritative follow-up decision",
    )
    validate_decision_route(decision)
    result, result_identity = _bound_json(
        args.quality_result,
        expected_sha256=args.expected_quality_result_sha256,
        label="quality bridge terminal result",
    )
    validate_quality_hold(result)
    if decision.get("source_reports", {}).get("quality_bridge_result") != result_identity:
        raise ValueError("authoritative decision quality-result binding differs")
    milestone_claim = decision.get("source_reports", {}).get("milestones", {}).get(
        "100000"
    )
    milestone_identity = _require_claimed_identity(
        milestone_claim, label="authoritative decision 100K milestone"
    )
    if result.get("source_reports", {}).get("milestone_100000") != milestone_identity:
        raise ValueError("terminal result 100K milestone binding differs")
    exposure_identity = _require_claimed_identity(
        decision.get("source_reports", {}).get("terminal_training_exposure"),
        label="authoritative terminal training exposure",
    )
    pair_monitor, pair_identity = _bound_json(
        quality_root / "pair_monitor.json",
        expected_sha256=None,
        label="quality bridge pair monitor",
    )
    if (
        pair_monitor.get("status") != "pass"
        or pair_monitor.get("stage") != "complete"
        or pair_monitor.get("issues") != []
    ):
        raise ValueError("quality bridge pair monitor is not a clean completion")
    methods = {
        method: _validate_method_sources(
            quality_root,
            method=method,
            layout=layout,
            training_git=training_git,
        )
        for method, layout in METHOD_LAYOUTS.items()
    }
    cofitok_common = _protocol_common(
        methods["cofitok"]["protocols"]["ddim100_10000"][
            "sampling_provenance"
        ]["sampling"]
    )
    dense_common = _protocol_common(
        methods["dense_identity"]["protocols"]["ddim100_10000"][
            "sampling_provenance"
        ]["sampling"]
    )
    if cofitok_common != dense_common:
        raise ValueError("matched methods use different shared sampling contracts")
    real_set = _real_set_evidence(methods)
    return {
        "evaluator_git": evaluator_git,
        "training_git": training_git,
        "quality_bridge_root": quality_root.resolve().as_posix(),
        "decision": {"identity": decision_identity, "report": decision},
        "quality_result": {"identity": result_identity, "report": result},
        "milestone_100000": milestone_identity,
        "terminal_training_exposure": exposure_identity,
        "pair_monitor": pair_identity,
        "methods": methods,
        "real_set": real_set,
    }


def prepare_hardlink_subset(
    source_images: list[Path],
    *,
    target_dir: Path,
) -> dict[str, Any]:
    if len(source_images) != INDEX_COUNT:
        raise ValueError("DDIM-100 subset must contain exactly 2,048 source images")
    target = reject_symlink_chain(target_dir, name="DDIM-100 subset directory")
    target.mkdir(parents=True, exist_ok=True)
    linked: list[Path] = []
    for index, source in enumerate(source_images):
        if source.name != f"{index:06d}.png" or source.is_symlink() or not source.is_file():
            raise ValueError("DDIM-100 source subset filename contract differs")
        destination = target / source.name
        if destination.exists():
            if destination.is_symlink() or not destination.is_file():
                raise ValueError("DDIM-100 subset contains a non-regular file")
            if not os.path.samefile(source, destination):
                raise ValueError("DDIM-100 subset contains a non-source hardlink")
        else:
            os.link(source, destination)
        linked.append(destination)
    unexpected = sorted(
        path.name for path in target.iterdir() if path.name not in {x.name for x in linked}
    )
    if unexpected:
        raise ValueError("DDIM-100 subset contains unexpected files")
    source_sha = sample_set_sha256(source_images)
    target_sha = sample_set_sha256(linked)
    if source_sha != target_sha:
        raise ValueError("DDIM-100 hardlink subset digest differs from source slice")
    return {
        "status": "verified",
        "link_mode": "hardlink_existing_png_view",
        "new_sampling_performed": False,
        "source_count": len(source_images),
        "target_count": len(linked),
        "source_sample_set_sha256": source_sha,
        "target_sample_set_sha256": target_sha,
        "target_dir": target.resolve().as_posix(),
        "images": linked,
    }


def paired_pixel_difference(
    left_images: list[Path],
    right_images: list[Path],
) -> dict[str, Any]:
    if len(left_images) != INDEX_COUNT or len(right_images) != INDEX_COUNT:
        raise ValueError("paired pixel comparison requires 2,048 images per protocol")
    squared_sum = 0.0
    absolute_sum = 0.0
    scalar_count = 0
    identical = 0
    maximum = 0
    for index, (left, right) in enumerate(zip(left_images, right_images, strict=True)):
        if left.name != right.name or left.name != f"{index:06d}.png":
            raise ValueError("paired pixel filenames differ")
        with Image.open(left) as image:
            left_array = np.asarray(image.convert("RGB"), dtype=np.int16)
        with Image.open(right) as image:
            right_array = np.asarray(image.convert("RGB"), dtype=np.int16)
        if left_array.shape != right_array.shape:
            raise ValueError("paired pixel image shapes differ")
        difference = left_array - right_array
        if not np.any(difference):
            identical += 1
        squared_sum += float(np.square(difference, dtype=np.int32).sum(dtype=np.float64))
        absolute_sum += float(np.abs(difference).sum(dtype=np.float64))
        scalar_count += int(difference.size)
        maximum = max(maximum, int(np.abs(difference).max()))
    mse_255 = squared_sum / scalar_count
    mse_unit = mse_255 / (255.0**2)
    return {
        "pair_count": INDEX_COUNT,
        "identical_pair_count": identical,
        "different_pair_count": INDEX_COUNT - identical,
        "mean_squared_error_unit_range": mse_unit,
        "mean_absolute_error_unit_range": absolute_sum / scalar_count / 255.0,
        "peak_signal_to_noise_ratio_db": (
            None if mse_unit == 0.0 else -10.0 * math.log10(mse_unit)
        ),
        "maximum_absolute_channel_difference": maximum,
    }


def _intermediate_expected(
    *,
    method: str,
    subset: dict[str, Any],
    source_sampling_identity: dict[str, Any],
    real_set: dict[str, Any],
    evaluator_git: dict[str, Any],
    evaluator_environment: dict[str, Any],
    evaluator_environment_sha256: str,
    batch_size: int,
    cache_root: Path,
    real_cache_name: str,
) -> dict[str, Any]:
    return {
        "schema_version": INTERMEDIATE_METRIC_SCHEMA_VERSION,
        "role": INTERMEDIATE_METRIC_ROLE,
        "status": "completed",
        "method": method,
        "protocol": "torch_fidelity_directory_metrics_existing_ddim100_first_2048",
        "implementation": {
            "package": "torch_fidelity",
            "version": TORCH_FIDELITY_VERSION,
        },
        "evaluator_git": evaluator_git,
        "runtime_environment": evaluator_environment,
        "runtime_environment_sha256": evaluator_environment_sha256,
        "source_sampling_report": source_sampling_identity,
        "subset": {
            key: value
            for key, value in subset.items()
            if key not in {"images"}
        },
        "real_set": {
            key: value for key, value in real_set.items() if key != "images"
        },
        "parameters": {
            "batch_size": batch_size,
            "cuda": False,
            "cpu_only": True,
            "precision_recall_enabled": False,
            "seed": 2027,
            "samples_find_deep": True,
            "samples_shuffle": False,
            "cache_root": cache_root.resolve().as_posix(),
            "real_cache_name": real_cache_name,
        },
    }


def evaluate_subset_metrics(
    *,
    method: str,
    subset: dict[str, Any],
    source_sampling_identity: dict[str, Any],
    real_set: dict[str, Any],
    evaluator_git: dict[str, Any],
    evaluator_environment: dict[str, Any],
    evaluator_environment_sha256: str,
    batch_size: int,
    cache_root: Path,
    real_cache_name: str,
    output: Path,
    metric_calculator: Callable[..., tuple[dict[str, float], str]] = calculate_metrics,
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected = _intermediate_expected(
        method=method,
        subset=subset,
        source_sampling_identity=source_sampling_identity,
        real_set=real_set,
        evaluator_git=evaluator_git,
        evaluator_environment=evaluator_environment,
        evaluator_environment_sha256=evaluator_environment_sha256,
        batch_size=batch_size,
        cache_root=cache_root,
        real_cache_name=real_cache_name,
    )
    if output.is_file():
        report = read_json_object(output, name=f"{method} subset metrics")
        if any(report.get(key) != value for key, value in expected.items()):
            raise ValueError(f"existing {method} subset metrics binding differs")
        metric_summary(report.get("metrics", {}), label=f"{method} subset metrics")
        return report, file_identity(output)
    start = time.time()
    metrics, version = metric_calculator(
        real_dir=Path(real_set["root"]),
        generated_dir=Path(subset["target_dir"]),
        batch_size=batch_size,
        prc_batch_size=10_000,
        seed=2027,
        cuda=False,
        cache_root=cache_root.resolve().as_posix(),
        real_cache_name=real_cache_name,
        prc=False,
    )
    if version != TORCH_FIDELITY_VERSION:
        raise ValueError("torch-fidelity version changed during subset evaluation")
    metric_summary(metrics, label=f"{method} subset metrics")
    report = {
        **expected,
        "metrics": metrics,
        "runtime": {
            "elapsed_seconds": time.time() - start,
            "torch_version": torch.__version__,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
        },
    }
    write_json_report(output, report)
    return report, file_identity(output)


class StatusHeartbeat:
    def __init__(self, path: Path, *, started_at: str, interval_seconds: float = 30.0):
        self.path = path
        self.started_at = started_at
        self.interval_seconds = interval_seconds
        self.phase = "initializing"
        self.detail = ""
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _payload(self, status: str = "running") -> dict[str, Any]:
        with self._lock:
            phase = self.phase
            detail = self.detail
        return {
            "schema_version": 1,
            "role": "generation_100k_cross_protocol_reconciliation_status",
            "status": status,
            "phase": phase,
            "detail": detail,
            "started_at": self.started_at,
            "updated_at": _utc_now(),
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "hostname": socket.gethostname(),
            "runtime": {
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
                "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
            },
            "claim_boundary": CLAIM_BOUNDARY,
        }

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            write_json_report(self.path, self._payload())

    def start(self) -> None:
        write_json_report(self.path, self._payload())
        self._thread.start()

    def set_phase(self, phase: str, detail: str = "") -> None:
        with self._lock:
            self.phase = phase
            self.detail = detail
        write_json_report(self.path, self._payload())

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=max(1.0, self.interval_seconds + 1.0))


def _final_source_projection(evidence: dict[str, Any]) -> dict[str, Any]:
    methods: dict[str, Any] = {}
    for method, source in evidence["methods"].items():
        protocols = {}
        for name in ("ddim50_2048", "ddim100_10000"):
            row = source["protocols"][name]
            protocols[name] = {
                "paths": row["paths"],
                "sample_set_sha256": row["sample_set_sha256"],
                "sampling_provenance": row["sampling_provenance"],
                "source_metrics": row["source_metrics"],
            }
        protocols["ddim100_first_2048"] = {
            key: value
            for key, value in source["protocols"]["ddim100_first_2048"].items()
            if key != "source_images"
        }
        methods[method] = {
            "method": method,
            "run_dir": source["run_dir"],
            "prefix_budget": source["prefix_budget"],
            "checkpoint": source["checkpoint"],
            "protocols": protocols,
        }
    return {
        "evaluator_git": evidence["evaluator_git"],
        "training_git": evidence["training_git"],
        "quality_bridge_root": evidence["quality_bridge_root"],
        "decision": evidence["decision"]["identity"],
        "quality_result": evidence["quality_result"]["identity"],
        "milestone_100000": evidence["milestone_100000"],
        "terminal_training_exposure": evidence["terminal_training_exposure"],
        "pair_monitor": evidence["pair_monitor"],
        "real_set": {key: value for key, value in evidence["real_set"].items() if key != "images"},
        "methods": methods,
    }


def build_final_report(
    *,
    evidence: dict[str, Any],
    subsets: dict[str, dict[str, Any]],
    subset_metrics: dict[str, tuple[dict[str, Any], dict[str, Any]]],
    pixel_differences: dict[str, dict[str, Any]],
    evaluator_environment: dict[str, Any],
    evaluator_environment_sha256: str,
    started_at: str,
) -> dict[str, Any]:
    comparison = build_reconciliation_comparison(
        cofitok_ddim50_2048=evidence["methods"]["cofitok"]["protocols"][
            "ddim50_2048"
        ]["source_metrics"]["metrics"],
        dense_ddim50_2048=evidence["methods"]["dense_identity"]["protocols"][
            "ddim50_2048"
        ]["source_metrics"]["metrics"],
        cofitok_ddim100_2048=subset_metrics["cofitok"][0]["metrics"],
        dense_ddim100_2048=subset_metrics["dense_identity"][0]["metrics"],
        cofitok_ddim100_10000=evidence["methods"]["cofitok"]["protocols"][
            "ddim100_10000"
        ]["source_metrics"]["metrics"],
        dense_ddim100_10000=evidence["methods"]["dense_identity"]["protocols"][
            "ddim100_10000"
        ]["source_metrics"]["metrics"],
    )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "role": REPORT_ROLE,
        "status": "completed",
        "operational_status": "pass",
        "terminal_status": "hold",
        "started_at": started_at,
        "completed_at": _utc_now(),
        "source_evidence": _final_source_projection(evidence),
        "evaluator_runtime_environment": evaluator_environment,
        "evaluator_runtime_environment_sha256": evaluator_environment_sha256,
        "implementation": {
            "metric_package": "torch_fidelity",
            "metric_version": TORCH_FIDELITY_VERSION,
            "subset_metric_device": "cpu",
            "new_model_load_performed": False,
            "new_sampling_performed": False,
        },
        "derived_ddim100_first_2048": {
            method: {
                "subset": {
                    key: value for key, value in subsets[method].items() if key != "images"
                },
                "metrics": subset_metrics[method][0]["metrics"],
                "metric_report": subset_metrics[method][1],
                "paired_ddim50_vs_ddim100_pixel_difference": pixel_differences[method],
            }
            for method in METHOD_LAYOUTS
        },
        "comparison": comparison,
        "interpretation": {
            "purpose": (
                "Isolate sampler-step and sample-count contributions to the matched "
                "100K FID ranking reversal using only pre-existing PNGs."
            ),
            "sampler_step_contrast": "DDIM-50/2048 versus DDIM-100/first-2048",
            "sample_count_contrast": "DDIM-100/first-2048 versus DDIM-100/10000",
            "formal_quality_status_unchanged": True,
            "terminal_result_remains_authoritative": True,
            "generation_advantage_proven": False,
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _validate_completed_final(
    report_path: Path,
    *,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    report = read_json_object(report_path, name="cross-protocol reconciliation report")
    if (
        report.get("schema_version") != REPORT_SCHEMA_VERSION
        or report.get("role") != REPORT_ROLE
        or report.get("status") != "completed"
        or report.get("operational_status") != "pass"
        or report.get("terminal_status") != "hold"
        or report.get("claim_boundary") != CLAIM_BOUNDARY
        or report.get("source_evidence") != _final_source_projection(evidence)
        or report.get("comparison", {}).get("generation_advantage_proven") is not False
    ):
        raise ValueError("completed cross-protocol reconciliation report differs")
    for method in METHOD_LAYOUTS:
        identity = report.get("derived_ddim100_first_2048", {}).get(method, {}).get(
            "metric_report"
        )
        _require_claimed_identity(identity, label=f"{method} subset metric report")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconcile the 100K DDIM-50/2,048 and DDIM-100/10,000 quality "
            "evidence using only existing PNGs and a CPU-only first-2,048 replay."
        )
    )
    parser.add_argument("--quality-bridge-root", type=Path, required=True)
    parser.add_argument("--training-checkout", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--expected-decision-sha256", required=True)
    parser.add_argument("--quality-result", type=Path, required=True)
    parser.add_argument("--expected-quality-result-sha256", required=True)
    parser.add_argument("--expected-evaluator-revision", required=True)
    parser.add_argument("--expected-evaluator-tree", required=True)
    parser.add_argument("--expected-evaluator-branch", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--torch-num-threads", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.torch_num_threads < 1:
        raise ValueError("batch size and torch thread count must be positive")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1"}:
        raise ValueError("cross-protocol reconciliation requires CUDA to be hidden")
    output_dir = reject_symlink_chain(args.output_dir, name="reconciliation output")
    if output_dir.exists() and not args.resume:
        raise FileExistsError(
            f"cross-protocol reconciliation output already exists: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / REPORT_FILENAME
    status_path = output_dir / STATUS_FILENAME
    started_at = _utc_now()
    torch.set_num_threads(args.torch_num_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    with exclusive_output_lock(output_dir, role=REPORT_ROLE):
        heartbeat = StatusHeartbeat(status_path, started_at=started_at)
        heartbeat.start()
        try:
            heartbeat.set_phase("verifying_source_evidence")
            evidence = collect_source_evidence(args)
            if report_path.is_file():
                if not args.resume:
                    raise FileExistsError(
                        f"cross-protocol reconciliation report exists: {report_path}"
                    )
                _validate_completed_final(report_path, evidence=evidence)
                heartbeat.stop()
                write_json_report(
                    status_path,
                    {
                        **heartbeat._payload(status="completed"),
                        "phase": "completed",
                        "detail": "existing_report_physically_revalidated",
                        "report": file_identity(report_path),
                    },
                )
                print(f"reused {report_path}")
                return
            evaluator_version = torch_fidelity_version()
            if evaluator_version != TORCH_FIDELITY_VERSION:
                raise ValueError("torch-fidelity 0.4.0 is required")
            evaluator_environment = capture_runtime_environment(
                torch.device("cpu"), project_root=PROJECT_ROOT
            )
            evaluator_environment_sha = runtime_environment_sha256(
                evaluator_environment
            )
            cache_root = reject_symlink_chain(
                args.cache_root, name="torch-fidelity cache root"
            )
            if not cache_root.is_dir():
                raise FileNotFoundError(f"torch-fidelity cache root is missing: {cache_root}")
            effective_cache_name = content_addressed_real_cache_name(
                REAL_CACHE_BASE, evidence["real_set"]["sha256"]
            )
            subsets: dict[str, dict[str, Any]] = {}
            for method in METHOD_LAYOUTS:
                heartbeat.set_phase("preparing_existing_png_subset", method)
                subsets[method] = prepare_hardlink_subset(
                    evidence["methods"][method]["protocols"][
                        "ddim100_first_2048"
                    ]["source_images"],
                    target_dir=output_dir / "subsets" / method / "ddim100_first_2048",
                )
                if (
                    subsets[method]["source_sample_set_sha256"]
                    != evidence["methods"][method]["protocols"][
                        "ddim100_first_2048"
                    ]["source_sample_set_sha256"]
                ):
                    raise ValueError(f"{method} staged subset source digest differs")
            pixel_differences: dict[str, dict[str, Any]] = {}
            for method in METHOD_LAYOUTS:
                heartbeat.set_phase("computing_paired_pixel_difference", method)
                pixel_differences[method] = paired_pixel_difference(
                    evidence["methods"][method]["protocols"]["ddim50_2048"][
                        "images"
                    ],
                    subsets[method]["images"],
                )
            metric_dir = output_dir / "metrics"
            metric_dir.mkdir(parents=True, exist_ok=True)
            subset_metrics: dict[
                str, tuple[dict[str, Any], dict[str, Any]]
            ] = {}
            for method in METHOD_LAYOUTS:
                heartbeat.set_phase("evaluating_ddim100_first_2048_cpu", method)
                subset_metrics[method] = evaluate_subset_metrics(
                    method=method,
                    subset=subsets[method],
                    source_sampling_identity=evidence["methods"][method]["protocols"][
                        "ddim100_10000"
                    ]["sampling_provenance"]["report_identity"],
                    real_set=evidence["real_set"],
                    evaluator_git=evidence["evaluator_git"],
                    evaluator_environment=evaluator_environment,
                    evaluator_environment_sha256=evaluator_environment_sha,
                    batch_size=args.batch_size,
                    cache_root=cache_root,
                    real_cache_name=effective_cache_name,
                    output=metric_dir / f"{method}_ddim100_first_2048_metrics.json",
                )
            heartbeat.set_phase("building_reconciliation_report")
            report = build_final_report(
                evidence=evidence,
                subsets=subsets,
                subset_metrics=subset_metrics,
                pixel_differences=pixel_differences,
                evaluator_environment=evaluator_environment,
                evaluator_environment_sha256=evaluator_environment_sha,
                started_at=started_at,
            )
            write_json_report(report_path, report)
            _validate_completed_final(report_path, evidence=evidence)
            heartbeat.stop()
            write_json_report(
                status_path,
                {
                    **heartbeat._payload(status="completed"),
                    "phase": "completed",
                    "detail": report["comparison"]["ranking_reversal_explanation"],
                    "report": file_identity(report_path),
                },
            )
            print(report["comparison"]["ranking_reversal_explanation"])
        except BaseException as error:
            heartbeat.stop()
            write_json_report(
                status_path,
                {
                    **heartbeat._payload(status="failed"),
                    "phase": "failed",
                    "detail": str(error),
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
            raise


if __name__ == "__main__":
    main()
