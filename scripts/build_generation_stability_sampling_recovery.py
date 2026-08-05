from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from cofitok.reporting import file_sha256, write_json_report


METHODS = ("cofitok", "dense_identity")
EXPECTED_ROLE = "non_authorizing_matched_inference_protocol_diagnostic"
EXPECTED_CLAIM_BOUNDARY = {
    "diagnostic_non_authorizing": True,
    "small_sample_fid_is_formal": False,
    "replaces_frozen_promotion_gate": False,
    "formal_protocol_change_allowed": False,
    "matched_10000_confirmation_required": True,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
}
EXPECTED_RANDOM_STREAM = {
    "scope": "per_global_sample_index",
    "seed_formula": "(seed + global_index) mod 2^63",
    "prefix_budgets_share_stream": True,
    "batch_size_invariant": True,
    "resume_index_invariant": True,
}
EXPECTED_SAMPLE_SET_DIGEST = {
    "algorithm": "sha256",
    "framing": "filename_utf8_nul_file_bytes_nul",
}
RANDOM_SEED_MODULUS = 2**63
EXPECTED_DIAGNOSTIC_KEYS = {
    "sample_count_per_case",
    "sample_steps",
    "batch_size",
    "metrics_batch_size",
    "metrics_seed",
    "seed",
    "start_index",
    "class_count",
    "balanced_modulo_exact_coverage_required",
    "class_schedule",
    "cfg_batch_mode",
    "eta",
    "clip_x0",
    "precision",
    "weights",
    "skip_precision_recall",
    "cases",
}
EXPECTED_CONFIRMATION_KEYS = {
    "sample_count",
    "seed",
    "start_index",
    "class_count",
    "balanced_modulo_exact_coverage_required",
    "disjoint_from_diagnostic_required",
    "disjoint_from_frozen_formal_required",
    "precision_recall_enabled",
}
EXPECTED_SELECTION_POLICY = {
    "shared_protocol_required": True,
    "baseline_case_id": "cfg150_r000",
    "objective": "minimize_worst_method_fid_ratio_to_shared_baseline",
    "eligibility": "strict_fid_improvement_for_both_methods",
    "tie_break": [
        "worst_method_ratio",
        "prefer_formal_baseline",
        "mean_method_ratio",
        "case_id",
    ],
    "per_method_protocol_selection_allowed": False,
    "automatic_formal_protocol_change_allowed": False,
}


def _exact_nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _random_stream_window(
    *,
    label: str,
    seed: Any,
    start_index: Any,
    sample_count: Any,
) -> dict[str, Any]:
    exact_seed = _exact_nonnegative_int(seed, label=f"{label} seed")
    exact_start = _exact_nonnegative_int(
        start_index, label=f"{label} start_index"
    )
    exact_count = _exact_nonnegative_int(
        sample_count, label=f"{label} sample_count"
    )
    if exact_count == 0:
        raise ValueError(f"{label} sample_count must be positive")
    index_stop = exact_start + exact_count
    derived_seed_start = exact_seed + exact_start
    derived_seed_stop = derived_seed_start + exact_count
    if derived_seed_stop > RANDOM_SEED_MODULUS:
        raise ValueError(f"{label} random stream must not wrap modulo 2^63")
    return {
        "seed": exact_seed,
        "start_index": exact_start,
        "sample_count": exact_count,
        "global_index_start_inclusive": exact_start,
        "global_index_stop_exclusive": index_stop,
        "derived_seed_start_inclusive": derived_seed_start,
        "derived_seed_stop_exclusive": derived_seed_stop,
        "wraparound": False,
    }


def _half_open_intervals_are_disjoint(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    start_key: str,
    stop_key: str,
) -> bool:
    return bool(
        int(left[stop_key]) <= int(right[start_key])
        or int(right[stop_key]) <= int(left[start_key])
    )


def _random_stream_disjointness(
    windows: dict[str, dict[str, Any]],
) -> dict[str, bool]:
    confirmation = windows["confirmation"]
    diagnostic = windows["selection_diagnostic"]
    formal = windows["frozen_formal"]
    return {
        "confirmation_global_indices_disjoint_from_selection_diagnostic": (
            _half_open_intervals_are_disjoint(
                confirmation,
                diagnostic,
                start_key="global_index_start_inclusive",
                stop_key="global_index_stop_exclusive",
            )
        ),
        "confirmation_derived_seeds_disjoint_from_selection_diagnostic": (
            _half_open_intervals_are_disjoint(
                confirmation,
                diagnostic,
                start_key="derived_seed_start_inclusive",
                stop_key="derived_seed_stop_exclusive",
            )
        ),
        "confirmation_global_indices_disjoint_from_frozen_formal": (
            _half_open_intervals_are_disjoint(
                confirmation,
                formal,
                start_key="global_index_start_inclusive",
                stop_key="global_index_stop_exclusive",
            )
        ),
        "confirmation_derived_seeds_disjoint_from_frozen_formal": (
            _half_open_intervals_are_disjoint(
                confirmation,
                formal,
                start_key="derived_seed_start_inclusive",
                stop_key="derived_seed_stop_exclusive",
            )
        ),
    }


def validate_random_stream_independence(
    evidence: Any,
    *,
    confirmation_protocol: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if (
        not isinstance(evidence, dict)
        or set(evidence)
        != {"schema_version", "seed_formula", "seed_modulus", "windows", "checks"}
        or evidence.get("schema_version") != 1
        or evidence.get("seed_formula") != EXPECTED_RANDOM_STREAM["seed_formula"]
        or evidence.get("seed_modulus") != RANDOM_SEED_MODULUS
    ):
        raise ValueError("sampling-confirmation random-stream evidence is malformed")
    declared_windows = evidence.get("windows")
    if not isinstance(declared_windows, dict) or set(declared_windows) != {
        "selection_diagnostic",
        "frozen_formal",
        "confirmation",
    }:
        raise ValueError("sampling-confirmation random-stream windows are malformed")
    windows: dict[str, dict[str, Any]] = {}
    for name, declared in declared_windows.items():
        if not isinstance(declared, dict):
            raise ValueError("sampling-confirmation random-stream window is malformed")
        rebuilt = _random_stream_window(
            label=name,
            seed=declared.get("seed"),
            start_index=declared.get("start_index"),
            sample_count=declared.get("sample_count"),
        )
        if declared != rebuilt:
            raise ValueError("sampling-confirmation random-stream window drifted")
        windows[name] = rebuilt
    expected_checks = _random_stream_disjointness(windows)
    if evidence.get("checks") != expected_checks or not all(expected_checks.values()):
        raise ValueError("sampling-confirmation random streams are not independent")
    if confirmation_protocol is not None:
        confirmation = windows["confirmation"]
        if any(
            confirmation[name] != int(confirmation_protocol.get(protocol_name, -1))
            for name, protocol_name in (
                ("seed", "seed"),
                ("start_index", "start_index"),
                ("sample_count", "num_samples"),
            )
        ):
            raise ValueError(
                "sampling-confirmation protocol differs from its random-stream evidence"
            )
    return evidence


def build_random_stream_independence(plan: dict[str, Any]) -> dict[str, Any]:
    source = plan["source"]
    formal = source["formal_protocol"]
    diagnostic = plan["diagnostic"]
    confirmation = plan["confirmation"]
    windows = {
        "selection_diagnostic": _random_stream_window(
            label="selection diagnostic",
            seed=diagnostic.get("seed"),
            start_index=diagnostic.get("start_index"),
            sample_count=diagnostic.get("sample_count_per_case"),
        ),
        "frozen_formal": _random_stream_window(
            label="frozen formal",
            seed=formal.get("seed"),
            start_index=formal.get("start_index"),
            sample_count=source.get("formal_sample_count"),
        ),
        "confirmation": _random_stream_window(
            label="confirmation",
            seed=confirmation.get("seed"),
            start_index=confirmation.get("start_index"),
            sample_count=confirmation.get("sample_count"),
        ),
    }
    evidence = {
        "schema_version": 1,
        "seed_formula": EXPECTED_RANDOM_STREAM["seed_formula"],
        "seed_modulus": RANDOM_SEED_MODULUS,
        "windows": windows,
        "checks": _random_stream_disjointness(windows),
    }
    return validate_random_stream_independence(evidence)


def _diagnostic_protocol(plan: dict[str, Any]) -> dict[str, Any]:
    diagnostic = plan["diagnostic"]
    return {
        "num_samples": int(diagnostic["sample_count_per_case"]),
        "sample_steps": int(diagnostic["sample_steps"]),
        "sampling_batch_size": int(diagnostic["batch_size"]),
        "metrics_batch_size": int(diagnostic["metrics_batch_size"]),
        "metrics_seed": int(diagnostic["metrics_seed"]),
        "seed": int(diagnostic["seed"]),
        "start_index": int(diagnostic["start_index"]),
        "class_count": int(diagnostic["class_count"]),
        "balanced_modulo_exact_coverage_required": diagnostic[
            "balanced_modulo_exact_coverage_required"
        ],
        "class_schedule": diagnostic["class_schedule"],
        "cfg_batch_mode": diagnostic["cfg_batch_mode"],
        "eta": float(diagnostic["eta"]),
        "clip_x0": diagnostic["clip_x0"],
        "precision": diagnostic["precision"],
        "weights": diagnostic["weights"],
        "precision_recall_enabled": not diagnostic["skip_precision_recall"],
    }


def _read_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _source(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": resolved.as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": file_sha256(resolved),
    }


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _matches_bound_real_set(actual: Any, expected: dict[str, Any]) -> bool:
    """Match the immutable real-set identity while allowing report-only metadata."""
    if not isinstance(actual, dict):
        return False
    return all(
        actual.get(name) == expected.get(name)
        for name in ("digest_schema", "image_count", "sha256")
    )


def _sampling_stream_is_exact(sampling: dict[str, Any]) -> bool:
    return (
        sampling.get("random_stream") == EXPECTED_RANDOM_STREAM
        and sampling.get("sample_set_digest") == EXPECTED_SAMPLE_SET_DIGEST
    )


def _git_identity(project: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(project), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "revision": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "tracked_dirty": bool(
            run("status", "--porcelain", "--untracked-files=no")
        ),
    }


def _require_git_identity(
    identity: dict[str, Any],
    *,
    expected_revision: str,
    expected_branch: str,
) -> None:
    if identity != {
        "revision": expected_revision,
        "branch": expected_branch,
        "tracked_dirty": False,
    }:
        raise ValueError("sampling-recovery diagnostic Git identity mismatch")


def validate_plan(plan: dict[str, Any]) -> dict[str, Any]:
    if (
        plan.get("schema_version") != 1
        or plan.get("role") != EXPECTED_ROLE
        or plan.get("source_profile") != "stability_scaling"
        or plan.get("selection_policy") != EXPECTED_SELECTION_POLICY
        or plan.get("claim_boundary") != EXPECTED_CLAIM_BOUNDARY
    ):
        raise ValueError("sampling-recovery plan contract mismatch")
    source = plan.get("source")
    methods = plan.get("methods")
    diagnostic = plan.get("diagnostic")
    confirmation = plan.get("confirmation")
    if (
        not isinstance(source, dict)
        or not isinstance(methods, dict)
        or set(methods) != set(METHODS)
        or not isinstance(diagnostic, dict)
        or not isinstance(confirmation, dict)
    ):
        raise ValueError("sampling-recovery plan sections are incomplete")
    for digest_name in (
        "promotion_gate_sha256",
        "sampling_runtime_environment_sha256",
        "metrics_runtime_environment_sha256",
    ):
        if not _is_sha256(source.get(digest_name)):
            raise ValueError(f"sampling-recovery source {digest_name} is invalid")
    if (
        source.get("required_gate_status") != "fail"
        or source.get("required_gate_decision") != "hold"
        or source.get("required_failed_gates") != ["absolute_fid_quality"]
        or int(source.get("formal_sample_count", -1)) != 10_000
    ):
        raise ValueError("sampling-recovery source gate contract is invalid")
    provenance = source.get("provenance_contract")
    real_set = source.get("real_set")
    evaluator = source.get("evaluator")
    if (
        not isinstance(provenance, dict)
        or set(provenance)
        != {
            "training_revision",
            "training_branch",
            "evaluation_revision",
            "evaluation_branch",
        }
        or not isinstance(real_set, dict)
        or real_set.get("digest_schema") != "cofitok_image_tree_sha256_v1"
        or int(real_set.get("image_count", -1)) != 50_000
        or not _is_sha256(real_set.get("sha256"))
        or evaluator != {"package": "torch_fidelity", "version": "0.4.0"}
    ):
        raise ValueError("sampling-recovery source provenance is invalid")
    if set(diagnostic) != EXPECTED_DIAGNOSTIC_KEYS:
        raise ValueError("sampling-recovery diagnostic protocol fields are invalid")
    diagnostic_count = _exact_nonnegative_int(
        diagnostic.get("sample_count_per_case"),
        label="diagnostic sample_count_per_case",
    )
    diagnostic_steps = _exact_nonnegative_int(
        diagnostic.get("sample_steps"), label="diagnostic sample_steps"
    )
    diagnostic_batch = _exact_nonnegative_int(
        diagnostic.get("batch_size"), label="diagnostic batch_size"
    )
    diagnostic_metrics_batch = _exact_nonnegative_int(
        diagnostic.get("metrics_batch_size"),
        label="diagnostic metrics_batch_size",
    )
    diagnostic_metrics_seed = _exact_nonnegative_int(
        diagnostic.get("metrics_seed"), label="diagnostic metrics_seed"
    )
    diagnostic_seed = _exact_nonnegative_int(
        diagnostic.get("seed"), label="diagnostic seed"
    )
    diagnostic_start = _exact_nonnegative_int(
        diagnostic.get("start_index"), label="diagnostic start_index"
    )
    diagnostic_class_count = _exact_nonnegative_int(
        diagnostic.get("class_count"), label="diagnostic class_count"
    )
    cases = diagnostic.get("cases")
    if (
        diagnostic_count != 1_000
        or diagnostic_steps != 100
        or diagnostic_batch != 32
        or diagnostic_metrics_batch != 64
        or diagnostic_metrics_seed != 2027
        or diagnostic_seed != 0
        or diagnostic_start != 0
        or diagnostic_class_count != 1_000
        or diagnostic_count != diagnostic_class_count
        or diagnostic_start % diagnostic_class_count != 0
        or diagnostic.get("balanced_modulo_exact_coverage_required") is not True
        or diagnostic.get("class_schedule") != "balanced_modulo"
        or diagnostic.get("cfg_batch_mode") != "batched"
        or diagnostic.get("eta") != 0.0
        or diagnostic.get("clip_x0") is not True
        or diagnostic.get("precision") != "bf16"
        or diagnostic.get("weights") != "ema"
        or diagnostic.get("skip_precision_recall") is not True
        or not isinstance(cases, list)
        or len(cases) < 2
    ):
        raise ValueError("sampling-recovery diagnostic protocol is invalid")
    case_ids: list[str] = []
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("sampling-recovery case is malformed")
        case_id = case.get("id")
        scale = case.get("guidance_scale")
        rescale = case.get("guidance_rescale")
        if (
            not isinstance(case_id, str)
            or not case_id
            or not isinstance(scale, (int, float))
            or not math.isfinite(float(scale))
            or float(scale) < 0.0
            or not isinstance(rescale, (int, float))
            or not math.isfinite(float(rescale))
            or not 0.0 <= float(rescale) <= 1.0
        ):
            raise ValueError("sampling-recovery case values are invalid")
        case_ids.append(case_id)
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("sampling-recovery case identifiers are not unique")
    baseline_case = next(
        (
            case
            for case in cases
            if case["id"] == EXPECTED_SELECTION_POLICY["baseline_case_id"]
        ),
        None,
    )
    formal_protocol = source.get("formal_protocol")
    if (
        not isinstance(baseline_case, dict)
        or not isinstance(formal_protocol, dict)
        or float(baseline_case["guidance_scale"])
        != float(formal_protocol.get("guidance_scale", math.nan))
        or float(baseline_case["guidance_rescale"])
        != float(formal_protocol.get("guidance_rescale", math.nan))
    ):
        raise ValueError(
            "sampling-recovery shared baseline differs from the frozen formal protocol"
        )
    if set(confirmation) != EXPECTED_CONFIRMATION_KEYS:
        raise ValueError("sampling-confirmation protocol fields are invalid")
    confirmation_count = _exact_nonnegative_int(
        confirmation.get("sample_count"),
        label="confirmation sample_count",
    )
    confirmation_class_count = _exact_nonnegative_int(
        confirmation.get("class_count"),
        label="confirmation class_count",
    )
    confirmation_start = _exact_nonnegative_int(
        confirmation.get("start_index"),
        label="confirmation start_index",
    )
    if (
        confirmation_count != int(source["formal_sample_count"])
        or confirmation_class_count != 1_000
        or confirmation_count % confirmation_class_count != 0
        or confirmation_start % confirmation_class_count != 0
        or confirmation.get("balanced_modulo_exact_coverage_required") is not True
        or confirmation.get("disjoint_from_diagnostic_required") is not True
        or confirmation.get("disjoint_from_frozen_formal_required") is not True
        or confirmation.get("precision_recall_enabled") is not True
    ):
        raise ValueError("sampling-confirmation protocol is invalid")
    build_random_stream_independence(plan)
    for method in METHODS:
        row = methods[method]
        if (
            not isinstance(row, dict)
            or int(row.get("checkpoint_step", -1)) != 50_000
            or int(row.get("prefix_budget", -1)) < 1
            or not _is_sha256(row.get("checkpoint_sha256"))
            or not _is_sha256(row.get("checkpoint_integrity_sha256"))
            or not _is_sha256(row.get("formal_metrics_sha256"))
            or not _is_sha256(row.get("formal_sample_set_sha256"))
            or not isinstance(row.get("formal_fid"), (int, float))
            or not math.isfinite(float(row["formal_fid"]))
        ):
            raise ValueError(f"sampling-recovery method {method} is invalid")
    return plan


def _validate_sampling(
    sampling: dict[str, Any],
    *,
    plan: dict[str, Any],
    method: str,
    guidance_scale: float,
    guidance_rescale: float,
    sample_count: int,
) -> None:
    diagnostic = plan["diagnostic"]
    expected = {
        "num_samples": sample_count,
        "batch_size": int(diagnostic["batch_size"]),
        "sample_steps": int(diagnostic["sample_steps"]),
        "seed": int(diagnostic["seed"]),
        "start_index": int(diagnostic["start_index"]),
        "num_classes": int(diagnostic["class_count"]),
        "class_schedule": diagnostic["class_schedule"],
        "guidance_scale": float(guidance_scale),
        "guidance_rescale": float(guidance_rescale),
        "cfg_batch_mode": diagnostic["cfg_batch_mode"],
        "eta": float(diagnostic["eta"]),
        "clip_x0": diagnostic["clip_x0"],
        "precision": diagnostic["precision"],
        "sampler": "ddim",
        "protocol_schema": "cofitok_ddim_sampling_v1",
        "prefix_budgets": [int(plan["methods"][method]["prefix_budget"])],
    }
    if not isinstance(sampling, dict) or any(
        sampling.get(name) != value for name, value in expected.items()
    ):
        raise ValueError(f"sampling-recovery {method} sampling protocol mismatch")
    if not _sampling_stream_is_exact(sampling):
        raise ValueError(f"sampling-recovery {method} random stream is invalid")


def _validate_formal_metrics(
    payload: dict[str, Any],
    *,
    path: Path,
    plan: dict[str, Any],
    method: str,
) -> dict[str, Any]:
    source = plan["source"]
    method_plan = plan["methods"][method]
    provenance = payload.get("sample_provenance")
    metrics = payload.get("metrics")
    counts = payload.get("counts")
    expected_git = {
        "revision": source["provenance_contract"]["evaluation_revision"],
        "branch": source["provenance_contract"]["evaluation_branch"],
        "tracked_dirty": False,
    }
    if (
        file_sha256(path) != method_plan["formal_metrics_sha256"]
        or payload.get("schema_version") != 2
        or payload.get("status") != "completed"
        or payload.get("git") != expected_git
        or payload.get("implementation") != source["evaluator"]
        or payload.get("runtime_environment_sha256")
        != source["metrics_runtime_environment_sha256"]
        or not isinstance(counts, dict)
        or int(counts.get("generated_image_count", -1))
        != int(source["formal_sample_count"])
        or int(counts.get("real_image_count", -1))
        != int(source["real_set"]["image_count"])
        or not _matches_bound_real_set(payload.get("real_set"), source["real_set"])
        or not isinstance(provenance, dict)
        or provenance.get("git") != expected_git
        or provenance.get("checkpoint_sha256")
        != method_plan["checkpoint_sha256"]
        or int(provenance.get("checkpoint_step", -1))
        != int(method_plan["checkpoint_step"])
        or provenance.get("sample_set_sha256")
        != method_plan["formal_sample_set_sha256"]
        or provenance.get("runtime_environment_sha256")
        != source["sampling_runtime_environment_sha256"]
        or provenance.get("selected_prefix_budget")
        != int(method_plan["prefix_budget"])
        or provenance.get("weights") != "ema"
        or not isinstance(metrics, dict)
        or float(metrics.get("frechet_inception_distance", math.nan))
        != float(method_plan["formal_fid"])
    ):
        raise ValueError(f"formal {method} metrics contract mismatch")
    formal_protocol = source["formal_protocol"]
    sampling = provenance.get("sampling")
    if not isinstance(sampling, dict):
        raise ValueError(f"formal {method} sampling provenance is missing")
    expected_formal = {
        "num_samples": int(source["formal_sample_count"]),
        "sample_steps": int(formal_protocol["sample_steps"]),
        "seed": int(formal_protocol["seed"]),
        "start_index": int(formal_protocol["start_index"]),
        "class_schedule": formal_protocol["class_schedule"],
        "guidance_scale": float(formal_protocol["guidance_scale"]),
        "guidance_rescale": float(formal_protocol["guidance_rescale"]),
        "cfg_batch_mode": formal_protocol["cfg_batch_mode"],
        "eta": float(formal_protocol["eta"]),
        "clip_x0": formal_protocol["clip_x0"],
        "precision": formal_protocol["precision"],
        "sampler": formal_protocol["sampler"],
        "protocol_schema": "cofitok_ddim_sampling_v1",
        "prefix_budgets": [int(method_plan["prefix_budget"])],
    }
    if any(
        sampling.get(name) != value for name, value in expected_formal.items()
    ) or not _sampling_stream_is_exact(sampling):
        raise ValueError(f"formal {method} sampling protocol mismatch")
    return {
        "source": _source(path),
        "checkpoint_sha256": method_plan["checkpoint_sha256"],
        "checkpoint_step": method_plan["checkpoint_step"],
        "sample_set_sha256": method_plan["formal_sample_set_sha256"],
        "sample_count": source["formal_sample_count"],
        "fid": float(metrics["frechet_inception_distance"]),
        "inception_score_mean": float(metrics["inception_score_mean"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
    }


def build_preflight(
    *,
    plan: dict[str, Any],
    plan_path: Path,
    promotion_gate: dict[str, Any],
    promotion_gate_path: Path,
    formal_metrics: dict[str, tuple[dict[str, Any], Path]],
    diagnostic_git: dict[str, Any],
    expected_diagnostic_revision: str,
    expected_diagnostic_branch: str,
) -> dict[str, Any]:
    validate_plan(plan)
    _require_git_identity(
        diagnostic_git,
        expected_revision=expected_diagnostic_revision,
        expected_branch=expected_diagnostic_branch,
    )
    source = plan["source"]
    failed_gates = sorted(
        str(row.get("name"))
        for row in promotion_gate.get("gates", [])
        if isinstance(row, dict) and row.get("passed") is False
    )
    if (
        file_sha256(promotion_gate_path) != source["promotion_gate_sha256"]
        or promotion_gate.get("schema_version") != 2
        or promotion_gate.get("status") != source["required_gate_status"]
        or promotion_gate.get("decision") != source["required_gate_decision"]
        or promotion_gate.get("stage") != "scaling"
        or promotion_gate.get("source_profile") != plan["source_profile"]
        or promotion_gate.get("provenance_contract")
        != source["provenance_contract"]
        or failed_gates != sorted(source["required_failed_gates"])
    ):
        raise ValueError("sampling-recovery promotion gate contract mismatch")
    formal = {
        method: _validate_formal_metrics(
            formal_metrics[method][0],
            path=formal_metrics[method][1],
            plan=plan,
            method=method,
        )
        for method in METHODS
    }
    cases = [
        {
            "id": case["id"],
            "guidance_scale": float(case["guidance_scale"]),
            "guidance_rescale": float(case["guidance_rescale"]),
            "formal_protocol_baseline": (
                case["id"] == plan["selection_policy"]["baseline_case_id"]
            ),
        }
        for case in plan["diagnostic"]["cases"]
    ]
    return {
        "schema_version": 1,
        "status": "pass",
        "role": "sampling_recovery_preflight",
        "plan": _source(plan_path),
        "promotion_gate": _source(promotion_gate_path),
        "gate_failure_is_exact_absolute_fid_only": True,
        "diagnostic_git": diagnostic_git,
        "formal_reference": formal,
        "expected_case_count": len(cases) * len(METHODS),
        "cases": cases,
        "diagnostic_protocol": _diagnostic_protocol(plan),
        "selection_policy": plan["selection_policy"],
        "claim_boundary": plan["claim_boundary"],
    }


def _validate_case(
    *,
    plan: dict[str, Any],
    method: str,
    case: dict[str, Any],
    case_root: Path,
    diagnostic_git: dict[str, Any],
) -> dict[str, Any]:
    case_dir = case_root / f"{method}_{case['id']}"
    sampling_path = case_dir / "sampling_report.json"
    metrics_path = case_dir / "metrics" / "generation_metrics_report.json"
    if not sampling_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError(f"sampling-recovery case is incomplete: {case_dir}")
    sampling_report = _read_object(sampling_path)
    metrics_report = _read_object(metrics_path)
    diagnostic = plan["diagnostic"]
    source = plan["source"]
    method_plan = plan["methods"][method]
    _validate_sampling(
        sampling_report.get("sampling"),
        plan=plan,
        method=method,
        guidance_scale=float(case["guidance_scale"]),
        guidance_rescale=float(case["guidance_rescale"]),
        sample_count=int(diagnostic["sample_count_per_case"]),
    )
    if (
        sampling_report.get("schema_version") != 6
        or sampling_report.get("status") != "completed"
        or sampling_report.get("weights") != "ema"
        or sampling_report.get("git") != diagnostic_git
        or sampling_report.get("checkpoint_sha256")
        != method_plan["checkpoint_sha256"]
        or int(sampling_report.get("checkpoint_step", -1))
        != int(method_plan["checkpoint_step"])
        or sampling_report.get("runtime_environment_sha256")
        != source["sampling_runtime_environment_sha256"]
    ):
        raise ValueError(f"sampling-recovery {method}/{case['id']} report mismatch")
    provenance = metrics_report.get("sample_provenance")
    counts = metrics_report.get("counts")
    metrics = metrics_report.get("metrics")
    parameters = metrics_report.get("parameters")
    budget_key = str(int(method_plan["prefix_budget"]))
    sample_sets = sampling_report.get("sample_sets")
    selected_sample_set = (
        sample_sets.get(budget_key) if isinstance(sample_sets, dict) else None
    )
    elapsed_seconds = float(sampling_report.get("elapsed_seconds", math.nan))
    if (
        metrics_report.get("schema_version") != 3
        or metrics_report.get("role") != "generation_directory_metrics_report"
        or metrics_report.get("protocol") != "torch_fidelity_directory_metrics"
        or metrics_report.get("status") != "completed"
        or metrics_report.get("git") != diagnostic_git
        or metrics_report.get("implementation") != source["evaluator"]
        or metrics_report.get("runtime_environment_sha256")
        != source["metrics_runtime_environment_sha256"]
        or not _matches_bound_real_set(
            metrics_report.get("real_set"), source["real_set"]
        )
        or not isinstance(parameters, dict)
        or parameters.get("precision_recall_enabled") is not False
        or int(parameters.get("batch_size", -1))
        != int(diagnostic["metrics_batch_size"])
        or int(parameters.get("min_samples", -1))
        != int(diagnostic["sample_count_per_case"])
        or int(parameters.get("seed", -1)) != int(diagnostic["metrics_seed"])
        or parameters.get("samples_find_deep") is not True
        or parameters.get("samples_shuffle") is not False
        or not isinstance(counts, dict)
        or int(counts.get("generated_image_count", -1))
        != int(diagnostic["sample_count_per_case"])
        or int(counts.get("real_image_count", -1))
        != int(source["real_set"]["image_count"])
        or not isinstance(provenance, dict)
        or provenance.get("git") != diagnostic_git
        or provenance.get("checkpoint_sha256")
        != method_plan["checkpoint_sha256"]
        or int(provenance.get("checkpoint_step", -1))
        != int(method_plan["checkpoint_step"])
        or provenance.get("runtime_environment_sha256")
        != source["sampling_runtime_environment_sha256"]
        or provenance.get("selected_prefix_budget")
        != int(method_plan["prefix_budget"])
        or provenance.get("weights") != "ema"
        or provenance.get("sampling") != sampling_report.get("sampling")
        or not _is_sha256(provenance.get("sample_set_sha256"))
        or not isinstance(selected_sample_set, dict)
        or int(selected_sample_set.get("count", -1))
        != int(diagnostic["sample_count_per_case"])
        or selected_sample_set.get("sha256") != provenance.get("sample_set_sha256")
        or not math.isfinite(elapsed_seconds)
        or elapsed_seconds <= 0.0
        or not isinstance(metrics, dict)
    ):
        raise ValueError(f"sampling-recovery {method}/{case['id']} metrics mismatch")
    values = {
        name: float(metrics.get(name, math.nan))
        for name in (
            "frechet_inception_distance",
            "inception_score_mean",
            "inception_score_std",
        )
    }
    if (
        not all(math.isfinite(value) for value in values.values())
        or values["frechet_inception_distance"] <= 0.0
    ):
        raise ValueError(f"sampling-recovery {method}/{case['id']} metrics are invalid")
    return {
        "case": case["id"],
        "method": method,
        "guidance_scale": float(case["guidance_scale"]),
        "guidance_rescale": float(case["guidance_rescale"]),
        "sample_count": int(diagnostic["sample_count_per_case"]),
        "sample_steps": int(diagnostic["sample_steps"]),
        "fid": values["frechet_inception_distance"],
        "inception_score_mean": values["inception_score_mean"],
        "inception_score_std": values["inception_score_std"],
        "sample_set_sha256": provenance["sample_set_sha256"],
        "sampling_elapsed_seconds": elapsed_seconds,
        "sampling_report": _source(sampling_path),
        "metrics_report": _source(metrics_path),
    }


def build_report(
    *,
    preflight: dict[str, Any],
    plan: dict[str, Any],
    case_root: Path,
    diagnostic_git: dict[str, Any],
) -> dict[str, Any]:
    validate_plan(plan)
    expected_cases = [
        {
            "id": case["id"],
            "guidance_scale": float(case["guidance_scale"]),
            "guidance_rescale": float(case["guidance_rescale"]),
            "formal_protocol_baseline": (
                case["id"] == plan["selection_policy"]["baseline_case_id"]
            ),
        }
        for case in plan["diagnostic"]["cases"]
    ]
    if (
        preflight.get("schema_version") != 1
        or preflight.get("status") != "pass"
        or preflight.get("role") != "sampling_recovery_preflight"
        or preflight.get("diagnostic_git") != diagnostic_git
        or preflight.get("expected_case_count")
        != len(expected_cases) * len(METHODS)
        or preflight.get("cases") != expected_cases
        or preflight.get("diagnostic_protocol") != _diagnostic_protocol(plan)
        or preflight.get("selection_policy") != EXPECTED_SELECTION_POLICY
        or preflight.get("claim_boundary") != EXPECTED_CLAIM_BOUNDARY
    ):
        raise ValueError("sampling-recovery preflight contract mismatch")
    rows = [
        _validate_case(
            plan=plan,
            method=method,
            case=case,
            case_root=case_root,
            diagnostic_git=diagnostic_git,
        )
        for method in METHODS
        for case in plan["diagnostic"]["cases"]
    ]
    if len(rows) != int(preflight["expected_case_count"]):
        raise ValueError("sampling-recovery completed case count differs")
    rankings = {
        method: [
            row["case"]
            for row in sorted(
                (candidate for candidate in rows if candidate["method"] == method),
                key=lambda candidate: candidate["fid"],
            )
        ]
        for method in METHODS
    }
    paired = []
    for case in plan["diagnostic"]["cases"]:
        indexed = {
            row["method"]: row
            for row in rows
            if row["case"] == case["id"]
        }
        cofitok_fid = float(indexed["cofitok"]["fid"])
        dense_fid = float(indexed["dense_identity"]["fid"])
        paired.append(
            {
                "case": case["id"],
                "cofitok_fid": cofitok_fid,
                "dense_fid": dense_fid,
                "cofitok_relative_to_dense": cofitok_fid / dense_fid - 1.0,
            }
        )
    policy = plan["selection_policy"]
    baseline_case_id = policy["baseline_case_id"]
    paired_by_case = {row["case"]: row for row in paired}
    baseline = paired_by_case.get(baseline_case_id)
    if (
        not isinstance(baseline, dict)
        or float(baseline["cofitok_fid"]) <= 0.0
        or float(baseline["dense_fid"]) <= 0.0
    ):
        raise ValueError("sampling-recovery shared baseline is missing or invalid")
    shared_rows = []
    for row in paired:
        cofitok_ratio = float(row["cofitok_fid"]) / float(baseline["cofitok_fid"])
        dense_ratio = float(row["dense_fid"]) / float(baseline["dense_fid"])
        worst_ratio = max(cofitok_ratio, dense_ratio)
        mean_ratio = (cofitok_ratio + dense_ratio) / 2.0
        shared_rows.append(
            {
                **row,
                "cofitok_fid_ratio_to_shared_baseline": cofitok_ratio,
                "dense_fid_ratio_to_shared_baseline": dense_ratio,
                "worst_method_fid_ratio_to_shared_baseline": worst_ratio,
                "mean_method_fid_ratio_to_shared_baseline": mean_ratio,
                "strictly_improves_both_methods": (
                    cofitok_ratio < 1.0 and dense_ratio < 1.0
                ),
            }
        )
    shared_ranked = sorted(
        shared_rows,
        key=lambda row: (
            row["worst_method_fid_ratio_to_shared_baseline"],
            0 if row["case"] == baseline_case_id else 1,
            row["mean_method_fid_ratio_to_shared_baseline"],
            row["case"],
        ),
    )
    shared_selected = shared_ranked[0]
    selected_case = next(
        case
        for case in plan["diagnostic"]["cases"]
        if case["id"] == shared_selected["case"]
    )
    eligible_for_confirmation = (
        shared_selected["case"] != baseline_case_id
        and shared_selected["strictly_improves_both_methods"] is True
    )
    return {
        "schema_version": 1,
        "status": "complete",
        "role": EXPECTED_ROLE,
        "diagnostic_git": diagnostic_git,
        "preflight": preflight,
        "sweep": {
            "sample_count_per_case": plan["diagnostic"]["sample_count_per_case"],
            "sample_steps": plan["diagnostic"]["sample_steps"],
            "rows": rows,
            "rankings_by_diagnostic_fid": rankings,
            "paired_cases": paired,
        },
        "selection": {
            "exploratory_best_case_by_method": {
                method: rankings[method][0] for method in METHODS
            },
            "per_method_protocol_selection_allowed": False,
            "shared_protocol": {
                "policy": policy,
                "formal_protocol_baseline_case": baseline_case_id,
                "ranked_cases": shared_ranked,
                "selected_case": shared_selected["case"],
                "selected_protocol": {
                    "guidance_scale": float(selected_case["guidance_scale"]),
                    "guidance_rescale": float(selected_case["guidance_rescale"]),
                },
                "selected_is_formal_protocol_baseline": (
                    shared_selected["case"] == baseline_case_id
                ),
                "eligible_for_matched_10000_confirmation": eligible_for_confirmation,
            },
            "automatic_formal_protocol_change_allowed": False,
            "matched_10000_confirmation_required_before_any_protocol_change": True,
        },
        "limitations": [
            (
                f"The {plan['diagnostic']['sample_count_per_case']}-sample FID "
                "and Inception scores are noisy configuration-selection diagnostics."
            ),
            "Rows are comparable only within this exact matched sweep and sample count.",
            "Per-method rankings are exploratory and cannot select different formal protocols.",
            "No row replaces the frozen 10K gate or authorizes full 300K training.",
        ],
        "claim_boundary": plan["claim_boundary"],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate or summarize the frozen matched sampling-recovery sweep."
    )
    parser.add_argument("--mode", choices=["preflight", "build"], required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--promotion-gate", type=Path, required=True)
    parser.add_argument("--cofitok-formal-metrics", type=Path, required=True)
    parser.add_argument("--dense-formal-metrics", type=Path, required=True)
    parser.add_argument("--case-root", type=Path)
    parser.add_argument("--expected-diagnostic-revision", required=True)
    parser.add_argument("--expected-diagnostic-branch", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    project = args.project.resolve()
    plan_path = args.plan.resolve()
    gate_path = args.promotion_gate.resolve()
    formal_paths = {
        "cofitok": args.cofitok_formal_metrics.resolve(),
        "dense_identity": args.dense_formal_metrics.resolve(),
    }
    diagnostic_git = _git_identity(project)
    plan = _read_object(plan_path)
    preflight = build_preflight(
        plan=plan,
        plan_path=plan_path,
        promotion_gate=_read_object(gate_path),
        promotion_gate_path=gate_path,
        formal_metrics={
            method: (_read_object(path), path)
            for method, path in formal_paths.items()
        },
        diagnostic_git=diagnostic_git,
        expected_diagnostic_revision=args.expected_diagnostic_revision,
        expected_diagnostic_branch=args.expected_diagnostic_branch,
    )
    if args.mode == "preflight":
        payload = preflight
    else:
        if args.case_root is None:
            raise ValueError("--case-root is required in build mode")
        payload = build_report(
            preflight=preflight,
            plan=plan,
            case_root=args.case_root.resolve(),
            diagnostic_git=diagnostic_git,
        )
    write_json_report(args.output, payload)
    print(json.dumps({"output": str(args.output), "status": payload["status"]}))


if __name__ == "__main__":
    main()
