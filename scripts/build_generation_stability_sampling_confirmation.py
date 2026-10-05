from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from cofitok.reporting import file_sha256, write_json_report

try:
    import build_generation_stability_sampling_recovery as recovery
except ModuleNotFoundError:  # Imported as scripts.<module> by tests and library callers.
    from scripts import build_generation_stability_sampling_recovery as recovery


METHODS = recovery.METHODS
EXPECTED_ROLE = "non_authorizing_matched_10000_sampling_confirmation"
EXPECTED_CLAIM_BOUNDARY = {
    "confirmation_non_authorizing": True,
    "selected_from_predeclared_shared_sweep": True,
    "replaces_frozen_promotion_gate": False,
    "confirmation_report_is_promotion_gate": False,
    "new_gate_required": True,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
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


def _source_matches(declared: Any, path: Path) -> bool:
    if not isinstance(declared, dict):
        return False
    actual = _source(path)
    return all(declared.get(name) == actual[name] for name in ("bytes", "sha256"))


def _relocatable_evidence(value: Any) -> Any:
    if isinstance(value, dict):
        normalized = {
            key: _relocatable_evidence(child)
            for key, child in value.items()
            if not (
                key == "path"
                and {"path", "bytes", "sha256"}.issubset(value)
            )
        }
        return normalized
    if isinstance(value, list):
        return [_relocatable_evidence(child) for child in value]
    return value


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
    label: str,
) -> None:
    if identity != {
        "revision": expected_revision,
        "branch": expected_branch,
        "tracked_dirty": False,
    }:
        raise ValueError(f"{label} Git identity mismatch")


def _validate_recovery_summary(
    *,
    summary: dict[str, Any],
    summary_path: Path,
    current_preflight: dict[str, Any],
    plan: dict[str, Any],
    plan_path: Path,
    promotion_gate_path: Path,
    formal_paths: dict[str, Path],
    expected_recovery_git: dict[str, Any],
) -> dict[str, Any]:
    preflight = summary.get("preflight")
    sweep = summary.get("sweep")
    selection = summary.get("selection")
    if (
        summary.get("schema_version") != 1
        or summary.get("status") != "complete"
        or summary.get("role") != recovery.EXPECTED_ROLE
        or summary.get("diagnostic_git") != expected_recovery_git
        or summary.get("claim_boundary") != recovery.EXPECTED_CLAIM_BOUNDARY
        or not isinstance(preflight, dict)
        or preflight.get("status") != "pass"
        or preflight.get("role") != "sampling_recovery_preflight"
        or preflight.get("diagnostic_git") != expected_recovery_git
        or preflight.get("selection_policy") != recovery.EXPECTED_SELECTION_POLICY
        or preflight.get("claim_boundary") != recovery.EXPECTED_CLAIM_BOUNDARY
        or preflight.get("expected_case_count") != current_preflight["expected_case_count"]
        or preflight.get("cases") != current_preflight["cases"]
        or not _source_matches(preflight.get("plan"), plan_path)
        or not _source_matches(preflight.get("promotion_gate"), promotion_gate_path)
        or not isinstance(sweep, dict)
        or int(sweep.get("sample_count_per_case", -1))
        != int(plan["diagnostic"]["sample_count_per_case"])
        or int(sweep.get("sample_steps", -1))
        != int(plan["diagnostic"]["sample_steps"])
        or not isinstance(sweep.get("rows"), list)
        or len(sweep["rows"]) != current_preflight["expected_case_count"]
        or not isinstance(selection, dict)
        or selection.get("per_method_protocol_selection_allowed") is not False
        or selection.get("automatic_formal_protocol_change_allowed") is not False
        or selection.get(
            "matched_10000_confirmation_required_before_any_protocol_change"
        )
        is not True
    ):
        raise ValueError("sampling-confirmation recovery summary contract mismatch")
    for method in METHODS:
        declared = preflight.get("formal_reference", {}).get(method, {}).get("source")
        if not _source_matches(declared, formal_paths[method]):
            raise ValueError(
                f"sampling-confirmation formal {method} source identity changed"
            )
        current = current_preflight["formal_reference"][method]
        observed = preflight["formal_reference"][method]
        for name in (
            "checkpoint_sha256",
            "checkpoint_step",
            "sample_set_sha256",
            "sample_count",
            "fid",
            "inception_score_mean",
            "precision",
            "recall",
        ):
            if observed.get(name) != current.get(name):
                raise ValueError(
                    f"sampling-confirmation formal {method} reference changed"
                )
    shared = selection.get("shared_protocol")
    if (
        not isinstance(shared, dict)
        or shared.get("policy") != recovery.EXPECTED_SELECTION_POLICY
        or shared.get("formal_protocol_baseline_case")
        != recovery.EXPECTED_SELECTION_POLICY["baseline_case_id"]
        or shared.get("selected_is_formal_protocol_baseline") is not False
        or shared.get("eligible_for_matched_10000_confirmation") is not True
        or not isinstance(shared.get("selected_case"), str)
        or not isinstance(shared.get("selected_protocol"), dict)
        or not isinstance(shared.get("ranked_cases"), list)
        or not shared["ranked_cases"]
        or shared["ranked_cases"][0].get("case") != shared["selected_case"]
        or shared["ranked_cases"][0].get("strictly_improves_both_methods") is not True
    ):
        raise ValueError("sampling-confirmation shared selection is not eligible")
    selected_case = next(
        (
            case
            for case in plan["diagnostic"]["cases"]
            if case["id"] == shared["selected_case"]
        ),
        None,
    )
    if (
        not isinstance(selected_case, dict)
        or shared["selected_protocol"]
        != {
            "guidance_scale": float(selected_case["guidance_scale"]),
            "guidance_rescale": float(selected_case["guidance_rescale"]),
        }
    ):
        raise ValueError("sampling-confirmation selected protocol differs from its case")
    return {
        "source": _source(summary_path),
        "diagnostic_git": expected_recovery_git,
        "selected_case": shared["selected_case"],
        "selected_protocol": shared["selected_protocol"],
        "selection_evidence": shared["ranked_cases"][0],
    }


def build_preflight(
    *,
    project: Path,
    plan: dict[str, Any],
    plan_path: Path,
    recovery_summary: dict[str, Any],
    recovery_summary_path: Path,
    recovery_case_root: Path,
    promotion_gate: dict[str, Any],
    promotion_gate_path: Path,
    formal_metrics: dict[str, tuple[dict[str, Any], Path]],
    confirmation_git: dict[str, Any],
    expected_recovery_revision: str,
    expected_recovery_branch: str,
    expected_confirmation_revision: str,
    expected_confirmation_branch: str,
) -> dict[str, Any]:
    del project
    recovery.validate_plan(plan)
    _require_git_identity(
        confirmation_git,
        expected_revision=expected_confirmation_revision,
        expected_branch=expected_confirmation_branch,
        label="sampling-confirmation",
    )
    recovery_git = {
        "revision": expected_recovery_revision,
        "branch": expected_recovery_branch,
        "tracked_dirty": False,
    }
    if recovery_git != confirmation_git:
        raise ValueError(
            "sampling-confirmation requires the exact recovery evaluator Git identity"
        )
    current_recovery_preflight = recovery.build_preflight(
        plan=plan,
        plan_path=plan_path,
        promotion_gate=promotion_gate,
        promotion_gate_path=promotion_gate_path,
        formal_metrics=formal_metrics,
        diagnostic_git=recovery_git,
        expected_diagnostic_revision=expected_recovery_revision,
        expected_diagnostic_branch=expected_recovery_branch,
    )
    selected = _validate_recovery_summary(
        summary=recovery_summary,
        summary_path=recovery_summary_path,
        current_preflight=current_recovery_preflight,
        plan=plan,
        plan_path=plan_path,
        promotion_gate_path=promotion_gate_path,
        formal_paths={method: formal_metrics[method][1] for method in METHODS},
        expected_recovery_git=recovery_git,
    )
    rebuilt_recovery_summary = recovery.build_report(
        preflight=current_recovery_preflight,
        plan=plan,
        case_root=recovery_case_root,
        diagnostic_git=recovery_git,
    )
    if _relocatable_evidence(rebuilt_recovery_summary) != _relocatable_evidence(
        recovery_summary
    ):
        raise ValueError(
            "sampling-confirmation recovery summary differs from physical case reports"
        )
    thresholds = promotion_gate.get("thresholds")
    if (
        not isinstance(thresholds, dict)
        or not isinstance(thresholds.get("max_absolute_fid"), (int, float))
        or not math.isfinite(float(thresholds["max_absolute_fid"]))
        or float(thresholds["max_absolute_fid"]) <= 0.0
        or not isinstance(thresholds.get("max_fid_regression"), (int, float))
        or not 0.0 <= float(thresholds["max_fid_regression"]) <= 1.0
    ):
        raise ValueError("sampling-confirmation quality thresholds are invalid")
    protocol = {
        "num_samples": 10_000,
        "sample_steps": int(plan["diagnostic"]["sample_steps"]),
        "sampling_batch_size": int(plan["diagnostic"]["batch_size"]),
        "metrics_batch_size": 64,
        "prc_batch_size": 10_000,
        "metrics_seed": 2027,
        "seed": int(plan["diagnostic"]["seed"]),
        "start_index": int(plan["diagnostic"]["start_index"]),
        "class_schedule": plan["diagnostic"]["class_schedule"],
        "guidance_scale": float(selected["selected_protocol"]["guidance_scale"]),
        "guidance_rescale": float(
            selected["selected_protocol"]["guidance_rescale"]
        ),
        "cfg_batch_mode": plan["diagnostic"]["cfg_batch_mode"],
        "eta": float(plan["diagnostic"]["eta"]),
        "clip_x0": plan["diagnostic"]["clip_x0"],
        "precision": plan["diagnostic"]["precision"],
        "weights": plan["diagnostic"]["weights"],
        "sampler": "ddim",
        "protocol_schema": "cofitok_ddim_sampling_v1",
        "precision_recall_enabled": True,
    }
    return {
        "schema_version": 1,
        "status": "pass",
        "role": "sampling_confirmation_preflight",
        "confirmation_git": confirmation_git,
        "recovery": selected,
        "sources": {
            "plan": _source(plan_path),
            "promotion_gate": _source(promotion_gate_path),
            "formal_metrics": {
                method: _source(formal_metrics[method][1]) for method in METHODS
            },
        },
        "methods": {
            method: {
                "checkpoint_step": int(plan["methods"][method]["checkpoint_step"]),
                "checkpoint_sha256": plan["methods"][method]["checkpoint_sha256"],
                "checkpoint_integrity_sha256": plan["methods"][method][
                    "checkpoint_integrity_sha256"
                ],
                "prefix_budget": int(plan["methods"][method]["prefix_budget"]),
                "formal_reference": current_recovery_preflight["formal_reference"][
                    method
                ],
            }
            for method in METHODS
        },
        "real_set": plan["source"]["real_set"],
        "sampling_runtime_environment_sha256": plan["source"][
            "sampling_runtime_environment_sha256"
        ],
        "metrics_runtime_environment_sha256": plan["source"][
            "metrics_runtime_environment_sha256"
        ],
        "evaluator": plan["source"]["evaluator"],
        "protocol": protocol,
        "thresholds": {
            "max_absolute_fid": float(thresholds["max_absolute_fid"]),
            "max_fid_regression": float(thresholds["max_fid_regression"]),
        },
        "claim_boundary": EXPECTED_CLAIM_BOUNDARY,
    }


def _validate_sampling(
    sampling: Any,
    *,
    preflight: dict[str, Any],
    method: str,
) -> None:
    protocol = preflight["protocol"]
    expected = {
        "num_samples": int(protocol["num_samples"]),
        "sample_steps": int(protocol["sample_steps"]),
        "seed": int(protocol["seed"]),
        "start_index": int(protocol["start_index"]),
        "class_schedule": protocol["class_schedule"],
        "guidance_scale": float(protocol["guidance_scale"]),
        "guidance_rescale": float(protocol["guidance_rescale"]),
        "cfg_batch_mode": protocol["cfg_batch_mode"],
        "eta": float(protocol["eta"]),
        "clip_x0": protocol["clip_x0"],
        "precision": protocol["precision"],
        "sampler": protocol["sampler"],
        "protocol_schema": protocol["protocol_schema"],
        "prefix_budgets": [int(preflight["methods"][method]["prefix_budget"])],
    }
    if (
        not isinstance(sampling, dict)
        or any(sampling.get(name) != value for name, value in expected.items())
        or not recovery._sampling_stream_is_exact(sampling)
    ):
        raise ValueError(f"sampling-confirmation {method} sampling protocol mismatch")


def _validate_method(
    *,
    preflight: dict[str, Any],
    method: str,
    case_root: Path,
    confirmation_git: dict[str, Any],
) -> dict[str, Any]:
    method_root = case_root / method
    sampling_path = method_root / "sampling_report.json"
    metrics_path = method_root / "metrics" / "generation_metrics_report.json"
    if not sampling_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError(f"sampling-confirmation {method} output is incomplete")
    sampling_report = _read_object(sampling_path)
    metrics_report = _read_object(metrics_path)
    method_plan = preflight["methods"][method]
    protocol = preflight["protocol"]
    sampling = sampling_report.get("sampling")
    _validate_sampling(sampling, preflight=preflight, method=method)
    budget_key = str(int(method_plan["prefix_budget"]))
    sample_sets = sampling_report.get("sample_sets")
    selected_sample_set = (
        sample_sets.get(budget_key) if isinstance(sample_sets, dict) else None
    )
    elapsed_seconds = float(sampling_report.get("elapsed_seconds", math.nan))
    if (
        sampling_report.get("schema_version") != 6
        or sampling_report.get("status") != "completed"
        or sampling_report.get("git") != confirmation_git
        or sampling_report.get("weights") != "ema"
        or sampling_report.get("checkpoint_sha256")
        != method_plan["checkpoint_sha256"]
        or int(sampling_report.get("checkpoint_step", -1))
        != int(method_plan["checkpoint_step"])
        or sampling_report.get("runtime_environment_sha256")
        != preflight["sampling_runtime_environment_sha256"]
        or not isinstance(selected_sample_set, dict)
        or int(selected_sample_set.get("count", -1)) != int(protocol["num_samples"])
        or not recovery._is_sha256(selected_sample_set.get("sha256"))
        or not math.isfinite(elapsed_seconds)
        or elapsed_seconds <= 0.0
    ):
        raise ValueError(f"sampling-confirmation {method} sampling report mismatch")
    provenance = metrics_report.get("sample_provenance")
    counts = metrics_report.get("counts")
    parameters = metrics_report.get("parameters")
    metrics = metrics_report.get("metrics")
    if (
        metrics_report.get("schema_version") != 3
        or metrics_report.get("role") != "generation_directory_metrics_report"
        or metrics_report.get("protocol") != "torch_fidelity_directory_metrics"
        or metrics_report.get("status") != "completed"
        or metrics_report.get("git") != confirmation_git
        or metrics_report.get("implementation") != preflight["evaluator"]
        or metrics_report.get("runtime_environment_sha256")
        != preflight["metrics_runtime_environment_sha256"]
        or not recovery._matches_bound_real_set(
            metrics_report.get("real_set"), preflight["real_set"]
        )
        or not isinstance(parameters, dict)
        or parameters.get("precision_recall_enabled") is not True
        or int(parameters.get("batch_size", -1))
        != int(protocol["metrics_batch_size"])
        or int(parameters.get("prc_batch_size", -1))
        != int(protocol["prc_batch_size"])
        or int(parameters.get("min_samples", -1)) != int(protocol["num_samples"])
        or int(parameters.get("seed", -1)) != int(protocol["metrics_seed"])
        or parameters.get("samples_find_deep") is not True
        or parameters.get("samples_shuffle") is not False
        or not isinstance(counts, dict)
        or int(counts.get("generated_image_count", -1))
        != int(protocol["num_samples"])
        or int(counts.get("real_image_count", -1))
        != int(preflight["real_set"]["image_count"])
        or not isinstance(provenance, dict)
        or provenance.get("git") != confirmation_git
        or provenance.get("checkpoint_sha256")
        != method_plan["checkpoint_sha256"]
        or int(provenance.get("checkpoint_step", -1))
        != int(method_plan["checkpoint_step"])
        or provenance.get("runtime_environment_sha256")
        != preflight["sampling_runtime_environment_sha256"]
        or provenance.get("selected_prefix_budget")
        != int(method_plan["prefix_budget"])
        or provenance.get("weights") != "ema"
        or provenance.get("sampling") != sampling
        or provenance.get("sample_set_sha256") != selected_sample_set["sha256"]
        or not isinstance(metrics, dict)
    ):
        raise ValueError(f"sampling-confirmation {method} metrics report mismatch")
    values = {
        name: float(metrics.get(name, math.nan))
        for name in (
            "frechet_inception_distance",
            "inception_score_mean",
            "inception_score_std",
            "precision",
            "recall",
        )
    }
    if (
        not all(math.isfinite(value) for value in values.values())
        or values["frechet_inception_distance"] <= 0.0
        or values["inception_score_mean"] <= 0.0
        or values["inception_score_std"] < 0.0
        or not 0.0 <= values["precision"] <= 1.0
        or not 0.0 <= values["recall"] <= 1.0
    ):
        raise ValueError(f"sampling-confirmation {method} metrics are invalid")
    return {
        "method": method,
        "checkpoint_sha256": method_plan["checkpoint_sha256"],
        "checkpoint_step": method_plan["checkpoint_step"],
        "prefix_budget": method_plan["prefix_budget"],
        "sample_count": protocol["num_samples"],
        "sample_set_sha256": selected_sample_set["sha256"],
        "sampling_elapsed_seconds": elapsed_seconds,
        "metrics": values,
        "sampling_report": _source(sampling_path),
        "metrics_report": _source(metrics_path),
    }


def build_report(
    *,
    preflight: dict[str, Any],
    case_root: Path,
    confirmation_git: dict[str, Any],
) -> dict[str, Any]:
    if (
        preflight.get("schema_version") != 1
        or preflight.get("status") != "pass"
        or preflight.get("role") != "sampling_confirmation_preflight"
        or preflight.get("confirmation_git") != confirmation_git
        or preflight.get("claim_boundary") != EXPECTED_CLAIM_BOUNDARY
    ):
        raise ValueError("sampling-confirmation preflight contract mismatch")
    rows = {
        method: _validate_method(
            preflight=preflight,
            method=method,
            case_root=case_root,
            confirmation_git=confirmation_git,
        )
        for method in METHODS
    }
    cofitok_fid = rows["cofitok"]["metrics"]["frechet_inception_distance"]
    dense_fid = rows["dense_identity"]["metrics"]["frechet_inception_distance"]
    cofitok_formal_fid = float(
        preflight["methods"]["cofitok"]["formal_reference"]["fid"]
    )
    dense_formal_fid = float(
        preflight["methods"]["dense_identity"]["formal_reference"]["fid"]
    )
    max_absolute = float(preflight["thresholds"]["max_absolute_fid"])
    max_regression = float(preflight["thresholds"]["max_fid_regression"])
    checks = {
        "cofitok_absolute_fid_quality": cofitok_fid <= max_absolute,
        "cofitok_within_candidate_dense_tolerance": (
            cofitok_fid / dense_fid - 1.0 <= max_regression
        ),
        "cofitok_strictly_improves_frozen_formal_fid": (
            cofitok_fid < cofitok_formal_fid
        ),
        "dense_strictly_improves_frozen_formal_fid": dense_fid < dense_formal_fid,
    }
    confirmed = all(checks.values())
    return {
        "schema_version": 1,
        "status": "pass" if confirmed else "hold",
        "role": EXPECTED_ROLE,
        "decision": (
            "candidate_quality_confirmed_non_authorizing"
            if confirmed
            else "candidate_quality_not_confirmed"
        ),
        "confirmation_git": confirmation_git,
        "preflight": preflight,
        "rows": rows,
        "comparison": {
            "selected_case": preflight["recovery"]["selected_case"],
            "selected_protocol": preflight["recovery"]["selected_protocol"],
            "cofitok_fid": cofitok_fid,
            "dense_fid": dense_fid,
            "cofitok_relative_to_dense": cofitok_fid / dense_fid - 1.0,
            "cofitok_fid_relative_to_frozen_formal": (
                cofitok_fid / cofitok_formal_fid - 1.0
            ),
            "dense_fid_relative_to_frozen_formal": (
                dense_fid / dense_formal_fid - 1.0
            ),
        },
        "quality_checks": checks,
        "quality_confirmed": confirmed,
        "next_boundary": {
            "candidate_protocol_formalized": False,
            "separate_new_gate_required": True,
            "full_training_launch_allowed": False,
            "full_300k_launch_allowed": False,
        },
        "claim_boundary": EXPECTED_CLAIM_BOUNDARY,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate or summarize a non-authorizing matched 10K confirmation "
            "of the predeclared shared sampling-recovery candidate."
        )
    )
    parser.add_argument("--mode", choices=["preflight", "build"], required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--recovery-plan", type=Path, required=True)
    parser.add_argument("--recovery-summary", type=Path, required=True)
    parser.add_argument("--recovery-case-root", type=Path, required=True)
    parser.add_argument("--promotion-gate", type=Path, required=True)
    parser.add_argument("--cofitok-formal-metrics", type=Path, required=True)
    parser.add_argument("--dense-formal-metrics", type=Path, required=True)
    parser.add_argument("--case-root", type=Path)
    parser.add_argument("--expected-recovery-revision", required=True)
    parser.add_argument("--expected-recovery-branch", required=True)
    parser.add_argument("--expected-confirmation-revision", required=True)
    parser.add_argument("--expected-confirmation-branch", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    project = args.project.resolve()
    plan_path = args.recovery_plan.resolve()
    summary_path = args.recovery_summary.resolve()
    gate_path = args.promotion_gate.resolve()
    formal_paths = {
        "cofitok": args.cofitok_formal_metrics.resolve(),
        "dense_identity": args.dense_formal_metrics.resolve(),
    }
    confirmation_git = _git_identity(project)
    preflight = build_preflight(
        project=project,
        plan=_read_object(plan_path),
        plan_path=plan_path,
        recovery_summary=_read_object(summary_path),
        recovery_summary_path=summary_path,
        recovery_case_root=args.recovery_case_root.resolve(),
        promotion_gate=_read_object(gate_path),
        promotion_gate_path=gate_path,
        formal_metrics={
            method: (_read_object(path), path) for method, path in formal_paths.items()
        },
        confirmation_git=confirmation_git,
        expected_recovery_revision=args.expected_recovery_revision,
        expected_recovery_branch=args.expected_recovery_branch,
        expected_confirmation_revision=args.expected_confirmation_revision,
        expected_confirmation_branch=args.expected_confirmation_branch,
    )
    if args.mode == "preflight":
        payload = preflight
    else:
        if args.case_root is None:
            raise ValueError("--case-root is required in build mode")
        payload = build_report(
            preflight=preflight,
            case_root=args.case_root.resolve(),
            confirmation_git=confirmation_git,
        )
    write_json_report(args.output, payload)
    print(json.dumps({"output": str(args.output), "status": payload["status"]}))


if __name__ == "__main__":
    main()
