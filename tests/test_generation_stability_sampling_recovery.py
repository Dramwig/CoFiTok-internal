from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from cofitok.reporting import file_sha256
from scripts import build_generation_stability_sampling_confirmation as confirmation
from scripts import build_generation_stability_sampling_recovery as recovery


TRAINING_REVISION = "1" * 40
TRAINING_BRANCH = "scale/generation-stability-50k-preflight"
EVALUATION_REVISION = "2" * 40
EVALUATION_BRANCH = "scale/generation-stability-50k-posteval-v4"
DIAGNOSTIC_REVISION = "3" * 40
DIAGNOSTIC_BRANCH = "scale/generation-large-capacity"
SAMPLING_RUNTIME = "4" * 64
METRICS_RUNTIME = "5" * 64
REAL_SET_SHA256 = "6" * 64
CONFIRMATION_REVISION = DIAGNOSTIC_REVISION


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _git(
    revision: str = DIAGNOSTIC_REVISION,
    branch: str = DIAGNOSTIC_BRANCH,
) -> dict:
    return {"revision": revision, "branch": branch, "tracked_dirty": False}


def _sampling(
    *,
    sample_count: int,
    sample_steps: int,
    prefix_budget: int,
    guidance_scale: float,
    guidance_rescale: float,
) -> dict:
    return {
        "num_samples": sample_count,
        "sample_steps": sample_steps,
        "seed": 0,
        "start_index": 0,
        "class_schedule": "balanced_modulo",
        "guidance_scale": guidance_scale,
        "guidance_rescale": guidance_rescale,
        "cfg_batch_mode": "batched",
        "eta": 0.0,
        "clip_x0": True,
        "precision": "bf16",
        "sampler": "ddim",
        "protocol_schema": "cofitok_ddim_sampling_v1",
        "prefix_budgets": [prefix_budget],
        "random_stream": deepcopy(recovery.EXPECTED_RANDOM_STREAM),
        "sample_set_digest": deepcopy(recovery.EXPECTED_SAMPLE_SET_DIGEST),
    }


def _plan() -> dict:
    cases = [
        {"id": "cfg100_r000", "guidance_scale": 1.0, "guidance_rescale": 0.0},
        {"id": "cfg125_r000", "guidance_scale": 1.25, "guidance_rescale": 0.0},
        {"id": "cfg150_r000", "guidance_scale": 1.5, "guidance_rescale": 0.0},
        {"id": "cfg150_r050", "guidance_scale": 1.5, "guidance_rescale": 0.5},
        {"id": "cfg150_r100", "guidance_scale": 1.5, "guidance_rescale": 1.0},
    ]
    return {
        "schema_version": 1,
        "name": "stability_50k_sampling_recovery_v1",
        "role": recovery.EXPECTED_ROLE,
        "source_profile": "stability_scaling",
        "source": {
            "promotion_gate_sha256": "0" * 64,
            "required_gate_status": "fail",
            "required_gate_decision": "hold",
            "required_failed_gates": ["absolute_fid_quality"],
            "provenance_contract": {
                "training_revision": TRAINING_REVISION,
                "training_branch": TRAINING_BRANCH,
                "evaluation_revision": EVALUATION_REVISION,
                "evaluation_branch": EVALUATION_BRANCH,
            },
            "real_set": {
                "digest_schema": "cofitok_image_tree_sha256_v1",
                "image_count": 50_000,
                "sha256": REAL_SET_SHA256,
            },
            "formal_sample_count": 10_000,
            "formal_protocol": {
                "sample_steps": 100,
                "sampler": "ddim",
                "seed": 0,
                "start_index": 0,
                "class_schedule": "balanced_modulo",
                "guidance_scale": 1.5,
                "guidance_rescale": 0.0,
                "cfg_batch_mode": "batched",
                "eta": 0.0,
                "clip_x0": True,
                "precision": "bf16",
                "weights": "ema",
            },
            "sampling_runtime_environment_sha256": SAMPLING_RUNTIME,
            "metrics_runtime_environment_sha256": METRICS_RUNTIME,
            "evaluator": {"package": "torch_fidelity", "version": "0.4.0"},
        },
        "methods": {
            "cofitok": {
                "checkpoint_step": 50_000,
                "checkpoint_sha256": _sha("cofitok-checkpoint"),
                "checkpoint_integrity_sha256": _sha("cofitok-integrity"),
                "prefix_budget": 8,
                "formal_metrics_sha256": "0" * 64,
                "formal_sample_set_sha256": _sha("cofitok-formal-samples"),
                "formal_fid": 138.0,
            },
            "dense_identity": {
                "checkpoint_step": 50_000,
                "checkpoint_sha256": _sha("dense-checkpoint"),
                "checkpoint_integrity_sha256": _sha("dense-integrity"),
                "prefix_budget": 1,
                "formal_metrics_sha256": "0" * 64,
                "formal_sample_set_sha256": _sha("dense-formal-samples"),
                "formal_fid": 151.0,
            },
        },
        "diagnostic": {
            "sample_count_per_case": 512,
            "sample_steps": 100,
            "batch_size": 32,
            "seed": 0,
            "start_index": 0,
            "class_schedule": "balanced_modulo",
            "cfg_batch_mode": "batched",
            "eta": 0.0,
            "clip_x0": True,
            "precision": "bf16",
            "weights": "ema",
            "skip_precision_recall": True,
            "cases": cases,
        },
        "selection_policy": deepcopy(recovery.EXPECTED_SELECTION_POLICY),
        "claim_boundary": deepcopy(recovery.EXPECTED_CLAIM_BOUNDARY),
    }


def _promotion_gate(plan: dict) -> dict:
    return {
        "schema_version": 2,
        "status": "fail",
        "decision": "hold",
        "stage": "scaling",
        "source_profile": "stability_scaling",
        "provenance_contract": deepcopy(plan["source"]["provenance_contract"]),
        "thresholds": {
            "max_absolute_fid": 100.0,
            "max_fid_regression": 0.05,
        },
        "gates": [
            {"name": "training_complete", "passed": True},
            {"name": "absolute_fid_quality", "passed": False},
        ],
    }


def _formal_metrics(plan: dict, method: str) -> dict:
    method_plan = plan["methods"][method]
    source = plan["source"]
    sampling = _sampling(
        sample_count=10_000,
        sample_steps=100,
        prefix_budget=int(method_plan["prefix_budget"]),
        guidance_scale=1.5,
        guidance_rescale=0.0,
    )
    return {
        "schema_version": 2,
        "status": "completed",
        "git": _git(EVALUATION_REVISION, EVALUATION_BRANCH),
        "implementation": deepcopy(source["evaluator"]),
        "runtime_environment_sha256": METRICS_RUNTIME,
        "parameters": {"precision_recall_enabled": True},
        "counts": {"generated_image_count": 10_000, "real_image_count": 50_000},
        "real_set": {
            **source["real_set"],
            "root": "/datasets/imagenet_256/val",
            "real_cache_name": "imagenet256-val-bound-cache",
        },
        "sample_provenance": {
            "git": _git(EVALUATION_REVISION, EVALUATION_BRANCH),
            "checkpoint_sha256": method_plan["checkpoint_sha256"],
            "checkpoint_step": 50_000,
            "sample_set_sha256": method_plan["formal_sample_set_sha256"],
            "runtime_environment_sha256": SAMPLING_RUNTIME,
            "selected_prefix_budget": method_plan["prefix_budget"],
            "weights": "ema",
            "sampling": sampling,
        },
        "metrics": {
            "frechet_inception_distance": method_plan["formal_fid"],
            "inception_score_mean": 8.0 if method == "cofitok" else 7.0,
            "inception_score_std": 0.2,
            "precision": 0.7,
            "recall": 0.01,
        },
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path) -> dict:
    plan = _plan()
    gate = _promotion_gate(plan)
    gate_path = tmp_path / "promotion_gate.json"
    _write_json(gate_path, gate)
    plan["source"]["promotion_gate_sha256"] = file_sha256(gate_path)

    formal_payloads = {}
    formal_paths = {}
    for method in recovery.METHODS:
        payload = _formal_metrics(plan, method)
        path = tmp_path / f"{method}_formal_metrics.json"
        _write_json(path, payload)
        plan["methods"][method]["formal_metrics_sha256"] = file_sha256(path)
        formal_payloads[method] = payload
        formal_paths[method] = path

    plan_path = tmp_path / "plan.json"
    _write_json(plan_path, plan)
    return {
        "plan": plan,
        "plan_path": plan_path,
        "gate": gate,
        "gate_path": gate_path,
        "formal_payloads": formal_payloads,
        "formal_paths": formal_paths,
        "diagnostic_git": _git(),
    }


def _build_preflight(fixture: dict) -> dict:
    return recovery.build_preflight(
        plan=fixture["plan"],
        plan_path=fixture["plan_path"],
        promotion_gate=fixture["gate"],
        promotion_gate_path=fixture["gate_path"],
        formal_metrics={
            method: (fixture["formal_payloads"][method], fixture["formal_paths"][method])
            for method in recovery.METHODS
        },
        diagnostic_git=fixture["diagnostic_git"],
        expected_diagnostic_revision=DIAGNOSTIC_REVISION,
        expected_diagnostic_branch=DIAGNOSTIC_BRANCH,
    )


def _case_reports(
    plan: dict,
    *,
    method: str,
    case: dict,
    fid: float,
) -> tuple[dict, dict]:
    method_plan = plan["methods"][method]
    sample_sha256 = _sha(f"{method}-{case['id']}-samples")
    sampling = _sampling(
        sample_count=512,
        sample_steps=100,
        prefix_budget=int(method_plan["prefix_budget"]),
        guidance_scale=float(case["guidance_scale"]),
        guidance_rescale=float(case["guidance_rescale"]),
    )
    sampling_report = {
        "schema_version": 6,
        "status": "completed",
        "git": _git(),
        "runtime_environment_sha256": SAMPLING_RUNTIME,
        "checkpoint_sha256": method_plan["checkpoint_sha256"],
        "checkpoint_step": 50_000,
        "checkpoint_integrity_manifest": f"/{method}/checkpoint_step_00050000.pt.integrity.json",
        "weights": "ema",
        "sampling": sampling,
        "sample_sets": {
            str(method_plan["prefix_budget"]): {"count": 512, "sha256": sample_sha256}
        },
        "elapsed_seconds": 100.0,
    }
    metrics_report = {
        "schema_version": 3,
        "role": "generation_directory_metrics_report",
        "protocol": "torch_fidelity_directory_metrics",
        "status": "completed",
        "git": _git(),
        "implementation": deepcopy(plan["source"]["evaluator"]),
        "runtime_environment_sha256": METRICS_RUNTIME,
        "parameters": {"precision_recall_enabled": False},
        "counts": {"generated_image_count": 512, "real_image_count": 50_000},
        "real_set": {
            **plan["source"]["real_set"],
            "root": "/datasets/imagenet_256/val",
        },
        "sample_provenance": {
            "git": _git(),
            "checkpoint_sha256": method_plan["checkpoint_sha256"],
            "checkpoint_step": 50_000,
            "checkpoint_integrity_manifest": sampling_report[
                "checkpoint_integrity_manifest"
            ],
            "runtime_environment_sha256": SAMPLING_RUNTIME,
            "selected_prefix_budget": method_plan["prefix_budget"],
            "weights": "ema",
            "sampling": deepcopy(sampling),
            "sample_set_sha256": sample_sha256,
        },
        "metrics": {
            "frechet_inception_distance": fid,
            "inception_score_mean": 8.0,
            "inception_score_std": 0.2,
        },
    }
    return sampling_report, metrics_report


def _write_cases(fixture: dict, case_root: Path) -> None:
    cofitok_fids = [120.0, 108.0, 101.0, 95.0, 112.0]
    dense_fids = [130.0, 126.0, 122.0, 115.0, 110.0]
    for method, fids in (
        ("cofitok", cofitok_fids),
        ("dense_identity", dense_fids),
    ):
        for case, fid in zip(fixture["plan"]["diagnostic"]["cases"], fids):
            sampling, metrics = _case_reports(
                fixture["plan"], method=method, case=case, fid=fid
            )
            case_dir = case_root / f"{method}_{case['id']}"
            _write_json(case_dir / "sampling_report.json", sampling)
            _write_json(case_dir / "metrics" / "generation_metrics_report.json", metrics)


def test_preflight_accepts_bound_real_set_with_report_metadata(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    report = _build_preflight(fixture)

    assert report["status"] == "pass"
    assert report["expected_case_count"] == 10
    assert report["gate_failure_is_exact_absolute_fid_only"] is True
    assert report["formal_reference"]["cofitok"]["fid"] == 138.0
    assert report["selection_policy"] == recovery.EXPECTED_SELECTION_POLICY
    assert report["claim_boundary"] == recovery.EXPECTED_CLAIM_BOUNDARY


def test_preflight_rejects_gate_sha_or_failed_gate_drift(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["gate_path"].write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="promotion gate contract"):
        _build_preflight(fixture)

    fixture = _fixture(tmp_path / "failed-set")
    fixture["gate"] = deepcopy(fixture["gate"])
    fixture["gate"]["gates"].append({"name": "another_gate", "passed": False})
    _write_json(fixture["gate_path"], fixture["gate"])
    fixture["plan"]["source"]["promotion_gate_sha256"] = file_sha256(
        fixture["gate_path"]
    )
    with pytest.raises(ValueError, match="promotion gate contract"):
        _build_preflight(fixture)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("checkpoint_sha256", "f" * 64),
        ("sample_set_sha256", "e" * 64),
        ("git", _git("9" * 40, EVALUATION_BRANCH)),
    ],
)
def test_preflight_rejects_formal_provenance_drift(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    fixture = _fixture(tmp_path)
    payload = fixture["formal_payloads"]["cofitok"]
    payload["sample_provenance"][field] = value
    _write_json(fixture["formal_paths"]["cofitok"], payload)
    fixture["plan"]["methods"]["cofitok"]["formal_metrics_sha256"] = file_sha256(
        fixture["formal_paths"]["cofitok"]
    )

    with pytest.raises(ValueError, match="formal cofitok metrics contract"):
        _build_preflight(fixture)


def test_build_report_summarizes_ten_matched_cases_without_authorization(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    preflight = _build_preflight(fixture)
    case_root = tmp_path / "cases"
    _write_cases(fixture, case_root)

    report = recovery.build_report(
        preflight=preflight,
        plan=fixture["plan"],
        case_root=case_root,
        diagnostic_git=fixture["diagnostic_git"],
    )
    rebuilt = recovery.build_report(
        preflight=preflight,
        plan=fixture["plan"],
        case_root=case_root,
        diagnostic_git=fixture["diagnostic_git"],
    )

    assert report["status"] == "complete"
    assert rebuilt == report
    assert len(report["sweep"]["rows"]) == 10
    assert len(report["sweep"]["paired_cases"]) == 5
    assert report["sweep"]["rankings_by_diagnostic_fid"]["cofitok"][0] == (
        "cfg150_r050"
    )
    assert report["selection"]["exploratory_best_case_by_method"] == {
        "cofitok": "cfg150_r050",
        "dense_identity": "cfg150_r100",
    }
    assert report["selection"]["per_method_protocol_selection_allowed"] is False
    shared = report["selection"]["shared_protocol"]
    assert shared["formal_protocol_baseline_case"] == "cfg150_r000"
    assert shared["selected_case"] == "cfg150_r050"
    assert shared["selected_protocol"] == {
        "guidance_scale": 1.5,
        "guidance_rescale": 0.5,
    }
    assert shared["selected_is_formal_protocol_baseline"] is False
    assert shared["eligible_for_matched_10000_confirmation"] is True
    assert shared["ranked_cases"][0]["strictly_improves_both_methods"] is True
    assert report["selection"]["automatic_formal_protocol_change_allowed"] is False
    assert (
        report["selection"][
            "matched_10000_confirmation_required_before_any_protocol_change"
        ]
        is True
    )
    assert report["claim_boundary"]["full_training_launch_allowed"] is False
    assert report["claim_boundary"]["full_300k_launch_allowed"] is False
    assert report["claim_boundary"]["replaces_frozen_promotion_gate"] is False


def test_shared_selection_keeps_formal_baseline_without_joint_improvement(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    preflight = _build_preflight(fixture)
    case_root = tmp_path / "cases"
    _write_cases(fixture, case_root)
    metrics_path = (
        case_root
        / "cofitok_cfg150_r050"
        / "metrics"
        / "generation_metrics_report.json"
    )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["metrics"]["frechet_inception_distance"] = 105.0
    _write_json(metrics_path, metrics)

    report = recovery.build_report(
        preflight=preflight,
        plan=fixture["plan"],
        case_root=case_root,
        diagnostic_git=fixture["diagnostic_git"],
    )

    shared = report["selection"]["shared_protocol"]
    assert shared["selected_case"] == "cfg150_r000"
    assert shared["selected_is_formal_protocol_baseline"] is True
    assert shared["eligible_for_matched_10000_confirmation"] is False


@pytest.mark.parametrize(
    ("target", "mutate", "match"),
    [
        (
            "sampling",
            lambda payload: payload["sampling"].__setitem__("guidance_scale", 9.0),
            "sampling protocol mismatch",
        ),
        (
            "sampling",
            lambda payload: payload["sampling"]["random_stream"].__setitem__(
                "scope", "per_batch"
            ),
            "random stream is invalid",
        ),
        (
            "metrics",
            lambda payload: payload["counts"].__setitem__(
                "generated_image_count", 511
            ),
            "metrics mismatch",
        ),
        (
            "sampling",
            lambda payload: payload.__setitem__(
                "runtime_environment_sha256", "a" * 64
            ),
            "report mismatch",
        ),
        (
            "metrics",
            lambda payload: payload["parameters"].__setitem__(
                "precision_recall_enabled", True
            ),
            "metrics mismatch",
        ),
    ],
)
def test_build_report_rejects_protocol_and_runtime_drift(
    tmp_path: Path,
    target: str,
    mutate,
    match: str,
) -> None:
    fixture = _fixture(tmp_path)
    preflight = _build_preflight(fixture)
    case_root = tmp_path / "cases"
    _write_cases(fixture, case_root)
    case_dir = case_root / "cofitok_cfg100_r000"
    path = (
        case_dir / "sampling_report.json"
        if target == "sampling"
        else case_dir / "metrics" / "generation_metrics_report.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    _write_json(path, payload)

    with pytest.raises(ValueError, match=match):
        recovery.build_report(
            preflight=preflight,
            plan=fixture["plan"],
            case_root=case_root,
            diagnostic_git=fixture["diagnostic_git"],
        )


def test_plan_rejects_any_authorizing_boundary(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["plan"]["claim_boundary"]["full_training_launch_allowed"] = True

    with pytest.raises(ValueError, match="plan contract mismatch"):
        _build_preflight(fixture)


def test_plan_rejects_per_method_selection_or_formal_baseline_drift(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    fixture["plan"]["selection_policy"][
        "per_method_protocol_selection_allowed"
    ] = True
    with pytest.raises(ValueError, match="plan contract mismatch"):
        _build_preflight(fixture)

    fixture = _fixture(tmp_path / "baseline")
    baseline = next(
        case
        for case in fixture["plan"]["diagnostic"]["cases"]
        if case["id"] == "cfg150_r000"
    )
    baseline["guidance_rescale"] = 0.25
    with pytest.raises(ValueError, match="shared baseline differs"):
        _build_preflight(fixture)


def test_runbook_is_gpu_deferred_frozen_and_non_authorizing() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root
        / "artifacts/runbooks/generation_stability_frozen_50k_sampling_recovery_diagnostic.sh"
    ).read_text(encoding="utf-8")

    assert "EXPECTED_SAMPLING_RECOVERY_REVISION=${" in source
    assert "EXPECTED_SAMPLING_RECOVERY_BRANCH=${" in source
    assert '[[ "$(git rev-parse HEAD)" == "$EXPECTED_SAMPLING_RECOVERY_REVISION" ]]' in source
    assert '[[ "$(git branch --show-current)" == "$EXPECTED_SAMPLING_RECOVERY_BRANCH" ]]' in source
    assert '[[ -z "$(git status --porcelain)" ]]' in source
    assert "flock -n 7" in source
    assert "nvidia-smi --query-compute-apps=pid,process_name,used_memory" in source
    assert "exit 75" in source
    assert "resolve_latest_checkpoint" in source
    assert "checkpoint_integrity_sha256" in source
    assert source.count("scripts/generate_samples.py") == 1
    assert source.count("scripts/evaluate_generation_metrics.py") == 1
    assert source.count("scripts/build_generation_stability_sampling_recovery.py") == 3
    assert "--num-samples 512" in source
    assert "--sample-steps 100" in source
    assert "--weights ema" in source
    assert "--skip-prc" in source
    assert "--resume" in source
    assert 'DIAGNOSTIC_ROOT="$CHECKPOINT_ROOT/stability_scaling_50k_sampling_recovery_v1"' in source
    assert '"full_training_launch_allowed": False' in source
    assert '"full_300k_launch_allowed": False' in source
    assert "train_generation.py" not in source
    assert "evaluate_generation_checkpoint.py" not in source
    assert "run_generation_stage_once.py" not in source
    assert "full_matched_300k" not in source


def _confirmation_fixture(tmp_path: Path) -> dict:
    fixture = _fixture(tmp_path)
    recovery_preflight = _build_preflight(fixture)
    recovery_case_root = tmp_path / "recovery-cases"
    _write_cases(fixture, recovery_case_root)
    recovery_summary = recovery.build_report(
        preflight=recovery_preflight,
        plan=fixture["plan"],
        case_root=recovery_case_root,
        diagnostic_git=fixture["diagnostic_git"],
    )
    recovery_summary_path = tmp_path / "recovery-summary.json"
    _write_json(recovery_summary_path, recovery_summary)
    confirmation_git = _git(CONFIRMATION_REVISION, DIAGNOSTIC_BRANCH)
    confirmation_preflight = confirmation.build_preflight(
        project=tmp_path,
        plan=fixture["plan"],
        plan_path=fixture["plan_path"],
        recovery_summary=recovery_summary,
        recovery_summary_path=recovery_summary_path,
        recovery_case_root=recovery_case_root,
        promotion_gate=fixture["gate"],
        promotion_gate_path=fixture["gate_path"],
        formal_metrics={
            method: (fixture["formal_payloads"][method], fixture["formal_paths"][method])
            for method in recovery.METHODS
        },
        confirmation_git=confirmation_git,
        expected_recovery_revision=DIAGNOSTIC_REVISION,
        expected_recovery_branch=DIAGNOSTIC_BRANCH,
        expected_confirmation_revision=CONFIRMATION_REVISION,
        expected_confirmation_branch=DIAGNOSTIC_BRANCH,
    )
    return {
        **fixture,
        "recovery_preflight": recovery_preflight,
        "recovery_case_root": recovery_case_root,
        "recovery_summary": recovery_summary,
        "recovery_summary_path": recovery_summary_path,
        "confirmation_git": confirmation_git,
        "confirmation_preflight": confirmation_preflight,
    }


def _confirmation_reports(
    preflight: dict,
    *,
    method: str,
    fid: float,
) -> tuple[dict, dict]:
    method_plan = preflight["methods"][method]
    protocol = preflight["protocol"]
    sample_sha256 = _sha(f"confirmation-{method}-samples")
    sampling = _sampling(
        sample_count=10_000,
        sample_steps=int(protocol["sample_steps"]),
        prefix_budget=int(method_plan["prefix_budget"]),
        guidance_scale=float(protocol["guidance_scale"]),
        guidance_rescale=float(protocol["guidance_rescale"]),
    )
    sampling_report = {
        "schema_version": 6,
        "status": "completed",
        "git": _git(CONFIRMATION_REVISION, DIAGNOSTIC_BRANCH),
        "runtime_environment_sha256": SAMPLING_RUNTIME,
        "checkpoint_sha256": method_plan["checkpoint_sha256"],
        "checkpoint_step": 50_000,
        "checkpoint_integrity_manifest": f"/{method}/checkpoint_step_00050000.pt.integrity.json",
        "weights": "ema",
        "sampling": sampling,
        "sample_sets": {
            str(method_plan["prefix_budget"]): {
                "count": 10_000,
                "sha256": sample_sha256,
            }
        },
        "elapsed_seconds": 1_000.0,
    }
    metrics_report = {
        "schema_version": 3,
        "role": "generation_directory_metrics_report",
        "protocol": "torch_fidelity_directory_metrics",
        "status": "completed",
        "git": _git(CONFIRMATION_REVISION, DIAGNOSTIC_BRANCH),
        "implementation": deepcopy(preflight["evaluator"]),
        "runtime_environment_sha256": METRICS_RUNTIME,
        "parameters": {
            "precision_recall_enabled": True,
            "batch_size": 64,
            "prc_batch_size": 10_000,
            "min_samples": 10_000,
            "seed": 2027,
            "samples_find_deep": True,
            "samples_shuffle": False,
        },
        "counts": {"generated_image_count": 10_000, "real_image_count": 50_000},
        "real_set": {**preflight["real_set"], "root": "/datasets/imagenet_256/val"},
        "sample_provenance": {
            "git": _git(CONFIRMATION_REVISION, DIAGNOSTIC_BRANCH),
            "checkpoint_sha256": method_plan["checkpoint_sha256"],
            "checkpoint_step": 50_000,
            "checkpoint_integrity_manifest": sampling_report[
                "checkpoint_integrity_manifest"
            ],
            "runtime_environment_sha256": SAMPLING_RUNTIME,
            "selected_prefix_budget": method_plan["prefix_budget"],
            "weights": "ema",
            "sampling": deepcopy(sampling),
            "sample_set_sha256": sample_sha256,
        },
        "metrics": {
            "frechet_inception_distance": fid,
            "inception_score_mean": 9.0,
            "inception_score_std": 0.2,
            "precision": 0.75,
            "recall": 0.02,
        },
    }
    return sampling_report, metrics_report


def _write_confirmation_cases(
    fixture: dict,
    case_root: Path,
    *,
    cofitok_fid: float = 90.0,
    dense_fid: float = 110.0,
) -> None:
    for method, fid in (("cofitok", cofitok_fid), ("dense_identity", dense_fid)):
        sampling, metrics = _confirmation_reports(
            fixture["confirmation_preflight"], method=method, fid=fid
        )
        method_root = case_root / method
        _write_json(method_root / "sampling_report.json", sampling)
        _write_json(method_root / "metrics" / "generation_metrics_report.json", metrics)


def test_confirmation_preflight_binds_shared_candidate_and_sources(
    tmp_path: Path,
) -> None:
    fixture = _confirmation_fixture(tmp_path)
    preflight = fixture["confirmation_preflight"]

    assert preflight["status"] == "pass"
    assert preflight["recovery"]["selected_case"] == "cfg150_r050"
    assert preflight["protocol"]["num_samples"] == 10_000
    assert preflight["protocol"]["guidance_scale"] == 1.5
    assert preflight["protocol"]["guidance_rescale"] == 0.5
    assert preflight["protocol"]["precision_recall_enabled"] is True
    assert preflight["claim_boundary"] == confirmation.EXPECTED_CLAIM_BOUNDARY
    assert preflight["claim_boundary"]["full_training_launch_allowed"] is False


def test_confirmation_preflight_allows_clean_checkout_relocation(
    tmp_path: Path,
) -> None:
    fixture = _confirmation_fixture(tmp_path)
    summary = deepcopy(fixture["recovery_summary"])
    summary["preflight"]["plan"]["path"] = "/relocated/clean-checkout/plan.json"
    _write_json(fixture["recovery_summary_path"], summary)

    preflight = confirmation.build_preflight(
        project=tmp_path,
        plan=fixture["plan"],
        plan_path=fixture["plan_path"],
        recovery_summary=summary,
        recovery_summary_path=fixture["recovery_summary_path"],
        recovery_case_root=fixture["recovery_case_root"],
        promotion_gate=fixture["gate"],
        promotion_gate_path=fixture["gate_path"],
        formal_metrics={
            method: (fixture["formal_payloads"][method], fixture["formal_paths"][method])
            for method in recovery.METHODS
        },
        confirmation_git=fixture["confirmation_git"],
        expected_recovery_revision=DIAGNOSTIC_REVISION,
        expected_recovery_branch=DIAGNOSTIC_BRANCH,
        expected_confirmation_revision=CONFIRMATION_REVISION,
        expected_confirmation_branch=DIAGNOSTIC_BRANCH,
    )

    assert preflight["status"] == "pass"
    assert preflight["recovery"]["selected_case"] == "cfg150_r050"


def test_confirmation_preflight_rejects_ineligible_or_per_method_selection(
    tmp_path: Path,
) -> None:
    fixture = _confirmation_fixture(tmp_path)
    summary = deepcopy(fixture["recovery_summary"])
    summary["selection"]["shared_protocol"][
        "eligible_for_matched_10000_confirmation"
    ] = False
    _write_json(fixture["recovery_summary_path"], summary)

    with pytest.raises(ValueError, match="shared selection is not eligible"):
        confirmation.build_preflight(
            project=tmp_path,
            plan=fixture["plan"],
            plan_path=fixture["plan_path"],
            recovery_summary=summary,
            recovery_summary_path=fixture["recovery_summary_path"],
            recovery_case_root=fixture["recovery_case_root"],
            promotion_gate=fixture["gate"],
            promotion_gate_path=fixture["gate_path"],
            formal_metrics={
                method: (
                    fixture["formal_payloads"][method],
                    fixture["formal_paths"][method],
                )
                for method in recovery.METHODS
            },
            confirmation_git=fixture["confirmation_git"],
            expected_recovery_revision=DIAGNOSTIC_REVISION,
            expected_recovery_branch=DIAGNOSTIC_BRANCH,
            expected_confirmation_revision=CONFIRMATION_REVISION,
            expected_confirmation_branch=DIAGNOSTIC_BRANCH,
        )


def test_confirmation_report_is_deterministic_pass_but_non_authorizing(
    tmp_path: Path,
) -> None:
    fixture = _confirmation_fixture(tmp_path)
    case_root = tmp_path / "confirmation-cases"
    _write_confirmation_cases(fixture, case_root)

    report = confirmation.build_report(
        preflight=fixture["confirmation_preflight"],
        case_root=case_root,
        confirmation_git=fixture["confirmation_git"],
    )
    rebuilt = confirmation.build_report(
        preflight=fixture["confirmation_preflight"],
        case_root=case_root,
        confirmation_git=fixture["confirmation_git"],
    )

    assert rebuilt == report
    assert report["status"] == "pass"
    assert report["decision"] == "candidate_quality_confirmed_non_authorizing"
    assert report["quality_confirmed"] is True
    assert all(report["quality_checks"].values())
    assert report["next_boundary"]["candidate_protocol_formalized"] is False
    assert report["next_boundary"]["separate_new_gate_required"] is True
    assert report["next_boundary"]["full_training_launch_allowed"] is False
    assert report["claim_boundary"]["confirmation_report_is_promotion_gate"] is False


def test_confirmation_report_holds_when_absolute_quality_is_not_recovered(
    tmp_path: Path,
) -> None:
    fixture = _confirmation_fixture(tmp_path)
    case_root = tmp_path / "confirmation-cases"
    _write_confirmation_cases(fixture, case_root, cofitok_fid=105.0)

    report = confirmation.build_report(
        preflight=fixture["confirmation_preflight"],
        case_root=case_root,
        confirmation_git=fixture["confirmation_git"],
    )

    assert report["status"] == "hold"
    assert report["quality_confirmed"] is False
    assert report["quality_checks"]["cofitok_absolute_fid_quality"] is False
    assert report["next_boundary"]["full_300k_launch_allowed"] is False


@pytest.mark.parametrize(
    ("target", "mutate", "match"),
    [
        (
            "sampling",
            lambda payload: payload["sampling"].__setitem__("guidance_rescale", 1.0),
            "sampling protocol mismatch",
        ),
        (
            "sampling",
            lambda payload: payload.__setitem__(
                "runtime_environment_sha256", "a" * 64
            ),
            "sampling report mismatch",
        ),
        (
            "metrics",
            lambda payload: payload["parameters"].__setitem__(
                "precision_recall_enabled", False
            ),
            "metrics report mismatch",
        ),
        (
            "metrics",
            lambda payload: payload["counts"].__setitem__(
                "generated_image_count", 9_999
            ),
            "metrics report mismatch",
        ),
    ],
)
def test_confirmation_rejects_protocol_runtime_prc_or_count_drift(
    tmp_path: Path,
    target: str,
    mutate,
    match: str,
) -> None:
    fixture = _confirmation_fixture(tmp_path)
    case_root = tmp_path / "confirmation-cases"
    _write_confirmation_cases(fixture, case_root)
    method_root = case_root / "cofitok"
    path = (
        method_root / "sampling_report.json"
        if target == "sampling"
        else method_root / "metrics" / "generation_metrics_report.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    _write_json(path, payload)

    with pytest.raises(ValueError, match=match):
        confirmation.build_report(
            preflight=fixture["confirmation_preflight"],
            case_root=case_root,
            confirmation_git=fixture["confirmation_git"],
        )


def test_confirmation_runbook_is_shared_formal_size_and_non_authorizing() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root
        / "artifacts/runbooks/generation_stability_frozen_50k_sampling_confirmation_10k.sh"
    ).read_text(encoding="utf-8")

    assert "EXPECTED_SAMPLING_PIPELINE_REVISION=${" in source
    assert "EXPECTED_SAMPLING_PIPELINE_BRANCH=${" in source
    assert '[[ "$(git rev-parse HEAD)" == "$EXPECTED_SAMPLING_PIPELINE_REVISION" ]]' in source
    assert '[[ -z "$(git status --porcelain)" ]]' in source
    assert "flock -n 6" in source
    assert "nvidia-smi --query-compute-apps=pid,process_name,used_memory" in source
    assert "exit 75" in source
    assert "resolve_latest_checkpoint" in source
    assert "--recovery-case-root \"$RECOVERY_CASE_ROOT\"" in source
    assert source.count("scripts/build_generation_stability_sampling_confirmation.py") == 3
    assert source.count("scripts/generate_samples.py") == 1
    assert source.count("scripts/evaluate_generation_metrics.py") == 1
    assert "--num-samples 10000" in source
    assert "--min-samples 10000" in source
    assert "--prc-batch-size 10000" in source
    assert "--skip-prc" not in source
    assert '[[ "$selected_case" != "cfg150_r000" ]]' in source
    assert 'CONFIRMATION_ROOT="$CHECKPOINT_ROOT/stability_scaling_50k_sampling_confirmation_10k_v1"' in source
    assert '"confirmation_report_is_promotion_gate": False' in source
    assert '"new_gate_required": True' in source
    assert '"full_training_launch_allowed": False' in source
    assert '"full_300k_launch_allowed": False' in source
    assert "train_generation.py" not in source
    assert "build_generation_gate_report.py" not in source
    assert "build_generation_full_launch_receipt.py" not in source
    assert "full_matched_300k" not in source
