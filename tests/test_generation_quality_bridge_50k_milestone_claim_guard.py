from __future__ import annotations

import json
import os
import shutil
import tempfile
from argparse import Namespace
from pathlib import Path

import pytest

from cofitok.diffusion import select_sampling_timesteps
from cofitok.generation import INFERENCE_API, SAMPLING_PROTOCOL_SCHEMA
from scripts.build_generation_milestone_report import (
    build_report,
    expected_source_report_suffixes,
    source_report_identity,
)
from scripts.wait_for_generation_quality_bridge_50k_milestone_claim_guard import (
    CLAIM_BOUNDARY,
    OPERATION_SCOPE,
    build_claim_guard_summary,
    run_once,
)


TRAINING_REVISION = "c" * 40
TRAINING_BRANCH = "scale/generation-stability-quality-bridge-100k"
CONTROL = {
    "checkout": {
        "path": "/control",
        "revision": "a" * 40,
        "tree": "b" * 40,
        "branch": "analysis/claim-guard",
        "tracked_dirty": False,
    },
    "source": {
        "path": "/control/scripts/waiter.py",
        "bytes": 123,
        "sha256": "d" * 64,
    },
}


@pytest.fixture
def case_root(tmp_path: Path):
    if os.name != "nt":
        yield tmp_path
        return
    root = Path(tempfile.mkdtemp(prefix="qbg-", dir=Path(__file__).parents[1]))
    try:
        yield root
    finally:
        shutil.rmtree(root)


def _git(*, revision: str = TRAINING_REVISION) -> dict:
    return {
        "revision": revision,
        "branch": TRAINING_BRANCH,
        "tracked_dirty": False,
    }


def _generation(*, sha: str, budget: int, fid: float, score: float) -> dict:
    return {
        "status": "completed",
        "git": _git(),
        "counts": {"generated_image_count": 2048},
        "metrics": {
            "frechet_inception_distance": fid,
            "inception_score_mean": score,
        },
        "sample_provenance": {
            "checkpoint": f"/checkpoints/{sha[:4]}.pt",
            "checkpoint_sha256": sha,
            "checkpoint_integrity_manifest": (
                f"/checkpoints/{sha[:4]}.pt.integrity.json"
            ),
            "checkpoint_step": 50_000,
            "weights": "ema",
            "selected_prefix_budget": budget,
            "sample_set_sha256": ("e" if budget > 1 else "f") * 64,
            "sampling": {
                "protocol_schema": SAMPLING_PROTOCOL_SCHEMA,
                "inference_api": INFERENCE_API,
                "sampler": "ddim",
                "num_samples": 2048,
                "start_index": 0,
                "batch_size": 32,
                "sample_steps": 50,
                "num_train_timesteps": 1000,
                "actual_timesteps": select_sampling_timesteps(1000, 50),
                "prefix_budgets": [budget],
                "guidance_scale": 1.5,
                "guidance_rescale": 0.0,
                "cfg_batch_mode": "batched",
                "eta": 0.0,
                "clip_x0": True,
                "seed": 0,
                "precision": "bf16",
                "image_shape": [3, 256, 256],
                "class_schedule": "balanced_modulo",
                "random_stream": {
                    "prefix_budgets_share_stream": True,
                    "batch_size_invariant": True,
                    "resume_index_invariant": True,
                },
            },
        },
    }


def _checkpoint_eval(
    *,
    sha: str,
    budget: int,
    endpoint: float,
    path_auc: float,
    rank: int,
) -> dict:
    return {
        "status": "completed",
        "git": _git(),
        "checkpoint_sha256": sha,
        "checkpoint_integrity_manifest": f"/checkpoints/{sha[:4]}.pt.integrity.json",
        "checkpoint_step": 50_000,
        "weights": "ema",
        "metrics": {
            "orders": {
                "ordered": {
                    "endpoint_clean_mse": endpoint,
                    "prefix_path_mse_auc": path_auc,
                }
            },
            "ordered_rank_by_path_auc": rank,
            "order_count": 18 if budget > 1 else 1,
            "zero_token_max_abs": 0.0,
            "shuffled_to_ordered_endpoint_ratio": 1.5,
        },
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _make_milestone(tmp_path: Path, *, wrong_git: bool = False) -> Path:
    payloads = {
        "cofitok_generation": _generation(
            sha="a" * 64,
            budget=8,
            fid=18.0,
            score=6.0,
        ),
        "dense_generation": _generation(
            sha="b" * 64,
            budget=1,
            fid=20.0,
            score=5.0,
        ),
        "cofitok_checkpoint_eval": _checkpoint_eval(
            sha="a" * 64,
            budget=8,
            endpoint=0.09,
            path_auc=0.12,
            rank=1,
        ),
        "dense_checkpoint_eval": _checkpoint_eval(
            sha="b" * 64,
            budget=1,
            endpoint=0.10,
            path_auc=0.10,
            rank=1,
        ),
    }
    if wrong_git:
        payloads["dense_generation"]["git"] = _git(revision="9" * 40)

    source_paths = {}
    for name, suffix in expected_source_report_suffixes(
        50_000,
        source_profile="quality_bridge",
    ).items():
        source = tmp_path / "source-root" / suffix
        _write_json(source, payloads[name])
        source_paths[name] = source

    report = build_report(
        cofitok_generation=payloads["cofitok_generation"],
        dense_generation=payloads["dense_generation"],
        cofitok_checkpoint_eval=payloads["cofitok_checkpoint_eval"],
        dense_checkpoint_eval=payloads["dense_checkpoint_eval"],
        source_reports={
            name: source_report_identity(path)
            for name, path in source_paths.items()
        },
        milestone_step=50_000,
        expected_samples=2048,
        source_profile="quality_bridge",
    )
    milestone = tmp_path / "reports" / "milestones" / "step_00050000.json"
    _write_json(milestone, report)
    return milestone


def _args(tmp_path: Path, milestone: Path) -> Namespace:
    return Namespace(
        milestone_report=milestone,
        summary_output=tmp_path / "guard" / "claim_guard.json",
        status_output=tmp_path / "guard" / "waiter_status.json",
        expected_training_revision=TRAINING_REVISION,
        expected_training_branch=TRAINING_BRANCH,
    )


def test_guard_replays_sources_and_publishes_descriptive_deltas(
    case_root: Path,
) -> None:
    milestone = _make_milestone(case_root)

    summary = build_claim_guard_summary(
        milestone,
        control=CONTROL,
        expected_training_revision=TRAINING_REVISION,
        expected_training_branch=TRAINING_BRANCH,
    )

    assert summary["status"] == "pass"
    assert summary["operational_scope"] == OPERATION_SCOPE
    assert summary["milestone"]["exact_replay"]["status"] == (
        "byte_equivalent_payload"
    )
    assert summary["descriptive_quality"]["fid"] == {
        "cofitok": 18.0,
        "dense_identity": 20.0,
        "cofitok_minus_dense": -2.0,
        "relative_change": -0.1,
        "lower_value": "cofitok",
    }
    assert summary["descriptive_quality"]["inception_score"]["higher_value"] == (
        "cofitok"
    )
    assert summary["scientific_interpretation"]["generation_advantage_proven"] is False


def test_guard_explicitly_prohibits_all_intermediate_authorizations(
    case_root: Path,
) -> None:
    summary = build_claim_guard_summary(
        _make_milestone(case_root),
        control=CONTROL,
        expected_training_revision=TRAINING_REVISION,
        expected_training_branch=TRAINING_BRANCH,
    )

    assert summary["claim_boundary"] == CLAIM_BOUNDARY
    for field in (
        "formal_generation_quality_claim_allowed",
        "broad_generation_superiority_claim_allowed",
        "promotion_authorization_allowed",
        "release_authorization_allowed",
        "sampling_launch_allowed",
        "training_launch_allowed",
        "full_training_launch_allowed",
        "full_300k_launch_allowed",
        "process_signals_allowed",
        "training_process_signals_allowed",
        "unrelated_process_signals_allowed",
        "cross_tier_numeric_ranking_allowed",
        "replaces_terminal_100k_result",
    ):
        assert summary["claim_boundary"][field] is False
    assert summary["claim_boundary"][
        "requires_terminal_100k_10000_sample_uncertainty"
    ] is True


def test_guard_rejects_tampered_milestone_values_even_with_intact_sources(
    case_root: Path,
) -> None:
    milestone = _make_milestone(case_root)
    report = json.loads(milestone.read_text(encoding="utf-8"))
    report["methods"]["cofitok"]["fid"] = 1.0
    report["matched_comparison"]["fid_relative_change"] = -0.95
    _write_json(milestone, report)

    with pytest.raises(ValueError, match="not an exact replay"):
        build_claim_guard_summary(
            milestone,
            control=CONTROL,
            expected_training_revision=TRAINING_REVISION,
            expected_training_branch=TRAINING_BRANCH,
        )


def test_guard_rejects_changed_bound_source(case_root: Path) -> None:
    milestone = _make_milestone(case_root)
    report = json.loads(milestone.read_text(encoding="utf-8"))
    source = Path(report["source_reports"]["cofitok_generation"]["path"])
    source.write_text("changed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="changed after binding"):
        build_claim_guard_summary(
            milestone,
            control=CONTROL,
            expected_training_revision=TRAINING_REVISION,
            expected_training_branch=TRAINING_BRANCH,
        )


def test_guard_rejects_wrong_training_revision_in_source(case_root: Path) -> None:
    milestone = _make_milestone(case_root, wrong_git=True)

    with pytest.raises(ValueError, match="Git revision differs"):
        build_claim_guard_summary(
            milestone,
            control=CONTROL,
            expected_training_revision=TRAINING_REVISION,
            expected_training_branch=TRAINING_BRANCH,
        )


def test_waiter_once_waits_without_publishing_when_milestone_is_missing(
    case_root: Path,
) -> None:
    args = _args(case_root, case_root / "absent.json")

    status, summary = run_once(args, control=CONTROL)

    assert status == "waiting"
    assert summary is None
    assert not args.summary_output.exists()
    waiter = json.loads(args.status_output.read_text(encoding="utf-8"))
    assert waiter["detail"] == "matched_50k_milestone_report_missing"
    assert waiter["claim_boundary"] == CLAIM_BOUNDARY


def test_waiter_publishes_once_and_rejects_changed_existing_summary(
    case_root: Path,
) -> None:
    args = _args(case_root, _make_milestone(case_root))

    status, summary = run_once(args, control=CONTROL)
    assert status == "completed"
    assert summary is not None
    waiter = json.loads(args.status_output.read_text(encoding="utf-8"))
    assert waiter["summary"]["byte_equivalent_replay"] is True

    published = json.loads(args.summary_output.read_text(encoding="utf-8"))
    published["descriptive_quality"]["fid"]["cofitok"] = 0.0
    _write_json(args.summary_output, published)
    with pytest.raises(ValueError, match="existing claim guard summary differs"):
        run_once(args, control=CONTROL)
