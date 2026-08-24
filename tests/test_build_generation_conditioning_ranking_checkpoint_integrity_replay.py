from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.build_generation_conditioning_ranking_checkpoint_integrity_replay import (
    REQUIRED_CHECKPOINT_STEPS,
    RUNS,
    SOURCE_PATHS,
    build_replay,
)


TRAINING_REVISION = "1" * 40
TRAINING_TREE = "2" * 40
TRAINING_BRANCH = "analysis/test-training"
DATASET_SHA = "3" * 64
RUNTIME_SHA = "4" * 64
CODE_IDENTITY = {
    "revision": "5" * 40,
    "tree": "6" * 40,
    "branch": "analysis/test-replay",
    "tracked_dirty": False,
}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_checkpoint(run_dir: Path, step: int) -> dict[str, object]:
    payload = f"{run_dir.name}-{step}".encode()
    checkpoint = run_dir / f"checkpoint_step_{step:08d}.pt"
    checkpoint.write_bytes(payload)
    sidecar = {
        "schema_version": 1,
        "checkpoint": checkpoint.name,
        "checkpoint_bytes": len(payload),
        "checkpoint_sha256": hashlib.sha256(payload).hexdigest(),
        "checkpoint_format_version": 1,
        "step": step,
        "runtime_environment_sha256": RUNTIME_SHA,
        "dataset_identity_sha256": DATASET_SHA,
        "git_revision": TRAINING_REVISION,
        "git_branch": TRAINING_BRANCH,
        "git_dirty": False,
    }
    _write_json(checkpoint.with_name(f"{checkpoint.name}.integrity.json"), sidecar)
    return sidecar


def _write_run(root: Path, run: str) -> None:
    run_dir = root / run
    run_dir.mkdir(parents=True)
    rows = []
    for event_index, step in enumerate((1, 250, 500, 750, 1_000)):
        rows.append(
            {
                "step": step,
                "samples_seen": step * 64,
                "total": 0.1,
                "epsilon": 0.08,
                "grad_norm": 0.2,
                "learning_rate": 1e-4,
                "elapsed_seconds": float(step),
                **(
                    {
                        "validation_epsilon_mse": 0.02,
                        "validation_event_index": event_index - 1,
                        "validation_batch_index": event_index - 1,
                        "validation_num_images": 16,
                        "validation_noise_seed": 102_030,
                    }
                    if step % 250 == 0
                    else {}
                ),
            }
        )
    (run_dir / "train_metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    sidecars = {step: _write_checkpoint(run_dir, step) for step in REQUIRED_CHECKPOINT_STEPS}
    latest = {
        **sidecars[1_000],
        "integrity_manifest": "checkpoint_step_00001000.pt.integrity.json",
    }
    _write_json(run_dir / "latest.json", latest)
    _write_json(
        run_dir / "training_report.json",
        {
            "completed_steps": 1_000,
            "target_steps": 1_000,
            "training_complete": True,
            "git": {
                "revision": TRAINING_REVISION,
                "branch": TRAINING_BRANCH,
                "dirty": False,
            },
            "runtime_environment_sha256": RUNTIME_SHA,
            "dataset_provenance": {"identity_sha256": DATASET_SHA},
            "final_metrics": rows[-1],
            "latest_checkpoint": latest,
        },
    )
    _write_json(root / "reports" / f"{run}_training_audit.json", {"status": "complete"})


def _write_sources(root: Path) -> dict[str, str]:
    clean_git = {
        "revision": TRAINING_REVISION,
        "branch": TRAINING_BRANCH,
        "tracked_dirty": False,
    }
    values = {
        "preparation": {
            "status": "pass",
            "git": clean_git,
            "tree": TRAINING_TREE,
        },
        "execution_authorization": {
            "status": "authorized",
            "generation_advantage_proven": False,
            "claim_boundary": {
                "diagnostic_non_authorizing": True,
                "followup_training_allowed": False,
            },
        },
        "training_status": {
            "status": "completed",
            "revision": TRAINING_REVISION,
            "generation_advantage_proven": False,
        },
        "postevaluation": {
            "status": "completed",
            "git": clean_git,
            "decision": {
                "shared_semantic_alignment_recovery_supported": False,
                "cofitok_specific_advantage_claim_allowed": False,
                "recommended_next_action": "revise_training_time_semantic_alignment_objective",
            },
        },
    }
    hashes = {}
    for name, relative in SOURCE_PATHS.items():
        path = root / relative
        _write_json(path, values[name])
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


@pytest.fixture
def evidence(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    for run in RUNS:
        _write_run(tmp_path, run)
    return tmp_path, _write_sources(tmp_path)


def _build(root: Path, hashes: dict[str, str]) -> dict[str, object]:
    return build_replay(
        project=Path(__file__).resolve().parents[1],
        output_root=root,
        code_identity=CODE_IDENTITY,
        training_revision=TRAINING_REVISION,
        training_tree=TRAINING_TREE,
        training_branch=TRAINING_BRANCH,
        expected_source_sha256=hashes,
    )


def test_replay_physically_verifies_all_twelve_checkpoints(evidence) -> None:
    root, hashes = evidence

    report = _build(root, hashes)

    assert report["status"] == "pass"
    assert report["physical_checkpoint_count"] == 12
    assert report["generation_advantage_proven"] is False
    assert all(
        value is False
        for key, value in report["claim_boundary"].items()
        if key != "diagnostic_non_authorizing"
    )
    assert report["claim_boundary"]["diagnostic_non_authorizing"] is True
    for run in RUNS:
        evidence_run = report["runs"][run]
        assert evidence_run["status"] == "verified"
        assert evidence_run["metrics"]["samples_seen_equals_step_times_64"] is True
        assert [
            item["step"]
            for item in evidence_run["required_checkpoint_integrity"]["checkpoints"]
        ] == [500, 750, 1_000]


def test_replay_rejects_historical_checkpoint_tampering(evidence) -> None:
    root, hashes = evidence
    checkpoint = root / RUNS[0] / "checkpoint_step_00000500.pt"
    checkpoint.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="Required checkpoint audit failed"):
        _build(root, hashes)


def test_replay_rejects_source_report_drift(evidence) -> None:
    root, hashes = evidence
    authorization = root / SOURCE_PATHS["execution_authorization"]
    value = json.loads(authorization.read_text(encoding="utf-8"))
    value["claim_boundary"]["followup_training_allowed"] = True
    _write_json(authorization, value)

    with pytest.raises(ValueError, match="execution_authorization SHA256 differs"):
        _build(root, hashes)


def test_replay_rejects_nonmonotonic_metrics(evidence) -> None:
    root, hashes = evidence
    path = root / RUNS[0] / "train_metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[-1]["step"] = rows[-2]["step"]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    with pytest.raises(ValueError, match="Required checkpoint audit failed"):
        _build(root, hashes)
