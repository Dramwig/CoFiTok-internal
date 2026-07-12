from __future__ import annotations

from pathlib import Path

import pytest

from scripts.write_generation_pipeline_status import build_status


ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_pipeline_status_requires_failure_exit_code() -> None:
    with pytest.raises(ValueError, match="requires an exit code"):
        build_status(
            status="failed",
            stage="promotion_gate",
            detail="gate held",
            exit_code=None,
            git_commit="a" * 40,
            hostname="server",
            updated_at="2026-07-12T00:00:00+00:00",
        )


def test_pipeline_status_captures_revision_and_stage() -> None:
    report = build_status(
        status="running",
        stage="full_training",
        detail="training",
        exit_code=None,
        git_commit="a" * 40,
        hostname="server",
        updated_at="2026-07-12T00:00:00+00:00",
    )

    assert report["pipeline"] == "generation_complete_after_10pct"
    assert report["stage"] == "full_training"
    assert report["git_commit"] == "a" * 40


def test_completion_runbook_serializes_and_orders_all_stages() -> None:
    runbook = _read("artifacts/runbooks/generation_complete_pipeline_after_10pct.sh")

    assert "flock -n 9" in runbook
    assert "trap record_failure EXIT" in runbook
    assert "scripts/validate_generation_training_pair.py" in runbook
    assert "--expected-steps 50000" in runbook
    assert "promote_to_full_imagenet256" in runbook
    assert "large_scale_generation_ready" in runbook
    ordered_markers = [
        "STAGE=posteval_10pct",
        "STAGE=promotion_gate",
        "STAGE=full_training",
        "STAGE=full_posteval",
        "STAGE=final_gate",
        "STAGE=complete",
    ]
    positions = [runbook.index(marker) for marker in ordered_markers]
    assert positions == sorted(positions)


def test_remote_deployer_guards_revision_training_and_duplicate_launch() -> None:
    deployer = _read(
        "artifacts/runbooks/deploy_generation_posttraining_pipeline_remote.sh"
    )

    assert "git diff --quiet" in deployer
    assert "git diff --cached --quiet" in deployer
    assert '"$current_commit" != "$TARGET_COMMIT"' in deployer
    assert '"$current_commit" == "$EXPECTED_COMMIT"' in deployer
    assert 'python "$VALIDATOR"' in deployer
    assert '--expected-revision "$EXPECTED_COMMIT"' in deployer
    assert deployer.index("conda activate pf-vlm") < deployer.index('python "$VALIDATOR"')
    assert "pgrep -af '[s]cripts/train_generation.py" in deployer
    assert "git bundle verify" in deployer
    assert "git merge --ff-only FETCH_HEAD" in deployer
    assert "python -m pytest -q" in deployer
    assert "kill -0 \"$previous_pid\"" in deployer
    assert "exit 0" in deployer
    assert "nohup bash \"$PIPELINE\"" in deployer
    assert "sleep 2" in deployer

    pipeline = _read("artifacts/runbooks/generation_complete_pipeline_after_10pct.sh")
    assert "PINNED_10PCT_REVISION=781a01444fddbf0d48a427ba58bdeed50167b5be" in pipeline
    assert "validate_generation_training_pair.py" in pipeline


def test_local_deployer_pins_current_training_revision_and_builds_bundle() -> None:
    deployer = _read("scripts/deploy_generation_posttraining_pipeline.ps1")

    assert "781a01444fddbf0d48a427ba58bdeed50167b5be" in deployer
    assert '"scale/generative-system"' in deployer
    assert "--untracked-files=no" in deployer
    assert '"bundle", "create"' in deployer
    assert "deploy_generation_posttraining_pipeline_remote.sh" in deployer
    assert "validate_generation_training_pair.py" in deployer
    assert "$RemoteValidator" in deployer
    assert "ExpectedRemoteCommit" in deployer
