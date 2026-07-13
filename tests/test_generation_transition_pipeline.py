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
    assert "--allow-legacy-missing-dataset-provenance" in runbook
    assert "promote_to_full_imagenet256" in runbook
    assert "large_scale_generation_ready" in runbook
    ordered_markers = [
        "STAGE=posteval_10pct",
        "STAGE=promotion_gate",
        "STAGE=full_training",
        "STAGE=full_posteval",
        "STAGE=final_gate",
        "STAGE=inference_export",
        "STAGE=completion_audit",
        "STAGE=complete",
    ]
    positions = [runbook.index(marker) for marker in ordered_markers]
    assert positions == sorted(positions)
    assert 'if [[ ! -f "$SCALING_GATE" ]]' in runbook
    assert 'if [[ ! -f "$FINAL_GATE" || ! -f "$FINAL_COMPARISON"' in runbook
    assert "FINAL_VISUAL_AUDIT" in runbook
    assert "generation_export_inference_artifacts.sh" in runbook

    full_training = _read(
        "artifacts/runbooks/generation_full_matched_300k_after_gate.sh"
    )
    assert full_training.count("validate_generation_milestone_report.py") == 2
    assert full_training.index("build_generation_milestone_report.py") < (
        full_training.rindex("validate_generation_milestone_report.py")
    )


def test_remote_deployer_guards_revision_training_and_duplicate_launch() -> None:
    deployer = _read(
        "artifacts/runbooks/deploy_generation_posttraining_pipeline_remote.sh"
    )

    assert "git diff --quiet" in deployer
    assert "git diff --cached --quiet" in deployer
    assert '"$current_commit" != "$TARGET_COMMIT"' in deployer
    assert '"$current_commit" == "$EXPECTED_COMMIT"' in deployer
    assert 'git archive "$TARGET_COMMIT"' in deployer
    assert "scripts/validate_generation_training_pair.py" in deployer
    assert "src/cofitok | tar -x" in deployer
    assert 'PYTHONPATH="$validator_pythonpath" python "$validator"' in deployer
    assert "mktemp -d /tmp/cofitok-generation-prevalidation" in deployer
    assert "trap cleanup_prevalidation EXIT" in deployer
    assert '--expected-revision "$EXPECTED_COMMIT"' in deployer
    assert "--expected-recipe-stage scaling" in deployer
    assert "--allow-legacy-missing-dataset-provenance" in deployer
    assert deployer.index("git fetch \"$BUNDLE\" HEAD") < deployer.index(
        'PYTHONPATH="$validator_pythonpath" python "$validator"'
    )
    assert deployer.index("conda activate pf-vlm") < deployer.index(
        'PYTHONPATH="$validator_pythonpath" python "$validator"'
    )
    assert deployer.index(
        'PYTHONPATH="$validator_pythonpath" python "$validator"'
    ) < deployer.index("git merge --ff-only FETCH_HEAD")
    assert "pgrep -af '[s]cripts/train_generation.py" in deployer
    assert "git bundle verify" in deployer
    assert "git ls-files --others --exclude-standard" in deployer
    assert "git ls-tree -r --name-only" in deployer
    assert "untracked files would conflict with target revision" in deployer
    assert "git merge --ff-only FETCH_HEAD" in deployer
    assert "python -m pytest -q" in deployer
    assert "kill -0 \"$previous_pid\"" in deployer
    assert "exit 0" in deployer
    assert "generation_completion_supervisor.sh" in deployer
    assert "bash -n \"$SUPERVISOR\"" in deployer
    assert "nohup bash \"$SUPERVISOR\"" in deployer
    assert "sleep 2" in deployer
    assert "write_generation_deployment_receipt.py" in deployer
    assert "generation_upgrade_deployment_receipt.json" in deployer
    assert deployer.index("python -m pytest -q") < deployer.index(
        "write_generation_deployment_receipt.py"
    )

    pipeline = _read("artifacts/runbooks/generation_complete_pipeline_after_10pct.sh")
    assert "PINNED_10PCT_REVISION=781a01444fddbf0d48a427ba58bdeed50167b5be" in pipeline
    assert "validate_generation_training_pair.py" in pipeline
    assert "--expected-recipe-stage scaling" in pipeline
    assert "--allow-legacy-missing-dataset-provenance" in pipeline
    assert "audit_large_scale_generation_completion.py" in pipeline
    assert '--expected-full-revision "$FULL_REVISION"' in pipeline


def test_local_deployer_pins_current_training_revision_and_builds_bundle() -> None:
    deployer = _read("scripts/deploy_generation_posttraining_pipeline.ps1")

    assert "781a01444fddbf0d48a427ba58bdeed50167b5be" in deployer
    assert '"scale/generative-system"' in deployer
    assert "--untracked-files=no" in deployer
    assert '"bundle",' in deployer
    assert '"create",' in deployer
    assert '"^$ExpectedRemoteCommit"' in deployer
    assert "git bundle list-heads" in deployer
    assert "BundleHeads.Count -ne 1" in deployer
    assert "$BundleHead -ne $TargetCommit" in deployer
    assert "deploy_generation_posttraining_pipeline_remote.sh" in deployer
    assert "$RemoteValidator" not in deployer
    assert "$Validator" not in deployer
    assert "ExpectedRemoteCommit" in deployer
