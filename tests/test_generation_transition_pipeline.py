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
    assert "--allow-legacy-missing-dataset-provenance" not in runbook
    assert runbook.count("validate_generation_gate_report.py") == 2
    assert runbook.count('validate_gate "$') == 2
    assert '--gate "$gate_path" --stage "$stage"' in runbook
    assert 'validate_gate "$SCALING_GATE" scaling' in runbook
    assert 'validate_gate "$FINAL_GATE" full' in runbook
    assert 'gate_sources_match "$SCALING_GATE" scaling' in runbook
    assert 'gate_sources_match "$FINAL_GATE" full' in runbook
    assert "--sources-only" in runbook
    ordered_markers = [
        "STAGE=scaling_training",
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

    scaling_posteval = _read(
        "artifacts/runbooks/generation_10pct_posteval_2026-07-12.sh"
    )
    assert "migrate_generation_checkpoint_integrity.py" not in scaling_posteval
    assert scaling_posteval.count("audit_generation_training_progress.py") == 2
    assert scaling_posteval.count("--integrity-policy required") == 2

    full_training = _read(
        "artifacts/runbooks/generation_full_matched_300k_after_gate.sh"
    )
    assert "validate_generation_gate_report.py" in full_training
    assert '--gate "$GATE" --stage scaling' in full_training
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
    assert "--expected-recipe-stage legacy_scaling" in deployer
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
    assert "bundle_prerequisites" in deployer
    assert 'bundle_header_line' in deployer
    assert '!= "$EXPECTED_COMMIT"' in deployer
    assert deployer.index("bundle_prerequisites=()") < deployer.index(
        'git bundle verify "$BUNDLE"'
    )
    assert "scripts/check_generation_deployment_conflicts.py" in deployer
    assert 'python "$conflict_checker"' in deployer
    assert '--current-commit "$current_commit"' in deployer
    assert '--target-commit "$fetched_commit"' in deployer
    assert '--output "$CONFLICT_REPORT"' in deployer
    assert deployer.index('python "$conflict_checker"') < deployer.index(
        "git merge --ff-only FETCH_HEAD"
    )
    assert "git merge --ff-only FETCH_HEAD" in deployer
    assert "python -m pytest -q --junitxml" in deployer
    assert "generation_upgrade_pytest.xml" in deployer
    assert "check_generation_runbook_syntax.py" in deployer
    assert '--output "$RUNBOOK_SYNTAX_REPORT"' in deployer
    assert "generation_upgrade_runbook_syntax.json" in deployer
    assert "generation_upgrade_conflict_scan.json" in deployer
    assert "DEPLOYMENT_EVIDENCE_ROOT" in deployer
    assert "ARCHIVED_BUNDLE" in deployer
    assert 'cp -- "$BUNDLE" "$bundle_temporary"' in deployer
    assert 'git bundle verify "$ARCHIVED_BUNDLE"' in deployer
    assert "kill -0 \"$previous_pid\"" in deployer
    assert "exit 0" in deployer
    assert "generation_completion_supervisor.sh" in deployer
    assert "nohup bash \"$SUPERVISOR\"" in deployer
    assert "sleep 2" in deployer
    assert "write_generation_deployment_receipt.py" in deployer
    assert "generation_upgrade_deployment_receipt.json" in deployer
    assert '--conflict-scan "$CONFLICT_REPORT"' in deployer
    assert '--runbook-syntax "$RUNBOOK_SYNTAX_REPORT"' in deployer
    assert '--pytest-report "$PYTEST_REPORT"' in deployer
    assert '--bundle "$ARCHIVED_BUNDLE"' in deployer
    assert deployer.index("python -m pytest -q") < deployer.index(
        "check_generation_runbook_syntax.py"
    )
    assert deployer.index("check_generation_runbook_syntax.py") < deployer.index(
        "write_generation_deployment_receipt.py"
    )

    pipeline = _read("artifacts/runbooks/generation_complete_pipeline_after_10pct.sh")
    assert "DEPLOYMENT_SOURCE_REVISION=781a01444fddbf0d48a427ba58bdeed50167b5be" in pipeline
    assert "validate_generation_training_pair.py" in pipeline
    assert "--expected-recipe-stage scaling" in pipeline
    assert "--allow-legacy-missing-dataset-provenance" not in pipeline
    assert "generation_10pct_matched_50k_2026-07-12.sh" in pipeline
    assert '--expected-revision "$FULL_REVISION"' in pipeline
    assert '--expected-deployment-source-revision "$DEPLOYMENT_SOURCE_REVISION"' in pipeline
    assert '--expected-10pct-revision "$FULL_REVISION"' in pipeline
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
