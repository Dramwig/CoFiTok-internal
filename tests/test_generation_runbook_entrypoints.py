from __future__ import annotations

import contextlib
import io
import os
import re
import runpy
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMMAND_PATTERN = re.compile(
    r'(?:python|"?\$PYTHON"?)\s+scripts/([A-Za-z0-9_]+\.py)'
)
RUNBOOKS = (
    "generation_attest_deployed_revision.sh",
    "generation_deploy_large_capacity_readiness_checkout.sh",
    "generation_10pct_matched_50k_2026-07-12.sh",
    "generation_10pct_posteval_2026-07-12.sh",
    "generation_full_milestone_eval.sh",
    "generation_full_posteval_50k.sh",
    "generation_export_inference_artifacts.sh",
    "generation_full_matched_300k_after_gate.sh",
    "generation_complete_pipeline_after_10pct.sh",
    "generation_stability_ema_teacher_matched_50k_after_gate.sh",
    "generation_stability_ema_teacher_dense_recovery_after_transition_failure.sh",
    "generation_stability_ema_teacher_50k_posteval_waiter.sh",
    "generation_stability_ema_teacher_50k_posteval_after_training.sh",
    "generation_stability_frozen_50k_supplemental_waiter.sh",
    "generation_stability_frozen_50k_supplemental_after_posteval.sh",
    "generation_stability_frozen_50k_class_fidelity_after_supplemental.sh",
    "generation_stability_frozen_50k_sampling_recovery_diagnostic.sh",
    "generation_stability_frozen_50k_sampling_confirmation_10k.sh",
    "generation_stability_frozen_existing_sample_support_audit.sh",
    "generation_stability_frozen_50k_convergence_audit.sh",
    "generation_stability_full_data_quality_bridge_100k_prepare.sh",
    "generation_stability_full_data_quality_bridge_100k_execute.sh",
    "generation_quality_bridge_followup_decision_after_result.sh",
    "generation_quality_bridge_exposure_followup_waiter.sh",
    "generation_quality_bridge_preserve_10k_capacity_references.sh",
    "generation_capacity_probe_prepare_after_decision.sh",
    "generation_stability_capacity_probe_250m_10k_execute.sh",
    "generation_stability_capacity_probe_250m_10k_supervisor.sh",
    "generation_capacity_scaling_decision_after_probe.sh",
    "generation_capacity_scaling_250m_50k_execute.sh",
    "generation_capacity_scaling_250m_50k_supervisor.sh",
    "generation_capacity_completion_250m_100k_execute.sh",
    "generation_capacity_completion_250m_100k_supervisor.sh",
    "generation_capacity_completion_100k_result_waiter.sh",
    "generation_capacity_full_300k_readiness_decision_waiter.sh",
    "generation_capacity_full_300k_readiness_after_decision.sh",
    "generation_capacity_full_300k_readiness_supervisor.sh",
    "generation_capacity_full_300k_execute.sh",
    "generation_capacity_full_300k_training_supervisor.sh",
    "generation_capacity_full_300k_posteval_50k.sh",
    "generation_capacity_full_300k_posteval_supervisor.sh",
    "generation_capacity_full_300k_export_inference_artifacts.sh",
    "generation_capacity_full_300k_completion_audit.sh",
    "generation_capacity_full_300k_finalize_after_gate.sh",
    "generation_capacity_full_300k_finalization_supervisor.sh",
    "generation_capacity_pipeline_lineage_observer.sh",
    "generation_capacity_control_plane_continuity_archive.sh",
    "generation_capacity_control_plane_process_snapshot.sh",
    "generation_capacity_control_plane_process_relaunch_after_restart.sh",
    "generation_stability_ema_teacher_full_readiness_after_gate.sh",
    "generation_stability_ema_teacher_full_readiness_bridge.sh",
    "generation_stability_ema_teacher_full_readiness_waiter.sh",
    "generation_stability_ema_teacher_posttraining_supervisor.sh",
    "generation_stability_ema_teacher_full_matched_300k_after_gate.sh",
    "generation_stability_ema_teacher_full_posteval_50k.sh",
    "generation_stability_ema_teacher_export_inference_artifacts.sh",
    "generation_stability_ema_teacher_completion_audit.sh",
)
ENTRYPOINTS = {
    "audit_generation_stability_completion.py",
    "audit_generation_training_progress.py",
    "audit_large_scale_generation_completion.py",
    "build_generation_gate_report.py",
    "build_generation_class_fidelity_qualification.py",
    "build_generation_large_capacity_deployment_receipt.py",
    "build_generation_full_launch_receipt.py",
    "build_generation_full_readiness.py",
    "build_generation_full_readiness_bridge.py",
    "build_generation_milestone_report.py",
    "build_generation_release_receipt.py",
    "build_generation_stability_50k_summary.py",
    "build_generation_stability_distribution_support.py",
    "build_generation_stability_frozen_supplemental.py",
    "build_generation_stability_sampling_recovery.py",
    "build_generation_stability_sampling_confirmation.py",
    "build_generation_frozen_sample_support_audit.py",
    "build_generation_stability_50k_convergence_audit.py",
    "build_generation_quality_bridge_preparation.py",
    "build_generation_quality_bridge_launch_receipt.py",
    "build_generation_quality_bridge_result.py",
    "build_generation_capacity_probe_execution_authorization.py",
    "build_generation_capacity_probe_launch_receipt.py",
    "build_generation_capacity_probe_result.py",
    "build_generation_quality_bridge_followup_decision.py",
    "build_generation_stability_qualification.py",
    "build_generation_visual_audit.py",
    "build_large_scale_generation_comparison.py",
    "check_generation_storage_capacity.py",
    "check_generation_deployment_conflicts.py",
    "check_generation_runbook_syntax.py",
    "evaluate_generation_checkpoint.py",
    "evaluate_generation_class_fidelity.py",
    "evaluate_generation_metrics.py",
    "evaluate_generation_rollout_stability.py",
    "export_generation_inference_artifact.py",
    "generate_samples.py",
    "infer_generation.py",
    "monitor_generation_pair.py",
    "monitor_generation_capacity_probe.py",
    "preflight_generation_sampling.py",
    "print_generation_workspace_paths.py",
    "run_generation_stability_50k_posteval_waiter.py",
    "run_generation_stability_frozen_supplemental_waiter.py",
    "run_generation_stability_full_readiness_waiter.py",
    "run_generation_stability_posttraining_supervisor.py",
    "run_generation_capacity_probe_execution_supervisor.py",
    "run_generation_capacity_scaling_50k_supervisor.py",
    "run_generation_capacity_completion_100k_supervisor.py",
    "wait_for_generation_capacity_completion_100k_result.py",
    "wait_for_generation_capacity_full_300k_readiness_decision.py",
    "validate_generation_capacity_full_300k_readiness_decision.py",
    "build_generation_capacity_full_300k_readiness.py",
    "verify_generation_capacity_full_300k_readiness.py",
    "run_generation_capacity_full_300k_readiness_supervisor.py",
    "verify_generation_capacity_full_300k_training_launch_receipt.py",
    "run_generation_capacity_full_300k_training_supervisor.py",
    "verify_generation_training_authorization.py",
    "run_generation_capacity_full_300k_posteval_supervisor.py",
    "run_generation_capacity_full_300k_finalization_supervisor.py",
    "observe_generation_capacity_pipeline_lineage.py",
    "build_generation_control_plane_continuity_archive.py",
    "capture_generation_control_plane_process_snapshot.py",
    "verify_generation_control_plane_continuity_archive.py",
    "verify_generation_control_plane_process_snapshot.py",
    "build_generation_control_plane_process_relaunch_readiness.py",
    "execute_generation_control_plane_process_relaunch.py",
    "audit_generation_capacity_full_completion.py",
    "build_generation_capacity_scaling_launch_receipt.py",
    "archive_generation_capacity_completion_sources.py",
    "restore_generation_capacity_completion_sources.py",
    "build_generation_capacity_completion_launch_receipt.py",
    "verify_generation_capacity_scaling_decision.py",
    "verify_generation_capacity_completion_decision.py",
    "verify_generation_capacity_scaling_launch_receipt.py",
    "verify_generation_capacity_completion_source_archive.py",
    "verify_generation_capacity_completion_launch_receipt.py",
    "validate_generation_capacity_scaling_training.py",
    "verify_generation_capacity_scaling_training.py",
    "validate_generation_capacity_completion_training.py",
    "verify_generation_capacity_completion_training.py",
    "wait_for_generation_capacity_scaling_decision.py",
    "run_generation_stage_once.py",
    "run_generation_training_watchdog.py",
    "wait_for_generation_quality_bridge_exposure_followup.py",
    "select_generation_sampling_batch.py",
    "select_generation_training_runtime.py",
    "train_generation.py",
    "validate_generation_configs.py",
    "validate_generation_full_launch_receipt.py",
    "validate_generation_full_readiness_bridge.py",
    "validate_generation_gate_report.py",
    "validate_generation_large_capacity_deployment.py",
    "validate_generation_milestone_report.py",
    "validate_generation_stability_scaling_decision.py",
    "validate_generation_stability_sampling_execution_approval.py",
    "validate_generation_quality_bridge_preparation.py",
    "validate_generation_quality_bridge_execution_approval.py",
    "validate_generation_quality_bridge_launch_receipt.py",
    "validate_generation_stability_frozen_posteval.py",
    "validate_generation_training_completion.py",
    "validate_generation_training_pair.py",
    "verify_generation_stability_frozen_class_fidelity.py",
    "verify_generation_quality_bridge_result.py",
    "verify_generation_quality_bridge_followup_decision.py",
    "verify_generation_capacity_probe_preparation.py",
    "verify_generation_capacity_probe_execution_authorization.py",
    "verify_generation_capacity_probe_launch_receipt.py",
    "verify_generation_capacity_probe_result.py",
    "validate_generation_capacity_probe_training.py",
    "verify_generation_stability_frozen_supplemental.py",
    "wait_for_generation_capacity_probe_preparation.py",
    "wait_for_generation_checkpoint_references.py",
    "write_generation_pipeline_status.py",
    "write_generation_deployment_receipt.py",
}
QUALITY_BRIDGE_DIRECT_ENTRYPOINTS = (
    "build_generation_quality_bridge_preparation.py",
    "validate_generation_quality_bridge_preparation.py",
    "validate_generation_quality_bridge_execution_approval.py",
    "build_generation_quality_bridge_launch_receipt.py",
    "validate_generation_quality_bridge_launch_receipt.py",
    "build_generation_quality_bridge_result.py",
    "verify_generation_quality_bridge_result.py",
    "build_generation_quality_bridge_followup_decision.py",
    "verify_generation_quality_bridge_followup_decision.py",
    "build_generation_capacity_probe_preparation.py",
    "verify_generation_capacity_probe_preparation.py",
    "wait_for_generation_capacity_probe_preparation.py",
    "build_generation_capacity_probe_execution_authorization.py",
    "verify_generation_capacity_probe_execution_authorization.py",
    "build_generation_capacity_probe_launch_receipt.py",
    "verify_generation_capacity_probe_launch_receipt.py",
    "build_generation_capacity_probe_result.py",
    "build_generation_capacity_scaling_decision.py",
    "verify_generation_capacity_scaling_decision.py",
    "verify_generation_capacity_probe_result.py",
    "monitor_generation_capacity_probe.py",
    "run_generation_capacity_probe_execution_supervisor.py",
    "wait_for_generation_capacity_scaling_decision.py",
    "build_generation_capacity_completion_decision.py",
    "verify_generation_capacity_completion_decision.py",
    "wait_for_generation_capacity_completion_decision.py",
    "validate_generation_capacity_probe_training.py",
    "archive_generation_capacity_completion_sources.py",
    "restore_generation_capacity_completion_sources.py",
    "verify_generation_capacity_completion_source_archive.py",
    "build_generation_capacity_completion_launch_receipt.py",
    "verify_generation_capacity_completion_launch_receipt.py",
    "validate_generation_capacity_completion_training.py",
    "verify_generation_capacity_completion_training.py",
    "run_generation_capacity_completion_100k_supervisor.py",
    "build_generation_capacity_completion_100k_result.py",
    "verify_generation_capacity_completion_100k_result.py",
    "wait_for_generation_capacity_completion_100k_result.py",
    "build_generation_capacity_full_300k_readiness_decision.py",
    "verify_generation_capacity_full_300k_readiness_decision.py",
    "wait_for_generation_capacity_full_300k_readiness_decision.py",
    "validate_generation_capacity_full_300k_readiness_decision.py",
    "build_generation_capacity_full_300k_readiness.py",
    "verify_generation_capacity_full_300k_readiness.py",
    "run_generation_capacity_full_300k_readiness_supervisor.py",
    "build_generation_capacity_full_300k_training_launch_receipt.py",
    "verify_generation_capacity_full_300k_training_launch_receipt.py",
    "run_generation_capacity_full_300k_training_supervisor.py",
    "verify_generation_training_authorization.py",
    "run_generation_capacity_full_300k_posteval_supervisor.py",
    "run_generation_capacity_full_300k_finalization_supervisor.py",
    "observe_generation_capacity_pipeline_lineage.py",
    "build_generation_control_plane_continuity_archive.py",
    "verify_generation_control_plane_continuity_archive.py",
    "restore_generation_control_plane_continuity_archive.py",
    "audit_generation_capacity_full_completion.py",
)


def _run_help(script: str) -> str:
    original_argv = sys.argv
    stdout = io.StringIO()
    stderr = io.StringIO()
    try:
        sys.argv = [str(ROOT / "scripts" / script), "--help"]
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                runpy.run_path(sys.argv[0], run_name="__main__")
            except SystemExit as error:
                assert error.code == 0, stderr.getvalue()
            else:
                raise AssertionError(f"{script} --help did not exit through argparse")
    finally:
        sys.argv = original_argv
    return stdout.getvalue()


def _runbook_contracts() -> dict[str, set[str]]:
    contracts: dict[str, set[str]] = {}
    for name in RUNBOOKS:
        lines = (ROOT / "artifacts/runbooks" / name).read_text(encoding="utf-8").splitlines()
        index = 0
        while index < len(lines):
            match = COMMAND_PATTERN.search(lines[index])
            if match is None:
                index += 1
                continue
            command_lines = [lines[index]]
            while command_lines[-1].rstrip().endswith("\\"):
                index += 1
                command_lines.append(lines[index])
            command = "\n".join(command_lines)
            parts = re.split(r"(?m)^\s*--\s*\\\s*$", command, maxsplit=1)
            for part in parts:
                nested_match = COMMAND_PATTERN.search(part)
                if nested_match is None:
                    continue
                script = nested_match.group(1)
                options = set(re.findall(r"--[a-z0-9-]+", part))
                contracts.setdefault(script, set()).update(options)
            index += 1
    return contracts


def test_formal_generation_runbook_entrypoints_have_live_cli_contracts() -> None:
    contracts = _runbook_contracts()
    assert set(contracts) == ENTRYPOINTS
    for script, required_options in contracts.items():
        assert required_options, f"{script} runbook command has no named options"
        help_text = _run_help(script)
        missing = sorted(option for option in required_options if option not in help_text)
        assert missing == [], f"{script} help is missing runbook options: {missing}"


def test_quality_bridge_direct_entrypoints_import_under_runbook_pythonpath() -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(ROOT), str(ROOT / "src"))
    )
    for script in QUALITY_BRIDGE_DIRECT_ENTRYPOINTS:
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script), "--help"],
            cwd=ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"{script}: {result.stderr}"
