from __future__ import annotations

import contextlib
import io
import re
import runpy
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
    "build_generation_large_capacity_deployment_receipt.py",
    "build_generation_full_launch_receipt.py",
    "build_generation_full_readiness.py",
    "build_generation_full_readiness_bridge.py",
    "build_generation_milestone_report.py",
    "build_generation_release_receipt.py",
    "build_generation_stability_50k_summary.py",
    "build_generation_stability_distribution_support.py",
    "build_generation_stability_frozen_supplemental.py",
    "build_generation_stability_qualification.py",
    "build_generation_visual_audit.py",
    "build_large_scale_generation_comparison.py",
    "check_generation_storage_capacity.py",
    "check_generation_deployment_conflicts.py",
    "check_generation_runbook_syntax.py",
    "evaluate_generation_checkpoint.py",
    "evaluate_generation_metrics.py",
    "evaluate_generation_rollout_stability.py",
    "export_generation_inference_artifact.py",
    "generate_samples.py",
    "infer_generation.py",
    "monitor_generation_pair.py",
    "preflight_generation_sampling.py",
    "print_generation_workspace_paths.py",
    "run_generation_stability_50k_posteval_waiter.py",
    "run_generation_stability_frozen_supplemental_waiter.py",
    "run_generation_stability_full_readiness_waiter.py",
    "run_generation_stability_posttraining_supervisor.py",
    "run_generation_stage_once.py",
    "run_generation_training_watchdog.py",
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
    "validate_generation_stability_frozen_posteval.py",
    "validate_generation_training_completion.py",
    "validate_generation_training_pair.py",
    "write_generation_pipeline_status.py",
    "write_generation_deployment_receipt.py",
}


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
