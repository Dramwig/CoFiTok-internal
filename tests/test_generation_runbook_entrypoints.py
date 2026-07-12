from __future__ import annotations

import contextlib
import io
import re
import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNBOOKS = (
    "generation_10pct_posteval_2026-07-12.sh",
    "generation_full_milestone_eval.sh",
    "generation_full_posteval_50k.sh",
    "generation_export_inference_artifacts.sh",
    "generation_full_matched_300k_after_gate.sh",
    "generation_complete_pipeline_after_10pct.sh",
)
ENTRYPOINTS = {
    "audit_generation_training_progress.py",
    "audit_large_scale_generation_completion.py",
    "build_generation_gate_report.py",
    "build_generation_milestone_report.py",
    "build_generation_visual_audit.py",
    "build_large_scale_generation_comparison.py",
    "evaluate_generation_checkpoint.py",
    "evaluate_generation_metrics.py",
    "export_generation_inference_artifact.py",
    "generate_samples.py",
    "infer_generation.py",
    "migrate_generation_checkpoint_integrity.py",
    "preflight_generation_sampling.py",
    "select_generation_sampling_batch.py",
    "select_generation_training_runtime.py",
    "train_generation.py",
    "validate_generation_training_pair.py",
    "write_generation_pipeline_status.py",
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
            match = re.search(r"python scripts/([A-Za-z0-9_]+\.py)", lines[index])
            if match is None:
                index += 1
                continue
            script = match.group(1)
            command_lines = [lines[index]]
            while command_lines[-1].rstrip().endswith("\\"):
                index += 1
                command_lines.append(lines[index])
            options = set(re.findall(r"--[a-z0-9-]+", "\n".join(command_lines)))
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
