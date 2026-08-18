from __future__ import annotations

import ast
from pathlib import Path

import pytest

from cofitok.generation.conditioning_ranking_training_confirmation import (
    EXPECTED_OUTPUT_ROOT,
    IDLE_GPU_EVIDENCE_ROLE,
)
from scripts import (
    run_generation_conditioning_ranking_training_confirmation_supervisor as supervisor,
)


ROOT = Path(__file__).resolve().parents[1]


def _git() -> dict:
    return {
        "revision": "a" * 40,
        "branch": "scale/generation-label-ranking-5k-training-confirmation-v1",
        "tracked_dirty": False,
    }


def _identity(name: str, character: str) -> dict:
    return {
        "path": f"/evidence/{name}",
        "bytes": 100 + len(name),
        "sha256": character * 64,
    }


def test_supervisor_gpu_parser_is_fail_closed() -> None:
    assert supervisor.parse_gpu_process_pids("") == []
    assert supervisor.parse_gpu_process_pids("123\n456\n123\n") == [123, 456]
    assert supervisor.parse_gpu_process_pids("123, python\n") == [123]
    with pytest.raises(ValueError, match="unparseable"):
        supervisor.parse_gpu_process_pids("not-a-pid\n")


def test_supervisor_idle_evidence_is_exactly_five_consecutive_empty_polls() -> None:
    observations = [
        {
            "poll_index": index + 1,
            "observed_at_unix": 1_000.0 + index,
            "gpu_compute_pids": [],
        }
        for index in range(5)
    ]
    report = supervisor.build_idle_gpu_evidence(
        git=_git(),
        output_root=EXPECTED_OUTPUT_ROOT,
        observations=observations,
        required_polls=5,
    )
    assert report["role"] == IDLE_GPU_EVIDENCE_ROLE
    assert report["status"] == "pass"
    assert report["observations"] == observations

    with pytest.raises(ValueError, match="exactly five"):
        supervisor.build_idle_gpu_evidence(
            git=_git(),
            output_root=EXPECTED_OUTPUT_ROOT,
            observations=observations,
            required_polls=4,
        )


def test_launch_receipt_is_one_shot_and_cannot_authorize_followups() -> None:
    config_identities = {
        name: _identity(f"{name}.json", f"{index:x}")
        for index, name in enumerate(supervisor.CONFIG_RELATIVE_PATHS, start=1)
    }
    report = supervisor.build_launch_receipt(
        git=_git(),
        output_root=EXPECTED_OUTPUT_ROOT,
        sampling_identity=_identity("sampling.json", "5"),
        preparation_identity=_identity("preparation.json", "6"),
        standing_identity=_identity("standing.json", "7"),
        idle_identity=_identity("idle.json", "8"),
        execution_receipt_identity=_identity("execution.json", "9"),
        runbook_identity=_identity("runbook.sh", "a"),
        config_identities=config_identities,
    )
    assert report["role"] == supervisor.LAUNCH_RECEIPT_ROLE
    assert report["status"] == "launch_reserved"
    boundary = report["authorization_boundary"]
    assert boundary["training_allowed"] is True
    assert boundary["runbook_launch_count_maximum"] == 1
    assert boundary["automatic_runbook_relaunch_allowed"] is False
    assert boundary["sampling_allowed"] is False
    assert boundary["followup_training_allowed"] is False
    assert boundary["full_100k_or_300k_launch_allowed"] is False
    assert boundary["release_authorization_allowed"] is False
    assert boundary["source_not_selected_gpu_work_allowed"] is False


def test_training_confirmation_runbook_contains_exact_fresh_four_arm_sequence() -> None:
    source = (
        ROOT
        / "artifacts"
        / "runbooks"
        / "generation_conditioning_ranking_four_arm_train5k_confirmation_v1.sh"
    ).read_text(encoding="utf-8")
    assert '[[ -z "$(git status --porcelain)" ]]' in source
    assert "nvidia-smi --query-compute-apps=pid" in source
    assert "prepare_generation_conditioning_ranking_training_confirmation.py" in source
    assert (
        "verify_generation_conditioning_ranking_training_confirmation_execution_receipt.py"
        in source
    )
    assert source.count("scripts/train_generation.py") == 1
    assert source.count("run_one control_") == 2
    assert source.count("run_one ranked_") == 2
    assert "--expected-steps 5000" in source
    assert "--checkpoint-interval 1250" in source
    assert "--evaluation-interval 1250" in source
    assert "--required-checkpoint-steps 1250,2500,5000" in source
    assert "--resume" not in source
    assert "generate_samples.py" not in source
    assert "full_100k_or_300k_launch_allowed" in source
    assert "authorizes_followup_training" in source


def test_supervisor_never_signals_processes_or_relaunches() -> None:
    source_path = (
        ROOT
        / "scripts"
        / "run_generation_conditioning_ranking_training_confirmation_supervisor.py"
    )
    source = source_path.read_text(encoding="utf-8")
    assert "os.kill" not in source
    assert "terminate(" not in source
    assert "kill(" not in source
    assert "launch_receipt_path.exists()" in source
    assert "resume=False" in source
    assert supervisor.SUPERVISOR_BOUNDARY["unrelated_process_signaling_allowed"] is False
    assert supervisor.SUPERVISOR_BOUNDARY[
        "automatic_runbook_relaunch_allowed"
    ] is False
    assert supervisor.SUPERVISOR_BOUNDARY[
        "failed_or_malformed_source_gpu_work_allowed"
    ] is False


def test_every_supervisor_status_write_supplies_the_output_path() -> None:
    source = (
        ROOT
        / "scripts"
        / "run_generation_conditioning_ranking_training_confirmation_supervisor.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_write_status"
    ]
    assert calls
    assert all(
        call.args or any(keyword.arg == "path" for keyword in call.keywords)
        for call in calls
    )
