from __future__ import annotations

import ast
import copy
import inspect
from pathlib import Path

import pytest

from cofitok.generation.conditioning_ranking_sampling import (
    EXPECTED_OUTPUT_ROOT,
    EXPECTED_POSTEVALUATION_DECISION,
    IDLE_GPU_EVIDENCE_ROLE,
)
from scripts import (
    run_generation_conditioning_ranking_sampling_validation_supervisor as supervisor,
)
from scripts.build_generation_conditioning_ranking_sampling_validation import (
    replay_postevaluation,
)


ROOT = Path(__file__).resolve().parents[1]


def _postevaluation(decision: dict) -> dict:
    return {
        "schema_version": 1,
        "role": "generation_conditioning_ranking_four_arm_postevaluation",
        "status": "completed",
        "decision": decision,
    }


def _git() -> dict:
    return {
        "revision": "a" * 40,
        "branch": "scale/generation-label-ranking-standing-authorization-v1",
        "tracked_dirty": False,
    }


def _identity(name: str, character: str) -> dict:
    return {
        "path": f"/evidence/{name}.json",
        "bytes": 100 + len(name),
        "sha256": character * 64,
    }


def test_supervisor_routes_only_the_exact_shared_passing_postevaluation() -> None:
    assert (
        supervisor.postevaluation_route(
            _postevaluation(copy.deepcopy(EXPECTED_POSTEVALUATION_DECISION))
        )
        == "selected"
    )
    asymmetric = copy.deepcopy(EXPECTED_POSTEVALUATION_DECISION)
    asymmetric["method_passes"]["cofitok"] = False
    asymmetric["shared_semantic_alignment_recovery_supported"] = False
    asymmetric["recommended_next_action"] = (
        "reject_shared_repair_due_method_asymmetry"
    )
    assert supervisor.postevaluation_route(_postevaluation(asymmetric)) == "not_selected"

    failed = copy.deepcopy(EXPECTED_POSTEVALUATION_DECISION)
    failed["method_passes"] = {"cofitok": False, "dense_identity": False}
    failed["shared_semantic_alignment_recovery_supported"] = False
    failed["recommended_next_action"] = (
        "revise_training_time_semantic_alignment_objective"
    )
    assert supervisor.postevaluation_route(_postevaluation(failed)) == "not_selected"

    malformed = copy.deepcopy(asymmetric)
    malformed["recommended_next_action"] = (
        "consider_separately_authorized_matched_sampling_validation"
    )
    assert supervisor.postevaluation_route(_postevaluation(malformed)) == "invalid"


def test_supervisor_gpu_parser_is_fail_closed() -> None:
    assert supervisor.parse_gpu_process_pids("") == []
    assert supervisor.parse_gpu_process_pids("123\n456\n123\n") == [123, 456]
    assert supervisor.parse_gpu_process_pids("123, python\n") == [123]
    with pytest.raises(ValueError, match="unparseable"):
        supervisor.parse_gpu_process_pids("not-a-pid\n")


def test_idle_gpu_evidence_is_exactly_five_ordered_empty_observations() -> None:
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

    busy = copy.deepcopy(observations)
    busy[-1]["gpu_compute_pids"] = [99]
    with pytest.raises(ValueError, match="observation differs"):
        supervisor.build_idle_gpu_evidence(
            git=_git(),
            output_root=EXPECTED_OUTPUT_ROOT,
            observations=busy,
            required_polls=5,
        )
    with pytest.raises(ValueError, match="exactly five"):
        supervisor.build_idle_gpu_evidence(
            git=_git(),
            output_root=EXPECTED_OUTPUT_ROOT,
            observations=observations,
            required_polls=4,
        )


def test_launch_receipt_is_one_shot_and_non_authorizing() -> None:
    classifier = {
        "name": "resnet50_imagenet1k_v2",
        "weights_path": "/weights/resnet50.pth",
    }
    report = supervisor.build_launch_receipt(
        git=_git(),
        output_root=EXPECTED_OUTPUT_ROOT,
        postevaluation_identity=_identity("postevaluation", "1"),
        preparation_identity=_identity("preparation", "2"),
        standing_identity=_identity("standing", "3"),
        idle_identity=_identity("idle", "4"),
        runbook_identity=_identity("runbook", "5"),
        classifier=classifier,
    )
    assert report["role"] == supervisor.LAUNCH_RECEIPT_ROLE
    assert report["status"] == "launch_reserved"
    boundary = report["authorization_boundary"]
    assert boundary["runbook_launch_count_maximum"] == 1
    assert boundary["automatic_runbook_relaunch_allowed"] is False
    assert boundary["training_allowed"] is False
    assert boundary["full_100k_or_300k_launch_allowed"] is False
    assert boundary["release_authorization_allowed"] is False


def test_replay_api_can_authenticate_a_nonselected_postevaluation() -> None:
    signature = inspect.signature(replay_postevaluation)
    assert signature.parameters["require_sampling_selected"].default is True


def test_sampling_runbook_contains_the_complete_resumable_physical_sequence() -> None:
    source = (
        ROOT
        / "artifacts"
        / "runbooks"
        / "generation_conditioning_ranking_four_arm_sampling5k_v1.sh"
    ).read_text(encoding="utf-8")
    assert '[[ -z "$(git status --porcelain)" ]]' in source
    assert "nvidia-smi --query-compute-apps=pid" in source
    assert "prepare_generation_conditioning_ranking_sampling_validation.py" in source
    assert "select_generation_conditioning_ranking_sampling_batch.py" in source
    assert "build_generation_conditioning_ranking_sampling_execution_receipt.py" in source
    assert "verify_generation_conditioning_ranking_sampling_execution_receipt.py" in source
    assert source.count("run_sampling ") == 4
    assert source.count("run_metrics ") == 4
    assert "--num-samples 5000" in source
    assert "--sample-steps 50" in source
    assert "--seed 406020" in source
    assert "--start-index 0" in source
    assert "--guidance-scale 1.5" in source
    assert "--skip-prc" in source
    assert "evaluate_generation_conditioning_ranking_samples.py" in source
    assert "CUDA_VISIBLE_DEVICES=-1" in source
    assert "--cpu" in source
    assert source.count(
        "build_generation_conditioning_ranking_sampling_validation.py"
    ) == 2
    assert "--resume" in source


def test_supervisor_never_signals_processes_or_authorizes_later_training() -> None:
    source_path = (
        ROOT
        / "scripts"
        / "run_generation_conditioning_ranking_sampling_validation_supervisor.py"
    )
    source = source_path.read_text(encoding="utf-8")
    assert "os.kill" not in source
    assert "terminate(" not in source
    assert "kill(" not in source
    assert "required_idle_polls" in source
    assert "launch_receipt_path.exists()" in source
    assert supervisor.SUPERVISOR_BOUNDARY["training_allowed"] is False
    assert supervisor.SUPERVISOR_BOUNDARY[
        "unrelated_process_signaling_allowed"
    ] is False
    assert supervisor.SUPERVISOR_BOUNDARY[
        "asymmetric_or_failed_postevaluation_gpu_work_allowed"
    ] is False


def test_every_supervisor_status_write_supplies_the_output_path() -> None:
    source = (
        ROOT
        / "scripts"
        / "run_generation_conditioning_ranking_sampling_validation_supervisor.py"
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
