from __future__ import annotations

import ast
import copy
from pathlib import Path

import pytest

from cofitok.generation.conditioning_ranking_posttraining_sampling import (
    EXPECTED_HELDOUT_DECISION,
    EXPECTED_OUTPUT_ROOT,
    IDLE_GPU_EVIDENCE_ROLE,
    RUNBOOK_RELATIVE_PATH,
    STAGE,
)
from scripts import (
    run_generation_conditioning_ranking_posttraining_sampling_confirmation_supervisor as supervisor,
)


ROOT = Path(__file__).resolve().parents[1]


def _git() -> dict:
    return {
        "revision": "a" * 40,
        "branch": (
            "scale/generation-label-ranking-posttraining-"
            "5k-sampling-confirmation-v1"
        ),
        "tracked_dirty": False,
    }


def _identity(path: str, character: str) -> dict:
    return {"path": path, "bytes": 123, "sha256": character * 64}


def _heldout(decision: dict) -> dict:
    return {"decision": decision}


def _observations() -> list[dict]:
    return [
        {
            "poll_index": index + 1,
            "observed_at_unix": 1_000.0 + index,
            "gpu_compute_pids": [],
        }
        for index in range(5)
    ]


def test_supervisor_selects_only_the_exact_shared_heldout_pass() -> None:
    assert (
        supervisor.heldout_route(
            _heldout(copy.deepcopy(EXPECTED_HELDOUT_DECISION))
        )
        == "selected"
    )

    asymmetric = copy.deepcopy(EXPECTED_HELDOUT_DECISION)
    asymmetric["method_passes"]["cofitok"] = False
    asymmetric["shared_semantic_alignment_recovery_supported"] = False
    asymmetric["recommended_next_action"] = (
        "reject_shared_repair_due_method_asymmetry"
    )
    assert supervisor.heldout_route(_heldout(asymmetric)) == "not_selected"

    failed = copy.deepcopy(EXPECTED_HELDOUT_DECISION)
    failed["method_passes"] = {"cofitok": False, "dense_identity": False}
    failed["shared_semantic_alignment_recovery_supported"] = False
    failed["recommended_next_action"] = (
        "revise_training_time_semantic_alignment_objective"
    )
    assert supervisor.heldout_route(_heldout(failed)) == "not_selected"

    malformed = copy.deepcopy(asymmetric)
    malformed["recommended_next_action"] = "launch_full_300k"
    assert supervisor.heldout_route(_heldout(malformed)) == "malformed"


def test_supervisor_gpu_parser_and_five_poll_evidence_are_fail_closed() -> None:
    assert supervisor.parse_gpu_process_pids("") == []
    assert supervisor.parse_gpu_process_pids("123\n456\n123\n") == [123, 456]
    with pytest.raises(ValueError, match="malformed"):
        supervisor.parse_gpu_process_pids("123, python\n")

    report = supervisor.build_idle_gpu_evidence(
        git=_git(),
        output_root=EXPECTED_OUTPUT_ROOT,
        observations=_observations(),
        required_polls=5,
    )
    assert report["role"] == IDLE_GPU_EVIDENCE_ROLE
    assert report["stage"] == STAGE
    assert report["observations"] == _observations()

    wrong_index = _observations()
    wrong_index[-1]["poll_index"] = 4
    with pytest.raises(ValueError, match="observation differs"):
        supervisor.build_idle_gpu_evidence(
            git=_git(),
            output_root=EXPECTED_OUTPUT_ROOT,
            observations=wrong_index,
            required_polls=5,
        )


def test_launch_receipt_is_one_shot_and_permanently_non_authorizing() -> None:
    report = supervisor.build_launch_receipt(
        git=_git(),
        output_root=EXPECTED_OUTPUT_ROOT,
        heldout_identity=_identity("/evidence/heldout.json", "1"),
        preparation_identity=_identity("/evidence/preparation.json", "2"),
        standing_identity=_identity("/evidence/standing.json", "3"),
        idle_identity=_identity("/evidence/idle.json", "4"),
        runbook_identity=_identity(
            f"/checkout/{RUNBOOK_RELATIVE_PATH}", "5"
        ),
        classifier={"name": "resnet50_imagenet1k_v2"},
    )

    assert report["status"] == "reserved"
    assert report["boundary"]["runbook_launch_count_maximum"] == 1
    assert report["boundary"]["automatic_runbook_relaunch_allowed"] is False
    assert report["boundary"]["training_allowed"] is False
    assert report["boundary"]["checkpoint_promotion_allowed"] is False
    assert report["boundary"]["full_100k_or_300k_launch_allowed"] is False
    assert report["boundary"]["release_authorization_allowed"] is False


def test_runbook_pins_step5000_independent_stream_and_complete_sequence() -> None:
    source = (ROOT / RUNBOOK_RELATIVE_PATH).read_text(encoding="utf-8")

    assert '[[ -z "$(git status --porcelain)" ]]' in source
    assert "nvidia-smi --query-compute-apps=pid" in source
    assert "checkpoint_step_00005000.pt" in source
    assert "--num-samples 5000" in source
    assert "--sample-steps 50" in source
    assert "--seed 506020" in source
    assert "--start-index 5000" in source
    assert "--sampling-stage \"$STAGE\"" in source
    assert source.count("run_sampling ") == 4
    assert source.count("run_metrics ") == 4
    assert source.count("run_paired_class_fidelity ") == 2
    assert source.count(
        "build_generation_conditioning_ranking_posttraining_sampling_confirmation.py"
    ) == 2
    assert "train_generation.py" not in source
    assert "--resume" in source


def test_supervisor_never_signals_or_relaunches_and_authenticates_training_receipt() -> None:
    source_path = (
        ROOT
        / "scripts"
        / "run_generation_conditioning_ranking_posttraining_sampling_confirmation_supervisor.py"
    )
    source = source_path.read_text(encoding="utf-8")

    assert "os.kill" not in source
    assert ".terminate(" not in source
    assert ".kill(" not in source
    assert "launch_path.exists()" in source
    assert source.count("subprocess.Popen(") == 1
    assert "name=\"5K training execution receipt\"" in source
    assert supervisor.SUPERVISOR_BOUNDARY[
        "automatic_runbook_relaunch_allowed"
    ] is False
    assert supervisor.SUPERVISOR_BOUNDARY["training_allowed"] is False
    assert supervisor.SUPERVISOR_BOUNDARY[
        "unrelated_process_signaling_allowed"
    ] is False


def test_every_supervisor_status_write_supplies_its_output_path() -> None:
    source = (
        ROOT
        / "scripts"
        / "run_generation_conditioning_ranking_posttraining_sampling_confirmation_supervisor.py"
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
