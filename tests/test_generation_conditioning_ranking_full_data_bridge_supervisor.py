from __future__ import annotations

import argparse
import ast
import copy
import json
from pathlib import Path

import pytest

from cofitok.generation.conditioning_ranking_full_data_bridge import (
    POSTTRAINING_DECISION,
    POSTTRAINING_GIT,
    POSTTRAINING_REPORT_ROLE,
)
from cofitok.generation.conditioning_ranking_posttraining_sampling import (
    CLAIM_BOUNDARY as POSTTRAINING_CLAIM_BOUNDARY,
    EXPECTED_OUTPUT_ROOT as POSTTRAINING_OUTPUT_ROOT,
    STAGE as POSTTRAINING_STAGE,
)
from cofitok.generation.conditioning_ranking_probe import (
    FOLLOWUP_DECISION_CATEGORY,
    FOLLOWUP_DECISION_ID,
    FOLLOWUP_DECISION_ROLE,
    QUALITY_BRIDGE_EXECUTION_GIT,
    STANDING_AUTHORIZATION_INTERPRETATION,
    STANDING_AUTHORIZATION_ROLE,
    STANDING_AUTHORIZATION_SAFETY_BOUNDARIES,
    STANDING_AUTHORIZATION_TEXT,
)
from cofitok.inference_replay import file_identity
from scripts import (
    run_generation_conditioning_ranking_full_data_bridge_preparation_supervisor
    as supervisor,
)


ROOT = Path(__file__).resolve().parents[1]
REVISION = "a" * 40
BRANCH = "scale/generation-conditioning-ranking-full-data-100k-v1"


def _identity(path: str, character: str = "b") -> dict:
    return {"path": path, "bytes": 123, "sha256": character * 64}


def _standing_authorization() -> dict:
    return {
        "schema_version": 1,
        "role": STANDING_AUTHORIZATION_ROLE,
        "status": "active",
        "instruction": {
            "language": "zh-CN",
            "exact_text": STANDING_AUTHORIZATION_TEXT,
            "interpretation": STANDING_AUTHORIZATION_INTERPRETATION,
            "received_at": "2026-08-13T00:11:00+08:00",
        },
        "preserved_safety_boundaries": copy.deepcopy(
            STANDING_AUTHORIZATION_SAFETY_BOUNDARIES
        ),
    }


def _followup(route_id: str = FOLLOWUP_DECISION_ID) -> dict:
    category = supervisor.KNOWN_FOLLOWUP_ROUTES[route_id]
    failed_name = (
        "class_fidelity"
        if route_id == FOLLOWUP_DECISION_ID
        else "cofitok_absolute_fid"
    )
    checks = [
        {"name": name, "passed": name != failed_name}
        for name in sorted(supervisor.EXPECTED_FOLLOWUP_CHECKS)
    ]
    return {
        "schema_version": 1,
        "status": "completed",
        "role": FOLLOWUP_DECISION_ROLE,
        "decision_builder_git": copy.deepcopy(
            supervisor.FOLLOWUP_DECISION_BUILDER_GIT
        ),
        "quality_bridge_execution_git": copy.deepcopy(
            QUALITY_BRIDGE_EXECUTION_GIT
        ),
        "source_reports": {
            "quality_bridge_result": _identity("/evidence/quality.json", "c"),
            "milestones": {
                "50000": _identity("/evidence/milestone_50000.json", "d"),
                "100000": _identity("/evidence/milestone_100000.json", "e"),
            },
        },
        "terminal_quality": {
            "status": "hold",
            "failed_checks": [failed_name],
            "checks": checks,
        },
        "recommended_next_stage": {
            "id": route_id,
            "category": category,
            "objective": "Collect the exact next bounded evidence.",
            "trigger": {"failed_checks": [failed_name]},
            "required_next_evidence": "A source-bound replayed diagnostic.",
            "execution_ready": False,
            "gpu_execution_allowed": False,
            "full_300k_launch_allowed": False,
            "release_authorization_allowed": False,
        },
        "claim_policy": {
            "experiment_selection_only": True,
            "milestone_trend_is_formal_quality_evidence": False,
            "terminal_result_is_promotion_gate": False,
            "cross_tier_numeric_ranking_allowed": False,
        },
        "authorization_boundary": copy.deepcopy(
            supervisor.FOLLOWUP_AUTHORIZATION_BOUNDARY
        ),
    }


def _posttraining(*, cofitok: bool, dense: bool) -> dict:
    passes = {"cofitok": cofitok, "dense_identity": dense}
    if cofitok and dense:
        decision = copy.deepcopy(POSTTRAINING_DECISION)
    else:
        decision = {
            "method_passes": passes,
            "shared_posttraining_generated_class_alignment_recovery_confirmed": False,
            "cofitok_specific_advantage_claim_allowed": False,
            "recommended_next_action": (
                "reject_shared_repair_due_posttraining_method_asymmetry"
                if cofitok or dense
                else "revise_training_time_semantic_alignment_objective"
            ),
        }
    return {
        "schema_version": 1,
        "role": POSTTRAINING_REPORT_ROLE,
        "status": "completed",
        "stage": POSTTRAINING_STAGE,
        "output_root": POSTTRAINING_OUTPUT_ROOT,
        "git": copy.deepcopy(POSTTRAINING_GIT),
        "sources": {},
        "methods": {
            "cofitok": {"pass": cofitok},
            "dense_identity": {"pass": dense},
        },
        "decision": decision,
        "claim_boundary": copy.deepcopy(POSTTRAINING_CLAIM_BOUNDARY),
    }


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _args(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> argparse.Namespace:
    project = tmp_path / "project"
    project.mkdir()
    control = tmp_path / "control"
    standing = tmp_path / "sources" / "standing.json"
    _write(standing, _standing_authorization())
    future_output_root = tmp_path / "future-ranked-output"
    monkeypatch.setattr(
        supervisor,
        "EXPECTED_OUTPUT_ROOT",
        future_output_root.as_posix(),
    )
    monkeypatch.setattr(
        supervisor,
        "_assert_exact_clean_git",
        lambda *_args, **_kwargs: None,
    )
    return argparse.Namespace(
        project=project,
        expected_revision=REVISION,
        expected_branch=BRANCH,
        standing_authorization=standing,
        expected_standing_authorization_sha256=file_identity(standing)["sha256"],
        quality_bridge_followup=tmp_path / "sources" / "followup.json",
        posttraining_confirmation=tmp_path / "sources" / "posttraining.json",
        output_root=future_output_root.as_posix(),
        preparation_output=control / "preparation.json",
        status_output=control / "supervisor_status.json",
        pid_file=control / "supervisor.pid.json",
        poll_seconds=1.0,
        timeout_seconds=100.0,
    )


def _patch_posttraining_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        supervisor,
        "replay_posttraining_sampling_confirmation",
        lambda path: (_read(path), file_identity(path)),
    )


def test_route_validation_is_exact_and_fail_closed() -> None:
    selected = _followup()
    assert supervisor.followup_route(selected) == "selected"

    other = _followup("prepare_matched_250m_capacity_qualification_probe")
    assert supervisor.followup_route(other) == "not_selected"

    malformed = copy.deepcopy(selected)
    malformed["authorization_boundary"]["full_300k_launch_allowed"] = True
    assert supervisor.followup_route(malformed) == "malformed"

    assert supervisor.posttraining_route(
        _posttraining(cofitok=True, dense=True)
    ) == "selected"
    assert supervisor.posttraining_route(
        _posttraining(cofitok=True, dense=False)
    ) == "not_selected"
    assert supervisor.posttraining_route(
        _posttraining(cofitok=False, dense=False)
    ) == "not_selected"

    malformed_posttraining = _posttraining(cofitok=True, dense=False)
    malformed_posttraining["decision"]["recommended_next_action"] = "launch_300k"
    assert supervisor.posttraining_route(malformed_posttraining) == "malformed"


def test_supervisor_waits_for_followup_then_exact_posttraining_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path, monkeypatch)

    def stop_after_wait(_seconds: float) -> None:
        raise StopIteration

    monkeypatch.setattr(supervisor.time, "sleep", stop_after_wait)
    with pytest.raises(StopIteration):
        supervisor._run(args)
    assert _read(args.status_output)["detail"] == (
        "waiting_for_quality_bridge_followup_decision"
    )
    assert not args.preparation_output.exists()

    _write(args.quality_bridge_followup, _followup())
    with pytest.raises(StopIteration):
        supervisor._run(args)
    assert _read(args.status_output)["detail"] == (
        "waiting_for_exact_posttraining_5k_sampling_confirmation"
    )
    assert not args.preparation_output.exists()


def test_wrong_followup_route_completes_without_preparation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path, monkeypatch)
    _write(
        args.quality_bridge_followup,
        _followup("prepare_matched_250m_capacity_qualification_probe"),
    )

    assert supervisor._run(args) == 0
    status = _read(args.status_output)
    assert status["status"] == "completed"
    assert status["detail"] == (
        "full_data_ranking_bridge_not_selected_by_quality_bridge"
    )
    assert not args.preparation_output.exists()
    assert not Path(args.output_root).exists()


@pytest.mark.parametrize("passes", [(True, False), (False, True), (False, False)])
def test_nonpassing_posttraining_completes_without_preparation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    passes: tuple[bool, bool],
) -> None:
    args = _args(tmp_path, monkeypatch)
    _write(args.quality_bridge_followup, _followup())
    _write(
        args.posttraining_confirmation,
        _posttraining(cofitok=passes[0], dense=passes[1]),
    )
    _patch_posttraining_replay(monkeypatch)

    assert supervisor._run(args) == 0
    status = _read(args.status_output)
    assert status["status"] == "completed"
    assert status["detail"] == (
        "full_data_ranking_bridge_not_selected_by_posttraining_sampling_confirmation"
    )
    assert not args.preparation_output.exists()
    assert not Path(args.output_root).exists()


def test_exact_double_pass_writes_once_and_replays_existing_preparation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path, monkeypatch)
    _write(args.quality_bridge_followup, _followup())
    _write(
        args.posttraining_confirmation,
        _posttraining(cofitok=True, dense=True),
    )
    _patch_posttraining_replay(monkeypatch)
    expected = {
        "schema_version": 1,
        "role": "test_full_data_ranking_preparation",
        "gpu_execution_authorized": False,
    }
    monkeypatch.setattr(
        supervisor,
        "_build_preparation_report",
        lambda **_kwargs: copy.deepcopy(expected),
    )
    monkeypatch.setattr(
        supervisor,
        "validate_full_data_ranked_bridge_preparation",
        lambda report, **_kwargs: copy.deepcopy(report),
    )

    assert supervisor._run(args) == 0
    first = args.preparation_output.read_bytes()
    assert _read(args.status_output)["detail"] == (
        "full_data_ranking_bridge_preparation_completed"
    )
    assert not Path(args.output_root).exists()

    assert supervisor._run(args) == 0
    assert args.preparation_output.read_bytes() == first

    _write(args.preparation_output, {"tampered": True})
    with pytest.raises(ValueError, match="resume manifest does not match"):
        supervisor._run(args)


def test_exact_clean_git_and_control_path_guards_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    expected = {
        "revision": REVISION,
        "branch": BRANCH,
        "tracked_dirty": False,
    }
    monkeypatch.setattr(
        supervisor,
        "git_provenance",
        lambda _project: {**expected, "revision": "b" * 40},
    )
    monkeypatch.setattr(supervisor, "_full_git_status", lambda _project: "")
    with pytest.raises(ValueError, match="exact fully clean"):
        supervisor._assert_exact_clean_git(project, expected_git=expected)

    monkeypatch.setattr(supervisor, "git_provenance", lambda _project: expected)
    monkeypatch.setattr(
        supervisor,
        "_full_git_status",
        lambda _project: "?? unexpected.json",
    )
    with pytest.raises(ValueError, match="exact fully clean"):
        supervisor._assert_exact_clean_git(project, expected_git=expected)

    future = tmp_path / "future"
    with pytest.raises(ValueError, match="outside the future"):
        supervisor._validate_control_paths(
            project=project,
            future_output_root=future,
            preparation_output=tmp_path / "control" / "preparation.json",
            status_output=future / "status.json",
            pid_file=tmp_path / "control" / "pid.json",
        )


def test_timeout_is_terminal_and_creates_no_preparation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path, monkeypatch)
    args.timeout_seconds = 1.0
    moments = iter([0.0, 2.0])
    monkeypatch.setattr(supervisor.time, "monotonic", lambda: next(moments))

    assert supervisor._run(args) == 4
    status = _read(args.status_output)
    assert status["status"] == "failed"
    assert status["detail"] == "timeout_before_full_data_ranking_preparation"
    assert not args.preparation_output.exists()


def test_supervisor_is_cpu_only_and_never_launches_or_signals_processes() -> None:
    source_path = (
        ROOT
        / "scripts"
        / (
            "run_generation_conditioning_ranking_full_data_bridge_"
            "preparation_supervisor.py"
        )
    )
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    assert "nvidia-smi" not in source
    assert "subprocess.Popen" not in source
    assert "os.kill" not in source
    assert ".terminate(" not in source
    assert ".kill(" not in source
    assert "train_generation.py" not in source
    assert "generate_samples.py" not in source
    assert supervisor.SUPERVISOR_BOUNDARY["cpu_only"] is True
    assert supervisor.SUPERVISOR_BOUNDARY["gpu_query_allowed"] is False
    assert supervisor.SUPERVISOR_BOUNDARY["training_allowed"] is False
    assert supervisor.SUPERVISOR_BOUNDARY["sampling_allowed"] is False
    assert supervisor.SUPERVISOR_BOUNDARY[
        "experiment_child_process_launch_allowed"
    ] is False
    assert supervisor.SUPERVISOR_BOUNDARY["process_signaling_allowed"] is False

    status_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_write_status"
    ]
    assert status_calls
    assert all(
        call.args or any(keyword.arg == "path" for keyword in call.keywords)
        for call in status_calls
    )
