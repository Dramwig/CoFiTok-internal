from __future__ import annotations

import ast
from pathlib import Path

import pytest

from cofitok.generation.conditioning_ranking_probe import (
    FOLLOWUP_AUTHORIZATION_BOUNDARY,
    FOLLOWUP_DECISION_BUILDER_GIT,
    QUALITY_BRIDGE_EXECUTION_GIT,
)
from scripts import run_generation_conditioning_ranking_probe_supervisor as supervisor


ROOT = Path(__file__).resolve().parents[1]


def _decision(identifier: str) -> dict:
    names = (
        "cofitok_absolute_fid",
        "matched_fid_tolerance",
        "cofitok_precision_floor",
        "cofitok_recall_floor",
        "matched_precision_tolerance",
        "matched_recall_tolerance",
        "matched_endpoint_tolerance",
        "ordered_prefix_rank",
        "coarse_token_utilization",
        "restricted_synthesis_zero_token",
        "shuffle_mismatch",
        "class_fidelity",
    )
    quality_identity = {
        "path": "/evidence/quality.json",
        "bytes": 100,
        "sha256": "a" * 64,
    }
    return {
        "schema_version": 2,
        "status": "completed",
        "role": "stability_quality_bridge_followup_experiment_decision",
        "decision_builder_git": dict(FOLLOWUP_DECISION_BUILDER_GIT),
        "quality_bridge_execution_git": dict(QUALITY_BRIDGE_EXECUTION_GIT),
        "source_reports": {
            "quality_bridge_result": quality_identity,
            "milestones": {
                "50000": {
                    "path": "/evidence/50000.json",
                    "bytes": 100,
                    "sha256": "b" * 64,
                },
                "100000": {
                    "path": "/evidence/100000.json",
                    "bytes": 100,
                    "sha256": "c" * 64,
                },
            },
            "terminal_training_exposure": {
                "path": "/evidence/exposure.json",
                "bytes": 100,
                "sha256": "d" * 64,
            },
        },
        "terminal_quality": {
            "failed_checks": ["class_fidelity"],
            "checks": [
                {"name": name, "passed": name != "class_fidelity"}
                for name in names
            ],
        },
        "recommended_next_stage": {
            "id": identifier,
            "category": "class_conditioning_recovery",
            "execution_ready": False,
            "gpu_execution_allowed": False,
            "full_300k_launch_allowed": False,
            "release_authorization_allowed": False,
            "trigger": {"failed_checks": ["class_fidelity"]},
        },
        "claim_policy": {
            "experiment_selection_only": True,
            "terminal_result_is_promotion_gate": False,
        },
        "authorization_boundary": dict(FOLLOWUP_AUTHORIZATION_BOUNDARY),
    }


def test_supervisor_selects_only_exact_class_conditioning_route() -> None:
    assert (
        supervisor.followup_route(
            _decision("run_class_conditioning_fidelity_diagnostic")
        )
        == "selected"
    )
    assert (
        supervisor.followup_route(
            _decision("prepare_matched_250m_capacity_qualification_probe")
        )
        == "not_selected"
    )
    malformed = _decision("run_class_conditioning_fidelity_diagnostic")
    malformed["decision_builder_git"]["revision"] = "0" * 40
    assert supervisor.followup_route(malformed) == "invalid"

    mixed = _decision("run_class_conditioning_fidelity_diagnostic")
    mixed["terminal_quality"]["failed_checks"] = [
        "matched_fid_tolerance",
        "class_fidelity",
    ]
    mixed["terminal_quality"]["checks"][1]["passed"] = False
    mixed["recommended_next_stage"]["trigger"] = {
        "failed_checks": ["matched_fid_tolerance", "class_fidelity"]
    }
    assert supervisor.followup_route(mixed) == "invalid"


def test_supervisor_gpu_process_parser_is_fail_closed() -> None:
    assert supervisor.parse_gpu_process_pids("") == []
    assert supervisor.parse_gpu_process_pids("123\n456\n123\n") == [123, 456]
    assert supervisor.parse_gpu_process_pids("123, python\n") == [123]
    with pytest.raises(ValueError, match="unparseable"):
        supervisor.parse_gpu_process_pids("not-a-pid\n")


def test_supervisor_never_signals_or_authorizes_later_stages() -> None:
    assert supervisor.AUTHORIZATION_BOUNDARY[
        "unrelated_process_signaling_allowed"
    ] is False
    assert supervisor.AUTHORIZATION_BOUNDARY["full_training_launch_allowed"] is False
    assert supervisor.AUTHORIZATION_BOUNDARY["full_300k_launch_allowed"] is False
    source = (
        ROOT
        / "scripts"
        / "run_generation_conditioning_ranking_probe_supervisor.py"
    ).read_text(encoding="utf-8")
    assert "os.kill" not in source
    assert "terminate(" not in source
    assert "kill(" not in source
    assert "required_idle_polls" in source
    assert "terminal_system_claim_guard" in source
    assert "requested_class_visual_audit_status" in source
    assert "fcntl.LOCK_EX | fcntl.LOCK_NB" in source
    assert "expected_tree" in source


def test_every_supervisor_status_write_supplies_the_output_path() -> None:
    source = (
        ROOT
        / "scripts"
        / "run_generation_conditioning_ranking_probe_supervisor.py"
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
