from __future__ import annotations

from pathlib import Path

import pytest

from scripts import run_generation_conditioning_ranking_probe_supervisor as supervisor


ROOT = Path(__file__).resolve().parents[1]


def _decision(identifier: str) -> dict:
    return {
        "schema_version": 1,
        "status": "completed",
        "role": "stability_quality_bridge_followup_experiment_decision",
        "quality_bridge_execution_git": {
            "revision": "cf0e5faa94bf4ab38d947b921935b3b765b5537a",
            "branch": "scale/generation-stability-quality-bridge-100k",
            "tracked_dirty": False,
        },
        "recommended_next_stage": {"id": identifier},
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
    malformed["quality_bridge_execution_git"]["revision"] = "0" * 40
    assert supervisor.followup_route(malformed) == "invalid"


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
