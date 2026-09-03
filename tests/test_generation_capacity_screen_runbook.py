from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNBOOK = ROOT / "artifacts/runbooks/generation_capacity_screen_10k.sh"


def test_capacity_screen_runbook_consumes_exact_locked_evidence() -> None:
    source = RUNBOOK.read_text(encoding="utf-8")

    assert 'EXECUTION_ENABLED="${CAPACITY_SCREEN_EXECUTION_ENABLED:-false}"' in source
    assert "--expected-preparation-sha256" in source
    assert "--expected-authorization-sha256" in source
    assert "--expected-launch-receipt-sha256" in source
    assert "--expected-runtime-selection-sha256" in source
    assert "--expected-live-snapshot-sha256" in source
    assert "--base128-cofitok-config" in source
    assert "--base128-dense-identity-config" in source
    assert "--base256-cofitok-config" in source
    assert "--base256-dense-identity-config" in source


def test_capacity_screen_runbook_cannot_create_or_broaden_authorization() -> None:
    source = RUNBOOK.read_text(encoding="utf-8")

    assert "build_generation_capacity_screen_stage_authorization.py" not in source
    assert "build_generation_capacity_screen_execution_authorization.py" not in source
    assert "build_generation_capacity_screen_launch_receipt.py" not in source
    assert "generation_stability_ema_teacher_full_matched_300k_after_gate.sh" not in source
    assert "full_matched" not in source


def test_capacity_screen_runbook_resume_is_explicitly_opt_in() -> None:
    source = RUNBOOK.read_text(encoding="utf-8")

    assert 'RESUME="${CAPACITY_SCREEN_RESUME:-false}"' in source
    assert "RESUME_ARGS+=(--resume)" in source


def test_capacity_screen_runbook_uses_fixed_formal_output_root() -> None:
    source = RUNBOOK.read_text(encoding="utf-8")

    assert (
        "CAPACITY_SCREEN_OUTPUT_ROOT:-/root/autodl-tmp/CoFiTok/checkpoints/"
        "generation/capacity_qualification_v1"
    ) in source
