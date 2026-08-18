from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNBOOKS = (
    "generation_stability_ema_teacher_matched_50k_after_gate.sh",
    "generation_stability_full_data_quality_bridge_100k_execute.sh",
    "generation_stability_ema_teacher_full_matched_300k_after_gate.sh",
    "generation_full_matched_300k_after_gate.sh",
)
EXACT_OPTIONS = (
    "--runbook-process-pid",
    "--runbook-process-start-ticks",
    "--runbook-process-executable",
    "--runbook-process-cwd",
    "--runbook-process-cmdline-sha256",
)


def test_formal_matched_runbooks_capture_exact_controller_identity() -> None:
    for name in RUNBOOKS:
        source = (ROOT / "artifacts/runbooks" / name).read_text(
            encoding="utf-8"
        )
        assert (
            'source "$PROJECT/artifacts/runbooks/lib/'
            'generation_exact_runbook_identity.sh"'
        ) in source
        assert "cofitok_capture_runbook_monitor_identity" in source
        for option in EXACT_OPTIONS:
            assert source.count(option) == 2, (name, option)


def test_formal_matched_runbooks_refuse_monitor_from_another_controller() -> None:
    for name in RUNBOOKS:
        source = (ROOT / "artifacts/runbooks" / name).read_text(
            encoding="utf-8"
        )
        assert "cofitok_monitor_report_matches_bound_controller" in source
        assert "belongs to another controller" in source or (
            "monitor PID file points to another process" in source
        )


def test_formal_matched_runbooks_require_exact_identity_in_terminal_pass() -> None:
    for name in RUNBOOKS:
        source = (ROOT / "artifacts/runbooks" / name).read_text(
            encoding="utf-8"
        )
        assert 'identity.get("role") != "generation_exact_runbook_process_identity"' in source
        assert 'identity.get("mode") != "exact_process_identity"' in source
        assert 'identity.get("status") != "active"' in source
        assert 'identity.get("mismatches") != []' in source


def test_identity_helper_uses_nul_delimited_cli_and_checks_all_fields() -> None:
    source = (
        ROOT
        / "artifacts/runbooks/lib/generation_exact_runbook_identity.sh"
    ).read_text(encoding="utf-8")

    assert "mapfile -d '' -t identity_args" in source
    assert "scripts/print_generation_process_identity.py" in source
    assert "--format monitor-args0" in source
    assert "generation_exact_runbook_process_identity" in source
    assert "exact_process_identity" in source
    for option in EXACT_OPTIONS:
        assert option in source
