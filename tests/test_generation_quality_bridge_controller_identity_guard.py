from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts import run_generation_quality_bridge_controller_identity_guard as guard


RUNBOOK_ARGV = [
    "bash",
    "artifacts/runbooks/generation_stability_full_data_quality_bridge_100k_execute.sh",
]


def raw_cmdline(argv: list[str] = RUNBOOK_ARGV) -> bytes:
    return b"\0".join(value.encode("utf-8") for value in argv) + b"\0"


def expected_identity() -> dict[str, object]:
    return {
        "pid": 618_821,
        "start_ticks": 1_660_953_145,
        "executable": "/usr/bin/bash",
        "cwd": "/tmp/quality-bridge/CoFiTok-internal",
        "cmdline_sha256": hashlib.sha256(raw_cmdline()).hexdigest(),
    }


def observed_identity() -> dict[str, object]:
    return {
        **expected_identity(),
        "ppid": 1,
        "argv": list(RUNBOOK_ARGV),
    }


def execution_status(status: str = "running") -> dict[str, object]:
    return {
        "schema_version": 1,
        "role": guard.EXECUTION_ROLE,
        "status": status,
        "detail": "quality bridge state",
        "exit_code": 1 if status == "failed" else None,
        "git": {
            "revision": "cf0e5fa",
            "branch": "scale/quality-bridge",
            "tracked_dirty": False,
        },
        "pid": 618_821,
        "quality_bridge_only": True,
        "full_training_launch_allowed": False,
        "full_300k_launch_allowed": False,
        "report_is_promotion_gate": False,
        "updated_at": "2026-08-18T00:00:00+00:00",
    }


def pair_monitor() -> dict[str, object]:
    return {
        "monitor": "quality_bridge_monitor",
        "status": "running",
        "stage": "cofitok_training",
        "git": {
            "revision": "cf0e5fa",
            "branch": "scale/quality-bridge",
            "tracked_dirty": False,
        },
        "processes": {
            "runbook": [
                "618821 bash artifacts/runbooks/controller.sh",
                "832803 python waiter.py --active-runbook controller.sh",
            ]
        },
        "updated_at": "2026-08-18T00:00:00+00:00",
    }


def test_decode_argv_preserves_exact_two_argument_command() -> None:
    assert guard._decode_argv(raw_cmdline()) == RUNBOOK_ARGV


def test_exact_process_identity_passes() -> None:
    assert guard.process_identity_mismatches(
        observed_identity(),
        expected=expected_identity(),
    ) == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("start_ticks", 1_660_953_146),
        ("executable", "/usr/bin/python"),
        ("cwd", "/tmp/other"),
        ("cmdline_sha256", "0" * 64),
    ],
)
def test_process_identity_detects_bound_field_changes(
    field: str,
    value: object,
) -> None:
    observed = observed_identity()
    observed[field] = value

    assert field in guard.process_identity_mismatches(
        observed,
        expected=expected_identity(),
    )


def test_process_identity_does_not_accept_a_waiter_argv() -> None:
    observed = observed_identity()
    observed["argv"] = [
        "python",
        "waiter.py",
        "--active-runbook",
        RUNBOOK_ARGV[1],
    ]

    assert "argv_shape" in guard.process_identity_mismatches(
        observed,
        expected=expected_identity(),
    )


def test_missing_bound_process_is_explicit() -> None:
    assert guard.process_identity_mismatches(
        None,
        expected=expected_identity(),
    ) == ["process_missing"]


def test_read_process_identity_binds_proc_start_exe_cwd_and_raw_cmdline(
    tmp_path: Path,
) -> None:
    proc = tmp_path / "618821"
    proc.mkdir()
    proc.joinpath("cmdline").write_bytes(raw_cmdline())
    stat_tail = ["S", "1"] + ["0"] * 17 + ["1660953145"] + ["0"] * 8
    proc.joinpath("stat").write_text(
        "618821 (bash worker) " + " ".join(stat_tail),
        encoding="ascii",
    )

    def fake_readlink(path: object) -> str:
        name = Path(path).name
        if name == "exe":
            return "/usr/bin/bash"
        if name == "cwd":
            return "/tmp/quality-bridge/CoFiTok-internal"
        raise AssertionError(name)

    with patch.object(guard.os, "readlink", side_effect=fake_readlink):
        observed = guard.read_process_identity(618_821, proc_root=tmp_path)

    assert observed == observed_identity()


def test_read_process_identity_treats_empty_cmdline_as_unavailable(
    tmp_path: Path,
) -> None:
    proc = tmp_path / "618821"
    proc.mkdir()
    proc.joinpath("cmdline").write_bytes(b"")
    stat_tail = ["S", "1"] + ["0"] * 17 + ["1660953145"] + ["0"] * 8
    proc.joinpath("stat").write_text(
        "618821 (bash) " + " ".join(stat_tail),
        encoding="ascii",
    )

    with patch.object(guard.os, "readlink", return_value="/usr/bin/bash"):
        assert guard.read_process_identity(618_821, proc_root=tmp_path) is None


def test_execution_status_accepts_exact_running_contract() -> None:
    assert guard.validate_execution_status(
        execution_status(),
        expected_pid=618_821,
        expected_revision="cf0e5fa",
        expected_branch="scale/quality-bridge",
    )["status"] == "running"


def test_execution_status_rejects_authorization_expansion() -> None:
    report = execution_status()
    report["full_300k_launch_allowed"] = True

    with pytest.raises(ValueError, match="contract differs"):
        guard.validate_execution_status(
            report,
            expected_pid=618_821,
            expected_revision="cf0e5fa",
            expected_branch="scale/quality-bridge",
        )


def test_pair_monitor_summary_exposes_substring_candidate_contamination() -> None:
    summary = guard.summarize_pair_monitor(
        pair_monitor(),
        expected_monitor_name="quality_bridge_monitor",
        expected_revision="cf0e5fa",
        expected_branch="scale/quality-bridge",
        bound_pid=618_821,
    )

    assert summary["reported_runbook_pids"] == [618_821, 832_803]
    assert summary["bound_pid_reported"] is True
    assert summary["nonbound_reported_pids"] == [832_803]
    assert summary["substring_candidate_contamination_observed"] is True


def test_guard_scope_is_cpu_only_non_authorizing_and_non_signaling() -> None:
    assert guard.SCOPE == {
        "cpu_only": True,
        "gpu_execution_allowed": False,
        "checkpoint_payload_loading_allowed": False,
        "process_signals_allowed": False,
        "controller_restart_allowed": False,
        "training_process_signals_allowed": False,
        "unrelated_process_signals_allowed": False,
        "promotion_authorization_allowed": False,
        "release_authorization_allowed": False,
        "full_training_launch_allowed": False,
        "full_300k_launch_allowed": False,
        "diagnostic_non_authorizing": True,
    }
    source = Path(guard.__file__).read_text(encoding="utf-8")
    assert "os.kill" not in source
    assert "send_signal" not in source
    assert "subprocess.Popen" not in source
    assert "nvidia-smi" not in source
    assert "torch.load" not in source


def test_guard_invocation_does_not_need_the_runbook_name_in_argv() -> None:
    source_name = Path(guard.__file__).name
    assert (
        "generation_stability_full_data_quality_bridge_100k_execute.sh"
        not in source_name
    )
    assert "--expected-controller-cmdline-sha256" in Path(
        guard.__file__
    ).read_text(encoding="utf-8")
