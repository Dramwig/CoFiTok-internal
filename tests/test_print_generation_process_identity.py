from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import print_generation_process_identity as identity_cli


IDENTITY = {
    "pid": 618_821,
    "start_ticks": 1_660_953_145,
    "argv": (
        "bash artifacts/runbooks/"
        "generation_stability_full_data_quality_bridge_100k_execute.sh"
    ),
    "executable": "/usr/bin/bash",
    "cwd": "/tmp/quality-bridge/CoFiTok-internal",
    "cmdline_sha256": "a" * 64,
}


ROOT = Path(__file__).resolve().parents[1]


def test_monitor_argument_tokens_are_complete_and_ordered() -> None:
    assert identity_cli.monitor_argument_tokens(IDENTITY) == [
        "--runbook-process-pid",
        "618821",
        "--runbook-process-start-ticks",
        "1660953145",
        "--runbook-process-executable",
        "/usr/bin/bash",
        "--runbook-process-cwd",
        "/tmp/quality-bridge/CoFiTok-internal",
        "--runbook-process-cmdline-sha256",
        "a" * 64,
    ]


def test_json_output_preserves_full_identity(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        identity_cli,
        "read_linux_process_identity",
        lambda pid, proc_root: dict(IDENTITY) if pid == 618_821 else None,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["print_generation_process_identity.py", "--pid", "618821"],
    )

    identity_cli.main()

    assert json.loads(capsys.readouterr().out) == IDENTITY


def test_monitor_args_output_is_nul_delimited(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
) -> None:
    monkeypatch.setattr(
        identity_cli,
        "read_linux_process_identity",
        lambda pid, proc_root: dict(IDENTITY) if pid == 618_821 else None,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "print_generation_process_identity.py",
            "--pid",
            "618821",
            "--format",
            "monitor-args0",
        ],
    )

    identity_cli.main()

    observed = capsysbinary.readouterr().out
    assert observed.endswith(b"\0")
    assert observed[:-1].split(b"\0") == [
        value.encode("utf-8")
        for value in identity_cli.monitor_argument_tokens(IDENTITY)
    ]


def test_missing_process_identity_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        identity_cli,
        "read_linux_process_identity",
        lambda pid, proc_root: None,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["print_generation_process_identity.py", "--pid", "618821"],
    )

    with pytest.raises(SystemExit, match="identity is unavailable"):
        identity_cli.main()


def test_cli_help_is_available() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/print_generation_process_identity.py"),
            "--help",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--pid" in result.stdout
    assert "--format" in result.stdout
