import json
import subprocess

from scripts.remote_queue_status import build_program, build_remote_command, parse_args, poll_remote


def test_build_remote_command_uses_requested_host_and_paths() -> None:
    command = build_remote_command("pro6000", "/root/project dir", "/env/bin/python")

    assert command[0] == "ssh"
    assert "pro6000" in command
    assert "cd '/root/project dir' && /env/bin/python -" in command[-1]


def test_build_program_embeds_poll_configuration() -> None:
    args = parse_args(
        [
            "--remote-code-dir",
            "/code",
            "--checkpoint-root",
            "/checkpoints",
            "--quality-images",
            "256",
            "--sample-count",
            "128",
            "--sample-steps",
            "20",
        ]
    )

    program = build_program(args)

    assert '"/code"' in program
    assert '"/checkpoints"' in program
    assert '"quality_images": 256' in program
    assert '"sample_count": 128' in program
    assert '"sample_steps": 20' in program


def test_poll_remote_parses_remote_json(monkeypatch) -> None:
    seen = {}

    def fake_run(command, input, capture_output, text, timeout):  # noqa: ANN001
        seen["command"] = command
        seen["input"] = input
        seen["timeout"] = timeout
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps({"status": "running", "queue": {"counts": {"complete": 4}}}),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    args = parse_args(["--host", "demo-host", "--timeout", "12"])

    payload, exit_code = poll_remote(args)

    assert exit_code == 0
    assert payload["status"] == "running"
    assert seen["command"][1 + 10] == "demo-host"
    assert "inspect_next_validation_queue" in seen["input"]
    assert seen["timeout"] == 12


def test_poll_remote_returns_structured_timeout(monkeypatch) -> None:
    def fake_run(*args, **kwargs):  # noqa: ANN002, ANN003
        raise subprocess.TimeoutExpired(cmd=["ssh", "demo"], timeout=5)

    monkeypatch.setattr(subprocess, "run", fake_run)
    args = parse_args(["--timeout", "5"])

    payload, exit_code = poll_remote(args)

    assert exit_code == 1
    assert payload["status"] == "unreachable"
    assert payload["reason"] == "timeout"


def test_poll_remote_can_allow_unreachable(monkeypatch) -> None:
    def fake_run(command, input, capture_output, text, timeout):  # noqa: ANN001
        return subprocess.CompletedProcess(command, 255, stdout="", stderr="timed out")

    monkeypatch.setattr(subprocess, "run", fake_run)
    args = parse_args(["--allow-unreachable"])

    payload, exit_code = poll_remote(args)

    assert exit_code == 0
    assert payload["status"] == "unreachable"
    assert payload["reason"] == "command_failed"

