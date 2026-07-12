import json
from pathlib import Path

from scripts.remote_recovery_launch import (
    build_extract_remote_command,
    build_launch_remote_command,
    build_plan,
    build_probe_remote_command,
    post_extract_checks_from_manifest,
)


def test_build_probe_and_extract_commands_use_project_root() -> None:
    probe = build_probe_remote_command("/root/autodl-tmp/CoFiTok")
    extract = build_extract_remote_command("/root/autodl-tmp/CoFiTok")

    assert "mkdir -p /root/autodl-tmp/CoFiTok" in probe
    assert "cd /root/autodl-tmp/CoFiTok" in probe
    assert "tar -xzf - -C /root/autodl-tmp/CoFiTok" in extract


def test_post_extract_checks_skip_cd_and_wrap_in_code_dir() -> None:
    manifest = {
        "post_extract_checks": [
            "cd /root/autodl-tmp/CoFiTok/CoFiTok-internal",
            "bash -n artifacts/runbooks/next_validation_queue_2026-07-08.sh",
            "PYTHONPATH=src python scripts/validate_dataset_conditions.py --require-ok",
        ]
    }

    checks = post_extract_checks_from_manifest(manifest, "/root/autodl-tmp/CoFiTok/CoFiTok-internal")

    assert len(checks) == 2
    assert all(check.startswith("cd /root/autodl-tmp/CoFiTok/CoFiTok-internal && ") for check in checks)
    assert "validate_dataset_conditions.py" in checks[1]


def test_build_launch_command_uses_tmux_or_nohup_without_blocking() -> None:
    command = build_launch_remote_command(
        "/root/autodl-tmp/CoFiTok/CoFiTok-internal",
        "artifacts/runbooks/next_validation_queue_2026-07-08.sh",
        "cofitok_next_validation_20260708",
    )

    assert "tmux new-session -d" in command
    assert "nvidia-smi" in command
    assert "GPU compute processes are active" in command
    assert "nohup bash artifacts/runbooks/next_validation_queue_2026-07-08.sh" in command
    assert "artifacts/logs/next_validation_queue_2026-07-08.log" in command
    assert "queue_log=artifacts/logs/next_validation_queue_2026-07-08.log" in command


def test_build_launch_command_can_skip_gpu_idle_guard() -> None:
    command = build_launch_remote_command(
        "/root/autodl-tmp/CoFiTok/CoFiTok-internal",
        "artifacts/runbooks/next_validation_queue_2026-07-08.sh",
        "cofitok_next_validation_20260708",
        require_idle_gpu=False,
    )

    assert "tmux new-session -d" in command
    assert "GPU compute processes are active" not in command


def test_build_plan_reads_manifest_and_includes_launch(tmp_path: Path) -> None:
    bundle = tmp_path / "remote_recovery.tar.gz"
    manifest = tmp_path / "remote_recovery.tar.gz.manifest.json"
    bundle.write_bytes(b"not a real tar for planning tests")
    manifest.write_text(
        json.dumps(
            {
                "post_extract_checks": [
                    "cd /root/autodl-tmp/CoFiTok/CoFiTok-internal",
                    "bash -n artifacts/runbooks/next_validation_queue_2026-07-08.sh",
                ]
            }
        ),
        encoding="utf-8",
    )

    plan = build_plan(
        host="pro6000",
        remote_root="/root/autodl-tmp/CoFiTok",
        bundle_path=bundle,
        manifest_path=manifest,
        runbook="artifacts/runbooks/next_validation_queue_2026-07-08.sh",
        session="cofitok_next_validation_20260708",
        include_launch=True,
    )

    assert plan.host == "pro6000"
    assert plan.code_dir == "/root/autodl-tmp/CoFiTok/CoFiTok-internal"
    assert plan.extract_command[0] == "ssh"
    assert plan.post_extract_commands
    assert plan.launch_command is not None
