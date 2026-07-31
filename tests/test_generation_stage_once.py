from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.run_generation_stage_once import (
    StageReplayError,
    _output_declarations,
    _output_identities,
    _request,
    _request_sha256,
    _worker_result_path,
    file_identity,
    run_stage_once,
)


ROOT = Path(__file__).resolve().parents[1]


def _git(*args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def _run(
    tmp_path: Path,
    *,
    command: list[str],
    input_file: Path,
    output_file: Path | None = None,
    output_tree: Path | None = None,
) -> dict:
    return run_stage_once(
        state_path=tmp_path / "stage.json",
        project=ROOT,
        cwd=tmp_path,
        command=command,
        input_files=[str(input_file)],
        input_trees=[],
        output_files=[] if output_file is None else [str(output_file)],
        output_trees=[] if output_tree is None else [str(output_tree)],
    )


def test_completed_stage_replays_without_rerunning_command(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    output = tmp_path / "output.txt"
    counter = tmp_path / "counter.txt"
    source.write_text("source", encoding="utf-8")
    command = [
        sys.executable,
        "-c",
        (
            "from pathlib import Path; "
            f"counter=Path({str(counter)!r}); output=Path({str(output)!r}); "
            "value=int(counter.read_text() if counter.exists() else '0')+1; "
            "counter.write_text(str(value)); output.write_text('stable')"
        ),
    ]

    first = _run(
        tmp_path,
        command=command,
        input_file=source,
        output_file=output,
    )
    second = _run(
        tmp_path,
        command=command,
        input_file=source,
        output_file=output,
    )

    assert first["reused"] is False
    assert second["reused"] is True
    assert counter.read_text(encoding="utf-8") == "1"
    assert output.read_text(encoding="utf-8") == "stable"


def test_completed_stage_rejects_input_or_output_drift(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    output = tmp_path / "output.txt"
    source.write_text("source", encoding="utf-8")
    command = [
        sys.executable,
        "-c",
        f"from pathlib import Path; Path({str(output)!r}).write_text('stable')",
    ]
    _run(
        tmp_path,
        command=command,
        input_file=source,
        output_file=output,
    )

    output.write_text("tampered", encoding="utf-8")
    with pytest.raises(StageReplayError, match="outputs differ"):
        _run(
            tmp_path,
            command=command,
            input_file=source,
            output_file=output,
        )

    output.write_text("stable", encoding="utf-8")
    source.write_text("changed", encoding="utf-8")
    with pytest.raises(StageReplayError, match="request differs"):
        _run(
            tmp_path,
            command=command,
            input_file=source,
            output_file=output,
        )


def test_failed_stage_archives_partial_outputs_before_retry(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    output = tmp_path / "output.txt"
    marker = tmp_path / "marker.txt"
    source.write_text("source", encoding="utf-8")
    command = [
        sys.executable,
        "-c",
        (
            "from pathlib import Path; import sys; "
            f"marker=Path({str(marker)!r}); output=Path({str(output)!r}); "
            "first=not marker.exists(); marker.write_text('seen'); "
            "output.write_text('partial' if first else 'complete'); "
            "sys.exit(7 if first else 0)"
        ),
    ]

    with pytest.raises(subprocess.CalledProcessError):
        _run(
            tmp_path,
            command=command,
            input_file=source,
            output_file=output,
        )
    assert not output.exists()
    archived = list(tmp_path.glob("output.txt.stage-attempt-001-failed*"))
    assert len(archived) == 1
    assert archived[0].read_text(encoding="utf-8") == "partial"

    result = _run(
        tmp_path,
        command=command,
        input_file=source,
        output_file=output,
    )
    receipt = json.loads((tmp_path / "stage.json").read_text(encoding="utf-8"))

    assert result["attempt"] == 2
    assert output.read_text(encoding="utf-8") == "complete"
    assert [attempt["status"] for attempt in receipt["attempts"]] == [
        "failed",
        "completed",
    ]


def test_interrupted_stage_archives_partial_tree_and_resumes(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    output = tmp_path / "outputs"
    source.write_text("source", encoding="utf-8")
    output.mkdir()
    (output / "partial.txt").write_text("partial", encoding="utf-8")
    command = [
        sys.executable,
        "-c",
        (
            "from pathlib import Path; "
            f"root=Path({str(output)!r}); root.mkdir(); "
            "(root/'complete.txt').write_text('complete')"
        ),
    ]
    receipt = {
        "schema_version": 1,
        "role": "generation_stage_receipt",
        "status": "running",
        "hostname": "stale-host",
        "pid": 99999999,
        "child_pid": 99999999,
        "attempt": 1,
        "request": {
            "project": ROOT.resolve().as_posix(),
            "cwd": tmp_path.resolve().as_posix(),
            "command": command,
            "git": {
                "revision": subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=ROOT,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip(),
                "branch": subprocess.run(
                    ["git", "branch", "--show-current"],
                    cwd=ROOT,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip(),
                "tracked_dirty": bool(
                    subprocess.run(
                        ["git", "status", "--porcelain", "--untracked-files=no"],
                        cwd=ROOT,
                        check=True,
                        capture_output=True,
                        text=True,
                    ).stdout.strip()
                ),
            },
            "inputs": [
                {
                    "kind": "file",
                    "path": source.resolve().as_posix(),
                    "bytes": source.stat().st_size,
                    "sha256": __import__("hashlib").sha256(source.read_bytes()).hexdigest(),
                }
            ],
            "outputs": [
                {"kind": "tree", "path": output.resolve().as_posix()}
            ],
        },
        "attempts": [{"attempt": 1, "status": "running"}],
    }
    (tmp_path / "stage.json").write_text(
        json.dumps(receipt),
        encoding="utf-8",
    )

    result = _run(
        tmp_path,
        command=command,
        input_file=source,
        output_tree=output,
    )

    assert result["attempt"] == 2
    assert (output / "complete.txt").read_text(encoding="utf-8") == "complete"
    archives = list(tmp_path.glob("outputs.stage-attempt-001-interrupted*"))
    assert len(archives) == 1
    assert (archives[0] / "partial.txt").is_file()


def test_unreceipted_output_is_never_adopted(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    output = tmp_path / "output.txt"
    source.write_text("source", encoding="utf-8")
    output.write_text("unknown", encoding="utf-8")

    with pytest.raises(StageReplayError, match="unreceipted"):
        _run(
            tmp_path,
            command=[sys.executable, "-c", "pass"],
            input_file=source,
            output_file=output,
        )


@pytest.mark.skipif(os.name == "nt", reason="Windows symlink creation needs elevated privileges")
def test_stage_rejects_symlinked_evidence_parent(tmp_path: Path) -> None:
    source_root = tmp_path / "source_root"
    source_root.mkdir()
    source = source_root / "source.txt"
    source.write_text("source", encoding="utf-8")
    alias = tmp_path / "source_alias"
    alias.symlink_to(source_root, target_is_directory=True)

    with pytest.raises(StageReplayError, match="must not contain a symlink"):
        _run(
            tmp_path,
            command=[sys.executable, "-c", "pass"],
            input_file=alias / "source.txt",
            output_file=tmp_path / "output.txt",
        )


def test_successful_worker_result_recovers_parent_crash_without_rerun(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    output = tmp_path / "output.txt"
    state_path = tmp_path / "stage.json"
    source.write_text("source", encoding="utf-8")
    output.write_text("completed-before-parent-crash", encoding="utf-8")
    command = [sys.executable, "-c", "raise AssertionError('must not rerun')"]
    outputs = _output_declarations([str(output)], [])
    request = _request(
        project=ROOT.resolve(),
        cwd=tmp_path.resolve(),
        command=command,
        inputs=[file_identity(source)],
        outputs=outputs,
    )
    worker_result_path = _worker_result_path(state_path.resolve(), 1)
    worker_result = {
        "schema_version": 1,
        "role": "generation_stage_worker_result",
        "status": "completed",
        "attempt": 1,
        "request_sha256": _request_sha256(request),
        "command": command,
        "cwd": tmp_path.resolve().as_posix(),
        "worker_pid": 99999999,
        "child_pid": 99999999,
        "exit_code": 0,
        "error_type": None,
        "error": None,
        "finished_at": "2026-07-31T00:00:00+00:00",
    }
    worker_result_path.write_text(json.dumps(worker_result), encoding="utf-8")
    state_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "role": "generation_stage_receipt",
                "status": "running",
                "hostname": "stale-host",
                "pid": 99999999,
                "child_pid": 99999999,
                "attempt": 1,
                "request": request,
                "attempts": [
                    {
                        "attempt": 1,
                        "status": "running",
                        "worker_result_path": worker_result_path.as_posix(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = _run(
        tmp_path,
        command=command,
        input_file=source,
        output_file=output,
    )
    receipt = json.loads(state_path.read_text(encoding="utf-8"))

    assert result["recovered"] is True
    assert result["reused"] is True
    assert output.read_text(encoding="utf-8") == "completed-before-parent-crash"
    assert receipt["status"] == "completed"
    assert receipt["completed_outputs"] == _output_identities(outputs)
