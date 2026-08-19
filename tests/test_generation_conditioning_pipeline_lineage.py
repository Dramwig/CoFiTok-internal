from __future__ import annotations

import copy
import json
from pathlib import Path

from cofitok.generation_conditioning_lineage import (
    AUTHORIZATION_BOUNDARY,
    build_conditioning_lineage_report,
    inspect_conditioning_stage,
    load_conditioning_lineage_plan,
)
from cofitok.inference_replay import file_identity

ROOT = Path(__file__).resolve().parents[1]
PLAN = (
    ROOT / "configs/generation/diagnostics/"
    "conditioning_generation_pipeline_lineage_v1.json"
)
OBSERVER = ROOT / "scripts/observe_generation_conditioning_pipeline_lineage.py"
RUNBOOK = (
    ROOT / "artifacts/runbooks/generation_conditioning_pipeline_lineage_observer.sh"
)
NOW = 1_787_113_200.0


def _write_json(path: Path, payload: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return file_identity(path)


def _fixture(tmp_path: Path, *, required_idle_polls: int = 5) -> tuple:
    project = tmp_path / "project"
    project.mkdir()
    source = project / "supervisor.py"
    runbook = project / "runbook.sh"
    source.write_text("print('supervisor')\n", encoding="utf-8")
    runbook.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    standing = tmp_path / "standing.json"
    standing_identity = _write_json(standing, {"authorized": True})
    preparation = tmp_path / "preparation.json"
    preparation_identity = _write_json(preparation, {"prepared": True})
    status_path = tmp_path / "status.json"
    output_root = tmp_path / "output"
    expected_git = {
        "path": project.resolve().as_posix(),
        "revision": "1" * 40,
        "tree": "2" * 40,
        "branch": "conditioning",
        "tracked_dirty": False,
    }
    expected_process = {
        "pid": 101,
        "start_ticks": 202,
        "cwd": project.resolve().as_posix(),
        "cmdline_sha256": "3" * 64,
    }
    spec = {
        "index": 0,
        "name": "probe",
        "role": "probe_supervisor",
        "status_path": status_path.resolve().as_posix(),
        "project": expected_git,
        "process": expected_process,
        "status_pid_required": True,
        "expected_status_output_root": output_root.resolve().as_posix(),
        "stage_output_path": output_root.resolve().as_posix(),
        "required_idle_polls": required_idle_polls,
        "max_launches": 1,
        "launch_receipt_path": None,
        "idle_gpu_evidence_path": None,
        "required_status_sources": ["preparation", "standing_authorization"],
        "source_files": [file_identity(source), file_identity(runbook)],
    }
    status = {
        "schema_version": 1,
        "role": "probe_supervisor",
        "status": "waiting",
        "detail": "waiting_for_upstream",
        "updated_at_unix": NOW - 10,
        "pid": 101,
        "child_pid": None,
        "project": project.resolve().as_posix(),
        "output_root": output_root.resolve().as_posix(),
        "idle_gpu_polls": 0,
        "sources": {
            "preparation": preparation_identity,
            "standing_authorization": standing_identity,
        },
    }
    _write_json(status_path, status)
    process = {**expected_process, "alive": True}
    return spec, status, standing_identity, expected_git, process


def _inspect(
    spec: dict,
    standing: dict,
    git: dict,
    process: dict,
    *,
    history: dict | None = None,
    now: float = NOW,
) -> tuple[dict, dict]:
    return inspect_conditioning_stage(
        spec,
        now_unix=now,
        stale_seconds=300.0,
        standing_authorization=standing,
        previous_history=history,
        git_inspector=lambda path: git,
        process_inspector=lambda pid: process,
    )


def test_exact_plan_covers_the_five_stage_repair_chain() -> None:
    plan = load_conditioning_lineage_plan(PLAN)
    assert [stage["name"] for stage in plan["stages"]] == [
        "conditioning_probe_1k",
        "conditioning_sampling_5k",
        "conditioning_training_5k",
        "conditioning_heldout_5k",
        "conditioning_posttraining_sampling_5k",
    ]
    assert [stage["required_idle_polls"] for stage in plan["stages"]] == [
        5,
        5,
        5,
        0,
        5,
    ]
    assert all(stage["max_launches"] == 1 for stage in plan["stages"])
    assert len({stage["process"]["pid"] for stage in plan["stages"]}) == 5


def test_waiting_stage_binds_process_git_sources_and_authorization(
    tmp_path: Path,
) -> None:
    spec, _, standing, git, process = _fixture(tmp_path)
    stage, history = _inspect(spec, standing, git, process)
    assert stage["health"] == "waiting"
    assert stage["issues"] == []
    assert stage["standing_authorization_bound"] is True
    assert stage["launch_count_lower_bound"] == 0
    assert history["observation_count"] == 1


def test_status_source_and_tracked_source_drift_fail_closed(tmp_path: Path) -> None:
    spec, status, standing, git, process = _fixture(tmp_path)
    status["sources"]["standing_authorization"] = {
        **status["sources"]["standing_authorization"],
        "sha256": "f" * 64,
    }
    _write_json(Path(spec["status_path"]), status)
    Path(spec["source_files"][0]["path"]).write_text("changed\n", encoding="utf-8")
    stage, _ = _inspect(spec, standing, git, process)
    assert "standing_authorization_status_binding_mismatch" in stage["issues"]
    assert any(
        issue.startswith("source_file_identity_mismatch") for issue in stage["issues"]
    )


def test_active_status_rejects_staleness_and_process_replacement(
    tmp_path: Path,
) -> None:
    spec, status, standing, git, process = _fixture(tmp_path)
    status["updated_at_unix"] = NOW - 301
    _write_json(Path(spec["status_path"]), status)
    replacement = {**process, "start_ticks": process["start_ticks"] + 1}
    stage, _ = _inspect(spec, standing, git, replacement)
    assert stage["health"] == "stalled"
    assert "active_status_stale" in stage["issues"]
    assert "active_supervisor_process_identity_mismatch" in stage["issues"]


def test_first_stage_launch_requires_an_observed_five_poll_idle_gate(
    tmp_path: Path,
) -> None:
    spec, status, standing, git, process = _fixture(tmp_path)
    status.update({"status": "running", "child_pid": 404, "idle_gpu_polls": 4})
    _write_json(Path(spec["status_path"]), status)
    stage, _ = _inspect(spec, standing, git, process)
    assert "launch_before_observed_idle_gpu_gate" in stage["issues"]


def test_immutable_idle_evidence_and_launch_receipt_bind_one_launch(
    tmp_path: Path,
) -> None:
    spec, status, standing, git, process = _fixture(tmp_path)
    idle = tmp_path / "idle.json"
    receipt = tmp_path / "launch.json"
    _write_json(
        idle,
        {
            "status": "pass",
            "required_consecutive_idle_polls": 5,
            "observations": [
                {
                    "poll_index": index,
                    "observed_at_unix": NOW + index,
                    "gpu_compute_pids": [],
                }
                for index in range(1, 6)
            ],
        },
    )
    _write_json(receipt, {"status": "reserved", "sources": {}})
    spec["idle_gpu_evidence_path"] = idle.resolve().as_posix()
    spec["launch_receipt_path"] = receipt.resolve().as_posix()
    status.update({"status": "running", "child_pid": 404, "idle_gpu_polls": 5})
    _write_json(Path(spec["status_path"]), status)
    stage, history = _inspect(spec, standing, git, process)
    assert stage["issues"] == []
    assert stage["launch_count_lower_bound"] == 1
    assert history["idle_gate_observed"] is True
    assert history["child_pids"] == [404]
    assert len(history["launch_receipt_sha256s"]) == 1


def test_history_rejects_a_second_child_even_after_status_replacement(
    tmp_path: Path,
) -> None:
    spec, status, standing, git, process = _fixture(tmp_path)
    status.update({"status": "running", "child_pid": 404, "idle_gpu_polls": 5})
    _write_json(Path(spec["status_path"]), status)
    first, history = _inspect(spec, standing, git, process)
    assert first["launch_count_lower_bound"] == 1
    status["child_pid"] = 405
    _write_json(Path(spec["status_path"]), status)
    second, history = _inspect(
        spec,
        standing,
        git,
        process,
        history=history,
        now=NOW + 1,
    )
    assert second["launch_count_lower_bound"] == 2
    assert "runbook_launch_count_exceeded" in second["issues"]
    assert history["child_pids"] == [404, 405]


def test_unbound_output_is_tolerated_only_as_a_short_launch_transition(
    tmp_path: Path,
) -> None:
    spec, _, standing, git, process = _fixture(tmp_path)
    Path(spec["stage_output_path"]).mkdir()
    history = None
    for offset in range(3):
        stage, history = _inspect(
            spec,
            standing,
            git,
            process,
            history=history,
            now=NOW + offset,
        )
    assert stage["launch_count_lower_bound"] == 1
    assert "stage_output_exists_without_observed_launch_binding" in stage["issues"]


def test_report_names_first_stage_and_never_authorizes_execution(
    tmp_path: Path,
) -> None:
    spec, _, standing, git, process = _fixture(tmp_path)
    stage, history = _inspect(spec, standing, git, process)
    report = build_conditioning_lineage_report(
        stages=[stage],
        histories={"probe": history},
        observed_at_unix=NOW,
        stale_seconds=300.0,
        plan_identity={"path": "/plan", "bytes": 1, "sha256": "a" * 64},
        observer_git=git,
        standing_authorization={"valid": True},
        gpu_compute_processes=[{"pid": 999}],
        gpu_query_complete=True,
        disk={"free_bytes": 1},
        hostname="host",
    )
    assert report["status"] == "waiting"
    assert report["progress"]["current_stage"] == "probe"
    assert report["authorization_boundary"] == AUTHORIZATION_BOUNDARY
    assert not any(
        report["authorization_boundary"][key]
        for key in (
            "training_launch_allowed",
            "sampling_or_evaluation_launch_allowed",
            "promotion_or_release_allowed",
            "process_signaling_allowed",
            "gpu_allocation_allowed",
        )
    )


def test_observer_and_runbook_are_cpu_only_and_non_signaling() -> None:
    observer = OBSERVER.read_text(encoding="utf-8")
    runbook = RUNBOOK.read_text(encoding="utf-8")
    joined = observer + runbook
    assert "flock -n" in runbook
    assert "nice -n 19" in runbook
    assert "CUDA_VISIBLE_DEVICES=-1" in runbook
    assert "OMP_NUM_THREADS=1" in runbook
    assert "nvidia-smi" in observer
    assert "train_generation.py" not in joined
    assert "generate_samples.py" not in joined
    assert "subprocess.Popen" not in joined
    assert "os.kill" not in joined
    assert ".terminate(" not in joined
    assert ".kill(" not in joined
    assert "pkill" not in joined


def test_completed_not_selected_stage_does_not_invent_a_launch(tmp_path: Path) -> None:
    spec, status, standing, git, process = _fixture(tmp_path)
    status.update(
        {
            "status": "completed",
            "detail": "conditioning_repair_not_selected",
            "child_pid": None,
        }
    )
    _write_json(Path(spec["status_path"]), status)
    stage, history = _inspect(spec, standing, git, process)
    assert stage["health"] == "pass"
    assert stage["launch_count_lower_bound"] == 0
    assert history["child_pids"] == []


def test_optional_missing_status_pid_is_explicitly_supported(tmp_path: Path) -> None:
    spec, status, standing, git, process = _fixture(tmp_path)
    spec["status_pid_required"] = False
    status.pop("pid")
    _write_json(Path(spec["status_path"]), status)
    stage, _ = _inspect(spec, standing, git, process)
    assert "status_pid_mismatch" not in stage["issues"]
    assert stage["health"] == "waiting"


def test_report_fails_when_any_stage_source_binding_fails(tmp_path: Path) -> None:
    spec, _, standing, git, process = _fixture(tmp_path)
    stage, history = _inspect(spec, standing, git, process)
    broken = copy.deepcopy(stage)
    broken["issues"] = ["source_file_identity_mismatch:supervisor.py"]
    broken["health"] = "failed"
    report = build_conditioning_lineage_report(
        stages=[broken],
        histories={"probe": history},
        observed_at_unix=NOW,
        stale_seconds=300.0,
        plan_identity={"path": "/plan", "bytes": 1, "sha256": "a" * 64},
        observer_git=git,
        standing_authorization={"valid": True},
        gpu_compute_processes=[],
        gpu_query_complete=True,
        disk={"free_bytes": 1},
        hostname="host",
    )
    assert report["status"] == "failed"
    assert report["complete"] is False
    assert report["issues"] == ["probe:source_file_identity_mismatch:supervisor.py"]
