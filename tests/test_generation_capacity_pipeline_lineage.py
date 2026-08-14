from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from cofitok.generation_pipeline_lineage import (
    AUTHORIZATION_BOUNDARY,
    build_lineage_report,
    inspect_lineage_stage,
    load_lineage_plan,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = (
    ROOT
    / "configs/generation/diagnostics/"
    "capacity_generation_pipeline_lineage_v1.json"
)
RUNBOOK = (
    ROOT / "artifacts/runbooks/generation_capacity_pipeline_lineage_observer.sh"
)
OBSERVER = ROOT / "scripts/observe_generation_capacity_pipeline_lineage.py"
NOW = datetime(2026, 8, 15, tzinfo=timezone.utc)


def _write_status(
    path: Path,
    *,
    role: str,
    status: str,
    detail: str = "detail",
    pid: int = 100,
    updated_at: datetime = NOW,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "role": role,
                "status": status,
                "detail": detail,
                "pid": pid,
                "updated_at": updated_at.isoformat(),
            }
        ),
        encoding="utf-8",
    )


def _spec(path: Path, *, index: int, name: str, role: str) -> dict:
    return {
        "index": index,
        "name": name,
        "role": role,
        "blocking": True,
        "status_path": path,
    }


def _inspect(spec: dict, *, alive: bool = True, stale_seconds: float = 240.0) -> dict:
    return inspect_lineage_stage(
        spec,
        now=NOW,
        stale_seconds=stale_seconds,
        process_exists=lambda pid: alive,
    )


def _report(stages: list[dict], *, external_issues: list[str] | None = None) -> dict:
    return build_lineage_report(
        stages=stages,
        plan_identity={"path": "/plan.json", "bytes": 1, "sha256": "a" * 64},
        observed_at=NOW,
        stale_seconds=240.0,
        observer_git={
            "revision": "1" * 40,
            "tree": "2" * 40,
            "branch": "observer",
            "tracked_dirty": False,
        },
        formal_checkout={
            "revision": "3" * 40,
            "branch": "formal",
            "porcelain_count": 0,
            "porcelain_sha256": "e" * 64,
        },
        gpu_compute_processes=[],
        gpu_query_complete=True,
        disk={"free_bytes": 1},
        hostname="host",
        external_issues=external_issues or [],
    )


def test_capacity_pipeline_plan_covers_the_complete_ordered_lineage() -> None:
    plan = load_lineage_plan(
        PLAN,
        variables={
            "checkpoint_root": "/checkpoints",
            "source_output_root": "/source",
            "quality_bridge_root": "/quality",
        },
    )
    names = [row["name"] for row in plan["stages"]]
    assert names[:4] == [
        "quality_bridge_100k_recovery",
        "quality_bridge_followup_decision",
        "capacity_probe_preparation",
        "capacity_probe_10k_execution",
    ]
    assert names[-4:] == [
        "capacity_full_300k_posteval",
        "capacity_full_300k_finalization",
        "quality_bridge_step10k_reference_preservation",
        "quality_bridge_requested_class_visual_audit",
    ]
    assert sum(row["blocking"] for row in plan["stages"]) == 14
    assert len({row["name"] for row in plan["stages"]}) == len(plan["stages"])


def test_capacity_pipeline_report_names_the_first_live_blocker(tmp_path: Path) -> None:
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    _write_status(first_path, role="first", status="completed")
    _write_status(
        second_path,
        role="second",
        status="waiting",
        detail="waiting_for_gpu_idle",
    )
    stages = [
        _inspect(_spec(first_path, index=0, name="first", role="first")),
        _inspect(_spec(second_path, index=1, name="second", role="second")),
    ]
    report = _report(stages)
    assert report["status"] == "waiting"
    assert report["progress"]["current_stage"] == "second"
    assert report["progress"]["successful_blocking_stage_count"] == 1
    assert "waiting_for_gpu_idle" in report["detail"]
    assert report["authorization_boundary"] == AUTHORIZATION_BOUNDARY


def test_capacity_pipeline_report_rejects_stale_active_status(tmp_path: Path) -> None:
    path = tmp_path / "stale.json"
    _write_status(
        path,
        role="stage",
        status="waiting",
        updated_at=NOW - timedelta(seconds=241),
    )
    stage = _inspect(_spec(path, index=0, name="stage", role="stage"))
    report = _report([stage])
    assert stage["health"] == "stalled"
    assert stage["issues"] == ["active_status_stale"]
    assert report["status"] == "stalled"


def test_capacity_pipeline_report_rejects_dead_downstream_waiter(tmp_path: Path) -> None:
    upstream = tmp_path / "upstream.json"
    downstream = tmp_path / "downstream.json"
    _write_status(upstream, role="upstream", status="waiting")
    _write_status(downstream, role="downstream", status="waiting", pid=200)
    stages = [
        _inspect(_spec(upstream, index=0, name="upstream", role="upstream")),
        _inspect(
            _spec(downstream, index=1, name="downstream", role="downstream"),
            alive=False,
        ),
    ]
    report = _report(stages)
    assert report["status"] == "failed"
    assert report["progress"]["current_stage"] == "downstream"
    assert "downstream:active_status_process_dead" in report["issues"]


def test_capacity_pipeline_report_preserves_hold_without_completion(tmp_path: Path) -> None:
    passed = tmp_path / "passed.json"
    held = tmp_path / "held.json"
    _write_status(passed, role="passed", status="pass")
    _write_status(held, role="held", status="hold")
    report = _report(
        [
            _inspect(_spec(passed, index=0, name="passed", role="passed")),
            _inspect(_spec(held, index=1, name="held", role="held")),
        ]
    )
    assert report["status"] == "hold"
    assert report["complete"] is False
    assert report["progress"]["current_stage"] == "held"


def test_capacity_pipeline_report_passes_only_when_every_blocker_passes(
    tmp_path: Path,
) -> None:
    stages = []
    for index in range(3):
        path = tmp_path / f"stage_{index}.json"
        _write_status(path, role=f"role_{index}", status="pass")
        stages.append(
            _inspect(
                _spec(
                    path,
                    index=index,
                    name=f"stage_{index}",
                    role=f"role_{index}",
                )
            )
        )
    report = _report(stages)
    assert report["status"] == "pass"
    assert report["complete"] is True
    assert report["progress"]["fraction"] == 1.0


def test_capacity_pipeline_missing_or_swapped_status_fails_closed(
    tmp_path: Path,
) -> None:
    missing = _inspect(
        _spec(tmp_path / "missing.json", index=0, name="missing", role="expected")
    )
    report = _report([missing])
    assert report["status"] == "failed"
    assert report["issues"] == ["missing:status_file_missing"]

    swapped_path = tmp_path / "swapped.json"
    _write_status(swapped_path, role="another", status="waiting")
    swapped = _inspect(
        _spec(swapped_path, index=0, name="swapped", role="expected")
    )
    assert "status_role_mismatch" in swapped["issues"]


def test_capacity_pipeline_observer_is_read_only_and_non_signaling() -> None:
    runbook = RUNBOOK.read_text(encoding="utf-8")
    observer = OBSERVER.read_text(encoding="utf-8")
    assert "flock -n" in runbook
    assert "nice -n 19" in runbook
    assert "observe_generation_capacity_pipeline_lineage.py" in runbook
    joined = runbook + observer
    assert "train_generation.py" not in joined
    assert "generate_samples.py" not in joined
    assert "os.kill" not in joined
    assert ".terminate(" not in joined
    assert ".kill(" not in joined
    assert "pkill" not in joined
    assert "nvidia-smi" in observer
    assert AUTHORIZATION_BOUNDARY["gpu_allocation_allowed"] is False
    assert AUTHORIZATION_BOUNDARY["process_signaling_allowed"] is False
