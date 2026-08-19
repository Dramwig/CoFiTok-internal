from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from cofitok.generation_capacity_supervisor_repair import (
    DOWNSTREAM_NAMES,
    FINALIZATION_NAME,
    FINALIZATION_RECEIPT_ROLE,
    POSTEVAL_NAME,
    POSTEVAL_RECEIPT_ROLE,
    REPAIR_SCOPE,
    TRAINING_NAME,
    TRAINING_RECEIPT_ROLE,
    build_supervisor_receipt_repair_approval,
    build_supervisor_receipt_repair_plan,
    execute_supervisor_receipt_repair,
)
from cofitok.generation_control_process_migration import (
    APPROVAL_ROLE,
    EXECUTION_ROLE,
    PLAN_ROLE,
    SUPERVISOR_NAMES,
    SUPERVISOR_SPECS,
)
from cofitok.inference_replay import file_identity


EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _git(path: Path, value: str, branch: str) -> dict[str, Any]:
    return {
        "path": path.resolve().as_posix(),
        "revision": value * 40,
        "tree": value.upper() * 40,
        "branch": branch,
        "tracked_dirty": False,
        "porcelain_count": 0,
        "porcelain_sha256": EMPTY_SHA256,
    }


def _core(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value[key]
        for key in ("path", "revision", "tree", "branch", "tracked_dirty")
    }


def _spec(name: str) -> dict[str, Any]:
    return dict(next(row for row in SUPERVISOR_SPECS if row["name"] == name))


class _Handle:
    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.returncode = None

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        return 0


def _fixture(tmp_path: Path, monkeypatch) -> dict[str, Any]:
    target = tmp_path / "target"
    formal = tmp_path / "formal"
    target.mkdir()
    formal.mkdir()
    target_git = _git(target, "a", "fix/repair")
    formal_git = _git(formal, "b", "scale/generative-system")
    git_states = {
        target.resolve().as_posix(): target_git,
        formal.resolve().as_posix(): formal_git,
    }
    monkeypatch.setattr(
        "cofitok.generation_capacity_supervisor_repair._git_state_with_porcelain",
        lambda path: copy.deepcopy(git_states[path.resolve().as_posix()]),
    )

    source_root = tmp_path / "capacity"
    reports = source_root / "reports"
    full_root = tmp_path / "full"
    logs = tmp_path / "logs"
    reports.mkdir(parents=True)
    logs.mkdir()
    source = tmp_path / "source.txt"
    source.write_text("source\n", encoding="utf-8")

    old_rows = []
    new_rows = []
    training_pid = 12_345
    training_runtime = None
    for index, raw_spec in enumerate(SUPERVISOR_SPECS):
        spec = dict(raw_spec)
        name = str(spec["name"])
        entrypoint = str(spec["entrypoint"])
        target_source = target / entrypoint
        target_source.parent.mkdir(parents=True, exist_ok=True)
        target_source.write_text(
            "from cofitok.process_monitoring import publish_child_heartbeat\n",
            encoding="utf-8",
        )
        old_checkout = tmp_path / f"old-{index}"
        old_source = old_checkout / entrypoint
        old_source.parent.mkdir(parents=True)
        old_source.write_text("# old\n", encoding="utf-8")
        status = reports / f"{name}_status.json"
        pid = training_pid if name == TRAINING_NAME else 10_000 + index
        argv = [
            sys.executable,
            entrypoint,
            "--status-output",
            status.resolve().as_posix(),
        ]
        barriers = []
        for barrier_index, option in enumerate(spec["barrier_options"]):
            barrier = full_root / f"{name}-{barrier_index}.json"
            argv.extend((str(option), barrier.resolve().as_posix()))
            barriers.append(
                {
                    "option": option,
                    "path": barrier.resolve().as_posix(),
                    "exists": False,
                }
            )
        if name in DOWNSTREAM_NAMES:
            argv.extend(
                (
                    "--expected-training-supervisor-deployment-sha256",
                    "1" * 64,
                )
            )
        if name == FINALIZATION_NAME:
            argv.extend(
                (
                    "--expected-posteval-supervisor-deployment-sha256",
                    "2" * 64,
                )
            )
        runtime = {
            "pid": pid,
            "ppid": 1,
            "process_group_id": pid,
            "session_id": pid,
            "nice": 19,
            "start_ticks": pid * 10,
            "umask": "0022",
            "cwd": old_checkout.resolve().as_posix(),
            "argv": argv,
            "executable_target": Path(sys.executable).resolve().as_posix(),
            "stdin_target": "/dev/null",
            "stdout_target": (logs / f"{name}.log").resolve().as_posix(),
            "stderr_target": (logs / f"{name}.log").resolve().as_posix(),
            "stdout_fd_flags": 1,
            "stderr_fd_flags": 1,
            "environment": {},
        }
        _write(
            status,
            {
                "role": spec["role"],
                "status": "waiting",
                "detail": spec["waiting_detail"],
                "pid": pid,
                "child_pid": None,
            },
        )
        if name == TRAINING_NAME:
            training_runtime = runtime
        row = {
            "name": name,
            "role": spec["role"],
            "waiting_detail": spec["waiting_detail"],
            "status_path": status.resolve().as_posix(),
            "barriers": barriers,
            "runtime": runtime,
            "relaunch_io": {
                "stdin_target": "/dev/null",
                "stdout_target": runtime["stdout_target"],
                "stderr_target": runtime["stderr_target"],
                "stdout_open_mode": "append",
                "stderr_open_mode": "append",
            },
            "entrypoint": {
                "argument": entrypoint,
                "argument_index": 1,
                "identity": file_identity(old_source),
            },
        }
        old_rows.append(copy.deepcopy(row))
        row["runtime"]["cwd"] = target.resolve().as_posix()
        row["runtime"]["environment"] = {
            "PYTHONPATH": os.pathsep.join(
                (target.resolve().as_posix(), (target / "src").resolve().as_posix())
            )
        }
        row["entrypoint"]["identity"] = file_identity(target_source)
        new_rows.append(row)
        lock_filename = spec.get("lock_filename")
        if lock_filename is not None:
            (source_root / str(lock_filename)).touch()

    assert training_runtime is not None
    migration_plan_path = tmp_path / "migration-plan.json"
    migration_plan = {
        "schema_version": 1,
        "role": PLAN_ROLE,
        "status": "ready",
        "selected_process_names": list(SUPERVISOR_NAMES),
        "old_processes": old_rows,
        "new_processes": new_rows,
    }
    _write(migration_plan_path, migration_plan)
    migration_plan_id = file_identity(migration_plan_path)
    migration_approval_path = tmp_path / "migration-approval.json"
    migration_approval = {
        "schema_version": 1,
        "role": APPROVAL_ROLE,
        "status": "approved",
        "plan": migration_plan_id,
    }
    _write(migration_approval_path, migration_approval)
    migration_approval_id = file_identity(migration_approval_path)
    migration_execution_path = tmp_path / "migration-execution.json"
    _write(
        migration_execution_path,
        {
            "schema_version": 1,
            "role": EXECUTION_ROLE,
            "status": "failed_restore_incomplete",
            "plan": migration_plan_id,
            "approval": migration_approval_id,
            "restoration_error": {
                "detail": "migrated supervisor exited before waiting status: "
                "capacity_full_300k_posteval:1"
            },
        },
    )

    pointers = tmp_path / "pointers"

    def receipt(name: str, role: str) -> tuple[Path, dict[str, Any]]:
        supervisor_name = {
            "training": TRAINING_NAME,
            "posteval": POSTEVAL_NAME,
            "finalization": FINALIZATION_NAME,
        }[name]
        supervisor_status = reports / f"{supervisor_name}_status.json"
        pointer = pointers / f"{name}.pid"
        pointer.parent.mkdir(parents=True, exist_ok=True)
        pointer.write_text("99999\n", encoding="utf-8")
        payload = {
            "schema_version": 1,
            "role": role,
            "status": "active",
            "checkout": {
                "path": old_rows[-1]["runtime"]["cwd"],
                "git": {"revision": "c" * 40},
            },
            "training_checkout": {
                "path": old_rows[-3]["runtime"]["cwd"],
                "git": {"revision": "d" * 40},
            },
            "full_output_root": full_root.resolve().as_posix(),
            "formal_checkout_snapshot": {
                "path": formal.resolve().as_posix(),
                "head": formal_git["revision"],
                "porcelain_count": 1,
                "porcelain_sha256": "f" * 64,
            },
            "authorization_boundary": {"training_launch_allowed": False},
            "sources": {"source": file_identity(source)},
            "supervisor": {
                "pid": 99_999,
                "pid_pointer": file_identity(pointer),
                "initial_status_identity": file_identity(supervisor_status),
                "status_path": supervisor_status.resolve().as_posix(),
            },
        }
        if name != "training":
            payload["sources"]["incremental_bundle"] = {
                "path": (tmp_path / "missing.bundle").resolve().as_posix(),
                "bytes": 1,
                "sha256": "e" * 64,
            }
            payload["incremental_bundle"] = payload["sources"][
                "incremental_bundle"
            ]
        path = reports / f"{name}-receipt.json"
        _write(path, payload)
        return path, payload

    training_receipt, _ = receipt("training", TRAINING_RECEIPT_ROLE)
    posteval_receipt, posteval_payload = receipt(
        "posteval", POSTEVAL_RECEIPT_ROLE
    )
    finalization_receipt, finalization_payload = receipt(
        "finalization", FINALIZATION_RECEIPT_ROLE
    )
    posteval_payload["sources"]["training_supervisor_deployment"] = file_identity(
        training_receipt
    )
    posteval_payload["training_supervisor_deployment"] = file_identity(
        training_receipt
    )
    _write(posteval_receipt, posteval_payload)
    finalization_payload["sources"]["training_supervisor_deployment"] = (
        file_identity(training_receipt)
    )
    finalization_payload["sources"]["posteval_supervisor_deployment"] = (
        file_identity(posteval_receipt)
    )
    finalization_payload["training_supervisor_deployment"] = file_identity(
        training_receipt
    )
    finalization_payload["posteval_supervisor_deployment"] = file_identity(
        posteval_receipt
    )
    _write(finalization_receipt, finalization_payload)

    runtimes = {training_pid: training_runtime}

    def inspect(pid: int) -> dict[str, Any]:
        return copy.deepcopy(runtimes[pid])

    plan = build_supervisor_receipt_repair_plan(
        migration_plan_path=migration_plan_path,
        expected_migration_plan_sha256=migration_plan_id["sha256"],
        migration_approval_path=migration_approval_path,
        expected_migration_approval_sha256=migration_approval_id["sha256"],
        migration_execution_path=migration_execution_path,
        expected_migration_execution_sha256=file_identity(migration_execution_path)[
            "sha256"
        ],
        target_project=target,
        expected_target_git=_core(target_git),
        formal_project=formal,
        expected_formal_git=_core(formal_git),
        live_training_pid=training_pid,
        training_receipt_path=training_receipt,
        expected_training_receipt_sha256=file_identity(training_receipt)["sha256"],
        posteval_receipt_path=posteval_receipt,
        expected_posteval_receipt_sha256=file_identity(posteval_receipt)["sha256"],
        finalization_receipt_path=finalization_receipt,
        expected_finalization_receipt_sha256=file_identity(finalization_receipt)[
            "sha256"
        ],
        archive_root=tmp_path / "archives",
        inspect_process=inspect,
        process_scanner=lambda _entrypoint: [],
    )
    plan_path = tmp_path / "repair-plan.json"
    _write(plan_path, plan)
    approval = build_supervisor_receipt_repair_approval(
        plan_path=plan_path,
        expected_plan_sha256=file_identity(plan_path)["sha256"],
    )
    approval_path = tmp_path / "repair-approval.json"
    _write(approval_path, approval)
    return {
        "target": target,
        "formal": formal,
        "git_states": git_states,
        "training_pid": training_pid,
        "runtimes": runtimes,
        "inspect": inspect,
        "plan": plan,
        "plan_path": plan_path,
        "approval_path": approval_path,
        "receipts": {
            "training": training_receipt,
            "posteval": posteval_receipt,
            "finalization": finalization_receipt,
        },
    }


def test_repair_plan_binds_failed_lineage_and_exact_receipts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    plan = fixture["plan"]
    assert plan["status"] == "ready"
    assert plan["scope"] == REPAIR_SCOPE
    assert set(plan["launch_rows"]) == set(DOWNSTREAM_NAMES)
    assert plan["live_training"]["runtime"]["pid"] == fixture["training_pid"]
    assert all(row["lifetime_lock"] for row in plan["launch_rows"].values())


def test_receipt_repair_archives_rotates_and_launches_locked_downstream(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    prior = {name: file_identity(path) for name, path in fixture["receipts"].items()}
    launched = []
    next_pid = 20_000

    def launcher(row: dict[str, Any]) -> _Handle:
        nonlocal next_pid
        next_pid += 1
        handle = _Handle(next_pid)
        runtime = copy.deepcopy(row["runtime"])
        runtime.update(
            {
                "pid": next_pid,
                "ppid": 1,
                "process_group_id": next_pid,
                "session_id": next_pid,
                "start_ticks": next_pid * 10,
            }
        )
        fixture["runtimes"][next_pid] = runtime
        launched.append((row, handle))
        return handle

    def waiter(row: dict[str, Any], process: _Handle, **_kwargs: Any) -> dict[str, Any]:
        status_path = Path(row["status_path"])
        _write(
            status_path,
            {
                "role": row["role"],
                "status": "waiting",
                "detail": row["waiting_detail"],
                "pid": process.pid,
                "child_pid": None,
            },
        )
        lock = copy.deepcopy(row["lifetime_lock"])
        lock.update(
            {
                "owner_pid": process.pid,
                "owner_descriptors": [9],
                "exclusive_nonblocking_probe": "contended",
            }
        )
        return {
            "name": row["name"],
            "role": row["role"],
            "pid": process.pid,
            "status": "waiting",
            "detail": row["waiting_detail"],
            "status_identity": file_identity(status_path),
            "lifetime_lock": lock,
        }

    published = []
    report = execute_supervisor_receipt_repair(
        approval_path=fixture["approval_path"],
        expected_approval_sha256=file_identity(fixture["approval_path"])["sha256"],
        publish=lambda value: published.append(copy.deepcopy(dict(value))),
        inspect_process=fixture["inspect"],
        process_scanner=lambda _entrypoint: [],
        launcher=launcher,
        status_waiter=waiter,
        terminator=lambda *_args, **_kwargs: {},
        lock_inspector=lambda pid, path: {
            "pid": pid,
            "path": path.resolve().as_posix(),
        },
    )
    assert report["status"] == "pass"
    assert [row[0]["name"] for row in launched] == list(DOWNSTREAM_NAMES)
    assert report["scope"] == REPAIR_SCOPE
    assert published[-1]["status"] == "pass"
    for name, old_identity in prior.items():
        canonical = file_identity(fixture["receipts"][name])
        assert canonical["sha256"] != old_identity["sha256"]
        archive = report["archives"][name]
        assert archive["sha256"] == old_identity["sha256"]
        assert archive["bytes"] == old_identity["bytes"]
    training = json.loads(fixture["receipts"]["training"].read_text())
    posteval = json.loads(fixture["receipts"]["posteval"].read_text())
    finalization = json.loads(fixture["receipts"]["finalization"].read_text())
    assert training["formal_checkout_snapshot"] == fixture["plan"][
        "formal_checkout_snapshot"
    ]
    assert posteval["training_supervisor_deployment"] == report[
        "canonical_receipts"
    ]["training"]
    assert finalization["training_supervisor_deployment"] == report[
        "canonical_receipts"
    ]["training"]
    assert finalization["posteval_supervisor_deployment"] == report[
        "canonical_receipts"
    ]["posteval"]
    assert "incremental_bundle" not in posteval["sources"]
    assert "incremental_bundle" in posteval["supersession"][
        "legacy_missing_sources"
    ]
    posteval_argv = launched[0][0]["runtime"]["argv"]
    finalization_argv = launched[1][0]["runtime"]["argv"]
    training_sha = report["canonical_receipts"]["training"]["sha256"]
    posteval_sha = report["canonical_receipts"]["posteval"]["sha256"]
    assert training_sha in posteval_argv
    assert training_sha in finalization_argv
    assert posteval_sha in finalization_argv
