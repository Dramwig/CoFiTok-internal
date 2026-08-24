from __future__ import annotations

import ast
import inspect
import textwrap
from pathlib import Path

import pytest

from cofitok.generation import (
    EPSILON_STABILITY_EXECUTION_ACTIONS,
    EPSILON_STABILITY_NON_AUTHORIZING_BOUNDARY,
)
from cofitok.generation_gate_sources import gate_source_report_identity
from cofitok.reporting import write_json_report
from scripts import recover_generation_epsilon_stability_sampling_recovery as recovery


def _write_source(path: Path, payload: dict) -> dict:
    write_json_report(path, payload)
    return {"identity": gate_source_report_identity(path), "payload": payload}


def test_failed_output_inventory_is_exact() -> None:
    identity = {
        "path": "/output/controller_receipt.json",
        "bytes": 7,
        "sha256": "a" * 64,
    }
    assert recovery._expected_failed_output_inventory(identity) == [
        {"path": "real_artifact_reference", "type": "directory"},
        {"path": "real_artifact_reference/images", "type": "directory"},
        {"path": "reports", "type": "directory"},
        {
            "path": "reports/controller_receipt.json",
            "type": "file",
            "identity": identity,
        },
    ]


def test_failed_attempt_replay_rejects_any_extra_output(
    tmp_path: Path,
) -> None:
    output = tmp_path / "failed-output"
    images = output / "real_artifact_reference/images"
    reports = output / "reports"
    images.mkdir(parents=True)
    reports.mkdir()
    failed_lock = tmp_path / "failed-output.lock"
    failed_lock.mkdir()
    controls = tmp_path / "controls"
    controls.mkdir()
    execution_identity = {
        "path": "/controls/execution_authorization.json",
        "bytes": 11,
        "sha256": "e" * 64,
    }
    status_payload = {
        "status": "failed",
        "stage": "failed",
        "completed_arms": 0,
        "child_pid": None,
        "pid": 321,
        "error": "CalledProcessError: JPEG materialization",
        "output_root": output.resolve().as_posix(),
        "execution_authorization": execution_identity,
        "generation_advantage_proven": False,
    }
    receipt_payload = {
        "status": "pass",
        "execution_authorization": execution_identity,
        "output_root": output.resolve().as_posix(),
        "authorized_actions": EPSILON_STABILITY_EXECUTION_ACTIONS,
        "result_authorization_boundary": EPSILON_STABILITY_NON_AUTHORIZING_BOUNDARY,
    }
    controller_receipt = _write_source(
        reports / "controller_receipt.json", receipt_payload
    )
    failure_payload = {
        "status": "failed",
        "pid": 321,
        "error": status_payload["error"],
        "process_signals_allowed": False,
    }
    sources = {
        "status": _write_source(controls / "status.json", status_payload),
        "receipt": controller_receipt,
        "failure": _write_source(failed_lock / "failure.json", failure_payload),
        "launch": _write_source(
            controls / "launch.json",
            {
                "status": "launched",
                "pid": 321,
                "output_root": output.resolve().as_posix(),
                "generation_advantage_proven": False,
            },
        ),
        "prelaunch": _write_source(
            controls / "prelaunch.json", {"status": "pass"}
        ),
    }
    log = controls / "controller.log"
    log.write_text(
        "prepare_generation_epsilon_stability_real_artifact_subset.py\n"
        "real artifact source image contract differs\n",
        encoding="utf-8",
    )
    log_source = {
        "identity": gate_source_report_identity(log),
        "payload": None,
    }

    result = recovery._validate_failed_attempt(
        output_root=output,
        failed_lock=failed_lock,
        status_source=sources["status"],
        log_source=log_source,
        controller_receipt_source=sources["receipt"],
        failure_source=sources["failure"],
        launch_source=sources["launch"],
        prelaunch_source=sources["prelaunch"],
        execution_identity=execution_identity,
    )
    assert result["status"] == "failed_before_gpu_sampling"

    (images / "unexpected.png").write_bytes(b"not an authorized artifact")
    with pytest.raises(ValueError, match="failed output inventory differs"):
        recovery._validate_failed_attempt(
            output_root=output,
            failed_lock=failed_lock,
            status_source=sources["status"],
            log_source=log_source,
            controller_receipt_source=sources["receipt"],
            failure_source=sources["failure"],
            launch_source=sources["launch"],
            prelaunch_source=sources["prelaunch"],
            execution_identity=execution_identity,
        )


def test_recovery_state_is_outside_preserved_failure_roots(
    tmp_path: Path,
) -> None:
    output = tmp_path / "diagnostic"
    failed_lock = tmp_path / "diagnostic.lock"
    contract = recovery._validate_recovery_state_paths(
        output_root=output,
        failed_lock=failed_lock,
        recovery_lock=tmp_path / "diagnostic.recovery.lock",
        recovery_status=tmp_path / "controls/recovery_status.json",
    )
    assert contract["failed_output"] == output.resolve().as_posix()
    with pytest.raises(ValueError, match="inside failed output"):
        recovery._validate_recovery_state_paths(
            output_root=output,
            failed_lock=failed_lock,
            recovery_lock=tmp_path / "diagnostic.recovery.lock",
            recovery_status=output / "recovery_status.json",
        )


def test_recovery_evaluator_drift_is_materializer_only(
    tmp_path: Path,
) -> None:
    original = tmp_path / "original"
    fixed = tmp_path / "fixed"
    relative_sampler = "scripts/generate_samples.py"
    for root in (original, fixed):
        (root / "scripts").mkdir(parents=True)
        (root / relative_sampler).write_text("sampling = 'original'\n", encoding="utf-8")
    original_materializer = original / recovery.MATERIALIZER_RELATIVE_PATH
    fixed_materializer = fixed / recovery.MATERIALIZER_RELATIVE_PATH
    original_materializer.write_text("png_only = True\n", encoding="utf-8")
    fixed_materializer.write_text("jpeg_and_png = True\n", encoding="utf-8")
    manifest = {
        "sources": {
            relative_sampler: gate_source_report_identity(original / relative_sampler),
            recovery.MATERIALIZER_RELATIVE_PATH: gate_source_report_identity(
                original_materializer
            ),
        }
    }
    result = recovery._validate_recovery_evaluator_equivalence(
        recovery_project=fixed,
        evaluator_manifest=manifest,
    )
    assert result["only_drifted_evaluator_source"] == (
        recovery.MATERIALIZER_RELATIVE_PATH
    )

    (fixed / relative_sampler).write_text("sampling = 'drifted'\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not materializer-only"):
        recovery._validate_recovery_evaluator_equivalence(
            recovery_project=fixed,
            evaluator_manifest=manifest,
        )


def test_duplicate_recovery_scan_binds_authorization(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    authorization = tmp_path / "recovery_authorization.json"
    for pid in (101, 102):
        (proc / str(pid)).mkdir(parents=True)
    prefix = (
        b"python\0scripts/"
        b"recover_generation_epsilon_stability_sampling_recovery.py\0run\0"
    )
    (proc / "101/cmdline").write_bytes(
        prefix + authorization.resolve().as_posix().encode()
    )
    (proc / "102/cmdline").write_bytes(prefix + b"/other/authorization.json")
    assert recovery.duplicate_recovery_controller_pids(
        authorization_path=authorization,
        own_pid=999,
        proc_root=proc,
    ) == [101]


def test_recovery_runbook_is_single_non_authorizing_controller() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "artifacts/runbooks/generation_epsilon_stability_sampling_recovery_pre_gpu_recovery_v1.sh"
    ).read_text(encoding="utf-8")
    assert source.count(
        "recover_generation_epsilon_stability_sampling_recovery.py run"
    ) == 1
    assert "--required-idle-polls 3" in source
    assert "nvidia-smi --query-compute-apps=pid" in source
    assert "train_generation.py" not in source
    assert "kill " not in source
    assert "pkill" not in source
    assert "10000" not in source
    assert "300k" not in source.lower()


def test_recovery_uses_original_sampling_checkout_only() -> None:
    run_source = textwrap.dedent(inspect.getsource(recovery.run_recovery))
    tree = ast.parse(run_source)
    bound_calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if not isinstance(node.func.value, ast.Name) or node.func.value.id != "base":
            continue
        if node.func.attr not in {
            "sampling_command",
            "evaluation_commands",
            "_build_observation",
        }:
            continue
        if node.func.attr == "_build_observation":
            argument = next(
                keyword.value for keyword in node.keywords if keyword.arg == "args"
            )
            assert isinstance(argument, ast.Name)
            assert argument.id == "args"
        else:
            project = next(
                keyword.value for keyword in node.keywords if keyword.arg == "project"
            )
            assert isinstance(project, ast.Name)
            assert project.id == "execution_project"
        bound_calls.append(node.func.attr)
    assert sorted(bound_calls) == [
        "_build_observation",
        "evaluation_commands",
        "sampling_command",
    ]
    assert run_source.count("cwd=execution_project") >= 3

    reference_source = inspect.getsource(recovery._prepare_real_reference)
    assert "recovery_project" in reference_source
    assert "MATERIALIZER_RELATIVE_PATH" in reference_source
    assert "cwd=recovery_project" in reference_source
    assert "cwd=args.project" in reference_source
    equivalence_source = inspect.getsource(
        recovery._validate_recovery_evaluator_equivalence
    )
    assert "only_drifted_evaluator_source" in equivalence_source
