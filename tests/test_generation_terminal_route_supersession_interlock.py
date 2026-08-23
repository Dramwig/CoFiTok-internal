from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/deploy_generation_terminal_route_supersession_interlock.py"
SPEC = importlib.util.spec_from_file_location("terminal_route_supersession", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
EVIDENCE_DIR = (
    ROOT
    / "artifacts/reports/generation/"
    "stability_full_data_100k_base128_quality_bridge_v1/"
    "terminal_route_supersession_interlock_v1"
)


def fake_route(tmp_path: Path, *, kind: str) -> object:
    output = tmp_path / ("legacy_output" if kind == "directory_with_marker" else "legacy_random")
    interlock = Path(f"{output}.lock") if kind == "directory_with_marker" else output
    return MODULE.LegacyRoute(
        name=f"route_{kind}",
        output_root=output,
        interlock_path=interlock,
        interlock_kind=kind,
        checkout=tmp_path,
        revision="a" * 40,
        tree="b" * 40,
        branch="analysis/test",
        supervisor_source=None,
        supervisor_sha256=None,
        runbook_source=tmp_path / "runbook.sh",
        runbook_sha256="c" * 64,
        status_path=None,
        expected_waiting_detail=None,
        process_needles=("unused",),
        cuda_hidden_value=None,
        supervisor_required_snippets=(),
        runbook_required_snippets=(),
    )


def marker(route: object) -> dict[str, object]:
    return MODULE.marker_payload(
        route,
        control_git={
            "path": "/checkout",
            "revision": "1" * 40,
            "tree": "2" * 40,
            "branch": "analysis/interlock",
            "tracked_dirty": False,
        },
        control_source={
            "path": "/checkout/scripts/deploy.py",
            "bytes": 123,
            "sha256": "3" * 64,
        },
    )


def test_scope_is_permanently_non_authorizing() -> None:
    assert MODULE.SCOPE == {
        "cpu_only": True,
        "diagnostic_non_authorizing": True,
        "static_filesystem_interlock_only": True,
        "checkpoint_payload_loading_allowed": False,
        "gpu_execution_allowed": False,
        "new_gpu_supervisor_launch_allowed": False,
        "process_signals_allowed": False,
        "training_launch_allowed": False,
        "sampling_launch_allowed": False,
        "full_training_launch_allowed": False,
        "full_300k_launch_allowed": False,
        "promotion_authorization_allowed": False,
        "inference_export_authorization_allowed": False,
        "release_authorization_allowed": False,
        "upstream_evidence_modified": False,
    }


def test_directory_interlock_is_atomic_replayable_and_blocks_mkdir(tmp_path: Path) -> None:
    route = fake_route(tmp_path, kind="directory_with_marker")
    payload = marker(route)
    first = MODULE.install_or_verify_interlock(route, payload)
    second = MODULE.install_or_verify_interlock(route, payload)

    assert first["active"] is True
    assert second["marker"] == first["marker"]
    assert route.interlock_path.is_dir()
    assert json.loads((route.interlock_path / "supersession_marker.json").read_text()) == payload
    with pytest.raises(FileExistsError):
        route.interlock_path.mkdir()


def test_regular_file_interlock_blocks_random_token_output_directory(tmp_path: Path) -> None:
    route = fake_route(tmp_path, kind="regular_file_marker")
    payload = marker(route)
    evidence = MODULE.install_or_verify_interlock(route, payload)

    assert evidence["active"] is True
    assert route.interlock_path.is_file()
    assert not route.interlock_path.is_dir()
    assert json.loads(route.interlock_path.read_text()) == payload
    with pytest.raises(FileExistsError):
        route.output_root.mkdir()


def test_unrecognized_existing_directory_interlock_fails_closed(tmp_path: Path) -> None:
    route = fake_route(tmp_path, kind="directory_with_marker")
    route.interlock_path.mkdir()
    with pytest.raises(RuntimeError, match="no recognized marker"):
        MODULE.install_or_verify_interlock(route, marker(route))


def test_existing_legacy_output_fails_before_directory_interlock(tmp_path: Path) -> None:
    route = fake_route(tmp_path, kind="directory_with_marker")
    route.output_root.mkdir()
    with pytest.raises(RuntimeError, match="legacy output root already exists"):
        MODULE.install_or_verify_interlock(route, marker(route))


def test_marker_binds_legacy_and_authoritative_guard_paths(tmp_path: Path) -> None:
    route = fake_route(tmp_path, kind="regular_file_marker")
    payload = marker(route)
    assert payload["legacy_terminal_guard"] == MODULE.LEGACY_TERMINAL_GUARD.as_posix()
    assert payload["authoritative_terminal_guard"] == MODULE.AUTHORITATIVE_TERMINAL_GUARD.as_posix()
    assert payload["reason"] == "legacy_consumer_binds_stale_runtime_v1_terminal_guard"
    assert payload["scope"] == MODULE.SCOPE


def test_random_token_contract_has_independent_non_directory_fail_closed_check() -> None:
    route = next(
        item for item in MODULE.LEGACY_ROUTES if item.name == "random_token_semantic_visual_v1"
    )
    assert route.interlock_kind == "regular_file_marker"
    assert route.interlock_path == route.output_root
    assert any("! -d" in snippet for snippet in route.runbook_required_snippets)
    assert any(
        "matched_factorization_quality_regression_diagnostic_completed" in snippet
        for snippet in route.runbook_required_snippets
    )


def test_build_receipt_does_not_promote_scientific_state() -> None:
    receipt = MODULE.build_receipt(preflight={"evidence": "test"}, interlocks={})
    assert receipt["status"] == "pass"
    assert receipt["generation_advantage_proven"] is False
    assert receipt["scope"] == MODULE.SCOPE
    assert receipt["guarantees"]["no_existing_process_was_signaled"] is True
    assert receipt["guarantees"]["no_gpu_process_was_started"] is True
    assert receipt["decision"] == "legacy_v1_gpu_route_consumers_superseded_fail_closed"


def test_exclusive_json_writer_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    first = MODULE.write_json_exclusive(path, {"status": "pass"})
    assert first["sha256"] == MODULE.sha256_file(path)
    with pytest.raises(FileExistsError):
        MODULE.write_json_exclusive(path, {"status": "different"})


def test_all_legacy_route_targets_are_distinct() -> None:
    outputs = [route.output_root for route in MODULE.LEGACY_ROUTES]
    interlocks = [route.interlock_path for route in MODULE.LEGACY_ROUTES]
    assert len(outputs) == len(set(outputs)) == 3
    assert len(interlocks) == len(set(interlocks)) == 3
    assert MODULE.QUALITY_ROOT not in outputs
    assert MODULE.CANONICAL_RECEIPT.parent == MODULE.CONTROL_DIR


def test_deployment_evidence_binds_the_byte_exact_remote_receipt() -> None:
    evidence = json.loads((EVIDENCE_DIR / "deployment_evidence.json").read_text())
    receipt_path = EVIDENCE_DIR / "supersession_receipt.json"
    receipt_payload = receipt_path.read_bytes()
    receipt = json.loads(receipt_payload)

    assert evidence["status"] == "pass"
    assert evidence["code"]["revision"] == "bdcab4b15bbbb18a78deaba85ff96eac6c129964"
    assert evidence["code"]["tree"] == "f94f85b227f2c6496f6d9c2ed845a03639df857b"
    assert evidence["receipt"]["bytes"] == len(receipt_payload) == 24968
    assert evidence["receipt"]["sha256"] == hashlib.sha256(receipt_payload).hexdigest()
    assert receipt["status"] == "pass"
    assert receipt["role"] == MODULE.ROLE
    assert receipt["scope"] == MODULE.SCOPE
    assert receipt["generation_advantage_proven"] is False


def test_deployment_evidence_keeps_every_launch_and_release_boundary_false() -> None:
    evidence = json.loads((EVIDENCE_DIR / "deployment_evidence.json").read_text())
    scope = evidence["scope"]
    for field in (
        "process_signals_allowed",
        "gpu_execution_allowed",
        "new_gpu_supervisor_launched",
        "training_launch_allowed",
        "sampling_launch_allowed",
        "full_training_launch_allowed",
        "full_300k_launch_allowed",
        "promotion_authorization_allowed",
        "release_authorization_allowed",
        "generation_advantage_proven",
    ):
        assert scope[field] is False
    assert evidence["verification"]["legacy_gpu_children_after_deployment"] == []
    assert set(evidence["interlocks"]) == {
        "factorization_quality_regression_v1",
        "conditioning_ranking_v1",
        "random_token_semantic_visual_v1",
    }
