from __future__ import annotations

import copy
import json
import subprocess
from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest

from cofitok.inference_replay import file_identity
from scripts import (
    build_generation_quality_bridge_terminal_completion_audit as builder,
)
from scripts import (
    wait_generation_quality_bridge_terminal_completion_audit as waiter,
)

ROOT = Path(__file__).resolve().parents[1]
SHA = "a" * 64


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), *arguments],
        text=True,
    ).strip()


def _write(path: Path, payload: dict[str, Any] | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _identity(path: Path) -> dict[str, Any]:
    return file_identity(path)


def _terminal_decision(status: str) -> str:
    return builder.TERMINAL_DECISIONS[status]


@pytest.mark.parametrize(("status", "advantage"), [("pass", True), ("hold", False)])
def test_terminal_claim_outcome_preserves_pass_or_hold(
    status: str,
    advantage: bool,
) -> None:
    outcome = builder.terminal_claim_outcome(
        status=status,
        decision=_terminal_decision(status),
    )

    assert outcome == {
        "terminal_status": status,
        "terminal_decision": _terminal_decision(status),
        "generation_advantage_proven": advantage,
    }


def test_terminal_claim_outcome_cannot_upgrade_hold() -> None:
    with pytest.raises(ValueError, match="inconsistent"):
        builder.terminal_claim_outcome(
            status="hold",
            decision=_terminal_decision("pass"),
        )


def test_authorization_boundary_is_permanently_non_authorizing() -> None:
    assert builder.AUTHORIZATION_BOUNDARY["diagnostic_non_authorizing"] is True
    assert builder.AUTHORIZATION_BOUNDARY["cpu_only_evidence_replay_allowed"] is True
    assert waiter.SCOPE["diagnostic_non_authorizing"] is True
    assert waiter.SCOPE["cpu_only_evidence_replay"] is True
    for boundary in (builder.AUTHORIZATION_BOUNDARY, waiter.SCOPE):
        for name in (
            "training_launch_allowed",
            "sampling_launch_allowed",
            "full_training_launch_allowed",
            "full_300k_launch_allowed",
            "promotion_authorization_allowed",
            "release_authorization_allowed",
            "inference_export_authorization_allowed",
            "process_signals_allowed",
            "upstream_decisions_modified",
        ):
            assert boundary[name] is False


@pytest.mark.parametrize("terminal_status", ["pass", "hold"])
def test_validate_status_bindings_accepts_exact_json_markdown_csv_outputs(
    tmp_path: Path,
    terminal_status: str,
) -> None:
    guard_path = tmp_path / "guard.json"
    comparison_path = tmp_path / "quality_bridge_comparison.json"
    markdown_path = tmp_path / "quality_bridge_comparison.md"
    csv_path = tmp_path / "quality_bridge_comparison.csv"
    _write(guard_path, {"status": terminal_status})
    _write(comparison_path, {"status": terminal_status})
    _write(markdown_path, "# comparison\n")
    _write(csv_path, "method\n")
    guard_identity = _identity(guard_path)
    comparison_identity = _identity(comparison_path)
    terminal_waiter_identity = {"path": "terminal", "bytes": 1, "sha256": SHA}
    comparison_waiter_identity = {
        "path": "comparison-status",
        "bytes": 1,
        "sha256": SHA,
    }
    terminal_guard = {
        "status": terminal_status,
        "decision": _terminal_decision(terminal_status),
    }
    terminal_waiter = {
        "schema_version": 1,
        "role": builder.TERMINAL_WAITER_ROLE,
        "status": "completed",
        "detail": "terminal_system_claim_guard_source_revalidated",
        "guard": guard_identity,
        "guard_status": terminal_status,
    }
    comparison_waiter = {
        "schema_version": 1,
        "role": builder.COMPARISON_WAITER_ROLE,
        "status": "pass",
        "detail": "terminal_quality_bridge_comparison_revalidated",
        "terminal_status": terminal_status,
        "comparison": {
            "json": comparison_identity,
            "markdown": _identity(markdown_path),
            "csv": _identity(csv_path),
        },
    }

    result = builder.validate_status_bindings(
        terminal_status_identity=terminal_waiter_identity,
        terminal_status=terminal_waiter,
        terminal_guard_identity=guard_identity,
        terminal_guard=terminal_guard,
        comparison_status_identity=comparison_waiter_identity,
        comparison_status=comparison_waiter,
        comparison_identity=comparison_identity,
    )

    assert result["terminal_status"] == terminal_status
    assert set(result["comparison_outputs"]) == {"json", "markdown", "csv"}

    comparison_waiter["comparison"]["unexpected"] = comparison_identity
    with pytest.raises(ValueError, match="comparison waiter status differs"):
        builder.validate_status_bindings(
            terminal_status_identity=terminal_waiter_identity,
            terminal_status=terminal_waiter,
            terminal_guard_identity=guard_identity,
            terminal_guard=terminal_guard,
            comparison_status_identity=comparison_waiter_identity,
            comparison_status=comparison_waiter,
            comparison_identity=comparison_identity,
        )


def test_validate_comparison_renderings_rejects_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    comparison = tmp_path / "quality_bridge_comparison.json"
    markdown = tmp_path / "quality_bridge_comparison.md"
    csv = tmp_path / "quality_bridge_comparison.csv"
    _write(comparison, {})
    _write(markdown, "expected markdown\n")
    _write(csv, "expected,csv\n")
    monkeypatch.setattr(
        builder.comparison_builder,
        "render_markdown",
        lambda _report: "expected markdown\n",
    )
    monkeypatch.setattr(
        builder.comparison_builder,
        "render_csv",
        lambda _report: "expected,csv\n",
    )
    outputs = {
        "json": _identity(comparison),
        "markdown": _identity(markdown),
        "csv": _identity(csv),
    }

    verified = builder.validate_comparison_renderings({}, outputs)
    assert verified == {"markdown": outputs["markdown"], "csv": outputs["csv"]}

    _write(csv, "drifted,csv\n")
    with pytest.raises(ValueError, match="does not replay exactly"):
        builder.validate_comparison_renderings({}, outputs)


def _terminal_replay(*, step: int = 100_000, samples_seen: int = 6_400_000) -> dict:
    metrics_identity = {
        "path": "/run/train_metrics.jsonl",
        "bytes": 123,
        "sha256": SHA,
        "last_step": step,
    }
    return {
        "latest_pointer_replay": {
            "relation": "exact_historical_pointer_still_current",
            "current_content": {"step": step},
        },
        "metrics_replay": {
            "current_observed_prefix": copy.deepcopy(metrics_identity),
            "audit_time_prefix": copy.deepcopy(metrics_identity),
            "target_row": {"step": step, "samples_seen": samples_seen},
        },
    }


def test_terminal_canonical_state_requires_exact_100k_latest_and_metrics() -> None:
    state = builder._terminal_canonical_state(
        _terminal_replay(),
        method="cofitok",
    )
    assert state["latest"]["current_content"]["step"] == 100_000

    with pytest.raises(ValueError, match="canonical terminal"):
        builder._terminal_canonical_state(
            _terminal_replay(step=99_999, samples_seen=6_399_936),
            method="cofitok",
        )

    with pytest.raises(ValueError, match="canonical terminal"):
        builder._terminal_canonical_state(
            _terminal_replay(samples_seen=6_399_999),
            method="cofitok",
        )


def _replay_receipt(step: int) -> dict[str, Any]:
    return {"path": f"/receipts/{step}.json", "bytes": step, "sha256": SHA}


def _replay_waiter_report(*, failed_step: int | None = None) -> dict[str, Any]:
    milestones = {}
    for step in builder.CHECKPOINT_STEPS:
        milestones[str(step)] = {
            "state": "failed" if step == failed_step else "pass",
            "detail": "failed" if step == failed_step else "replay_completed",
            "checkpoint_step": step,
            "replay_output": _replay_receipt(step),
        }
    return {
        "schema_version": 1,
        "role": builder.REPLAY_WAITER_ROLE,
        "status": "pass",
        "detail": "all_checkpoint_integrity_replays_completed",
        "completed_count": 3,
        "expected_count": 3,
        "milestones": milestones,
    }


def test_dense_replay_waiter_requires_all_three_locked_receipts() -> None:
    expected = {step: _replay_receipt(step) for step in builder.CHECKPOINT_STEPS}
    result = builder._validate_dense_replay_status(
        identity={"path": "/status", "bytes": 1, "sha256": SHA},
        report=_replay_waiter_report(),
        expected_receipts=expected,
    )
    assert set(result["milestones"]) == {"90000", "95000", "100000"}

    with pytest.raises(ValueError, match="milestone differs"):
        builder._validate_dense_replay_status(
            identity={"path": "/status", "bytes": 1, "sha256": SHA},
            report=_replay_waiter_report(failed_step=95_000),
            expected_receipts=expected,
        )


def test_final_checkpoint_source_revalidation_detects_post_replay_drift(
    tmp_path: Path,
) -> None:
    auditor = tmp_path / "auditor.py"
    verifier = tmp_path / "verifier.py"
    replay_waiter = tmp_path / "replay_waiter.json"
    for path in (auditor, verifier, replay_waiter):
        _write(path, f"{path.name}\n")
    methods: dict[str, Any] = {}
    tracked_payload: Path | None = None
    for method in builder.EXPECTED_METHODS:
        steps: dict[str, Any] = {}
        for step in builder.CHECKPOINT_STEPS:
            stem = f"{method}_{step}"
            audit = tmp_path / f"{stem}_audit.json"
            status = tmp_path / f"{stem}_status.json"
            payload = tmp_path / f"{stem}.pt"
            sidecar = tmp_path / f"{stem}.integrity.json"
            for path in (audit, status, payload, sidecar):
                _write(path, f"{path.name}\n")
            row = {
                "audit_report": _identity(audit),
                "waiter_status": _identity(status),
                "checkpoint": {
                    "payload": _identity(payload),
                    "integrity_manifest": _identity(sidecar),
                },
            }
            if method == "dense_identity":
                receipt = tmp_path / f"{stem}_replay.json"
                _write(receipt, f"{receipt.name}\n")
                row["independent_replay_receipt"] = _identity(receipt)
            steps[str(step)] = row
            tracked_payload = payload
        latest = tmp_path / f"{method}_latest.json"
        metrics = tmp_path / f"{method}_metrics.jsonl"
        _write(latest, "latest\n")
        _write(metrics, "metrics\n")
        methods[method] = {
            "steps": steps,
            "canonical_latest": {
                **_identity(latest),
                "step": 100_000,
            },
            "canonical_metrics": {
                **_identity(metrics),
                "last_step": 100_000,
            },
        }
    chain = {
        "methods": methods,
        "dense_replay_waiter": {"identity": _identity(replay_waiter)},
        "physical_auditor": {"source": _identity(auditor)},
        "checkpoint_replay": {"verifier_source": _identity(verifier)},
    }

    verified = builder.revalidate_checkpoint_chain_sources(chain)
    assert set(verified["methods"]) == set(builder.EXPECTED_METHODS)

    assert tracked_payload is not None
    _write(tracked_payload, "changed\n")
    with pytest.raises(ValueError, match="changed after physical replay"):
        builder.revalidate_checkpoint_chain_sources(chain)


def test_class_fidelity_classifier_sources_physically_rehash_weights(
    tmp_path: Path,
) -> None:
    weights = tmp_path / "resnet50-11ad3fa6.pth"
    _write(weights, "classifier weights\n")
    physical = _identity(weights)
    classifier = {
        "name": "torchvision_resnet50_imagenet1k_v2",
        "weights_enum": "ResNet50_Weights.IMAGENET1K_V2",
        "weights_path": physical["path"],
        "weights_bytes": physical["bytes"],
        "weights_sha256": physical["sha256"],
    }
    source_reports = {}
    for name in (
        "class_fidelity_qualification",
        "cofitok_class_fidelity",
        "dense_class_fidelity",
    ):
        path = tmp_path / f"{name}.json"
        _write(path, {"classifier": classifier})
        source_reports[name] = _identity(path)

    verified = builder.verify_class_fidelity_classifier_sources(source_reports)
    assert verified["physical_weights"] == physical
    assert verified["matched_report_classifier_identity"] is True
    assert verified["physical_weight_sha256_replayed"] is True

    _write(weights, "drifted classifier weights\n")
    with pytest.raises(ValueError, match="physical identity differs"):
        builder.verify_class_fidelity_classifier_sources(source_reports)


def test_class_fidelity_classifier_sources_reject_report_mismatch(
    tmp_path: Path,
) -> None:
    weights = tmp_path / "resnet50-11ad3fa6.pth"
    _write(weights, "classifier weights\n")
    physical = _identity(weights)
    source_reports = {}
    for index, name in enumerate(
        (
            "class_fidelity_qualification",
            "cofitok_class_fidelity",
            "dense_class_fidelity",
        )
    ):
        classifier = {
            "weights_path": physical["path"],
            "weights_bytes": physical["bytes"],
            "weights_sha256": (
                physical["sha256"] if index < 2 else "f" * 64
            ),
        }
        path = tmp_path / f"{name}.json"
        _write(path, {"classifier": classifier})
        source_reports[name] = _identity(path)

    with pytest.raises(ValueError, match="classifier identities differ"):
        builder.verify_class_fidelity_classifier_sources(source_reports)


def _build_args(tmp_path: Path) -> Namespace:
    return Namespace(
        project=ROOT,
        quality_output_root=tmp_path / "quality",
        terminal_system_guard=tmp_path / "guard.json",
        expected_terminal_dir_name="terminal_system_claim_guard_v1",
        expected_terminal_system_guard_sha256=SHA,
        terminal_waiter_status=tmp_path / "terminal-status.json",
        comparison=tmp_path / "comparison.json",
        expected_comparison_dir_name="quality_bridge_comparison_v1",
        expected_comparison_sha256=SHA,
        comparison_waiter_status=tmp_path / "comparison-status.json",
        checkpoint_audit_dir=tmp_path / "audits",
        checkpoint_replay_waiter_status=tmp_path / "replay-status.json",
        metrics_trust_receipt=tmp_path / "metrics-trust-receipt.json",
        expected_metrics_trust_receipt_sha256="e" * 64,
        physical_auditor_checkout=ROOT,
        checkpoint_replay_checkout=ROOT,
        training_checkout=ROOT,
        expected_revision="1" * 40,
        expected_tree="2" * 40,
        expected_branch="audit",
        expected_training_revision="3" * 40,
        expected_training_tree="4" * 40,
        expected_training_branch="training",
        expected_physical_auditor_revision="5" * 40,
        expected_physical_auditor_tree="6" * 40,
        expected_physical_auditor_branch="",
        expected_physical_auditor_source_sha256="7" * 64,
        expected_checkpoint_replay_revision="8" * 40,
        expected_checkpoint_replay_tree="9" * 40,
        expected_checkpoint_replay_branch="replay",
        expected_checkpoint_replay_verifier_source_sha256="b" * 64,
        expected_dataset_sha256="c" * 64,
        expected_runtime_sha256="d" * 64,
        effective_batch=64,
    )


def test_build_audit_rejects_noncanonical_effective_batch(tmp_path: Path) -> None:
    args = _build_args(tmp_path)
    args.effective_batch = 32

    with pytest.raises(ValueError, match="must be 64"):
        builder.build_audit(args)


def test_builder_canonical_paths_accept_explicit_versioned_source_dirs(
    tmp_path: Path,
) -> None:
    args = _build_args(tmp_path)
    root = args.quality_output_root
    reports = root / "reports"
    terminal_dir = reports / "terminal_system_claim_guard_v2_runtime_strict"
    comparison_dir = reports / "quality_bridge_comparison_v3_runtime_strict"
    args.expected_terminal_dir_name = terminal_dir.name
    args.expected_comparison_dir_name = comparison_dir.name
    args.terminal_system_guard = terminal_dir / "terminal_system_claim_guard.json"
    args.terminal_waiter_status = terminal_dir / "waiter_status.json"
    args.comparison = comparison_dir / "quality_bridge_comparison.json"
    args.comparison_waiter_status = comparison_dir / "waiter_status.json"
    args.checkpoint_audit_dir = reports / "checkpoint_audits"
    args.checkpoint_replay_waiter_status = (
        args.checkpoint_audit_dir
        / "dense_checkpoint_integrity_replay_waiter_status.json"
    )
    args.metrics_trust_receipt = (
        reports / "metrics_trust_boundary_v1" / "metrics_trust_receipt.json"
    )

    paths = builder._canonical_paths(args)

    assert paths["terminal_guard"] == args.terminal_system_guard.resolve()
    assert paths["comparison"] == args.comparison.resolve()


def test_builder_canonical_paths_reject_unsafe_explicit_source_dir(
    tmp_path: Path,
) -> None:
    args = _build_args(tmp_path)
    args.expected_terminal_dir_name = "../terminal_system_claim_guard_v2"

    with pytest.raises(ValueError, match="directory name is invalid"):
        builder._canonical_paths(args)


@pytest.mark.parametrize(
    ("terminal_status", "advantage"), [("pass", True), ("hold", False)]
)
def test_build_audit_preserves_terminal_claim_and_never_authorizes_followup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    terminal_status: str,
    advantage: bool,
) -> None:
    args = _build_args(tmp_path)
    quality_root = args.quality_output_root.resolve()
    paths = {
        "root": quality_root,
        "terminal_guard": args.terminal_system_guard.resolve(),
        "terminal_status": args.terminal_waiter_status.resolve(),
        "comparison": args.comparison.resolve(),
        "comparison_status": args.comparison_waiter_status.resolve(),
        "audit_dir": args.checkpoint_audit_dir.resolve(),
        "replay_status": args.checkpoint_replay_waiter_status.resolve(),
        "metrics_trust_receipt": args.metrics_trust_receipt.resolve(),
    }
    git = {
        "revision": "1" * 40,
        "tree": "2" * 40,
        "branch": "audit",
        "tracked_dirty": False,
    }
    guard_identity = {"path": "/guard", "bytes": 1, "sha256": SHA}
    comparison_identity = {"path": "/comparison", "bytes": 1, "sha256": SHA}
    metrics_trust_identity = {"path": "/metrics-trust", "bytes": 1, "sha256": "e" * 64}
    metrics_trust_receipt = {"status": "pass"}
    quality_identity = {"path": "/quality", "bytes": 1, "sha256": SHA}
    terminal_guard = {
        "schema_version": 1,
        "role": builder.TERMINAL_GUARD_ROLE,
        "status": terminal_status,
        "decision": _terminal_decision(terminal_status),
        "sources": {"quality_bridge_result": quality_identity},
    }
    quality = {
        "physical_evidence": {
            method: {
                "sampling_report": {},
                "sampling_manifest": {},
                "sampling_progress": {},
                "sample_set_sha256": SHA,
                "sample_count": 10_000,
            }
            for method in builder.EXPECTED_METHODS
        },
        "source_reports": {
            "class_fidelity_qualification": {},
            "cofitok_class_fidelity": {},
            "dense_class_fidelity": {},
        },
        "generation_metric_reports": {
            "cofitok": {},
            "dense_identity": {},
        },
    }
    _write(paths["terminal_status"], {"status": "completed"})
    _write(paths["comparison_status"], {"status": "pass"})
    monkeypatch.setattr(builder, "_canonical_paths", lambda _args: paths)
    monkeypatch.setattr(builder, "require_git_identity", lambda *args, **kwargs: git)
    monkeypatch.setattr(
        builder,
        "_bound_json",
        lambda path, **kwargs: (
            (guard_identity, terminal_guard)
            if path == paths["terminal_guard"]
            else (
                (metrics_trust_identity, metrics_trust_receipt)
                if path == paths["metrics_trust_receipt"]
                else (comparison_identity, {"status": terminal_status})
            )
        ),
    )
    monkeypatch.setattr(
        builder,
        "_replay_identity",
        lambda descriptor, **kwargs: (quality_identity, {}),
    )
    monkeypatch.setattr(
        builder,
        "validate_status_bindings",
        lambda **kwargs: {
            "terminal_waiter_status": {},
            "comparison_waiter_status": {},
            "comparison_outputs": {"json": {}, "markdown": {}, "csv": {}},
            "terminal_status": terminal_status,
            "terminal_decision": _terminal_decision(terminal_status),
        },
    )
    monkeypatch.setattr(builder, "replay_quality_result", lambda *args: quality)
    monkeypatch.setattr(
        builder,
        "verify_class_fidelity_classifier_sources",
        lambda _sources: {"status": "verified"},
    )
    monkeypatch.setattr(
        builder,
        "verify_metrics_trust_receipt",
        lambda _receipt: {"status": "verified"},
    )
    monkeypatch.setattr(
        builder,
        "validate_generation_reports_against_metrics_trust",
        lambda *args, **kwargs: {"status": "verified"},
    )
    monkeypatch.setattr(
        builder,
        "replay_comparison",
        lambda **kwargs: {
            "status": terminal_status,
            "decision": _terminal_decision(terminal_status),
        },
    )
    monkeypatch.setattr(
        builder,
        "validate_comparison_renderings",
        lambda *args: {"markdown": {}, "csv": {}},
    )
    monkeypatch.setattr(
        builder,
        "verify_checkpoint_chain",
        lambda **kwargs: {"dense_replay_waiter": {"identity": {}}},
    )
    monkeypatch.setattr(
        builder,
        "revalidate_checkpoint_chain_sources",
        lambda _checkpoint_chain: {},
    )

    report = builder.build_audit(args)

    assert report["status"] == "pass"
    assert report["terminal_status"] == terminal_status
    assert report["generation_advantage_proven"] is advantage
    assert (
        report["claim_policy"]["matched_distribution_quality_claim_allowed"]
        is advantage
    )
    assert report["claim_policy"]["full_300k_launch_allowed"] is False
    assert report["claim_policy"]["promotion_or_release_allowed"] is False


def _waiter_args(tmp_path: Path) -> Namespace:
    quality = tmp_path / "quality"
    output_dir = quality / "reports" / "terminal_completion_audit_v1"
    metrics_trust_receipt = (
        quality / "reports" / "metrics_trust_boundary_v1" / "metrics_trust_receipt.json"
    )
    _write(metrics_trust_receipt, {"status": "pass"})
    revision = _git("rev-parse", "HEAD")
    tree = _git("rev-parse", "HEAD^{tree}")
    branch = _git("branch", "--show-current")
    auditor_source = ROOT / "scripts" / "wait_generation_checkpoint_integrity_audit.py"
    replay_source = (
        ROOT / "scripts" / "verify_generation_checkpoint_integrity_audit_replay.py"
    )
    return Namespace(
        project=ROOT,
        quality_output_root=quality,
        terminal_system_guard=quality
        / "reports"
        / "terminal_system_claim_guard_v1"
        / "terminal_system_claim_guard.json",
        expected_terminal_dir_name="terminal_system_claim_guard_v1",
        terminal_waiter_status=quality
        / "reports"
        / "terminal_system_claim_guard_v1"
        / "waiter_status.json",
        comparison=quality
        / "reports"
        / "quality_bridge_comparison_v1"
        / "quality_bridge_comparison.json",
        expected_comparison_dir_name="quality_bridge_comparison_v1",
        comparison_waiter_status=quality
        / "reports"
        / "quality_bridge_comparison_v1"
        / "waiter_status.json",
        checkpoint_audit_dir=quality / "reports" / "checkpoint_audits",
        checkpoint_replay_waiter_status=quality
        / "reports"
        / "checkpoint_audits"
        / "dense_checkpoint_integrity_replay_waiter_status.json",
        metrics_trust_receipt=metrics_trust_receipt,
        expected_metrics_trust_receipt_sha256=_identity(metrics_trust_receipt)[
            "sha256"
        ],
        physical_auditor_checkout=ROOT,
        checkpoint_replay_checkout=ROOT,
        training_checkout=ROOT,
        expected_revision=revision,
        expected_tree=tree,
        expected_branch=branch,
        expected_training_revision=revision,
        expected_training_tree=tree,
        expected_training_branch=branch,
        expected_physical_auditor_revision=revision,
        expected_physical_auditor_tree=tree,
        expected_physical_auditor_branch=branch,
        expected_physical_auditor_source_sha256=_identity(auditor_source)["sha256"],
        expected_checkpoint_replay_revision=revision,
        expected_checkpoint_replay_tree=tree,
        expected_checkpoint_replay_branch=branch,
        expected_checkpoint_replay_verifier_source_sha256=_identity(replay_source)[
            "sha256"
        ],
        expected_dataset_sha256="c" * 64,
        expected_runtime_sha256="d" * 64,
        effective_batch=64,
        expected_output_dir_name="terminal_completion_audit_v1",
        output=output_dir / "terminal_completion_audit.json",
        status_output=output_dir / "waiter_status.json",
        deployment_receipt_output=output_dir / "deployment_receipt.json",
        lock=output_dir / "waiter.lock",
        expected_waiter_source_sha256=_identity(Path(waiter.__file__))["sha256"],
        expected_builder_source_sha256=_identity(Path(builder.__file__))["sha256"],
        poll_seconds=0.01,
        timeout_seconds=1.0,
    )


def _install_ready_upstreams(args: Namespace) -> None:
    _write(args.terminal_waiter_status, {"status": "completed"})
    _write(args.terminal_system_guard, {"status": "hold"})
    _write(args.comparison_waiter_status, {"status": "pass"})
    _write(args.comparison, {"status": "hold"})
    _write(args.comparison.parent / "quality_bridge_comparison.md", "# report\n")
    _write(args.comparison.parent / "quality_bridge_comparison.csv", "method\n")
    for alias in waiter.ALIASES:
        for step in waiter.CHECKPOINT_STEPS:
            stem = f"{alias}_checkpoint_step_{step:08d}"
            _write(
                args.checkpoint_audit_dir / f"{stem}_waiter_status.json",
                {"status": "pass"},
            )
            _write(
                args.checkpoint_audit_dir / f"{stem}_physical_integrity_audit.json",
                {"status": "pass"},
            )
    _write(args.checkpoint_replay_waiter_status, {"status": "pass"})
    for step in waiter.CHECKPOINT_STEPS:
        _write(
            args.checkpoint_audit_dir
            / f"dense_checkpoint_step_{step:08d}_physical_integrity_audit_replay.json",
            {"status": "pass"},
        )


def _mock_waiter_clean_git(
    monkeypatch: pytest.MonkeyPatch,
    args: Namespace,
) -> None:
    def clean_git(_project: Path) -> dict[str, Any]:
        return {
            "revision": args.expected_revision,
            "branch": args.expected_branch,
            "tracked_dirty": False,
        }

    monkeypatch.setattr(waiter, "git_provenance", clean_git)
    monkeypatch.setattr(waiter.builder, "git_provenance", clean_git)


def test_waiter_static_context_binds_all_three_checkouts_and_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _waiter_args(tmp_path)
    _mock_waiter_clean_git(monkeypatch, args)

    context = waiter._canonical_context(args)

    assert context["training_git"]["revision"] == args.expected_training_revision
    assert context["auditor_source"]["sha256"] == (
        args.expected_physical_auditor_source_sha256
    )
    assert context["replay_verifier_source"]["sha256"] == (
        args.expected_checkpoint_replay_verifier_source_sha256
    )
    assert context["metrics_trust_identity"]["sha256"] == (
        args.expected_metrics_trust_receipt_sha256
    )

    args.expected_checkpoint_replay_verifier_source_sha256 = "0" * 64
    with pytest.raises(ValueError, match="verifier source SHA256 differs"):
        waiter._canonical_context(args)


def test_waiter_static_context_accepts_explicit_versioned_source_and_output_dirs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _waiter_args(tmp_path)
    reports = args.quality_output_root / "reports"
    terminal_dir = reports / "terminal_system_claim_guard_v2_runtime_strict"
    comparison_dir = reports / "quality_bridge_comparison_v3_runtime_strict"
    output_dir = reports / "terminal_completion_audit_v2_runtime_strict"
    args.expected_terminal_dir_name = terminal_dir.name
    args.expected_comparison_dir_name = comparison_dir.name
    args.expected_output_dir_name = output_dir.name
    args.terminal_system_guard = terminal_dir / "terminal_system_claim_guard.json"
    args.terminal_waiter_status = terminal_dir / "waiter_status.json"
    args.comparison = comparison_dir / "quality_bridge_comparison.json"
    args.comparison_waiter_status = comparison_dir / "waiter_status.json"
    args.output = output_dir / "terminal_completion_audit.json"
    args.status_output = output_dir / "waiter_status.json"
    args.deployment_receipt_output = output_dir / "deployment_receipt.json"
    args.lock = output_dir / "waiter.lock"
    _mock_waiter_clean_git(monkeypatch, args)

    context = waiter._canonical_context(args)

    assert context["terminal_guard"] == args.terminal_system_guard.resolve()
    assert context["comparison"] == args.comparison.resolve()
    assert context["output"] == args.output.resolve()


@pytest.mark.parametrize(
    "name",
    ["../terminal_completion_audit_v2", "terminal_completion_audit_v2/child"],
)
def test_waiter_static_context_rejects_unsafe_explicit_output_dir_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    args = _waiter_args(tmp_path)
    args.expected_output_dir_name = name
    _mock_waiter_clean_git(monkeypatch, args)

    with pytest.raises(ValueError, match="directory name is invalid"):
        waiter._canonical_context(args)


def test_waiter_observes_waiting_failure_and_ready_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _waiter_args(tmp_path)
    _mock_waiter_clean_git(monkeypatch, args)
    context = waiter._canonical_context(args)

    waiting = waiter.observe_upstreams(context)
    assert waiting["state"] == "waiting"
    assert "terminal_system_guard_waiter" in waiting["missing"]

    _install_ready_upstreams(args)
    assert waiter.observe_upstreams(context)["state"] == "ready"

    _write(
        args.checkpoint_audit_dir / "dense_checkpoint_step_00095000_waiter_status.json",
        {"status": "failed", "detail": "physical hash mismatch"},
    )
    failed = waiter.observe_upstreams(context)
    assert failed["state"] == "failed"
    assert "physical hash mismatch" in failed["failures"][0]


def test_waiter_metadata_snapshot_detects_source_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _waiter_args(tmp_path)
    _mock_waiter_clean_git(monkeypatch, args)
    _install_ready_upstreams(args)
    context = waiter._canonical_context(args)
    before = waiter.snapshot_upstream_metadata(context)

    _write(args.comparison.parent / "quality_bridge_comparison.csv", "drift\n")

    assert waiter.snapshot_upstream_metadata(context) != before


def test_waiter_rejects_metrics_trust_receipt_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _waiter_args(tmp_path)
    _mock_waiter_clean_git(monkeypatch, args)
    _install_ready_upstreams(args)
    context = waiter._canonical_context(args)

    _write(args.metrics_trust_receipt, {"status": "failed"})

    with pytest.raises(ValueError, match="metrics trust receipt SHA256 differs"):
        waiter._canonical_context(args)
    failed = waiter.observe_upstreams(context)
    assert failed["state"] == "failed"
    assert "metrics trust receipt:failed" in failed["failures"][0]


def test_deployment_receipt_allows_exact_restart_pid_runtime_normalization(
    tmp_path: Path,
) -> None:
    path = tmp_path / "deployment.json"
    original = {
        "schema_version": 1,
        "created_at": "first",
        "pid": 10,
        "runtime": {"pid": 10, "ppid": 1},
        "git": {"revision": "a" * 40},
    }
    restarted = {
        **copy.deepcopy(original),
        "created_at": "second",
        "pid": 20,
        "runtime": {"pid": 20, "ppid": 1},
    }
    first = waiter.prepare_deployment_receipt(path, original)
    second = waiter.prepare_deployment_receipt(path, restarted)

    assert second == first
    assert json.loads(path.read_text(encoding="utf-8"))["pid"] == 10

    changed = copy.deepcopy(restarted)
    changed["git"] = {"revision": "b" * 40}
    with pytest.raises(ValueError, match="differs"):
        waiter.prepare_deployment_receipt(path, changed)


def _fake_context(tmp_path: Path) -> dict[str, Any]:
    output_dir = tmp_path / "reports" / "terminal_completion_audit_v1"
    terminal_guard = tmp_path / "terminal_system_claim_guard.json"
    comparison = tmp_path / "quality_bridge_comparison.json"
    metrics_trust_receipt = tmp_path / "metrics_trust_receipt.json"
    _write(terminal_guard, {"status": "hold"})
    _write(comparison, {"status": "hold"})
    _write(metrics_trust_receipt, {"status": "pass"})
    return {
        "git": {"revision": "a" * 40},
        "waiter_source": {"path": "/waiter", "bytes": 1, "sha256": SHA},
        "builder_source": {"path": "/builder", "bytes": 1, "sha256": SHA},
        "training_git": {"revision": "b" * 40},
        "auditor_git": {"revision": "c" * 40},
        "replay_git": {"revision": "d" * 40},
        "auditor_source": {"path": "/auditor", "bytes": 1, "sha256": SHA},
        "replay_verifier_source": {
            "path": "/replay",
            "bytes": 1,
            "sha256": SHA,
        },
        "metrics_trust_identity": _identity(metrics_trust_receipt),
        "metrics_trust_receipt": metrics_trust_receipt,
        "terminal_guard": terminal_guard,
        "comparison": comparison,
        "status": output_dir / "waiter_status.json",
        "deployment": output_dir / "deployment_receipt.json",
        "output": output_dir / "terminal_completion_audit.json",
    }


def test_run_locked_waits_runs_and_publishes_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = Namespace(poll_seconds=0.01, timeout_seconds=1.0)
    context = _fake_context(tmp_path)
    observations = iter(
        [
            {"state": "waiting", "detail": "waiting", "missing": ["source"]},
            {"state": "ready", "detail": "ready", "terminal_status": "hold"},
        ]
    )
    published: list[dict[str, Any]] = []
    monkeypatch.setattr(waiter, "validate_runtime", lambda **kwargs: {"pid": 1})
    monkeypatch.setattr(waiter, "_canonical_context", lambda _args: context)
    monkeypatch.setattr(waiter, "deployment_payload", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        waiter,
        "prepare_deployment_receipt",
        lambda *args, **kwargs: {"path": "/deployment", "bytes": 1, "sha256": SHA},
    )
    monkeypatch.setattr(waiter, "observe_upstreams", lambda _ctx: next(observations))
    monkeypatch.setattr(waiter.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        waiter,
        "snapshot_upstream_metadata",
        lambda _ctx: {"source": {"sha256": SHA}},
    )
    monkeypatch.setattr(
        waiter.builder,
        "build_audit",
        lambda _args: {
            "terminal_status": "hold",
            "terminal_decision": _terminal_decision("hold"),
            "generation_advantage_proven": False,
        },
    )
    monkeypatch.setattr(
        waiter,
        "prepare_manifest",
        lambda *args, **kwargs: {"path": "/audit", "bytes": 1, "sha256": SHA},
    )
    monkeypatch.setattr(
        waiter,
        "publish",
        lambda _path, **kwargs: published.append(kwargs),
    )

    assert waiter.run_locked(args, require_detached=False) == 0
    assert [row["status"] for row in published] == ["waiting", "running", "pass"]
    assert published[-1]["audit"]["generation_advantage_proven"] is False


def test_run_locked_fails_closed_on_upstream_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = Namespace(poll_seconds=0.01, timeout_seconds=1.0)
    context = _fake_context(tmp_path)
    published: list[dict[str, Any]] = []
    monkeypatch.setattr(waiter, "validate_runtime", lambda **kwargs: {"pid": 1})
    monkeypatch.setattr(waiter, "_canonical_context", lambda _args: context)
    monkeypatch.setattr(waiter, "deployment_payload", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        waiter,
        "prepare_deployment_receipt",
        lambda *args, **kwargs: {"path": "/deployment", "bytes": 1, "sha256": SHA},
    )
    monkeypatch.setattr(
        waiter,
        "observe_upstreams",
        lambda _ctx: {"state": "failed", "detail": "source_failed"},
    )
    monkeypatch.setattr(
        waiter,
        "publish",
        lambda _path, **kwargs: published.append(kwargs),
    )

    assert waiter.run_locked(args, require_detached=False) == 1
    assert published[-1]["status"] == "failed"


def test_run_locked_rejects_metadata_drift_during_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = Namespace(poll_seconds=0.01, timeout_seconds=1.0)
    context = _fake_context(tmp_path)
    snapshots = iter([{"sha": "before"}, {"sha": "after"}])
    monkeypatch.setattr(waiter, "validate_runtime", lambda **kwargs: {"pid": 1})
    monkeypatch.setattr(waiter, "_canonical_context", lambda _args: context)
    monkeypatch.setattr(waiter, "deployment_payload", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        waiter,
        "prepare_deployment_receipt",
        lambda *args, **kwargs: {"path": "/deployment", "bytes": 1, "sha256": SHA},
    )
    monkeypatch.setattr(
        waiter,
        "observe_upstreams",
        lambda _ctx: {"state": "ready", "detail": "ready"},
    )
    monkeypatch.setattr(
        waiter,
        "snapshot_upstream_metadata",
        lambda _ctx: next(snapshots),
    )
    monkeypatch.setattr(
        waiter.builder,
        "build_audit",
        lambda _args: {
            "terminal_status": "hold",
            "terminal_decision": _terminal_decision("hold"),
            "generation_advantage_proven": False,
        },
    )
    monkeypatch.setattr(waiter, "publish", lambda *args, **kwargs: None)

    with pytest.raises(ValueError, match="metadata sources changed"):
        waiter.run_locked(args, require_detached=False)
