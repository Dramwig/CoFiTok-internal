from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

from cofitok.generation.factorization_quality_regression import (
    CHECKPOINT_STEP,
    RUN_DIRS,
    build_source_binding,
)
from cofitok.inference_replay import (
    file_identity,
    prepare_manifest,
    read_json_object,
    reject_symlink_chain,
)

try:
    import build_generation_quality_bridge_followup_decision as followup_builder
    import build_generation_terminal_system_claim_guard as terminal_builder
except ModuleNotFoundError:  # Imported as scripts.<module> by tests.
    from scripts import build_generation_quality_bridge_followup_decision as followup_builder
    from scripts import build_generation_terminal_system_claim_guard as terminal_builder


def _bound_json(
    path: Path,
    *,
    label: str,
    expected_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = reject_symlink_chain(path, name=label).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"{label} is missing: {source}")
    identity = file_identity(source)
    if expected_sha256 is not None and identity["sha256"] != expected_sha256:
        raise ValueError(f"{label} SHA256 differs")
    return read_json_object(source, name=label), identity


def _physical_source(
    sources: Mapping[str, Any],
    name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected = sources.get(name)
    if not isinstance(expected, Mapping):
        raise ValueError(f"quality result source {name} is missing")
    report, identity = _bound_json(
        Path(str(expected.get("path", ""))),
        label=f"quality result source {name}",
    )
    if identity != dict(expected):
        raise ValueError(f"quality result source {name} identity differs")
    return report, identity


def _checkpoint_reference(
    training: Mapping[str, Any],
    *,
    method: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    latest = training.get("latest_checkpoint")
    if not isinstance(latest, Mapping):
        raise ValueError(f"{method} latest checkpoint is missing")
    run_dir = Path(RUN_DIRS[method])
    checkpoint = reject_symlink_chain(
        run_dir / str(latest.get("checkpoint", "")),
        name=f"{method} 100K checkpoint",
    ).resolve()
    integrity = reject_symlink_chain(
        run_dir / str(latest.get("integrity_manifest", "")),
        name=f"{method} 100K checkpoint integrity",
    ).resolve()
    if not checkpoint.is_file() or not integrity.is_file():
        raise FileNotFoundError(f"{method} 100K checkpoint or integrity sidecar is missing")
    sidecar = read_json_object(integrity, name=f"{method} checkpoint integrity")
    expected_checkpoint_name = "checkpoint_step_00100000.pt"
    if (
        sidecar.get("schema_version") != 1
        or sidecar.get("checkpoint") != expected_checkpoint_name
        or int(sidecar.get("step", -1)) != CHECKPOINT_STEP
        or int(sidecar.get("checkpoint_bytes", -1)) != checkpoint.stat().st_size
        or sidecar.get("checkpoint_sha256") != latest.get("checkpoint_sha256")
        or sidecar.get("git_revision") != training.get("git", {}).get("revision")
        or sidecar.get("git_branch") != training.get("git", {}).get("branch")
        or sidecar.get("git_dirty") is not False
    ):
        raise ValueError(f"{method} checkpoint integrity metadata differs")
    return (
        {
            "checkpoint": checkpoint.as_posix(),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "checkpoint_sha256": sidecar["checkpoint_sha256"],
            "checkpoint_step": CHECKPOINT_STEP,
            "integrity_manifest": file_identity(integrity),
        },
        file_identity(integrity),
    )


def _replay_terminal_guard(
    guard: Mapping[str, Any],
    *,
    quality_root: Path,
) -> dict[str, Any]:
    sources = guard.get("sources")
    if not isinstance(sources, Mapping):
        raise ValueError("terminal-system guard sources are missing")

    def source(name: str) -> tuple[Path, str]:
        identity = sources.get(name)
        if not isinstance(identity, Mapping):
            raise ValueError(f"terminal-system guard source {name} is missing")
        return Path(str(identity.get("path", ""))), str(identity.get("sha256", ""))

    quality_path, quality_sha = source("quality_bridge_result")
    statistical_path, statistical_sha = source("statistical_claim_language_guard")
    visual_path, visual_sha = source("requested_class_visual_audit_waiter_status")
    runtime_path, runtime_sha = source("runtime_compute_claim_guard")
    rebuilt = terminal_builder.build_guard(
        quality_result_path=quality_path,
        expected_quality_result_sha256=quality_sha,
        statistical_claim_guard_path=statistical_path,
        expected_statistical_claim_guard_sha256=statistical_sha,
        visual_audit_waiter_status_path=visual_path,
        expected_visual_audit_waiter_status_sha256=visual_sha,
        runtime_claim_guard_path=runtime_path,
        expected_runtime_claim_guard_sha256=runtime_sha,
        quality_output_root=quality_root,
    )
    if dict(guard) != rebuilt:
        raise ValueError("terminal-system guard is not reproducible")
    return rebuilt


def build_from_paths(
    *,
    preparation_path: Path,
    expected_preparation_sha256: str,
    followup_decision_path: Path,
    terminal_system_guard_path: Path,
    expected_revision: str,
    expected_branch: str,
) -> dict[str, Any]:
    preparation, preparation_identity = _bound_json(
        preparation_path,
        label="factorization-regression preparation",
        expected_sha256=expected_preparation_sha256,
    )
    decision, decision_identity = _bound_json(
        followup_decision_path,
        label="quality-bridge follow-up decision",
    )
    result_source = decision.get("source_reports", {}).get("quality_bridge_result")
    if not isinstance(result_source, Mapping):
        raise ValueError("follow-up decision does not bind a quality result")
    result_path = Path(str(result_source.get("path", "")))
    result_sha = str(result_source.get("sha256", ""))
    rebuilt_decision = followup_builder.build_from_sources(
        quality_bridge_result_path=result_path,
        expected_quality_bridge_result_sha256=result_sha,
        decision_git=dict(decision.get("decision_builder_git", {})),
    )
    if decision != rebuilt_decision:
        raise ValueError("quality-bridge follow-up decision is not reproducible")
    quality_result, quality_identity = followup_builder.replay_quality_bridge_result(
        result_path,
        expected_sha256=result_sha,
    )
    if quality_identity != dict(result_source):
        raise ValueError("physical quality result identity differs from follow-up decision")
    terminal_guard, terminal_identity = _bound_json(
        terminal_system_guard_path,
        label="terminal-system claim guard",
    )
    _replay_terminal_guard(
        terminal_guard,
        quality_root=Path(str(preparation["quality_bridge_root"])),
    )
    result_sources = quality_result.get("source_reports")
    if not isinstance(result_sources, Mapping):
        raise ValueError("quality result source reports are missing")
    cofitok_training, cofitok_training_identity = _physical_source(
        result_sources,
        "cofitok_training",
    )
    dense_training, dense_training_identity = _physical_source(
        result_sources,
        "dense_training",
    )
    cofitok_eval, cofitok_eval_identity = _physical_source(
        result_sources,
        "cofitok_checkpoint_eval",
    )
    dense_eval, dense_eval_identity = _physical_source(
        result_sources,
        "dense_checkpoint_eval",
    )
    cofitok_checkpoint, cofitok_integrity_identity = _checkpoint_reference(
        cofitok_training,
        method="cofitok",
    )
    dense_checkpoint, dense_integrity_identity = _checkpoint_reference(
        dense_training,
        method="dense_identity",
    )
    sources = {
        "quality_result": quality_identity,
        "followup_decision": decision_identity,
        "terminal_system_guard": terminal_identity,
        "cofitok_training": cofitok_training_identity,
        "dense_training": dense_training_identity,
        "cofitok_checkpoint_evaluation": cofitok_eval_identity,
        "dense_checkpoint_evaluation": dense_eval_identity,
        "cofitok_checkpoint_integrity": cofitok_integrity_identity,
        "dense_checkpoint_integrity": dense_integrity_identity,
    }
    return build_source_binding(
        preparation=preparation,
        preparation_identity=preparation_identity,
        followup_decision=decision,
        followup_decision_identity=decision_identity,
        terminal_system_guard=terminal_guard,
        terminal_system_guard_identity=terminal_identity,
        quality_result=quality_result,
        quality_result_identity=quality_identity,
        training_reports={
            "cofitok": cofitok_training,
            "dense_identity": dense_training,
        },
        checkpoint_evaluations={
            "cofitok": cofitok_eval,
            "dense_identity": dense_eval,
        },
        checkpoint_references={
            "cofitok": cofitok_checkpoint,
            "dense_identity": dense_checkpoint,
        },
        source_identities=sources,
        expected_revision=expected_revision,
        expected_branch=expected_branch,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay and bind the exact terminal 100K sources selected for the matched "
            "factorization quality-regression diagnostic."
        )
    )
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--expected-preparation-sha256", required=True)
    parser.add_argument("--followup-decision", type=Path, required=True)
    parser.add_argument("--terminal-system-guard", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_from_paths(
        preparation_path=args.preparation,
        expected_preparation_sha256=args.expected_preparation_sha256,
        followup_decision_path=args.followup_decision,
        terminal_system_guard_path=args.terminal_system_guard,
        expected_revision=args.expected_revision,
        expected_branch=args.expected_branch,
    )
    identity = prepare_manifest(
        args.output,
        report,
        resume=args.resume,
        overwrite=False,
    )
    print(identity["sha256"])


if __name__ == "__main__":
    main()
