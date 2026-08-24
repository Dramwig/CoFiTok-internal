from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping

from cofitok.generation.conditioning_ranking_terminal_rebind import (
    EPSILON_INPUT_NAMES,
    EXPECTED_SOURCE_SHA256,
    LEGACY_PREPARATION_GIT,
    OFFICIAL_EPSILON_VALIDATOR_GIT,
    OFFICIAL_EPSILON_VALIDATOR_SHA256,
    OUTPUT_ROOT,
    build_terminal_rebind_authorization,
)
from cofitok.inference_replay import (
    file_identity,
    prepare_manifest,
    read_json_object,
    reject_symlink_chain,
)
from cofitok.reporting import git_provenance


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_NAMES = (
    "preparation",
    "legacy_preparation",
    "standing_authorization",
    "quality_bridge_result",
    "post_reconciliation_decision",
    "post_reconciliation_verification",
    "epsilon_stability_result",
    "epsilon_recovery_status",
    "conditioning_gain_comparison",
    "supersession_receipt",
    "supersession_marker",
    "runbook",
)


def _full_git(project: Path) -> dict[str, Any]:
    project = reject_symlink_chain(project, name="terminal-rebind checkout").resolve()
    provenance = git_provenance(project)
    full_status = subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=project,
        text=True,
    ).strip()
    if provenance["tracked_dirty"] or full_status:
        raise ValueError(f"terminal-rebind checkout is not fully clean: {project}")
    return {
        **provenance,
        "tree": subprocess.check_output(
            ["git", "rev-parse", "HEAD^{tree}"],
            cwd=project,
            text=True,
        ).strip(),
        "path": project.as_posix(),
    }


def _without_path(git: Mapping[str, Any]) -> dict[str, Any]:
    return {key: git[key] for key in ("revision", "tree", "branch", "tracked_dirty")}


def _load_source(
    path: str | Path,
    *,
    label: str,
    expected_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = reject_symlink_chain(path, name=label).resolve()
    identity = file_identity(source)
    if expected_sha256 is not None and identity["sha256"] != expected_sha256:
        raise ValueError(f"{label} SHA256 differs")
    return read_json_object(source, name=label), identity


def _replay_descriptor(
    descriptor: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    source = reject_symlink_chain(
        Path(str(descriptor.get("path", ""))),
        name=label,
    ).resolve()
    identity = file_identity(source)
    if identity != dict(descriptor):
        raise ValueError(f"{label} identity changed")
    return identity


def _replay_gain_sources(report: Mapping[str, Any]) -> dict[str, Any]:
    sources = report.get("sources")
    if not isinstance(sources, Mapping):
        raise ValueError("conditioning-gain sources are missing")
    return {
        name: _replay_descriptor(descriptor, label=f"conditioning-gain {name}")
        for name, descriptor in sources.items()
        if isinstance(descriptor, Mapping)
    }


def _replay_legacy_configs(
    *,
    project: Path,
    preparation: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    checkout = _full_git(project)
    if _without_path(checkout) != LEGACY_PREPARATION_GIT:
        raise ValueError("legacy conditioning-ranking checkout identity differs")
    configs = preparation.get("configs")
    if not isinstance(configs, Mapping):
        raise ValueError("legacy conditioning-ranking configs are missing")
    replayed: dict[str, Any] = {}
    for name, descriptor in configs.items():
        if not isinstance(descriptor, Mapping):
            raise ValueError(f"legacy conditioning-ranking config is malformed: {name}")
        relative = Path(str(descriptor.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"legacy conditioning-ranking config path is invalid: {name}")
        identity = file_identity(project / relative)
        if (
            identity["bytes"] != descriptor.get("bytes")
            or identity["sha256"] != descriptor.get("sha256")
        ):
            raise ValueError(f"legacy conditioning-ranking config changed: {name}")
        replayed[name] = identity
    return checkout, replayed


def replay_official_epsilon_result(
    *,
    project: Path,
    python: Path,
    result: Mapping[str, Any],
    result_identity: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    checkout = _full_git(project)
    if _without_path(checkout) != OFFICIAL_EPSILON_VALIDATOR_GIT:
        raise ValueError("official epsilon validator checkout identity differs")
    validator = project / "scripts" / "validate_generation_epsilon_stability_sampling_result.py"
    validator_identity = file_identity(validator)
    if validator_identity["sha256"] != OFFICIAL_EPSILON_VALIDATOR_SHA256:
        raise ValueError("official epsilon validator source SHA256 differs")
    if not python.is_file():
        raise FileNotFoundError(f"terminal-rebind Python is missing: {python}")
    bindings = result.get("source_bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("epsilon-stability result source bindings are missing")
    input_graph = {
        name: _replay_descriptor(
            bindings.get(name, {}),
            label=f"official epsilon input {name}",
        )
        for name in EPSILON_INPUT_NAMES
    }
    command = [
        python.resolve().as_posix(),
        validator.resolve().as_posix(),
        "--result",
        str(result_identity["path"]),
        "--expected-result-sha256",
        str(result_identity["sha256"]),
        "--design",
        input_graph["design"]["path"],
        "--expected-design-sha256",
        input_graph["design"]["sha256"],
        "--execution-authorization",
        input_graph["execution_authorization"]["path"],
        "--expected-execution-authorization-sha256",
        input_graph["execution_authorization"]["sha256"],
        "--observation-manifest",
        input_graph["observation_manifest"]["path"],
        "--expected-observation-manifest-sha256",
        input_graph["observation_manifest"]["sha256"],
        "--real-artifact-reference",
        input_graph["real_artifact_reference"]["path"],
        "--expected-real-artifact-reference-sha256",
        input_graph["real_artifact_reference"]["sha256"],
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "-1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "PYTHONPATH": f"{project / 'src'}:{project}",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    completed = subprocess.run(
        command,
        cwd=project,
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        timeout=600,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError(
            "official epsilon result replay failed: " + completed.stderr[-2000:]
        )
    replay = {
        "status": "verified",
        "checkout_git": _without_path(checkout),
        "validator": validator_identity,
        "result": dict(result_identity),
        "input_graph": input_graph,
        "cuda_visible_devices": "-1",
        "returncode": 0,
    }
    return replay, checkout


def _assemble(
    *,
    sources: Mapping[str, tuple[dict[str, Any], dict[str, Any]]],
    project: Path,
    legacy_project: Path,
    official_epsilon_project: Path,
    python: Path,
    expected_revision: str,
    expected_tree: str,
    expected_branch: str,
    expected_output_root: str,
) -> dict[str, Any]:
    builder_checkout = _full_git(project)
    expected_git = {
        "revision": expected_revision,
        "tree": expected_tree,
        "branch": expected_branch,
        "tracked_dirty": False,
    }
    if _without_path(builder_checkout) != expected_git:
        raise ValueError("terminal-rebind builder checkout identity differs")
    if expected_output_root != OUTPUT_ROOT:
        raise ValueError("terminal-rebind output root differs")
    legacy_checkout, legacy_configs = _replay_legacy_configs(
        project=legacy_project,
        preparation=sources["legacy_preparation"][0],
    )
    gain_sources = _replay_gain_sources(sources["conditioning_gain_comparison"][0])
    official_replay, official_checkout = replay_official_epsilon_result(
        project=official_epsilon_project,
        python=python,
        result=sources["epsilon_stability_result"][0],
        result_identity=sources["epsilon_stability_result"][1],
    )
    runbook_identity = sources["runbook"][1]
    control_plane = {
        "builder_checkout": builder_checkout,
        "legacy_candidate_checkout": legacy_checkout,
        "legacy_candidate_configs": legacy_configs,
        "official_epsilon_validator_checkout": official_checkout,
        "official_epsilon_validator": official_replay["validator"],
        "python": python.resolve().as_posix(),
    }
    return build_terminal_rebind_authorization(
        preparation=sources["preparation"][0],
        preparation_identity=sources["preparation"][1],
        legacy_preparation=sources["legacy_preparation"][0],
        legacy_preparation_identity=sources["legacy_preparation"][1],
        standing_authorization=sources["standing_authorization"][0],
        standing_authorization_identity=sources["standing_authorization"][1],
        quality_result=sources["quality_bridge_result"][0],
        quality_result_identity=sources["quality_bridge_result"][1],
        post_decision=sources["post_reconciliation_decision"][0],
        post_decision_identity=sources["post_reconciliation_decision"][1],
        post_verification=sources["post_reconciliation_verification"][0],
        post_verification_identity=sources["post_reconciliation_verification"][1],
        epsilon_result=sources["epsilon_stability_result"][0],
        epsilon_result_identity=sources["epsilon_stability_result"][1],
        epsilon_recovery_status=sources["epsilon_recovery_status"][0],
        epsilon_recovery_status_identity=sources["epsilon_recovery_status"][1],
        gain_report=sources["conditioning_gain_comparison"][0],
        gain_report_identity=sources["conditioning_gain_comparison"][1],
        gain_replayed_sources=gain_sources,
        supersession_receipt=sources["supersession_receipt"][0],
        supersession_receipt_identity=sources["supersession_receipt"][1],
        supersession_marker=sources["supersession_marker"][0],
        supersession_marker_identity=sources["supersession_marker"][1],
        official_epsilon_replay=official_replay,
        runbook_identity=runbook_identity,
        authorization_git=expected_git,
        expected_revision=expected_revision,
        expected_tree=expected_tree,
        expected_branch=expected_branch,
        expected_output_root=expected_output_root,
        control_plane=control_plane,
    )


def build_from_authorization(
    authorization: Mapping[str, Any],
    *,
    project: Path,
    expected_revision: str,
    expected_tree: str,
    expected_branch: str,
    expected_output_root: str,
) -> dict[str, Any]:
    descriptors = authorization.get("source_reports")
    control = authorization.get("source_control")
    if not isinstance(descriptors, Mapping) or set(descriptors) != set(SOURCE_NAMES):
        raise ValueError("terminal-rebind authorization source set differs")
    if not isinstance(control, Mapping):
        raise ValueError("terminal-rebind authorization control plane is missing")
    sources: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for name in SOURCE_NAMES:
        descriptor = descriptors[name]
        if not isinstance(descriptor, Mapping):
            raise ValueError(f"terminal-rebind source descriptor is malformed: {name}")
        identity = _replay_descriptor(descriptor, label=f"terminal-rebind {name}")
        if name == "runbook":
            sources[name] = ({}, identity)
        else:
            sources[name] = (
                read_json_object(identity["path"], name=f"terminal-rebind {name}"),
                identity,
            )
    legacy_checkout = control.get("legacy_candidate_checkout")
    official_checkout = control.get("official_epsilon_validator_checkout")
    python = control.get("python")
    if (
        not isinstance(legacy_checkout, Mapping)
        or not isinstance(official_checkout, Mapping)
        or not isinstance(python, str)
        or not python
    ):
        raise ValueError("terminal-rebind control checkout binding is malformed")
    return _assemble(
        sources=sources,
        project=project,
        legacy_project=Path(str(legacy_checkout.get("path", ""))),
        official_epsilon_project=Path(str(official_checkout.get("path", ""))),
        python=Path(python),
        expected_revision=expected_revision,
        expected_tree=expected_tree,
        expected_branch=expected_branch,
        expected_output_root=expected_output_root,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the source-bound terminal rebind for the four-arm 1K "
            "training-time semantic-alignment probe."
        )
    )
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--expected-preparation-sha256", required=True)
    parser.add_argument("--legacy-preparation", type=Path, required=True)
    parser.add_argument("--standing-authorization", type=Path, required=True)
    parser.add_argument("--quality-bridge-result", type=Path, required=True)
    parser.add_argument("--post-reconciliation-decision", type=Path, required=True)
    parser.add_argument("--post-reconciliation-verification", type=Path, required=True)
    parser.add_argument("--epsilon-stability-result", type=Path, required=True)
    parser.add_argument("--epsilon-recovery-status", type=Path, required=True)
    parser.add_argument("--conditioning-gain-comparison", type=Path, required=True)
    parser.add_argument("--supersession-receipt", type=Path, required=True)
    parser.add_argument("--supersession-marker", type=Path, required=True)
    parser.add_argument("--legacy-project", type=Path, required=True)
    parser.add_argument("--official-epsilon-project", type=Path, required=True)
    parser.add_argument("--runbook", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-tree", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def build_from_paths(args: argparse.Namespace) -> dict[str, Any]:
    path_map = {
        "preparation": args.preparation,
        "legacy_preparation": args.legacy_preparation,
        "standing_authorization": args.standing_authorization,
        "quality_bridge_result": args.quality_bridge_result,
        "post_reconciliation_decision": args.post_reconciliation_decision,
        "post_reconciliation_verification": args.post_reconciliation_verification,
        "epsilon_stability_result": args.epsilon_stability_result,
        "epsilon_recovery_status": args.epsilon_recovery_status,
        "conditioning_gain_comparison": args.conditioning_gain_comparison,
        "supersession_receipt": args.supersession_receipt,
        "supersession_marker": args.supersession_marker,
    }
    sources: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for name, path in path_map.items():
        expected = (
            args.expected_preparation_sha256
            if name == "preparation"
            else EXPECTED_SOURCE_SHA256[name]
        )
        sources[name] = _load_source(path, label=name, expected_sha256=expected)
    runbook = reject_symlink_chain(args.runbook, name="terminal-rebind runbook").resolve()
    sources["runbook"] = ({}, file_identity(runbook))
    return _assemble(
        sources=sources,
        project=PROJECT_ROOT,
        legacy_project=args.legacy_project,
        official_epsilon_project=args.official_epsilon_project,
        python=args.python,
        expected_revision=args.expected_revision,
        expected_tree=args.expected_tree,
        expected_branch=args.expected_branch,
        expected_output_root=args.output_root,
    )


def main() -> None:
    args = parse_args()
    report = build_from_paths(args)
    identity = prepare_manifest(
        args.output,
        report,
        resume=args.resume,
        overwrite=False,
    )
    print(identity["sha256"])


if __name__ == "__main__":
    main()
