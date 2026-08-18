from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
from typing import Any, Mapping

from cofitok.generation.conditioning_ranking_sampling import (
    METHOD_PREFIX_BUDGETS,
    PREPARATION_ROLE,
    RUN_METHODS,
    RUN_NAMES,
    SAMPLING_PROTOCOL,
    SCHEMA_VERSION,
    STAGE,
    clean_git,
    identity,
)
from cofitok.generation import conditioning_ranking_posttraining_sampling as posttraining
from cofitok.inference_replay import (
    file_identity,
    read_json_object,
    reject_symlink_chain,
)
from cofitok.output_lock import exclusive_output_locks
from cofitok.reporting import git_provenance, write_json_report
from scripts.select_generation_sampling_batch import (
    _checkpoint_identity,
    _preflight_matches,
    _protocol,
    _run_preflight,
    _validated_runtime_environment_sha,
    parse_candidates,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_ROLE = (
    "generation_conditioning_ranking_four_arm_sampling_batch_selection"
)


def _stage_contract(preparation: Mapping[str, Any]) -> dict[str, Any]:
    stage = preparation.get("stage")
    if stage == STAGE:
        return {
            "stage": STAGE,
            "preparation_role": PREPARATION_ROLE,
            "sampling_protocol": SAMPLING_PROTOCOL,
            "method_prefix_budgets": METHOD_PREFIX_BUDGETS,
            "run_methods": RUN_METHODS,
            "sample_run_name": "samples_5000_ddim50_cfg15",
        }
    if stage == posttraining.STAGE:
        return {
            "stage": posttraining.STAGE,
            "preparation_role": posttraining.PREPARATION_ROLE,
            "sampling_protocol": posttraining.SAMPLING_PROTOCOL,
            "method_prefix_budgets": posttraining.METHOD_PREFIX_BUDGETS,
            "run_methods": posttraining.RUN_METHODS,
            "sample_run_name": posttraining.SAMPLE_RUN_NAME,
        }
    raise ValueError("conditioning-ranking sampling stage is unsupported")


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def select_four_arm_sampling_batch(
    candidates: list[dict[str, Any]],
    *,
    baseline_batch_size: int,
    max_memory_fraction: float,
) -> dict[str, Any]:
    if not 0.0 < max_memory_fraction < 1.0:
        raise ValueError("maximum memory fraction must be between zero and one")
    normalized: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    environment_shas_seen: list[str] = []
    for source in candidates:
        row = copy.deepcopy(source)
        batch_size = int(row.get("batch_size", -1))
        arms = row.get("arms")
        if not isinstance(arms, Mapping) or set(arms) != set(RUN_NAMES):
            raise ValueError("four-arm preflight candidate set differs")
        reasons: list[str] = []
        throughputs: list[float] = []
        memory_fractions: list[float] = []
        protocols: list[dict[str, Any]] = []
        environment_shas: list[str] = []
        for run in RUN_NAMES:
            arm = arms[run]
            report = arm.get("report") if isinstance(arm, Mapping) else None
            if not isinstance(report, Mapping) or report.get("status") != "passed":
                reasons.append(f"{run}_preflight_incomplete")
                continue
            request = report.get("request")
            result = report.get("result")
            if not isinstance(request, Mapping) or not isinstance(result, Mapping):
                reasons.append(f"{run}_preflight_malformed")
                continue
            if int(request.get("batch_size", -1)) != batch_size:
                reasons.append(f"{run}_batch_mismatch")
            throughput = _finite(
                result.get("output_images_per_second"),
                label=f"{run} preflight throughput",
            )
            if throughput <= 0.0:
                reasons.append(f"{run}_invalid_throughput")
            else:
                throughputs.append(throughput)
            memory = result.get("cuda_memory_after_forward")
            peak = int(memory.get("peak_allocated_bytes", 0)) if isinstance(memory, Mapping) else 0
            total = int(result.get("device_total_memory_bytes", 0))
            if peak <= 0 or total <= 0 or peak > total:
                reasons.append(f"{run}_invalid_memory_accounting")
            else:
                fraction = peak / total
                memory_fractions.append(fraction)
                if fraction > max_memory_fraction:
                    reasons.append(f"{run}_memory_headroom")
            protocols.append(_protocol(dict(report)))
            environment_sha = _validated_runtime_environment_sha(dict(report))
            if environment_sha is None:
                reasons.append(f"{run}_runtime_environment_invalid")
            else:
                environment_shas.append(environment_sha)
        if len(protocols) == len(RUN_NAMES) and any(
            protocol != protocols[0] for protocol in protocols[1:]
        ):
            reasons.append("four_arm_sampling_protocol_mismatch")
        if len(environment_shas) == len(RUN_NAMES):
            if len(set(environment_shas)) != 1:
                reasons.append("four_arm_sampling_environment_mismatch")
            else:
                environment_shas_seen.append(environment_shas[0])
        row["runtime_environment_sha256"] = (
            environment_shas[0]
            if len(environment_shas) == len(RUN_NAMES)
            and len(set(environment_shas)) == 1
            else None
        )
        row["eligible"] = not reasons
        row["ineligible_reasons"] = sorted(set(reasons))
        row["selection_score_images_per_second"] = (
            min(throughputs) if len(throughputs) == len(RUN_NAMES) else None
        )
        row["max_memory_fraction"] = (
            max(memory_fractions)
            if len(memory_fractions) == len(RUN_NAMES)
            else None
        )
        normalized.append(row)
        if row["eligible"]:
            eligible.append(row)
    if len(set(environment_shas_seen)) > 1:
        raise ValueError("runtime environment changed across four-arm candidates")
    if not eligible:
        raise ValueError("no shared four-arm sampling batch passed")
    baseline = next(
        (row for row in normalized if row["batch_size"] == baseline_batch_size),
        None,
    )
    if baseline is None or not baseline["eligible"]:
        raise ValueError("conservative four-arm sampling baseline did not pass")
    selected = max(
        eligible,
        key=lambda row: (
            float(row["selection_score_images_per_second"]),
            -int(row["batch_size"]),
        ),
    )
    return {
        "policy": {
            "shared_candidate_required": True,
            "score": "maximize_worst_four_arm_output_images_per_second",
            "baseline_batch_size": baseline_batch_size,
            "max_memory_fraction": max_memory_fraction,
            "batch_size_invariant_random_stream_required": True,
        },
        "selected": {
            "batch_size": int(selected["batch_size"]),
            "selection_score_images_per_second": float(
                selected["selection_score_images_per_second"]
            ),
            "max_memory_fraction": float(selected["max_memory_fraction"]),
            "estimated_speedup_over_baseline": (
                float(selected["selection_score_images_per_second"])
                / float(baseline["selection_score_images_per_second"])
            ),
        },
        "runtime_environment_sha256": selected["runtime_environment_sha256"],
        "candidates": normalized,
    }


def _load_preparation(path: Path, expected_sha256: str) -> tuple[dict, dict]:
    source = reject_symlink_chain(path, name="sampling preparation").resolve()
    source_identity = file_identity(source)
    if source_identity["sha256"] != expected_sha256:
        raise ValueError("sampling preparation SHA256 differs")
    report = read_json_object(source, name="sampling preparation")
    contract = _stage_contract(report)
    if (
        report.get("schema_version") != SCHEMA_VERSION
        or report.get("role") != contract["preparation_role"]
        or report.get("status") != "pass"
        or report.get("valid") is not True
        or report.get("stage") != contract["stage"]
        or report.get("sampling_protocol") != contract["sampling_protocol"]
    ):
        raise ValueError("sampling preparation contract differs")
    return report, source_identity


def _physical_checkpoint_contract(preparation: Mapping[str, Any]) -> dict[str, Any]:
    sources = preparation.get("source_reports")
    checkpoints = sources.get("training_checkpoints") if isinstance(sources, Mapping) else None
    if not isinstance(checkpoints, Mapping) or set(checkpoints) != set(RUN_NAMES):
        raise ValueError("sampling preparation checkpoints are missing")
    for run in RUN_NAMES:
        expected = checkpoints[run]
        checkpoint = reject_symlink_chain(
            expected["checkpoint"]["path"],
            name=f"{run} sampling checkpoint",
        ).resolve()
        actual = _checkpoint_identity(checkpoint)
        if (
            actual["path"] != expected["checkpoint"]["path"]
            or actual["sha256"] != expected["checkpoint"]["sha256"]
            or actual["step"] != expected["step"]
            or actual["integrity_manifest"]
            != expected["integrity_manifest"]["path"]
            or file_identity(Path(actual["integrity_manifest"]))
            != expected["integrity_manifest"]
        ):
            raise ValueError(f"{run} sampling checkpoint identity differs")
    return copy.deepcopy(dict(checkpoints))


def _expected_preflight(
    *,
    preparation: Mapping[str, Any],
    run: str,
    batch_size: int,
) -> dict[str, Any]:
    stage_contract = _stage_contract(preparation)
    protocol = stage_contract["sampling_protocol"]
    checkpoint = preparation["source_reports"]["training_checkpoints"][run]
    method = stage_contract["run_methods"][run]
    return {
        "checkpoint_identity": {
            "path": checkpoint["checkpoint"]["path"],
            "sha256": checkpoint["checkpoint"]["sha256"],
            "step": checkpoint["step"],
            "integrity_manifest": checkpoint["integrity_manifest"]["path"],
        },
        "expected_revision": preparation["git"]["revision"],
        "batch_size": batch_size,
        "prefix_budget": stage_contract["method_prefix_budgets"][method],
        "guidance_scale": protocol["guidance_scale"],
        "guidance_rescale": protocol["guidance_rescale"],
        "cfg_batch_mode": protocol["cfg_batch_mode"],
        "weights": protocol["weights"],
        "precision": protocol["precision"],
        "warmup_forwards": protocol["warmup_forwards"],
        "measured_forwards": protocol["measured_forwards"],
    }


def validate_completed_selection(
    selection: Mapping[str, Any],
    *,
    preparation: Mapping[str, Any],
    preparation_identity: Mapping[str, Any],
) -> dict[str, Any]:
    stage_contract = _stage_contract(preparation)
    protocol = stage_contract["sampling_protocol"]
    if (
        selection.get("schema_version") != SCHEMA_VERSION
        or selection.get("role") != REPORT_ROLE
        or selection.get("status") != "selected"
        or selection.get("stage") != stage_contract["stage"]
        or selection.get("output_root") != preparation.get("output_root")
        or selection.get("git") != preparation.get("git")
        or selection.get("sampling_protocol") != protocol
        or selection.get("checkpoints")
        != preparation.get("source_reports", {}).get("training_checkpoints")
        or selection.get("source_reports", {}).get("preparation")
        != identity(preparation_identity, label="sampling preparation")
    ):
        raise ValueError("completed four-arm sampling selection differs")
    candidates = selection.get("candidates")
    if (
        not isinstance(candidates, list)
        or [int(row.get("batch_size", -1)) for row in candidates]
        != protocol["candidate_batch_sizes"]
    ):
        raise ValueError("four-arm sampling selection candidates differ")
    replay_rows: list[dict[str, Any]] = []
    for row in candidates:
        batch_size = int(row["batch_size"])
        arms = row.get("arms")
        if not isinstance(arms, Mapping) or set(arms) != set(RUN_NAMES):
            raise ValueError("four-arm selection preflight set differs")
        replay_arms: dict[str, Any] = {}
        for run in RUN_NAMES:
            arm = arms[run]
            if not isinstance(arm, Mapping) or not isinstance(arm.get("report"), Mapping):
                raise ValueError(f"{run} selection preflight row is malformed")
            report = dict(arm["report"])
            descriptor = arm.get("identity")
            if isinstance(descriptor, Mapping):
                expected_identity = identity(
                    descriptor,
                    label=f"{run} batch {batch_size} preflight",
                )
                source = reject_symlink_chain(
                    expected_identity["path"],
                    name=f"{run} batch {batch_size} preflight",
                ).resolve()
                if file_identity(source) != expected_identity:
                    raise ValueError(f"{run} preflight identity changed")
                physical = read_json_object(source, name=f"{run} preflight")
                if physical != report:
                    raise ValueError(f"{run} preflight replay differs")
            elif report.get("status") == "passed":
                raise ValueError(f"{run} passed preflight identity is missing")
            if report.get("status") == "passed" and not _preflight_matches(
                report,
                **_expected_preflight(
                    preparation=preparation,
                    run=run,
                    batch_size=batch_size,
                ),
            ):
                raise ValueError(f"{run} preflight contract differs")
            replay_arms[run] = {"report": report, "identity": descriptor}
        replay_rows.append({"batch_size": batch_size, "arms": replay_arms})
    recomputed = select_four_arm_sampling_batch(
        replay_rows,
        baseline_batch_size=protocol["baseline_batch_size"],
        max_memory_fraction=protocol["maximum_memory_fraction"],
    )
    for field in ("policy", "selected", "runtime_environment_sha256", "candidates"):
        if selection.get(field) != recomputed[field]:
            raise ValueError(f"four-arm sampling selection {field} is not reproducible")
    return copy.deepcopy(dict(selection))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select one shared batch across all four conditioning-ranking arms."
    )
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--expected-preparation-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-script", default="scripts/preflight_generation_sampling.py")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preparation, preparation_identity = _load_preparation(
        args.preparation,
        args.expected_preparation_sha256,
    )
    stage_contract = _stage_contract(preparation)
    protocol = stage_contract["sampling_protocol"]
    output_root = reject_symlink_chain(
        args.output_root,
        name="conditioning-ranking sampling output root",
    ).resolve()
    output = reject_symlink_chain(
        args.output,
        name="conditioning-ranking sampling batch selection",
    ).resolve()
    if output_root.as_posix() != preparation["output_root"]:
        raise ValueError("conditioning-ranking sampling output root differs")
    if output != output_root / "reports" / "sampling_batch_selection.json":
        raise ValueError("conditioning-ranking sampling selection output differs")
    current_git = git_provenance(PROJECT_ROOT)
    if clean_git(current_git, label="sampling batch selector") != preparation["git"]:
        raise ValueError("sampling batch selector Git differs from preparation")
    checkpoints = _physical_checkpoint_contract(preparation)
    if output.is_file():
        if not args.resume:
            raise FileExistsError("sampling batch selection exists; pass --resume")
        validate_completed_selection(
            read_json_object(output, name="four-arm sampling batch selection"),
            preparation=preparation,
            preparation_identity=preparation_identity,
        )
        print(read_json_object(output, name="four-arm sampling batch selection")["selected"]["batch_size"])
        return
    candidates = parse_candidates(
        ",".join(str(value) for value in protocol["candidate_batch_sizes"]),
        baseline_batch_size=protocol["baseline_batch_size"],
    )
    preflight_root = output_root / "reports" / "sampling_preflight"
    preflight_script = (PROJECT_ROOT / args.preflight_script).resolve()
    sampling_output_dirs = [
        output_root / run / stage_contract["sample_run_name"] for run in RUN_NAMES
    ]
    rows: list[dict[str, Any]] = []
    with exclusive_output_locks(
        sampling_output_dirs,
        role="generation_conditioning_ranking_four_arm_sampling_batch_selector",
    ):
        for batch_size in candidates:
            arms: dict[str, Any] = {}
            for run in RUN_NAMES:
                checkpoint = Path(checkpoints[run]["checkpoint"]["path"])
                report = _run_preflight(
                    method=run,
                    checkpoint=checkpoint,
                    checkpoint_identity={
                        "path": checkpoints[run]["checkpoint"]["path"],
                        "sha256": checkpoints[run]["checkpoint"]["sha256"],
                        "step": checkpoints[run]["step"],
                        "integrity_manifest": checkpoints[run]["integrity_manifest"]["path"],
                    },
                    expected_revision=current_git["revision"],
                    batch_size=batch_size,
                    prefix_budget=stage_contract["method_prefix_budgets"][
                        stage_contract["run_methods"][run]
                    ],
                    output_root=preflight_root,
                    preflight_script=preflight_script,
                    project_root=PROJECT_ROOT,
                    guidance_scale=protocol["guidance_scale"],
                    guidance_rescale=protocol["guidance_rescale"],
                    cfg_batch_mode=protocol["cfg_batch_mode"],
                    weights=protocol["weights"],
                    precision=protocol["precision"],
                    warmup_forwards=protocol["warmup_forwards"],
                    measured_forwards=protocol["measured_forwards"],
                    timeout_seconds=args.timeout_seconds,
                )
                report_path = preflight_root / run / f"batch_{batch_size}" / "sampling_preflight.json"
                arms[run] = {
                    "report": report,
                    "identity": file_identity(report_path) if report_path.is_file() else None,
                }
            rows.append({"batch_size": batch_size, "arms": arms})
    selected = select_four_arm_sampling_batch(
        rows,
        baseline_batch_size=protocol["baseline_batch_size"],
        max_memory_fraction=protocol["maximum_memory_fraction"],
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "role": REPORT_ROLE,
        "status": "selected",
        "stage": stage_contract["stage"],
        "output_root": output_root.as_posix(),
        "git": preparation["git"],
        "source_reports": {"preparation": preparation_identity},
        "sampling_protocol": copy.deepcopy(protocol),
        "checkpoints": checkpoints,
        **selected,
    }
    write_json_report(output, report)
    print(report["selected"]["batch_size"])


if __name__ == "__main__":
    main()
