"""Shared argument and path helpers for the exposure continuation result CLIs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from cofitok.generation.exposure_capacity_authorization import (
    checkout_identity,
    identity,
    read_object,
)
from cofitok.inference_replay import reject_symlink_chain


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument(
        "--validation-receipt",
        type=Path,
        help=(
            "Optional immutable receipt written by the validator after the result "
            "has been rebuilt from all bound source evidence."
        ),
    )
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--expected-authorization-sha256", required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--expected-gate-sha256", required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--expected-preparation-sha256", required=True)
    parser.add_argument("--standing-authorization", type=Path, required=True)
    parser.add_argument("--expected-standing-authorization-sha256", required=True)
    parser.add_argument("--stage-authorization", type=Path, required=True)
    parser.add_argument("--expected-stage-authorization-sha256", required=True)
    parser.add_argument("--source-project-root", type=Path, required=True)
    parser.add_argument("--execution-project-root", type=Path, required=True)
    parser.add_argument(
        "--validator-project-root",
        type=Path,
        help=(
            "Clean checkout running the result validator. This may be a newer "
            "revision than the immutable execution checkout recorded by the result."
        ),
    )
    parser.add_argument("--cofitok-config", type=Path, required=True)
    parser.add_argument("--dense-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--classifier-checkpoint", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build or validate a bounded 100K to 110K continuation result."
    )
    add_arguments(parser)
    return parser.parse_args()


def layout(output_root: Path, method: str) -> dict[str, Path]:
    root = reject_symlink_chain(output_root, name="continuation result output root").resolve()
    run_name = (
        "cofitok_rgbtail3_rollout_x0_u2_ema_teacher"
        if method == "cofitok"
        else "dense_rollout_x0_u2_ema_teacher"
    )
    prefix = 8 if method == "cofitok" else 1
    evaluation = root / "evaluations" / method
    sampling = evaluation / "samples_10000_ddim100_cfg15"
    return {
        "run": root / run_name,
        "training_report": root / run_name / "training_report.json",
        "sampling": sampling,
        "sampling_report": sampling / "sampling_report.json",
        "generated": sampling / f"prefix_{prefix}",
        "metrics": evaluation / "metrics",
        "metrics_report": evaluation / "metrics" / "generation_metrics_report.json",
        "class_fidelity": evaluation / "class_fidelity",
        "class_report": evaluation / "class_fidelity" / "class_fidelity_report.json",
        "mechanism": evaluation / "mechanism",
        "checkpoint_report": evaluation / "mechanism" / "checkpoint_evaluation_report.json",
        "rollout": evaluation / "rollout",
        "rollout_report": evaluation / "rollout" / "rollout_stability_report.json",
    }


def result_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    root = reject_symlink_chain(args.output_root, name="continuation result output root").resolve()
    stage_identity = identity(args.stage_authorization)
    authorization = read_object(args.authorization, name="execution authorization")
    if authorization.get("stage_authorization_identity") != stage_identity:
        raise ValueError("execution authorization binds another stage authorization")
    cofitok = layout(root, "cofitok")
    dense = layout(root, "dense_identity")
    return {
        "authorization": authorization,
        "gate": read_object(args.gate, name="execution gate"),
        "preparation": read_object(args.preparation, name="preparation"),
        "preparation_identity": identity(args.preparation),
        "gate_identity": identity(args.gate),
        "standing_identity": identity(args.standing_authorization),
        "execution_checkout": checkout_identity(args.execution_project_root),
        "source_checkout": checkout_identity(args.source_project_root),
        "config_identities": {
            "cofitok": identity(args.cofitok_config),
            "dense_identity": identity(args.dense_config),
        },
        "authorization_identity": identity(args.authorization),
        "candidate_gate_identity": identity(args.gate),
        "cofitok_run_dir": cofitok["run"],
        "dense_run_dir": dense["run"],
        "cofitok_training_report": cofitok["training_report"],
        "dense_training_report": dense["training_report"],
        "cofitok_sampling_report": cofitok["sampling_report"],
        "dense_sampling_report": dense["sampling_report"],
        "cofitok_metrics_report": cofitok["metrics_report"],
        "dense_metrics_report": dense["metrics_report"],
        "cofitok_class_report": cofitok["class_report"],
        "dense_class_report": dense["class_report"],
        "cofitok_checkpoint_report": cofitok["checkpoint_report"],
        "dense_checkpoint_report": dense["checkpoint_report"],
        "cofitok_rollout_report": cofitok["rollout_report"],
        "dense_rollout_report": dense["rollout_report"],
    }
