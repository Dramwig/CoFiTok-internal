from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cofitok.environment import capture_runtime_environment, runtime_environment_sha256
from cofitok.generation import export_ema_inference_artifact
from cofitok.reporting import git_provenance, write_json_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a verified EMA-only CoFiTok generation artifact."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument(
        "--release-gate",
        default="",
        help="Passing full generation gate required for a formal deployment artifact.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume only a manifest-bound interrupted export; a completed exact "
            "artifact is replayed without rewriting bytes."
        ),
    )
    args = parser.parse_args()
    runtime_environment = capture_runtime_environment(
        torch.device("cpu"),
        project_root=PROJECT_ROOT,
    )
    execution = {
        "git": git_provenance(PROJECT_ROOT),
        "runtime_environment": runtime_environment,
        "runtime_environment_sha256": runtime_environment_sha256(
            runtime_environment
        ),
    }
    report = export_ema_inference_artifact(
        args.checkpoint,
        args.output,
        release_gate=args.release_gate or None,
        resume=args.resume,
        execution=execution,
    )
    write_json_report(Path(args.report), report)
    print(args.report)


if __name__ == "__main__":
    main()
