from __future__ import annotations

import argparse
import gc
import subprocess
from pathlib import Path
from typing import Any

from cofitok.configs import config_to_dict, load_config
from cofitok.generation.semantic_residual_alignment_probe import (
    OUTPUT_ROOT,
    semantic_residual_alignment_probe_contract,
)
from cofitok.inference_replay import file_identity, prepare_manifest
from cofitok.models import CoFiTokTiny
from cofitok.reporting import git_provenance


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the exact four-arm 1K semantic residual-alignment probe."
        )
    )
    parser.add_argument("--control-cofitok", type=Path, required=True)
    parser.add_argument("--control-dense", type=Path, required=True)
    parser.add_argument("--residual-cofitok", type=Path, required=True)
    parser.add_argument("--residual-dense", type=Path, required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _parameter_count(path: Path) -> int:
    model = CoFiTokTiny(load_config(path).model)
    count = sum(parameter.numel() for parameter in model.parameters())
    del model
    gc.collect()
    return count


def _full_git() -> dict[str, Any]:
    provenance = git_provenance(PROJECT_ROOT)
    status = subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        text=True,
    ).strip()
    if provenance["tracked_dirty"] or status:
        raise ValueError("residual-alignment preparation requires a fully clean checkout")
    return {
        **provenance,
        "tree": subprocess.check_output(
            ["git", "rev-parse", "HEAD^{tree}"],
            cwd=PROJECT_ROOT,
            text=True,
        ).strip(),
    }


def main() -> None:
    args = parse_args()
    if args.output_root != OUTPUT_ROOT:
        raise ValueError("residual-alignment preparation output root differs")
    paths = {
        "control_cofitok": args.control_cofitok,
        "control_dense": args.control_dense,
        "residual_cofitok": args.residual_cofitok,
        "residual_dense": args.residual_dense,
    }
    configs = {name: config_to_dict(load_config(path)) for name, path in paths.items()}
    contract = semantic_residual_alignment_probe_contract(
        control_cofitok=configs["control_cofitok"],
        control_dense=configs["control_dense"],
        residual_cofitok=configs["residual_cofitok"],
        residual_dense=configs["residual_dense"],
    )
    if contract["valid"] is not True:
        raise ValueError(
            "residual-alignment config contract failed: "
            + "; ".join(contract["issues"])
        )
    parameter_counts = {name: _parameter_count(path) for name, path in paths.items()}
    if parameter_counts["control_cofitok"] != parameter_counts["residual_cofitok"]:
        raise ValueError("CoFiTok control/residual parameter counts differ")
    if parameter_counts["control_dense"] != parameter_counts["residual_dense"]:
        raise ValueError("dense control/residual parameter counts differ")
    report = {
        **contract,
        "git": _full_git(),
        "output_root": args.output_root,
        "configs": {name: file_identity(path) for name, path in paths.items()},
        "parameter_counts": parameter_counts,
    }
    identity = prepare_manifest(
        args.output,
        report,
        resume=args.resume,
        overwrite=False,
    )
    print(identity["sha256"])


if __name__ == "__main__":
    main()
