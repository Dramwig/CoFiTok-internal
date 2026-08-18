from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Any

from cofitok.configs import config_to_dict, load_config
from cofitok.generation.conditioning_ranking_probe import (
    conditioning_ranking_probe_contract,
)
from cofitok.models import CoFiTokTiny
from cofitok.reporting import file_sha256, git_provenance, write_json_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the non-authorizing four-arm class-ranking probe."
    )
    parser.add_argument("--control-cofitok", type=Path, required=True)
    parser.add_argument("--control-dense", type=Path, required=True)
    parser.add_argument("--ranked-cofitok", type=Path, required=True)
    parser.add_argument("--ranked-dense", type=Path, required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _source(path: Path) -> dict[str, Any]:
    return {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _parameter_count(config_path: Path) -> int:
    model = CoFiTokTiny(load_config(config_path).model)
    count = sum(parameter.numel() for parameter in model.parameters())
    del model
    gc.collect()
    return count


def main() -> None:
    args = _parse_args()
    paths = {
        "control_cofitok": args.control_cofitok,
        "control_dense": args.control_dense,
        "ranked_cofitok": args.ranked_cofitok,
        "ranked_dense": args.ranked_dense,
    }
    configs = {name: config_to_dict(load_config(path)) for name, path in paths.items()}
    contract = conditioning_ranking_probe_contract(
        control_cofitok=configs["control_cofitok"],
        control_dense=configs["control_dense"],
        ranked_cofitok=configs["ranked_cofitok"],
        ranked_dense=configs["ranked_dense"],
    )
    git = git_provenance(PROJECT_ROOT)
    if git["tracked_dirty"]:
        raise ValueError("probe preparation requires a clean tracked worktree")
    if contract["valid"] is not True:
        raise ValueError("probe config contract failed: " + "; ".join(contract["issues"]))
    parameter_counts = {name: _parameter_count(path) for name, path in paths.items()}
    if parameter_counts["control_cofitok"] != parameter_counts["ranked_cofitok"]:
        raise ValueError("CoFiTok control/ranked parameter counts differ")
    if parameter_counts["control_dense"] != parameter_counts["ranked_dense"]:
        raise ValueError("dense control/ranked parameter counts differ")
    report = {
        **contract,
        "git": git,
        "output_root": args.output_root,
        "configs": {name: _source(path) for name, path in paths.items()},
        "parameter_counts": parameter_counts,
        "claim_boundary": {
            "training_quality_claim_allowed": False,
            "sample_quality_claim_allowed": False,
            "promotion_authorization_allowed": False,
            "full_training_launch_allowed": False,
            "required_next_evidence": (
                "an exact user approval sentinel followed by the four fresh 1K runs"
            ),
        },
    }
    write_json_report(args.output, report)
    print(args.output)


if __name__ == "__main__":
    main()
