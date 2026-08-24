from __future__ import annotations

import argparse
import gc
import subprocess
from pathlib import Path

from cofitok.configs import config_to_dict, load_config
from cofitok.generation.conditioning_ranking_probe import (
    conditioning_ranking_probe_contract,
)
from cofitok.generation.conditioning_ranking_terminal_rebind import (
    OUTPUT_ROOT,
    PREPARATION_CLAIM_BOUNDARY,
)
from cofitok.inference_replay import (
    file_identity,
    read_json_object,
    reject_symlink_chain,
)
from cofitok.models import CoFiTokTiny
from cofitok.reporting import git_provenance, write_json_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the immutable four-arm 1K semantic-alignment probe for the "
            "post-reconciliation terminal rebind."
        )
    )
    parser.add_argument("--control-cofitok", type=Path, required=True)
    parser.add_argument("--control-dense", type=Path, required=True)
    parser.add_argument("--ranked-cofitok", type=Path, required=True)
    parser.add_argument("--ranked-dense", type=Path, required=True)
    parser.add_argument("--legacy-preparation", type=Path, required=True)
    parser.add_argument("--expected-legacy-preparation-sha256", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _parameter_count(path: Path) -> int:
    model = CoFiTokTiny(load_config(path).model)
    count = sum(parameter.numel() for parameter in model.parameters())
    del model
    gc.collect()
    return count


def _tree() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=PROJECT_ROOT,
        text=True,
    ).strip()


def main() -> None:
    args = parse_args()
    if args.output_root != OUTPUT_ROOT:
        raise ValueError("terminal-rebind preparation output root differs")
    legacy_path = reject_symlink_chain(
        args.legacy_preparation,
        name="legacy conditioning-ranking preparation",
    ).resolve()
    legacy_identity = file_identity(legacy_path)
    if legacy_identity["sha256"] != args.expected_legacy_preparation_sha256:
        raise ValueError("legacy conditioning-ranking preparation SHA256 differs")
    legacy = read_json_object(
        legacy_path,
        name="legacy conditioning-ranking preparation",
    )

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
    full_status = subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        text=True,
    ).strip()
    if git["tracked_dirty"] or full_status:
        raise ValueError("terminal-rebind preparation requires a fully clean checkout")
    if contract["valid"] is not True:
        raise ValueError("probe config contract failed: " + "; ".join(contract["issues"]))
    parameter_counts = {name: _parameter_count(path) for name, path in paths.items()}
    if parameter_counts["control_cofitok"] != parameter_counts["ranked_cofitok"]:
        raise ValueError("CoFiTok control/ranked parameter counts differ")
    if parameter_counts["control_dense"] != parameter_counts["ranked_dense"]:
        raise ValueError("dense control/ranked parameter counts differ")
    config_identities = {name: file_identity(path) for name, path in paths.items()}
    legacy_configs = legacy.get("configs")
    if not isinstance(legacy_configs, dict) or set(legacy_configs) != set(config_identities):
        raise ValueError("legacy preparation config set differs")
    for name, identity in config_identities.items():
        previous = legacy_configs[name]
        if (
            identity["bytes"] != previous.get("bytes")
            or identity["sha256"] != previous.get("sha256")
        ):
            raise ValueError(f"candidate config differs from legacy preparation: {name}")
    if parameter_counts != legacy.get("parameter_counts"):
        raise ValueError("candidate parameter counts differ from legacy preparation")

    report = {
        **contract,
        "git": git,
        "tree": _tree(),
        "output_root": args.output_root,
        "configs": config_identities,
        "parameter_counts": parameter_counts,
        "legacy_candidate_source": legacy_identity,
        "claim_boundary": PREPARATION_CLAIM_BOUNDARY,
    }
    write_json_report(args.output, report)
    print(args.output)


if __name__ == "__main__":
    main()
