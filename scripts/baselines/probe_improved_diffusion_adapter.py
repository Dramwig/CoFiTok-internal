from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe the CoFiTok improved-diffusion adapter.")
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path("/root/autodl-tmp/CoFiTok/baselines/repos/improved_diffusion"),
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=Path("baselines/adapters/improved_diffusion"),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.adapter.resolve()))
    sys.path.insert(1, str(args.repo.resolve()))

    result: dict[str, Any] = {
        "repo": str(args.repo),
        "adapter": str(args.adapter),
        "imports": {},
    }
    for module_name in ("blobfile", "mpi4py", "improved_diffusion.script_util"):
        try:
            module = __import__(module_name, fromlist=["*"])
            result["imports"][module_name] = {
                "status": "ok",
                "file": getattr(module, "__file__", ""),
            }
        except Exception as error:  # noqa: BLE001 - probe records exact failure
            result["imports"][module_name] = {"status": "failed", "error": repr(error)}

    try:
        from improved_diffusion.script_util import model_and_diffusion_defaults

        defaults = model_and_diffusion_defaults()
        result["model_defaults"] = {
            "image_size": defaults["image_size"],
            "num_channels": defaults["num_channels"],
            "diffusion_steps": defaults["diffusion_steps"],
        }
    except Exception as error:  # noqa: BLE001
        result["model_defaults_error"] = repr(error)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
