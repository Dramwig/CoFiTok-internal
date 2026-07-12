from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe the isolated EDM baseline repository.")
    parser.add_argument("--repo", type=Path, default=Path("/root/autodl-tmp/CoFiTok/baselines/repos/edm"))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _repo_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", repo.as_posix(), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return ""


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    if not repo.is_dir():
        raise FileNotFoundError(f"EDM repo does not exist: {repo}")
    sys.path.insert(0, repo.as_posix())

    imports: dict[str, Any] = {}
    for module_name in ["dnnlib", "torch_utils.distributed", "training.networks", "training.dataset"]:
        module = importlib.import_module(module_name)
        imports[module_name] = {
            "status": "ok",
            "file": str(getattr(module, "__file__", "")),
        }

    payload = {
        "baseline": "edm",
        "repo": repo.as_posix(),
        "repo_commit": _repo_commit(repo),
        "imports": imports,
        "scripts": {
            "train": (repo / "train.py").as_posix(),
            "generate": (repo / "generate.py").as_posix(),
            "dataset_tool": (repo / "dataset_tool.py").as_posix(),
        },
        "status": "ok",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
