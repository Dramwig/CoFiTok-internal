from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any


FORMAL64_DATASETS = ["downsampled_imagenet_64", "ffhq_64", "afhqv2_64"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe D-AR formal64 baseline feasibility.")
    parser.add_argument("--repo", type=Path, default=Path("/root/autodl-tmp/CoFiTok/baselines/repos/d_ar"))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _repo_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", repo.as_posix(), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return ""


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _extract_image_size_choices(source: str) -> list[int]:
    pattern = re.compile(
        r"add_argument\(\s*[\"']--image-size[\"'].*?choices\s*=\s*\[([^\]]+)\]",
        re.DOTALL,
    )
    match = pattern.search(source)
    if not match:
        return []
    return [int(item) for item in re.findall(r"\d+", match.group(1))]


def _contains(path: Path, needle: str) -> bool:
    return needle in _read(path)


def build_report(repo: Path) -> dict[str, Any]:
    repo = repo.resolve()
    required_files = {
        "readme": "README.md",
        "getting_started": "GETTING_STARTED.md",
        "requirements": "requirements.txt",
        "tokenizer_config": "configs/tokenizer_v1.yaml",
        "tokenizer_train": "tokenizer/tokenizer_image/vq_train_accelerate.py",
        "ar_train": "autoregressive/train/train_c2i_accelerate.py",
        "ar_sample": "autoregressive/sample/sample_c2i.py",
        "imagenet_dataset": "dataset/imagenet.py",
    }
    missing = [rel for rel in required_files.values() if not (repo / rel).is_file()]
    if missing:
        return {
            "baseline": "d_ar",
            "repo": repo.as_posix(),
            "repo_commit": _repo_commit(repo),
            "status": "missing_files",
            "missing_files": missing,
        }

    tokenizer_train = _read(repo / required_files["tokenizer_train"])
    ar_train = _read(repo / required_files["ar_train"])
    ar_sample = _read(repo / required_files["ar_sample"])
    tokenizer_config = _read(repo / required_files["tokenizer_config"])
    readme = _read(repo / required_files["readme"])

    image_size_choices = {
        "tokenizer_train": _extract_image_size_choices(tokenizer_train),
        "ar_train": _extract_image_size_choices(ar_train),
        "ar_sample": _extract_image_size_choices(ar_sample),
    }

    blockers = []
    if 64 not in image_size_choices["ar_train"]:
        blockers.append("AR training image-size choices exclude 64.")
    if 64 not in image_size_choices["ar_sample"]:
        blockers.append("AR sampling image-size choices exclude 64.")
    if 64 not in image_size_choices["tokenizer_train"]:
        blockers.append("Tokenizer training image-size choices exclude 64.")
    if "256x256" in readme or "256x256" in tokenizer_config:
        blockers.append("Official pretrained tokenizer and D-AR checkpoints target ImageNet-256.")
    if _contains(repo / required_files["ar_train"], "from dataset.imagenet import build_imagenet"):
        blockers.append("AR training path is class-conditional ImageFolder/ImageNet-style.")
    if "DINO" in _read(repo / required_files["getting_started"]) or "dino_weight" in tokenizer_train:
        blockers.append("Tokenizer training uses extra perceptual/DINO losses before AR training.")

    return {
        "baseline": "d_ar",
        "repo": repo.as_posix(),
        "repo_commit": _repo_commit(repo),
        "status": "feasibility_only",
        "formal64_status": "not_protocol_compatible_without_code_and_protocol_changes",
        "formal64_datasets": FORMAL64_DATASETS,
        "required_files": {key: (repo / rel).as_posix() for key, rel in required_files.items()},
        "image_size_choices": image_size_choices,
        "blockers": blockers,
        "compatible_followups": [
            "ImageNet-256 eval-only comparison with official tokenizer and D-AR checkpoints.",
            "ImageNet-256 training comparison with explicit tokenizer-training budget.",
            "A separate 128/256 protocol if the paper expands beyond MVP formal64.",
        ],
        "decision": (
            "Do not enter D-AR as a completed formal64 baseline. Record it as the nearest "
            "diffusion-as-AR threat until a separate compatible protocol is defined."
        ),
    }


def main() -> None:
    args = parse_args()
    if not args.repo.is_dir():
        raise FileNotFoundError(f"D-AR repo does not exist: {args.repo}")
    report = build_report(args.repo)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
