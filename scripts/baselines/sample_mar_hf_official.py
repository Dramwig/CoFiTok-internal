#!/usr/bin/env python
"""Sample community-converted MAR Hugging Face safetensors auditably."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.util
import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from PIL import Image


DEFAULT_HF_REPO = Path(
    "/root/autodl-tmp/CoFiTok/checkpoints/baselines/mar/"
    "official_imagenet256_eval_only/hf_repo"
)
DEFAULT_OUTPUT = Path(
    "/root/autodl-tmp/CoFiTok/checkpoints/baselines/mar/"
    "official_imagenet256_eval_only/samples_50k_hf"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate ImageNet-256 samples from the verified MAR-B Hugging Face "
            "safetensors without modifying the external MAR repository."
        )
    )
    parser.add_argument("--hf-repo-dir", type=Path, default=DEFAULT_HF_REPO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-filename", default="mar-base.safetensors")
    parser.add_argument("--vae-filename", default="kl16.safetensors")
    parser.add_argument("--num-images", type=int, default=50_000)
    parser.add_argument("--class-num", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--decode-batch-size",
        type=int,
        default=16,
        help="Decode sampled VAE latents in chunks to bound peak CUDA memory.",
    )
    parser.add_argument("--num-iter", type=int, default=256)
    parser.add_argument("--num-sampling-steps", type=int, default=100)
    parser.add_argument("--cfg", type=float, default=2.9)
    parser.add_argument("--cfg-schedule", choices=["linear", "constant"], default="linear")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Stop after this many generated batches; 0 runs until complete.",
    )
    parser.add_argument(
        "--pack-npz",
        type=Path,
        default=None,
        help="Pack all completed PNGs into an ADM-evaluator arr_0 NPZ.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Progress report path; defaults to OUTPUT_DIR/baseline_eval_report.json.",
    )
    parser.add_argument("--progress", action="store_true", help="Show MAR token sampling progress bars.")
    return parser.parse_args()


def balanced_class_labels(num_images: int, class_num: int) -> np.ndarray:
    if num_images <= 0:
        raise ValueError("num_images must be positive")
    if class_num <= 0:
        raise ValueError("class_num must be positive")
    if num_images % class_num != 0:
        raise ValueError(
            f"num_images ({num_images}) must be divisible by class_num ({class_num}) "
            "for the official balanced ImageNet protocol"
        )
    return np.arange(class_num, dtype=np.int64).repeat(num_images // class_num)


def batch_ranges(num_images: int, batch_size: int) -> Iterator[tuple[int, int]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, num_images, batch_size):
        yield start, min(start + batch_size, num_images)


def sample_path(output_dir: Path, index: int) -> Path:
    return output_dir / f"{index:05d}.png"


def batch_is_complete(output_dir: Path, start: int, end: int) -> bool:
    return all(sample_path(output_dir, index).is_file() for index in range(start, end))


def count_completed(output_dir: Path, num_images: int) -> int:
    return sum(sample_path(output_dir, index).is_file() for index in range(num_images))


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_hf_package(repo_dir: Path, package_name: str = "_cofitok_mar_hf") -> tuple[Any, Any]:
    init_path = repo_dir / "__init__.py"
    if not init_path.is_file():
        raise FileNotFoundError(f"HF package init file is missing: {init_path}")

    for module_name in list(sys.modules):
        if module_name == package_name or module_name.startswith(package_name + "."):
            del sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(
        package_name,
        init_path,
        submodule_search_locations=[str(repo_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load MAR HF package from {repo_dir}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)
    mar_module = importlib.import_module(f"{package_name}.mar")
    vae_module = importlib.import_module(f"{package_name}.vae")
    return mar_module, vae_module


def build_models(args: argparse.Namespace, device: Any) -> tuple[Any, Any, dict[str, Any]]:
    import torch
    from safetensors.torch import load_file

    model_path = args.hf_repo_dir / args.model_filename
    vae_path = args.hf_repo_dir / args.vae_filename
    for path in (model_path, vae_path):
        if not path.is_file():
            raise FileNotFoundError(f"required MAR asset is missing: {path}")

    mar_module, vae_module = load_hf_package(args.hf_repo_dir)
    model = mar_module.mar_base(
        img_size=256,
        vae_stride=16,
        patch_size=1,
        vae_embed_dim=16,
        buffer_size=64,
        diffloss_d=6,
        diffloss_w=1024,
        num_sampling_steps=str(args.num_sampling_steps),
    )
    model.load_state_dict(load_file(str(model_path), device="cpu"), strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    vae = vae_module.AutoencoderKL(
        embed_dim=16,
        ch_mult=(1, 1, 2, 2, 4),
        ckpt_path=None,
    )
    vae.load_state_dict(load_file(str(vae_path), device="cpu"), strict=True)
    vae.to(device).eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)

    assets = {
        "model": {
            "path": str(model_path),
            "bytes": model_path.stat().st_size,
            "sha256": sha256_file(model_path),
        },
        "vae": {
            "path": str(vae_path),
            "bytes": vae_path.stat().st_size,
            "sha256": sha256_file(vae_path),
        },
    }
    return model, vae, assets


def autocast_context(torch_module: Any, precision: str) -> Any:
    if precision == "fp32":
        return nullcontext()
    dtype = torch_module.float16 if precision == "fp16" else torch_module.bfloat16
    return torch_module.autocast(device_type="cuda", dtype=dtype)


def save_batch(images: Any, output_dir: Path, start: int) -> None:
    image_array = (
        ((images.detach().float().cpu().clamp(-1, 1) + 1.0) * 127.5)
        .round()
        .byte()
        .permute(0, 2, 3, 1)
        .numpy()
    )
    for offset, image in enumerate(image_array):
        destination = sample_path(output_dir, start + offset)
        temporary = destination.with_name(destination.name + ".tmp")
        Image.fromarray(image, mode="RGB").save(temporary, format="PNG")
        os.replace(temporary, destination)


def decode_and_save_batch(
    torch_module: Any,
    vae: Any,
    sampled_tokens: Any,
    output_dir: Path,
    start: int,
    decode_batch_size: int,
    precision: str,
) -> None:
    if decode_batch_size <= 0:
        raise ValueError("decode_batch_size must be positive")
    for offset in range(0, sampled_tokens.shape[0], decode_batch_size):
        end = min(offset + decode_batch_size, sampled_tokens.shape[0])
        with torch_module.inference_mode():
            with autocast_context(torch_module, precision):
                decoded = vae.decode(sampled_tokens[offset:end] / 0.2325)
        save_batch(decoded, output_dir, start + offset)
        del decoded
        torch_module.cuda.empty_cache()


def validate_and_pack_pngs(
    output_dir: Path,
    output_npz: Path,
    num_images: int,
) -> dict[str, Any]:
    if count_completed(output_dir, num_images) != num_images:
        raise RuntimeError("cannot pack NPZ before all expected PNG samples exist")

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    temporary_npy = output_npz.with_name(output_npz.name + ".packing.npy")
    temporary_npz = output_npz.with_name(output_npz.name + ".tmp")
    array = np.lib.format.open_memmap(
        temporary_npy,
        mode="w+",
        dtype=np.uint8,
        shape=(num_images, 256, 256, 3),
    )
    try:
        for index in range(num_images):
            with Image.open(sample_path(output_dir, index)) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            if rgb.shape != (256, 256, 3):
                raise ValueError(
                    f"sample {index} has shape {rgb.shape}; expected (256, 256, 3)"
                )
            array[index] = rgb
        array.flush()
        with temporary_npz.open("wb") as handle:
            np.savez(handle, arr_0=array)
        os.replace(temporary_npz, output_npz)
    finally:
        del array
        temporary_npy.unlink(missing_ok=True)
        temporary_npz.unlink(missing_ok=True)

    return {
        "path": str(output_npz),
        "bytes": output_npz.stat().st_size,
        "sha256": sha256_file(output_npz),
        "shape": [num_images, 256, 256, 3],
        "dtype": "uint8",
        "key": "arr_0",
    }


def progress_payload(
    args: argparse.Namespace,
    assets: dict[str, Any],
    completed: int,
    elapsed_seconds: float,
    status: str,
    batch_seconds: list[float],
    measured_images: int,
    packed_npz: dict[str, Any] | None = None,
) -> dict[str, Any]:
    generation_seconds = sum(batch_seconds)
    images_per_second = measured_images / generation_seconds if generation_seconds else None
    remaining = max(0, args.num_images - completed)
    eta_seconds = remaining / images_per_second if images_per_second else None
    protocol = getattr(args, "protocol", "community_hf_safetensors_eval_only")
    source_root = getattr(args, "source_root", args.hf_repo_dir)
    metric_note = getattr(
        args,
        "metric_note",
        (
            "MAR-B ImageNet-256 sampling parameters through community-converted "
            "Hugging Face safetensors. This is eval-only and not same-budget P0 "
            "retraining."
        ),
    )
    return {
        "schema_version": 1,
        "report_type": "baseline_eval",
        "baseline": "mar",
        "dataset": "imagenet_256",
        "protocol": protocol,
        "status": status,
        "paper_table_role": "secondary_related_method_only",
        "parameters": {
            "model_type": "mar_base",
            "num_images": args.num_images,
            "class_num": args.class_num,
            "balanced_images_per_class": args.num_images // args.class_num,
            "batch_size": args.batch_size,
            "decode_batch_size": args.decode_batch_size,
            "num_ar_steps": args.num_iter,
            "num_sampling_steps": args.num_sampling_steps,
            "cfg_scale": args.cfg,
            "cfg_schedule": args.cfg_schedule,
            "temperature": args.temperature,
            "seed": args.seed,
            "precision": args.precision,
            "resume_seed_rule": "batch_seed=global_seed+batch_start_index",
        },
        "artifacts": {
            "hf_repo_dir": str(args.hf_repo_dir),
            "source_root": str(source_root),
            "samples_dir": str(args.output_dir),
            "weights": assets,
            "sample_npz": packed_npz,
        },
        "progress": {
            "completed_images": completed,
            "expected_images": args.num_images,
            "elapsed_seconds": elapsed_seconds,
            "measured_generation_seconds": generation_seconds,
            "images_per_second": images_per_second,
            "eta_seconds": eta_seconds,
        },
        "metric_notes": {
            "protocol": metric_note
        },
    }


def run(
    args: argparse.Namespace,
    model_builder: Any = build_models,
) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("MAR official sampling requires CUDA")
    if args.max_batches < 0:
        raise ValueError("max_batches must be non-negative")
    if args.decode_batch_size <= 0:
        raise ValueError("decode_batch_size must be positive")
    labels = balanced_class_labels(args.num_images, args.class_num)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.report or (args.output_dir / "baseline_eval_report.json")

    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    started = time.perf_counter()
    model, vae, assets = model_builder(args, device)
    batch_seconds: list[float] = []
    generated_batches = 0
    generated_images = 0

    completed_indices = {
        index
        for index in range(args.num_images)
        if sample_path(args.output_dir, index).is_file()
    }
    initial_completed = len(completed_indices)
    write_json_atomic(
        report_path,
        progress_payload(
            args,
            assets,
            initial_completed,
            0.0,
            "running",
            batch_seconds,
            generated_images,
        ),
    )

    for start, end in batch_ranges(args.num_images, args.batch_size):
        if all(index in completed_indices for index in range(start, end)):
            continue
        if args.max_batches and generated_batches >= args.max_batches:
            break

        batch_seed = args.seed + start
        torch.manual_seed(batch_seed)
        torch.cuda.manual_seed_all(batch_seed)
        np.random.seed(batch_seed % (2**32 - 1))
        batch_labels = torch.from_numpy(labels[start:end]).to(device=device, dtype=torch.long)

        torch.cuda.synchronize()
        batch_started = time.perf_counter()
        with torch.inference_mode():
            with autocast_context(torch, args.precision):
                sampled_tokens = model.sample_tokens(
                    bsz=end - start,
                    num_iter=args.num_iter,
                    cfg=args.cfg,
                    cfg_schedule=args.cfg_schedule,
                    labels=batch_labels,
                    temperature=args.temperature,
                    progress=args.progress,
                )
        gc.collect()
        torch.cuda.empty_cache()
        decode_and_save_batch(
            torch,
            vae,
            sampled_tokens,
            args.output_dir,
            start,
            args.decode_batch_size,
            args.precision,
        )
        torch.cuda.synchronize()
        batch_seconds.append(time.perf_counter() - batch_started)
        del sampled_tokens, batch_labels
        generated_batches += 1
        generated_images += end - start
        completed_indices.update(range(start, end))

        completed = len(completed_indices)
        elapsed = time.perf_counter() - started
        payload = progress_payload(
            args,
            assets,
            completed,
            elapsed,
            "running",
            batch_seconds,
            generated_images,
        )
        write_json_atomic(report_path, payload)
        rate = payload["progress"]["images_per_second"]
        eta = payload["progress"]["eta_seconds"]
        print(
            f"MAR progress: {completed}/{args.num_images}; "
            f"{rate:.4f} image/s; ETA {eta / 3600:.2f} h",
            flush=True,
        )

    completed = len(completed_indices)
    packed_npz = None
    if args.pack_npz is not None and completed == args.num_images:
        packed_npz = validate_and_pack_pngs(args.output_dir, args.pack_npz, args.num_images)
    status = "samples_completed" if completed == args.num_images else "running_or_partial"
    if packed_npz is not None:
        status = "samples_and_npz_completed_eval_pending"
    payload = progress_payload(
        args,
        assets,
        completed,
        time.perf_counter() - started,
        status,
        batch_seconds,
        generated_images,
        packed_npz,
    )
    payload["runtime"] = {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(device),
        "peak_memory_bytes": torch.cuda.max_memory_allocated(device),
    }
    write_json_atomic(report_path, payload)
    return payload


def main() -> None:
    args = parse_args()
    payload = run(args)
    print(json.dumps(payload["progress"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
