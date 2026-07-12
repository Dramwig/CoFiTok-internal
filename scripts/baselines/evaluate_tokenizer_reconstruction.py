from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    script_dir = Path(__file__).resolve().parents[1]
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(script_dir))
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn.functional as F

from cofitok.configs import DataConfig
from cofitok.data import build_dataloader
from cofitok.metrics import frechet_distance_from_features, lowres_image_features, psnr_from_mse
from cofitok.reporting import write_json_report
from scripts.evaluate_generated_samples import _inception_features, _make_inception
from train_short import _batch_images


MODEL_DEFAULTS = {
    "ml_flextok": "EPFL-VILAB/flextok_d12_d12_in1k",
    "titok_1d_tokenizer": "yucornetto/tokenizer_titok_l32_imagenet",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pretrained tokenizer reconstruction baselines.")
    parser.add_argument("--baseline", required=True, choices=sorted(MODEL_DEFAULTS))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-root", default="/root/autodl-tmp/CoFiTok/datasets")
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-id", default="")
    parser.add_argument("--split", default="val", choices=["train", "val", "validation", "test"])
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--source-resolution", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--feature-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=139)
    parser.add_argument("--flextok-keep", type=int, default=-1)
    parser.add_argument("--flextok-timesteps", type=int, default=4)
    parser.add_argument("--flextok-guidance-scale", type=float, default=1.0)
    parser.add_argument("--flextok-norm-guidance", action="store_true")
    parser.add_argument("--enable-inception-fid", action="store_true")
    return parser.parse_args()


def _resolve_device(requested: str) -> torch.device:
    requested = requested.strip().lower()
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _git_commit(repo_dir: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def _load_flextok(repo_dir: Path, model_id: str, device: torch.device) -> Any:
    sys.path.insert(0, str(repo_dir))
    from flextok.flextok_wrapper import FlexTokFromHub

    model = FlexTokFromHub.from_pretrained(model_id)
    return model.to(device).eval().requires_grad_(False)


def _load_titok(repo_dir: Path, model_id: str, device: torch.device) -> Any:
    sys.path.insert(0, str(repo_dir))
    from modeling.titok import TiTok

    model = TiTok.from_pretrained(model_id)
    return model.to(device).eval().requires_grad_(False)


@torch.no_grad()
def _reconstruct_flextok(
    model: Any,
    images: torch.Tensor,
    keep: int,
    timesteps: int,
    guidance_scale: float,
    norm_guidance: bool,
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, Any]]:
    token_ids = model.tokenize(images)
    full_lengths = [int(tokens.shape[1]) for tokens in token_ids]
    if keep > 0:
        token_ids = [tokens[:, : min(keep, tokens.shape[1])] for tokens in token_ids]
    kept_lengths = [int(tokens.shape[1]) for tokens in token_ids]
    recon = model.detokenize(
        token_ids,
        timesteps=timesteps,
        guidance_scale=guidance_scale,
        perform_norm_guidance=norm_guidance,
        generator=generator,
        verbose=False,
    )
    return recon.clamp(-1.0, 1.0), {
        "full_token_lengths": full_lengths,
        "kept_token_lengths": kept_lengths,
        "decode_timesteps": timesteps,
        "guidance_scale": guidance_scale,
        "perform_norm_guidance": norm_guidance,
    }


@torch.no_grad()
def _reconstruct_titok(model: Any, images: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    images_01 = (images.clamp(-1.0, 1.0) + 1.0) * 0.5
    encoded = model.encode(images_01)
    payload = encoded[1]
    if getattr(model, "quantize_mode", "") == "vq":
        tokens = payload["min_encoding_indices"]
        recon_01 = model.decode_tokens(tokens)
        token_shape = list(tokens.shape)
    elif getattr(model, "quantize_mode", "") == "vae":
        tokens = payload.sample()
        recon_01 = model.decode_tokens(tokens)
        token_shape = list(tokens.shape)
    else:
        raise NotImplementedError(f"Unsupported TiTok quantize_mode: {getattr(model, 'quantize_mode', None)}")
    recon = recon_01.clamp(0.0, 1.0) * 2.0 - 1.0
    return recon.clamp(-1.0, 1.0), {
        "quantize_mode": getattr(model, "quantize_mode", ""),
        "token_shape": token_shape,
    }


def _metric_sums(candidate: torch.Tensor, reference: torch.Tensor) -> tuple[float, float]:
    mse_per_image = F.mse_loss(candidate, reference, reduction="none").flatten(1).mean(dim=1)
    return float(mse_per_image.sum().detach().cpu().item()), float(mse_per_image.mean().detach().cpu().item())


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.max_images < 1:
        raise ValueError("--max-images must be >= 1")
    repo_dir = Path(args.repo_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_id = args.model_id or MODEL_DEFAULTS[args.baseline]
    device = _resolve_device(args.device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    start = time.time()

    data_config = DataConfig(
        dataset=args.dataset,
        root=args.data_root,
        image_size=args.image_size,
        channels=3,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    loader = build_dataloader(data_config, split=args.split, drop_last=False)
    if args.baseline == "ml_flextok":
        model = _load_flextok(repo_dir, model_id, device)
    elif args.baseline == "titok_1d_tokenizer":
        model = _load_titok(repo_dir, model_id, device)
    else:  # pragma: no cover - argparse enforces choices
        raise ValueError(args.baseline)

    inception_model = None
    inception_status: dict[str, Any] = {"available": False, "reason": "not requested"}
    if args.enable_inception_fid:
        inception_model, inception_status = _make_inception(device)

    reference_lowres = []
    recon_lowres = []
    reference_inception = []
    recon_inception = []
    mse_sum = 0.0
    last_batch_mse = 0.0
    image_count = 0
    batch_count = 0
    token_records: list[dict[str, Any]] = []

    for batch in loader:
        if image_count >= args.max_images:
            break
        images = _batch_images(batch, device)
        remaining = args.max_images - image_count
        images = images[:remaining].clamp(-1.0, 1.0)
        if args.baseline == "ml_flextok":
            recon, token_info = _reconstruct_flextok(
                model,
                images,
                keep=args.flextok_keep,
                timesteps=args.flextok_timesteps,
                guidance_scale=args.flextok_guidance_scale,
                norm_guidance=args.flextok_norm_guidance,
                generator=generator,
            )
        else:
            recon, token_info = _reconstruct_titok(model, images)

        batch_mse_sum, last_batch_mse = _metric_sums(recon, images)
        mse_sum += batch_mse_sum
        reference_lowres.append(lowres_image_features(images, feature_size=args.feature_size).cpu())
        recon_lowres.append(lowres_image_features(recon, feature_size=args.feature_size).cpu())
        if inception_model is not None:
            reference_inception.append(_inception_features(inception_model, images))
            recon_inception.append(_inception_features(inception_model, recon))
        if len(token_records) < 4:
            token_records.append(token_info)
        image_count += int(images.shape[0])
        batch_count += 1

    if image_count == 0:
        raise RuntimeError("No images evaluated")

    mse = mse_sum / image_count
    metrics: dict[str, Any] = {
        "reconstruction_mse": mse,
        "reconstruction_psnr_db": psnr_from_mse(mse, max_value=2.0),
        "lowres_frechet_proxy": frechet_distance_from_features(
            torch.cat(reference_lowres, dim=0),
            torch.cat(recon_lowres, dim=0),
        ),
    }
    if reference_inception and recon_inception:
        metrics["inception_frechet"] = frechet_distance_from_features(
            torch.cat(reference_inception, dim=0),
            torch.cat(recon_inception, dim=0),
        )

    run_name = f"{args.baseline}_{args.dataset}_recon{args.image_size}_{image_count}img"
    if args.baseline == "ml_flextok" and args.flextok_keep > 0:
        run_name += f"_k{args.flextok_keep}"
    repo_commit = _git_commit(repo_dir)

    train_report = {
        "baseline": args.baseline,
        "dataset": args.dataset,
        "report_type": "baseline_train",
        "status": "completed",
        "run_name": run_name,
        "repo": str(repo_dir),
        "repo_commit": repo_commit,
        "output_dir": str(output_dir),
        "parameters": {
            "train_steps": 0,
            "batch_size": args.batch_size,
            "learning_rate": None,
            "pretrained_model_id": model_id,
            "eval_only": True,
            "official_pretrained_weights": True,
        },
        "notes": (
            "Eval-only official pretrained tokenizer baseline. No project-local training was run; "
            "this report is present so matrix accounting can distinguish eval-only P1 evidence "
            "from missing adapter work."
        ),
    }
    eval_report = {
        "baseline": args.baseline,
        "dataset": args.dataset,
        "report_type": "baseline_eval",
        "status": "completed",
        "run_name": run_name,
        "checkpoint": model_id,
        "samples_dir": "streamed_reconstructions",
        "evaluation": {
            "split": args.split,
            "image_count": image_count,
            "sample_image_count": image_count,
            "real_image_count": image_count,
            "available_sample_images": image_count,
            "feature_size": args.feature_size,
            "seed": args.seed,
            "eval_image_size": args.image_size,
            "source_resolution": args.source_resolution if args.source_resolution > 0 else None,
        },
        "parameters": {
            "sample_steps": args.flextok_timesteps if args.baseline == "ml_flextok" else 1,
            "sampler": "tokenizer_reconstruction",
            "sampler_nfe": args.flextok_timesteps if args.baseline == "ml_flextok" else 1,
            "model_id": model_id,
            "flextok_keep": args.flextok_keep if args.baseline == "ml_flextok" else None,
        },
        "metrics": metrics,
        "metric_notes": {
            "lowres_frechet_proxy": "Small-scale reconstruction Frechet proxy over deterministic low-resolution RGB/color features.",
            "inception_frechet": inception_status,
            "protocol": (
                "Tokenizer reconstruction, not unconditional generation. Datasets are resized to eval_image_size "
                "before tokenization so 32/64 source datasets are not directly comparable to native-resolution P0 rows."
            ),
            "last_batch_mse": last_batch_mse,
        },
        "tokens": token_records,
        "runtime": {
            "requested_device": args.device.strip(),
            "actual_device": str(device),
            "elapsed_seconds": time.time() - start,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
    }
    write_json_report(output_dir / "baseline_train_report.json", train_report)
    write_json_report(output_dir / "baseline_eval_report.json", eval_report)
    print(f"wrote {output_dir / 'baseline_eval_report.json'}")


if __name__ == "__main__":
    main()
