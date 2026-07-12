from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from cofitok.configs import config_to_dict, load_config
from cofitok.data import build_dataloader
from cofitok.diffusion import DiffusionSchedule
from cofitok.metrics import (
    energy_distribution_metrics,
    frechet_distance_from_features,
    lowres_image_features,
    normalized_curve_auc,
    psnr_from_mse,
)
from cofitok.models import CoFiTokTiny
from cofitok.reporting import write_json_report
from cofitok.training.losses import _denoise_path_prefix_targets
from cofitok.utils.seed import seed_everything
from train_short import _batch_images, _resolve_device, _select_component_order


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate cross-batch prefix reconstruction quality.")
    parser.add_argument("--config", required=True, help="Path to JSON experiment config.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint_final.pt.")
    parser.add_argument("--output-dir", required=True, help="Directory for quality report.")
    parser.add_argument("--split", default="val", choices=["train", "val", "validation", "test"])
    parser.add_argument("--max-batches", type=int, default=8)
    parser.add_argument("--max-images", type=int, default=256)
    parser.add_argument("--timestep", type=int, default=-1, help="Fixed diffusion timestep; default is midpoint.")
    parser.add_argument("--prefix-budgets", default="", help="Comma-separated prefix budgets; default is 1,2,4,8,K.")
    parser.add_argument("--feature-size", type=int, default=8, help="Spatial size for low-res Frechet proxy features.")
    parser.add_argument(
        "--evaluation-progress-power",
        type=float,
        default=None,
        help=(
            "Denoise-path progress power used only for evaluation targets. "
            "By default, use the training config value. Set this when comparing "
            "models trained with different progress powers against one common path."
        ),
    )
    parser.add_argument(
        "--component-order",
        choices=["ordered", "reverse", "random"],
        default="ordered",
        help="Order used when accumulating components into prefixes.",
    )
    parser.add_argument("--random-order-seed", type=int, default=0)
    parser.add_argument("--enable-lpips", action="store_true", help="Try LPIPS if the optional package is installed.")
    parser.add_argument(
        "--enable-inception-fid",
        action="store_true",
        help="Try torchvision Inception-V3 pool features for a formal FID-style Frechet metric.",
    )
    return parser.parse_args()


def _prefix_epsilons_from_components(components: list[torch.Tensor]) -> list[torch.Tensor]:
    prefix_epsilons = []
    running = torch.zeros_like(components[0])
    for component in components:
        running = running + component
        prefix_epsilons.append(running)
    return prefix_epsilons


def _prefix_budgets(raw: str, token_count: int) -> list[int]:
    if raw.strip():
        budgets = [int(value.strip()) for value in raw.split(",") if value.strip()]
    else:
        budgets = [1, 2, 4, 8, token_count]
    budgets = sorted({budget for budget in budgets if 1 <= budget <= token_count})
    if token_count not in budgets:
        budgets.append(token_count)
    return budgets


def _make_lpips(device: torch.device) -> tuple[Any | None, dict[str, Any]]:
    try:
        import lpips  # type: ignore[import-not-found]
    except Exception as error:  # pragma: no cover - optional dependency
        return None, {"available": False, "reason": repr(error)}
    try:
        model = lpips.LPIPS(net="alex").to(device).eval()
    except Exception as error:  # pragma: no cover - optional dependency
        return None, {"available": False, "reason": repr(error)}
    return model, {"available": True, "model": "lpips.LPIPS(net='alex')"}


def _make_inception(device: torch.device) -> tuple[nn.Module | None, dict[str, Any]]:
    try:
        from torchvision.models import Inception_V3_Weights, inception_v3
    except Exception as error:  # pragma: no cover - optional dependency
        return None, {"available": False, "reason": repr(error)}
    try:
        weights = Inception_V3_Weights.IMAGENET1K_V1
        model = inception_v3(weights=weights, transform_input=False)
        model.fc = nn.Identity()
        model = model.to(device).eval()
    except Exception as error:  # pragma: no cover - optional weight download
        return None, {"available": False, "reason": repr(error)}
    return model, {"available": True, "model": "torchvision.models.inception_v3/IMAGENET1K_V1"}


def _inception_features(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    images = (images.detach().clamp(-1.0, 1.0) + 1.0) * 0.5
    images = F.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)
    mean = images.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = images.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    images = (images - mean) / std
    features = model(images)
    if isinstance(features, tuple):
        features = features[0]
    return features.detach().to(dtype=torch.float64).cpu()


def _batch_mse_sum(candidate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    per_image = F.mse_loss(candidate, reference, reduction="none").flatten(1).mean(dim=1)
    return per_image.sum()


def _component_abs_cosines(components: list[torch.Tensor]) -> torch.Tensor:
    if len(components) < 2:
        return components[0].new_zeros(0)
    flattened = torch.stack([component.flatten(1) for component in components], dim=0)
    normalized = F.normalize(flattened, dim=-1, eps=1e-12)
    correlations = torch.einsum("kbn,lbn->klb", normalized, normalized).abs()
    values = []
    for first in range(len(components)):
        for second in range(first + 1, len(components)):
            values.append(correlations[first, second])
    return torch.cat(values, dim=0)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.max_batches < 1:
        raise ValueError("--max-batches must be >= 1")
    if args.max_images < 1:
        raise ValueError("--max-images must be >= 1")

    config = load_config(args.config)
    seed_everything(config.runtime.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(config.runtime.device)
    model = CoFiTokTiny(config.model).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    schedule = DiffusionSchedule(config.diffusion, device=device)
    timestep = args.timestep if args.timestep >= 0 else schedule.num_train_timesteps // 2
    if not 0 <= timestep < schedule.num_train_timesteps:
        raise ValueError(f"timestep must be in [0, {schedule.num_train_timesteps}), got {timestep}")
    budgets = _prefix_budgets(args.prefix_budgets, config.model.token_count)
    evaluation_progress_power = (
        config.loss.denoise_path_progress_power
        if args.evaluation_progress_power is None
        else float(args.evaluation_progress_power)
    )
    if evaluation_progress_power <= 0.0:
        raise ValueError("--evaluation-progress-power must be > 0")
    eval_loader = build_dataloader(config.data, split=args.split, drop_last=False)

    lpips_model = None
    lpips_status: dict[str, Any] = {"available": False, "reason": "not requested"}
    if args.enable_lpips:
        lpips_model, lpips_status = _make_lpips(device)
    inception_model = None
    inception_status: dict[str, Any] = {"available": False, "reason": "not requested"}
    if args.enable_inception_fid:
        inception_model, inception_status = _make_inception(device)

    mse_sums = {budget: 0.0 for budget in budgets}
    denoise_path_mse_sums = {budget: 0.0 for budget in budgets}
    lpips_sums = {budget: 0.0 for budget in budgets}
    clean_features = []
    candidate_features = {budget: [] for budget in budgets}
    clean_inception_features = []
    candidate_inception_features = {budget: [] for budget in budgets}
    image_count = 0
    batch_count = 0
    component_corr_sum = 0.0
    component_corr_count = 0
    component_corr_max = 0.0
    component_order_indices: list[int] | None = None
    original_component_energy_sum = 0.0
    component_energy_sums = [0.0 for _ in range(config.model.token_count)]
    zero_component_energy_sum = 0.0
    endpoint_epsilon_mse_sum = 0.0
    start = time.time()

    for batch in eval_loader:
        if batch_count >= args.max_batches or image_count >= args.max_images:
            break
        clean_images = _batch_images(batch, device)
        remaining = args.max_images - image_count
        clean_images = clean_images[:remaining]
        timesteps = torch.full((clean_images.shape[0],), timestep, dtype=torch.long, device=device)
        noise = torch.randn_like(clean_images)
        noisy_images = schedule.add_noise(clean_images, noise, timesteps)
        output = model(noisy_images, timesteps)
        ordered_components, batch_order_indices = _select_component_order(
            output.components,
            mode=args.component_order,
            random_seed=args.random_order_seed,
        )
        if component_order_indices is None:
            component_order_indices = batch_order_indices
        elif component_order_indices != batch_order_indices:
            raise RuntimeError("component order changed across evaluation batches")
        prefix_epsilons = _prefix_epsilons_from_components(ordered_components)
        denoise_path_targets = _denoise_path_prefix_targets(
            schedule,
            noisy_images,
            clean_images,
            timesteps,
            count=len(output.components),
            progress_power=evaluation_progress_power,
        )
        zero_components = model.synthesis.zero_components_like(output.tokens)
        batch_component_energies = [
            float(component.square().sum().detach().cpu().item())
            for component in output.components
        ]
        for index, energy in enumerate(batch_component_energies):
            component_energy_sums[index] += energy
        original_component_energy_sum += sum(batch_component_energies)
        zero_component_energy_sum += sum(
            float(component.square().sum().detach().cpu().item())
            for component in zero_components
        )
        endpoint_epsilon_mse_sum += float(
            _batch_mse_sum(prefix_epsilons[-1], output.epsilon).detach().cpu().item()
        )
        component_corr = _component_abs_cosines(output.components)
        if component_corr.numel() > 0:
            component_corr_sum += float(component_corr.sum().detach().cpu().item())
            component_corr_count += int(component_corr.numel())
            component_corr_max = max(component_corr_max, float(component_corr.max().detach().cpu().item()))

        clean_features.append(lowres_image_features(clean_images, feature_size=args.feature_size).cpu())
        if inception_model is not None:
            clean_inception_features.append(_inception_features(inception_model, clean_images))
        for budget in budgets:
            prefix_image = schedule.predict_x0_from_epsilon(
                noisy_images,
                prefix_epsilons[budget - 1],
                timesteps,
            )
            mse_sums[budget] += float(_batch_mse_sum(prefix_image, clean_images).detach().cpu().item())
            denoise_path_mse_sums[budget] += float(
                _batch_mse_sum(prefix_image, denoise_path_targets[budget - 1])
                .detach()
                .cpu()
                .item()
            )
            candidate_features[budget].append(
                lowres_image_features(prefix_image, feature_size=args.feature_size).cpu()
            )
            if inception_model is not None:
                candidate_inception_features[budget].append(_inception_features(inception_model, prefix_image))
            if lpips_model is not None:
                values = lpips_model(prefix_image.clamp(-1.0, 1.0), clean_images.clamp(-1.0, 1.0))
                lpips_sums[budget] += float(values.view(values.shape[0], -1).mean(dim=1).sum().detach().cpu().item())

        image_count += int(clean_images.shape[0])
        batch_count += 1

    if image_count == 0:
        raise RuntimeError("No images were evaluated")

    reference_features = torch.cat(clean_features, dim=0)
    reference_inception_features = (
        torch.cat(clean_inception_features, dim=0)
        if clean_inception_features
        else None
    )
    metrics_by_prefix: dict[str, dict[str, float]] = {}
    mse_curve = []
    denoise_path_mse_curve = []
    psnr_curve = []
    frechet_curve = []
    lpips_curve = []
    inception_curve = []
    for budget in budgets:
        mse = mse_sums[budget] / image_count
        denoise_path_mse = denoise_path_mse_sums[budget] / image_count
        psnr = psnr_from_mse(mse, max_value=2.0)
        candidate = torch.cat(candidate_features[budget], dim=0)
        lowres_frechet = frechet_distance_from_features(reference_features, candidate)
        entry = {
            "mse": mse,
            "denoise_path_mse": denoise_path_mse,
            "psnr_db": psnr,
            "lowres_frechet_proxy": lowres_frechet,
        }
        if lpips_model is not None:
            lpips_value = lpips_sums[budget] / image_count
            entry["lpips_alex"] = lpips_value
            lpips_curve.append(lpips_value)
        if reference_inception_features is not None:
            inception_candidate = torch.cat(candidate_inception_features[budget], dim=0)
            inception_frechet = frechet_distance_from_features(reference_inception_features, inception_candidate)
            entry["inception_frechet"] = inception_frechet
            inception_curve.append(inception_frechet)
        metrics_by_prefix[str(budget)] = entry
        mse_curve.append(mse)
        denoise_path_mse_curve.append(denoise_path_mse)
        psnr_curve.append(psnr)
        frechet_curve.append(lowres_frechet)

    component_energy_total = sum(component_energy_sums)
    component_energy_ratios = [
        energy / max(component_energy_total, 1e-30)
        for energy in component_energy_sums
    ]

    report = {
        "config": config_to_dict(config),
        "checkpoint": args.checkpoint,
        "evaluation": {
            "split": args.split,
            "image_count": image_count,
            "batch_count": batch_count,
            "max_batches": args.max_batches,
            "max_images": args.max_images,
            "fixed_timestep": timestep,
            "prefix_budgets": budgets,
            "component_order": args.component_order,
            "component_order_indices": component_order_indices,
            "random_order_seed": args.random_order_seed,
            "training_denoise_path_progress_power": config.loss.denoise_path_progress_power,
            "evaluation_denoise_path_progress_power": evaluation_progress_power,
            # Kept for compatibility with older report readers.
            "denoise_path_progress_power": evaluation_progress_power,
            "image_range": "[-1, 1]",
        },
        "metrics_by_prefix": metrics_by_prefix,
        "curve_auc": {
            "mse": normalized_curve_auc(mse_curve),
            "denoise_path_mse": normalized_curve_auc(denoise_path_mse_curve),
            "psnr_db": normalized_curve_auc(psnr_curve),
            "lowres_frechet_proxy": normalized_curve_auc(frechet_curve),
            **({"lpips_alex": normalized_curve_auc(lpips_curve)} if lpips_curve else {}),
            **({"inception_frechet": normalized_curve_auc(inception_curve)} if inception_curve else {}),
        },
        "component_correlation": {
            "mean_abs_cosine": component_corr_sum / max(component_corr_count, 1),
            "max_abs_cosine": component_corr_max,
            "pair_image_count": component_corr_count,
        },
        "component_energy": {
            "squared_energy_sums": component_energy_sums,
            "ratios": component_energy_ratios,
            **energy_distribution_metrics(component_energy_ratios),
        },
        "diagnostics": {
            "zero_token_component_energy_ratio": (
                zero_component_energy_sum / max(original_component_energy_sum, 1e-30)
            ),
            "endpoint_epsilon_sum_mse": endpoint_epsilon_mse_sum / image_count,
        },
        "metric_notes": {
            "psnr_db": "Computed from MSE in normalized image range [-1, 1] with max_value=2.",
            "lowres_frechet_proxy": (
                "Small-scale rFID proxy over deterministic low-resolution RGB/color features; "
                "not a substitute for Inception/FID."
            ),
            "lpips": lpips_status,
            "inception_frechet": {
                **inception_status,
                "cache_hint": "Set TORCH_HOME to a project checkpoint/cache directory before first download.",
            },
        },
        "runtime": {
            "requested_device": config.runtime.device,
            "actual_device": str(device),
            "elapsed_seconds": time.time() - start,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
    }
    report_path = output_dir / "quality_report.json"
    write_json_report(report_path, report)
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
