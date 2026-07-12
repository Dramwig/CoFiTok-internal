import json
from pathlib import Path

import pytest

from scripts.validate_mvp_evidence import validate_evidence


def _train(dataset: str, variant: str, k: int, seed: int, steps: int, **extra):
    row = {
        "dataset": dataset,
        "variant": variant,
        "token_count": k,
        "seed": seed,
        "steps": steps,
        "final_clean_mse": 0.1,
        "path_auc": 0.05 if variant == "light_denoise_path" else 1.0,
        "effective_tokens": max(1.0, k * 0.8),
        "zero_token_ratio": 0.0,
    }
    row.update(extra)
    return row


def _order(k: int, seed: int, order: str, variant: str = "light_denoise_path"):
    auc = {"ordered": 0.05, "random": 0.20, "reverse": 0.70}[order]
    return {
        "dataset": "imagenet_1k_64x64_hf",
        "variant": variant,
        "token_count": k,
        "seed": seed,
        "component_order": order,
        "path_auc": auc,
        "zero_token_ratio": 0.0,
    }


def _quality(variant: str, k: int):
    return {
        "dataset": "imagenet_1k_64x64_hf",
        "variant": variant,
        "token_count": k,
        "seed": 103,
        "final_mse": 0.12,
        "final_psnr_db": 15.0,
        "final_inception_frechet": 320.0,
        "final_lpips_alex": 0.55,
        "mse_auc": 5.0,
        "lpips_available": True,
        "inception_available": True,
    }


def _summary_payload() -> dict:
    train = [
        _train("cifar10", "light_denoise_path", 8, 103, 3000),
    ]
    for dataset in ["tiny_imagenet_200", "imagenet_1k_64x64_hf"]:
        train.append(_train(dataset, "epsilon_only", 8, 139, 20000, path_auc=1.0, final_clean_mse=0.10))
        train.append(_train(dataset, "light_denoise_path", 8, 139, 20000, path_auc=0.04, final_clean_mse=0.105))
    for seed in [103, 139]:
        train.append(_train("imagenet_1k_64x64_hf", "deep_synthesis_ablation", 8, seed, 5000, zero_token_ratio=0.05))
    order_eval = []
    for k in [4, 8, 16]:
        for seed in [103, 139]:
            for order in ["ordered", "random", "reverse"]:
                order_eval.append(_order(k, seed, order))
    for seed in [103, 139]:
        for order in ["ordered", "random", "reverse"]:
            order_eval.append(_order(8, seed, order, variant="simultaneous_predictor"))
    quality = [
        _quality("epsilon_only", 8),
        _quality("light_denoise_path", 4),
        _quality("light_denoise_path", 8),
        _quality("light_denoise_path", 16),
        _quality("simultaneous_predictor", 8),
        _quality("deep_synthesis_ablation", 8),
    ]
    sampling = [
        {"dataset": dataset, "variant": "light_denoise_path"}
        for dataset in ["cifar10", "tiny_imagenet_200", "imagenet_1k_64x64_hf"]
    ]
    generated_quality = []
    for dataset in ["tiny_imagenet_200", "imagenet_1k_64x64_hf"]:
        for variant in ["epsilon_only", "light_denoise_path"]:
            generated_quality.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "token_count": 8,
                    "steps": 20000,
                    "sample_image_count": 1024,
                    "real_image_count": 4096,
                    "inception_frechet": 240.0,
                    "lowres_frechet_proxy": 6.0,
                }
            )
    return {
        "train": train,
        "order_eval": order_eval,
        "quality": quality,
        "sampling": sampling,
        "generated_quality": generated_quality,
    }


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_validate_mvp_evidence_accepts_complete_summary(tmp_path) -> None:
    path = tmp_path / "summary.json"
    _write(path, _summary_payload())

    result = validate_evidence(path)

    assert result["status"] == "ok"
    assert result["check_count"] == 6


def test_validate_mvp_evidence_rejects_missing_generated_quality(tmp_path) -> None:
    payload = _summary_payload()
    payload["generated_quality"] = []
    path = tmp_path / "summary.json"
    _write(path, payload)

    with pytest.raises(AssertionError, match="missing row"):
        validate_evidence(path)
