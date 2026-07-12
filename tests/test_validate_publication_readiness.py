import json
from pathlib import Path

from scripts.validate_publication_readiness import ReadinessCriteria, render_markdown, validate_readiness


MAIN_DATASETS = ("tiny_imagenet_200", "imagenet_1k_64x64_hf")
MAIN_VARIANTS = ("epsilon_only", "light_denoise_path")


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _train(dataset: str, variant: str, seed: int, steps: int, predictor_type: str = "tiny_conv") -> dict:
    return {
        "dataset": dataset,
        "variant": variant,
        "token_count": 8,
        "seed": seed,
        "steps": steps,
        "predictor_type": predictor_type,
        "synthesis_mode": "restricted",
        "final_clean_mse": 0.1,
        "path_auc": 0.05 if variant == "light_denoise_path" else 1.0,
        "effective_tokens": 6.8,
        "zero_token_ratio": 0.0,
    }


def _quality(dataset: str, variant: str, steps: int, predictor_type: str = "tiny_conv") -> dict:
    return {
        "dataset": dataset,
        "variant": variant,
        "token_count": 8,
        "steps": steps,
        "predictor_type": predictor_type,
        "image_count": 1024,
        "final_inception_frechet": 240.0,
        "final_lpips_alex": 0.45,
        "inception_available": True,
        "lpips_available": True,
    }


def _order(dataset: str, order: str) -> dict:
    return {
        "dataset": dataset,
        "variant": "light_denoise_path",
        "token_count": 8,
        "steps": 20000,
        "predictor_type": "tiny_conv",
        "seed": 103,
        "component_order": order,
        "path_auc": {"ordered": 0.05, "random": 0.25, "reverse": 0.65}[order],
    }


def _order_variant(dataset: str, variant: str, order: str, steps: int = 10000) -> dict:
    return {
        "dataset": dataset,
        "variant": variant,
        "token_count": 8,
        "steps": steps,
        "predictor_type": "tiny_conv",
        "seed": 157,
        "component_order": order,
        "path_auc": {"ordered": 0.05, "random": 0.25, "reverse": 0.65}[order],
    }


def _sampling(dataset: str, variant: str, steps: int, predictor_type: str = "tiny_conv") -> dict:
    return {
        "dataset": dataset,
        "variant": variant,
        "token_count": 8,
        "steps": steps,
        "predictor_type": predictor_type,
        "num_samples": 2048,
        "sample_steps": 50,
    }


def _generated(dataset: str, variant: str, steps: int, config_name: str) -> dict:
    return {
        "dataset": dataset,
        "variant": variant,
        "token_count": 8,
        "steps": steps,
        "config_name": config_name,
        "sample_image_count": 2048,
        "real_image_count": 8192,
        "inception_frechet": 240.0,
        "lowres_frechet_proxy": 6.0,
        "inception_available": True,
    }


def _complete_summary() -> dict:
    train = []
    quality = []
    order_eval = []
    sampling = []
    generated_quality = []
    for dataset in MAIN_DATASETS:
        for variant in MAIN_VARIANTS:
            train.append(_train(dataset, variant, seed=139, steps=20000))
            train.append(_train(dataset, variant, seed=103, steps=20000))
            quality.append(_quality(dataset, variant, steps=20000))
            sampling.append(_sampling(dataset, variant, steps=20000))
            generated_quality.append(_generated(dataset, variant, steps=20000, config_name=f"{dataset}_{variant}_20k"))
        for order in ("ordered", "random", "reverse"):
            order_eval.append(_order(dataset, order))
        train.append(_train(dataset, "light_denoise_path", seed=151, steps=10000, predictor_type="multiscale_unet"))
        quality.append(_quality(dataset, "light_denoise_path", steps=10000, predictor_type="multiscale_unet"))
        sampling.append(_sampling(dataset, "light_denoise_path", steps=10000, predictor_type="multiscale_unet"))
        generated_quality.append(
            _generated(
                dataset,
                "light_denoise_path",
                steps=10000,
                config_name=f"train_{dataset}_k8_denoisepath_p150_light_multiscale_10k_cuda",
            )
        )
        for variant in ("no_prefix_loss_ablation", "clean_monotonic_ablation"):
            train.append(_train(dataset, variant, seed=157, steps=10000))
            quality.append(_quality(dataset, variant, steps=10000))
            for order in ("ordered", "random", "reverse"):
                order_eval.append(_order_variant(dataset, variant, order))
    return {
        "train": train,
        "quality": quality,
        "order_eval": order_eval,
        "sampling": sampling,
        "generated_quality": generated_quality,
    }


def test_publication_readiness_accepts_complete_summary(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    _write(path, _complete_summary())

    result = validate_readiness(path)

    assert result["status"] == "ready"
    assert result["missing_count"] == 0
    assert result["check_count"] == 7


def test_publication_readiness_reports_missing_smoke_generation(tmp_path: Path) -> None:
    payload = _complete_summary()
    for row in payload["generated_quality"]:
        row["sample_image_count"] = 1024
        row["real_image_count"] = 4096
    path = tmp_path / "summary.json"
    _write(path, payload)

    result = validate_readiness(path)

    assert result["status"] == "not_ready"
    missing_names = {check["name"] for check in result["checks"] if check["status"] == "missing"}
    assert "generated_quality_scale" in missing_names
    assert "multiscale_pilots" in missing_names


def test_publication_readiness_defaults_missing_predictor_type_to_tiny_conv(tmp_path: Path) -> None:
    payload = _complete_summary()
    for section in ("train", "quality", "order_eval", "sampling"):
        for row in payload[section]:
            if row.get("predictor_type") == "tiny_conv":
                row.pop("predictor_type")
    path = tmp_path / "summary.json"
    _write(path, payload)

    result = validate_readiness(path, ReadinessCriteria())

    assert result["status"] == "ready"


def test_publication_readiness_markdown_reports_missing_items(tmp_path: Path) -> None:
    payload = _complete_summary()
    payload["sampling"] = []
    path = tmp_path / "summary.json"
    _write(path, payload)

    result = validate_readiness(path)
    markdown = render_markdown(result)

    assert "# CoFiTok Publication-Readiness Gap Report" in markdown
    assert "Status: `not_ready`" in markdown
    assert "`sampling_scale`" in markdown
    assert "need >= 2048 DDIM samples" in markdown
    assert "next_validation_queue_2026-07-08.sh" in markdown
