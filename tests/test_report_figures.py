import json
from pathlib import Path

from PIL import Image

from scripts.make_report_figures import (
    make_figures,
    select_20k_generated_rows,
    select_20k_training_rows,
    select_order_rows,
)


def _summary_payload() -> dict:
    train_rows = []
    generated_rows = []
    for dataset in ["tiny_imagenet_200", "imagenet_1k_64x64_hf"]:
        train_rows.extend(
            [
                {
                    "dataset": dataset,
                    "variant": "epsilon_only",
                    "token_count": 8,
                    "steps": 20000,
                    "final_clean_mse": 0.12,
                    "path_auc": 1.1,
                    "effective_tokens": 5.2,
                },
                {
                    "dataset": dataset,
                    "variant": "light_denoise_path",
                    "token_count": 8,
                    "steps": 20000,
                    "final_clean_mse": 0.13,
                    "path_auc": 0.04,
                    "effective_tokens": 6.8,
                },
            ]
        )
        generated_rows.extend(
            [
                {
                    "dataset": dataset,
                    "variant": "epsilon_only",
                    "token_count": 8,
                    "steps": 20000,
                    "sample_image_count": 256,
                    "real_image_count": 1024,
                    "inception_frechet": 263.0,
                    "lowres_frechet_proxy": 5.4,
                },
                {
                    "dataset": dataset,
                    "variant": "light_denoise_path",
                    "token_count": 8,
                    "steps": 20000,
                    "sample_image_count": 256,
                    "real_image_count": 1024,
                    "inception_frechet": 281.0,
                    "lowres_frechet_proxy": 6.8,
                },
                {
                    "dataset": dataset,
                    "variant": "epsilon_only",
                    "token_count": 8,
                    "steps": 20000,
                    "sample_image_count": 1024,
                    "real_image_count": 4096,
                    "inception_frechet": 227.0,
                    "lowres_frechet_proxy": 5.2,
                },
                {
                    "dataset": dataset,
                    "variant": "light_denoise_path",
                    "token_count": 8,
                    "steps": 20000,
                    "sample_image_count": 1024,
                    "real_image_count": 4096,
                    "inception_frechet": 245.0,
                    "lowres_frechet_proxy": 6.4,
                },
                {
                    "dataset": dataset,
                    "variant": "epsilon_only",
                    "token_count": 8,
                    "steps": 20000,
                    "sample_image_count": 8192,
                    "real_image_count": 8192,
                    "inception_frechet": 241.0,
                    "lowres_frechet_proxy": 6.5,
                },
                {
                    "dataset": dataset,
                    "variant": "light_denoise_path",
                    "token_count": 8,
                    "steps": 20000,
                    "sample_image_count": 8192,
                    "real_image_count": 8192,
                    "inception_frechet": 269.0,
                    "lowres_frechet_proxy": 6.7,
                },
            ]
        )
    generated_rows.extend(
        [
            {
                "dataset": "imagenet_1k_64x64_hf",
                "variant": "epsilon_only",
                "token_count": 8,
                "steps": 20000,
                "sample_image_count": 50000,
                "real_image_count": 50000,
                "inception_frechet": 241.0,
                "lowres_frechet_proxy": 6.5,
            },
            {
                "dataset": "imagenet_1k_64x64_hf",
                "variant": "light_denoise_path",
                "token_count": 8,
                "steps": 20000,
                "sample_image_count": 50000,
                "real_image_count": 50000,
                "inception_frechet": 269.0,
                "lowres_frechet_proxy": 6.7,
            },
        ]
    )
    order_rows = []
    for token_count in [4, 8, 16]:
        for component_order, path_auc in [("ordered", 0.06), ("random", 0.24), ("reverse", 0.70)]:
            order_rows.append(
                {
                    "dataset": "imagenet_1k_64x64_hf",
                    "variant": "light_denoise_path",
                    "token_count": token_count,
                    "seed": 139,
                    "component_order": component_order,
                    "path_auc": path_auc,
                }
            )
    return {"train": train_rows, "generated_quality": generated_rows, "order_eval": order_rows}


def test_select_report_figure_rows() -> None:
    summary = _summary_payload()

    assert len(select_20k_training_rows(summary)) == 4
    assert len(select_20k_generated_rows(summary)) == 4
    assert len(select_20k_generated_rows(summary, sample_image_count=1024, real_image_count=4096)) == 4
    assert len(select_20k_generated_rows(summary, sample_image_count=8192, real_image_count=8192)) == 4
    assert len(select_20k_generated_rows(summary, sample_image_count=50000, real_image_count=50000)) == 2
    assert len(select_order_rows(summary)) == 9


def test_make_figures_writes_pngs_and_manifest(tmp_path) -> None:
    summary_path = tmp_path / "experiment_summary.json"
    summary_path.write_text(json.dumps(_summary_payload()), encoding="utf-8")

    manifest = make_figures(summary_path, tmp_path / "figures")

    assert manifest["figure_count"] == 8
    for item in manifest["figures"]:
        path = Path(item["path"])
        assert path.exists()
        with Image.open(path) as image:
            assert image.size == (1260, 760)
    assert (tmp_path / "figures" / "figure_manifest.json").exists()
