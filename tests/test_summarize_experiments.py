import json

from scripts.summarize_experiments import (
    collect_generated_quality_rows,
    collect_official_fid_rows,
    collect_quality_rows,
    collect_sample_rows,
    collect_train_rows,
    infer_variant,
)


def test_infer_variant_prefers_explicit_ablation_flags() -> None:
    assert infer_variant({"name": "x", "model": {"synthesis_mode": "dense_identity", "predictor_use_feedback": False}, "loss": {}}) == "dense_monolithic"
    assert infer_variant({"name": "x", "model": {"synthesis_mode": "deep_decoder"}, "loss": {}}) == "deep_synthesis_ablation"
    assert infer_variant({"name": "x", "model": {"predictor_use_feedback": False}, "loss": {}}) == "simultaneous_predictor"


def test_infer_variant_from_loss_and_name() -> None:
    assert infer_variant({"name": "train_cifar10_densehead_3k_cuda", "model": {}, "loss": {}}) == "dense_monolithic"
    assert infer_variant({"name": "train_cifar10_channelmask", "model": {}, "loss": {}}) == "channel_mask"
    assert infer_variant({"name": "train_cifar10_epsilononly", "model": {}, "loss": {}}) == "epsilon_only"
    assert (
        infer_variant(
            {
                "name": "train_cifar10_denoisepath_light",
                "model": {},
                "loss": {"denoise_path_prefix_weight": 0.15},
            }
        )
        == "light_denoise_path"
    )
    assert (
        infer_variant(
            {
                "name": "train_cifar10_denoisepath_light_nopathprefix",
                "model": {},
                "loss": {"denoise_path_component_weight": 0.3},
            }
        )
        == "no_prefix_loss_ablation"
    )
    assert (
        infer_variant(
            {
                "name": "train_cifar10_denoisepath_light_cleanmono",
                "model": {},
                "loss": {"denoise_path_prefix_weight": 0.15, "monotonic_weight": 0.02},
            }
        )
        == "clean_monotonic_ablation"
    )
    assert (
        infer_variant(
            {
                "name": "train_cifar10_denoisepath_light_decor",
                "model": {},
                "loss": {"denoise_path_prefix_weight": 0.15, "component_decorrelation_weight": 0.05},
            }
        )
        == "component_decorrelation_ablation"
    )


def test_collect_sample_rows_reads_sampling_reports(tmp_path) -> None:
    report_dir = tmp_path / "samples"
    report_dir.mkdir()
    (report_dir / "sample_report.json").write_text(
        json.dumps(
            {
                "checkpoint": "/tmp/checkpoint.pt",
                "config": {
                    "name": "train_cifar10_k8_denoisepath_p150_light",
                    "data": {"dataset": "cifar10", "image_size": 32},
                    "model": {"token_count": 8, "token_channels": 16},
                    "loss": {"denoise_path_prefix_weight": 0.15},
                    "runtime": {"seed": 13, "steps": 3000},
                },
                "sampling": {
                    "num_samples": 16,
                    "batch_size": 16,
                    "sample_steps": 20,
                    "actual_timestep_count": 20,
                    "first_timestep": 999,
                    "last_timestep": 0,
                    "prefix_budgets": [1, 4, 8],
                    "eta": 0.0,
                    "clip_x0": True,
                    "seed": 20260708,
                },
                "runtime": {"elapsed_seconds": 1.5},
                "artifacts": {"samples_prefix_8": "/tmp/samples_prefix_8.png"},
            }
        ),
        encoding="utf-8",
    )

    rows = collect_sample_rows(tmp_path)

    assert len(rows) == 1
    assert rows[0]["dataset"] == "cifar10"
    assert rows[0]["variant"] == "light_denoise_path"
    assert rows[0]["prefix_budgets"] == [1, 4, 8]
    assert rows[0]["artifact_count"] == 1


def test_collect_train_rows_records_predictor_type(tmp_path) -> None:
    report_dir = tmp_path / "train"
    report_dir.mkdir()
    (report_dir / "report.json").write_text(
        json.dumps(
            {
                "config": {
                    "name": "train_tiny_multiscale",
                    "data": {"dataset": "tiny_imagenet_200", "image_size": 64},
                    "model": {
                        "token_count": 8,
                        "token_channels": 16,
                        "predictor_type": "multiscale_unet",
                        "predictor_multiscale_levels": 2,
                    },
                    "loss": {"denoise_path_prefix_weight": 0.15},
                    "runtime": {"seed": 151, "steps": 10000},
                },
                "final_losses": {
                    "epsilon": 0.1,
                    "component_decorrelation": 0.03,
                    "component_decorrelation_effective_weight": 0.005,
                    "total": 0.2,
                },
                "prefix_summary": {
                    "prefix_mse_to_clean": [1.0, 0.5],
                    "prefix_mse_to_denoise_path_auc": 0.05,
                    "diagnostics": {"zero_token_component_energy_ratio": 0.0},
                },
                "artifacts": {"checkpoint": "/tmp/checkpoint.pt"},
            }
        ),
        encoding="utf-8",
    )

    rows = collect_train_rows(tmp_path)

    assert len(rows) == 1
    assert rows[0]["predictor_type"] == "multiscale_unet"
    assert rows[0]["predictor_multiscale_levels"] == 2
    assert rows[0]["final_component_decorrelation_loss"] == 0.03
    assert rows[0]["final_component_decorrelation_effective_weight"] == 0.005
    assert rows[0]["parameter_count"] > 0


def test_collect_quality_rows_records_component_correlation(tmp_path) -> None:
    report_dir = tmp_path / "quality"
    report_dir.mkdir()
    (report_dir / "quality_report.json").write_text(
        json.dumps(
            {
                "config": {
                    "name": "train_cifar10_k8_denoisepath_p150_light_decor",
                    "data": {"dataset": "cifar10", "image_size": 32},
                    "model": {"token_count": 8, "token_channels": 16},
                    "loss": {"denoise_path_prefix_weight": 0.15, "component_decorrelation_weight": 0.05},
                    "runtime": {"seed": 137, "steps": 3000},
                },
                "evaluation": {
                    "split": "val",
                    "image_count": 256,
                    "fixed_timestep": 500,
                    "prefix_budgets": [1, 2, 4, 8],
                    "component_order": "ordered",
                    "component_order_indices": list(range(8)),
                    "random_order_seed": 0,
                },
                "checkpoint": "/tmp/checkpoint_final.pt",
                "metrics_by_prefix": {
                    "8": {"mse": 0.02, "psnr_db": 23.0, "lowres_frechet_proxy": 0.1},
                },
                "curve_auc": {"mse": 0.03, "psnr_db": 21.0, "lowres_frechet_proxy": 0.2},
                "component_correlation": {
                    "mean_abs_cosine": 0.12,
                    "max_abs_cosine": 0.44,
                    "pair_image_count": 7168,
                },
                "component_energy": {
                    "ratios": [0.125] * 8,
                    "energy_effective_token_count": 8.0,
                },
                "diagnostics": {
                    "zero_token_component_energy_ratio": 0.0,
                    "endpoint_epsilon_sum_mse": 0.0,
                },
                "metric_notes": {
                    "lpips": {"available": False},
                    "inception_frechet": {"available": False},
                },
            }
        ),
        encoding="utf-8",
    )

    rows = collect_quality_rows(tmp_path)

    assert len(rows) == 1
    assert rows[0]["variant"] == "component_decorrelation_ablation"
    assert rows[0]["component_mean_abs_cosine"] == 0.12
    assert rows[0]["component_max_abs_cosine"] == 0.44
    assert rows[0]["component_order"] == "ordered"
    assert rows[0]["energy_effective_token_count"] == 8.0
    assert rows[0]["zero_token_ratio"] == 0.0
    assert rows[0]["checkpoint"] == "/tmp/checkpoint_final.pt"


def test_collect_generated_quality_rows_reads_sample_distribution_reports(tmp_path) -> None:
    report_dir = tmp_path / "generated_quality"
    report_dir.mkdir()
    (report_dir / "generated_quality_report.json").write_text(
        json.dumps(
            {
                "config_name": "train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_20k_cuda",
                "dataset": "imagenet_1k_64x64_hf",
                "samples_dir": "/tmp/samples_prefix_8",
                "evaluation": {
                    "split": "val",
                    "real_image_count": 256,
                    "sample_image_count": 64,
                    "available_sample_images": 64,
                    "feature_size": 8,
                },
                "sampling": {
                    "sample_steps": 50,
                    "prefix_budget": 8,
                    "batch_size": 16,
                    "eta": 0.0,
                },
                "metrics": {
                    "lowres_frechet_proxy": 7.25,
                    "inception_frechet": 313.8,
                },
                "metric_notes": {"inception_frechet": {"available": True}},
                "runtime": {"elapsed_seconds": 4.5},
            }
        ),
        encoding="utf-8",
    )

    rows = collect_generated_quality_rows(tmp_path)

    assert len(rows) == 1
    assert rows[0]["dataset"] == "imagenet_1k_64x64_hf"
    assert rows[0]["variant"] == "light_denoise_path"
    assert rows[0]["token_count"] == 8
    assert rows[0]["steps"] == 20000
    assert rows[0]["predictor_type"] == "tiny_conv"
    assert rows[0]["sample_image_count"] == 64
    assert rows[0]["sample_steps"] == 50
    assert rows[0]["prefix_budget"] == 8
    assert rows[0]["inception_available"] is True


def test_collect_generated_quality_rows_infers_multiscale_backbone(tmp_path) -> None:
    report_dir = tmp_path / "generated_quality_stream_tiny_epsilononly_multiscale_20k_10000_ddim50_2026-07-08"
    report_dir.mkdir()
    (report_dir / "generated_quality_report.json").write_text(
        json.dumps(
            {
                "config_name": "train_tiny_imagenet_k8_epsilononly_p150eval_multiscale_20k_cuda",
                "dataset": "tiny_imagenet_200",
                "evaluation": {
                    "split": "val",
                    "real_image_count": 10000,
                    "sample_image_count": 10000,
                    "available_sample_images": 10000,
                    "feature_size": 8,
                },
                "metrics": {"lowres_frechet_proxy": 5.0},
                "metric_notes": {"inception_frechet": {"available": False}},
            }
        ),
        encoding="utf-8",
    )

    rows = collect_generated_quality_rows(tmp_path)

    assert len(rows) == 1
    assert rows[0]["variant"] == "epsilon_only"
    assert rows[0]["predictor_type"] == "multiscale_unet"
    assert rows[0]["steps"] == 20000


def test_collect_generated_quality_rows_recognizes_direct_dense_head(tmp_path) -> None:
    report_dir = tmp_path / "generated_dense"
    report_dir.mkdir()
    (report_dir / "generated_quality_report.json").write_text(
        json.dumps(
            {
                "config_name": "train_tiny_imagenet_k8_densehead_p150eval_5k_cuda",
                "dataset": "tiny_imagenet_200",
                "sampling": {
                    "sample_steps": 50,
                    "prefix_budget": 1,
                    "batch_size": 64,
                    "eta": 0.0,
                },
                "evaluation": {
                    "split": "val",
                    "real_image_count": 4096,
                    "sample_image_count": 1024,
                    "available_sample_images": 1024,
                    "feature_size": 8,
                    "seed": 13,
                },
                "metrics": {
                    "lowres_frechet_proxy": 7.0,
                    "inception_frechet": 300.0,
                },
                "metric_notes": {"inception_frechet": {"available": True}},
            }
        ),
        encoding="utf-8",
    )

    rows = collect_generated_quality_rows(tmp_path)

    assert rows[0]["variant"] == "dense_monolithic"
    assert rows[0]["token_count"] == 1
    assert rows[0]["sample_steps"] == 50


def test_collect_official_fid_rows_reads_protocol_exports(tmp_path) -> None:
    export_dir = (
        tmp_path
        / "official_fid_protocol_2026-07-09"
        / "official_fid_export_imagenet_hf_k8_light_multiscale_20k_50000_ddim50_2026-07-08"
    )
    export_dir.mkdir(parents=True)
    (export_dir / "official_fid_report.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "metric": "fid",
                "metrics": {"fid": 134.08},
                "counts": {"real_image_count": 50000, "generated_image_count": 50000},
                "implementation": {"available": True, "package": "pytorch-fid"},
                "parameters": {"batch_size": 64, "device": "cuda", "dims": 2048},
                "runtime": {"elapsed_seconds": 45.5},
                "paths": {"real_dir": "/tmp/real", "generated_dir": "/tmp/generated"},
            }
        ),
        encoding="utf-8",
    )
    (export_dir / "official_fid_export_manifest.json").write_text(
        json.dumps(
            {
                "config": {
                    "name": "train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_multiscale_20k_cuda",
                    "data": {"dataset": "imagenet_1k_64x64_hf"},
                    "model": {"token_count": 8, "predictor_type": "multiscale_unet"},
                    "loss": {"denoise_path_prefix_weight": 0.15},
                    "runtime": {"steps": 20000},
                },
                "sampling": {"sample_steps": 50, "prefix_budget": 8},
            }
        ),
        encoding="utf-8",
    )

    rows = collect_official_fid_rows(tmp_path)

    assert len(rows) == 1
    assert rows[0]["report_type"] == "official_fid"
    assert rows[0]["dataset"] == "imagenet_1k_64x64_hf"
    assert rows[0]["variant"] == "light_denoise_path"
    assert rows[0]["predictor_type"] == "multiscale_unet"
    assert rows[0]["official_fid"] == 134.08
    assert rows[0]["implementation_package"] == "pytorch-fid"
