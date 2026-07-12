import json
from pathlib import Path

from scripts.summarize_remote_queue_progress import build_progress_summary, render_markdown, write_progress_summary


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_progress_summary_extracts_completed_run(tmp_path) -> None:
    reports = tmp_path / "reports"
    _write_json(
        reports / "train_tiny_imagenet_k8_epsilononly_p150eval_20k_seed2_2026-07-08/report.json",
        {
            "config": {
                "data": {"dataset": "tiny_imagenet_200"},
                "runtime": {"seed": 103, "steps": 20000},
                "model": {"token_count": 8, "synthesis_mode": "restricted", "predictor_type": "tiny_conv"},
            },
            "final_losses": {"total": 0.2, "epsilon": 0.1},
            "prefix_summary": {
                "prefix_mse_to_clean_auc": 5.0,
                "prefix_mse_to_denoise_path_auc": 0.4,
                "prefix_mse_to_clean": [3.0, 1.0],
                "diagnostics": {
                    "zero_token_component_energy_ratio": 0.0,
                    "random_token_component_energy_ratio": 2.0,
                },
                "shuffled_final_mse_ratio": 9.0,
                "energy_effective_token_count": 4.0,
                "tail_energy_ratio": 0.75,
            },
        },
    )
    _write_json(
        reports / "quality_tiny_epsilononly_20k_seed2_1024_t500_lpips_inception_2026-07-08/quality_report.json",
        {
            "evaluation": {"image_count": 1024},
            "curve_auc": {"mse": 4.5},
            "metrics_by_prefix": {
                "1": {"mse": 2.0, "lpips_alex": 0.9, "inception_frechet": 300.0},
                "8": {"mse": 0.2, "lpips_alex": 0.5, "inception_frechet": 250.0},
            },
        },
    )
    _write_json(
        reports / "generated_tiny_epsilononly_20k_seed2_2048_ddim50_2026-07-08/sample_report.json",
        {"sampling": {"num_samples": 2048, "sample_steps": 50}},
    )
    _write_json(
        reports
        / "generated_quality_tiny_epsilononly_20k_seed2_2048_ddim50_2026-07-08/generated_quality_report.json",
        {
            "evaluation": {"sample_image_count": 2048, "real_image_count": 8192},
            "metrics": {"inception_frechet": 240.0, "lowres_frechet_proxy": 7.0},
        },
    )

    summary = build_progress_summary(reports, "2026-07-08")

    assert summary["completed_labels"] == ["tiny_epsilononly_20k_seed2"]
    run = summary["runs"]["tiny_epsilononly_20k_seed2"]
    assert run["quality_final_clean_mse"] == 0.2
    assert run["quality_final_lpips_alex"] == 0.5
    assert run["train_prefix_auc_denoise_path"] == 0.4
    assert run["generated_quality_real_count"] == 8192
    assert "tiny_epsilononly_20k_seed2" in render_markdown(summary)


def test_write_progress_summary_writes_json_and_markdown(tmp_path) -> None:
    summary = {"date": "2026-07-08", "completed_labels": [], "runs": {}}

    outputs = write_progress_summary(summary, tmp_path)

    assert outputs["json"].exists()
    assert outputs["markdown"].exists()
