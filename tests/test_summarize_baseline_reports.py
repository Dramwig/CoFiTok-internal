from __future__ import annotations

import json
from pathlib import Path

from scripts.baselines.summarize_baseline_reports import collect_rows


def test_collect_rows_merges_train_and_eval_reports(tmp_path: Path) -> None:
    run_dir = tmp_path / "improved_diffusion" / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "baseline_train_report.json").write_text(
        json.dumps(
            {
                "baseline": "improved_diffusion",
                "dataset": "ffhq_64",
                "repo_commit": "abc123",
                "parameters": {
                    "train_steps": 5000,
                    "batch_size": 32,
                    "microbatch": 8,
                    "learning_rate": "1e-4",
                    "diffusion_steps": 1000,
                    "num_channels": 64,
                    "num_res_blocks": 2,
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "baseline_eval_report.json").write_text(
        json.dumps(
            {
                "baseline": "improved_diffusion",
                "dataset": "ffhq_64",
                "run_name": "run",
                "status": "completed",
                "parameters": {"sample_steps": 50, "sampler_nfe": 50, "sampler": "ddim"},
                "evaluation": {"sample_image_count": 1024, "real_image_count": 4096},
                "metrics": {"lowres_frechet_proxy": 1.25, "inception_frechet": 2.5},
                "checkpoint": "/tmp/model.pt",
                "samples_dir": "/tmp/samples",
            }
        ),
        encoding="utf-8",
    )

    rows = collect_rows(
        tmp_path,
        {
            "rows": [
                {
                    "baseline": "improved_diffusion",
                    "run_name": "run",
                    "parameter_count": 123456,
                    "status": "completed",
                }
            ]
        },
    )

    assert rows == [
        {
            "baseline": "improved_diffusion",
            "dataset": "ffhq_64",
            "run_name": "run",
            "status": "completed",
            "repo_commit": "abc123",
            "eval_only": False,
            "train_steps": 5000,
            "batch_size": 32,
            "microbatch": 8,
            "learning_rate": "1e-4",
            "diffusion_steps": 1000,
            "num_channels": 64,
            "num_res_blocks": 2,
            "parameter_count": 123456,
            "sample_steps": 50,
            "sampler_nfe": 50,
            "sampler": "ddim",
            "sample_image_count": 1024,
            "real_image_count": 4096,
            "eval_image_size": None,
            "source_resolution": None,
            "lowres_frechet_proxy": 1.25,
            "inception_frechet": 2.5,
            "reconstruction_mse": None,
            "reconstruction_psnr_db": None,
            "eval_protocol": None,
            "actual_device": None,
            "checkpoint": "/tmp/model.pt",
            "samples_dir": "/tmp/samples",
            "eval_report": (run_dir / "baseline_eval_report.json").as_posix(),
            "train_report": (run_dir / "baseline_train_report.json").as_posix(),
        }
    ]
