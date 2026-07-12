from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "build_formal64_paper_table.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("build_formal64_paper_table", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_rows_merges_internal_and_external_evidence() -> None:
    module = load_script_module()
    summary = {
        "train": [
            {
                "dataset": "ffhq_64",
                "variant": "light_denoise_path",
                "steps": 5000,
                "token_count": 8,
                "path_auc": 31.0,
                "clean_auc": 22.0,
                "effective_tokens": 6.5,
                "zero_token_ratio": 0.01,
                "shuffled_final_ratio": 0.9,
            }
        ],
        "quality": [
            {
                "dataset": "ffhq_64",
                "variant": "light_denoise_path",
                "steps": 5000,
                "token_count": 8,
                "final_psnr_db": 18.25,
                "final_lpips_alex": 0.42,
            }
        ],
        "generated_quality": [
            {
                "dataset": "ffhq_64",
                "variant": "light_denoise_path",
                "steps": 5000,
                "token_count": 8,
                "lowres_frechet_proxy": 8.75,
                "inception_frechet": 425.0,
            }
        ],
    }
    baseline_summary = {
        "rows": [
            {
                "baseline": "edm",
                "dataset": "ffhq_64",
                "train_steps": 5000,
                "sampler_nfe": 79,
                "lowres_frechet_proxy": 2.5,
                "inception_frechet": 325.0,
            }
        ]
    }

    rows = module.build_rows(summary, baseline_summary)
    cofitok = next(row for row in rows if row["dataset"] == "ffhq_64" and row["method"] == "cofitok_light")
    edm = next(row for row in rows if row["dataset"] == "ffhq_64" and row["method"] == "edm")

    assert cofitok["train_steps"] == 5000
    assert cofitok["sample_nfe"] == 50
    assert cofitok["sample_lowres_frechet"] == 8.75
    assert cofitok["denoise_psnr"] == 18.25
    assert cofitok["path_auc"] == 31.0
    assert cofitok["effective_tokens"] == 6.5
    assert cofitok["evidence"] == "train,quality,generated_quality"
    assert edm["train_steps"] == 5000
    assert edm["sample_nfe"] == 79
    assert edm["sample_inception_frechet"] == 325.0
    assert edm["evidence"] == "baseline_train,baseline_eval"


def test_main_writes_table_files(tmp_path: Path, monkeypatch) -> None:
    module = load_script_module()
    summary_path = tmp_path / "summary.json"
    baseline_path = tmp_path / "baseline_summary.json"
    output_dir = tmp_path / "table"
    summary_path.write_text(json.dumps({"train": [], "quality": [], "generated_quality": []}), encoding="utf-8")
    baseline_path.write_text(json.dumps({"rows": []}), encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_formal64_paper_table.py",
            "--summary",
            str(summary_path),
            "--baseline-summary",
            str(baseline_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    module.main()

    assert (output_dir / "formal64_paper_table.json").exists()
    assert (output_dir / "formal64_paper_table.csv").exists()
    assert (output_dir / "formal64_paper_table.md").exists()
