import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "baselines" / "build_official_related_methods_table.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "build_official_related_methods_table",
        SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_rows_keeps_completed_mar_hf_50k_as_non_ema_audit(
    tmp_path: Path, monkeypatch
) -> None:
    module = load_module()
    mar_root = (
        tmp_path
        / "checkpoints/baselines/mar/official_imagenet256_eval_only/samples_50k_hf"
    )
    mar_dir = mar_root / module.MAR_SAMPLE_STEM
    mar_dir.mkdir(parents=True)
    mar_npz = mar_root / f"{module.MAR_SAMPLE_STEM}.npz"
    mar_txt = mar_root / f"{module.MAR_SAMPLE_STEM}.txt"
    mar_npz.write_bytes(b"npz")
    mar_txt.write_text(
        "Inception Score: 281.7\n"
        "FID: 2.31\n"
        "sFID: 5.0\n"
        "Precision: 0.8\n"
        "Recall: 0.6\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        module,
        "count_pngs",
        lambda path: 50_000 if path == mar_dir else 0,
    )
    monkeypatch.setattr(module, "adm_npz_is_valid", lambda path: path == mar_npz)
    rows = module.build_rows(tmp_path)
    mar = next(row for row in rows if row["alias"] == "mar")

    assert mar["status"] == "community_non_ema_audit_completed_official_ema_pending"
    assert mar["sample_count"] == 50_000
    assert mar["fid"] == 2.31
    assert mar["inception_score"] == 281.7
    assert mar["npz"] == str(mar_npz)
    assert mar["source_kind"] == "community_hf_safetensors_non_ema"


def test_build_rows_prefers_completed_official_mar_ema(tmp_path: Path, monkeypatch) -> None:
    module = load_module()
    monkeypatch.setattr(module, "MAR_OFFICIAL_SAMPLE_STEM", "official-ema")
    official_root = (
        tmp_path
        / "checkpoints/baselines/mar/official_imagenet256_eval_only/samples_50k_official_pth"
    )
    official_dir = official_root / module.MAR_OFFICIAL_SAMPLE_STEM
    official_dir.mkdir(parents=True)
    official_npz = official_root / f"{module.MAR_OFFICIAL_SAMPLE_STEM}.npz"
    official_txt = official_root / f"{module.MAR_OFFICIAL_SAMPLE_STEM}.txt"
    official_npz.write_bytes(b"npz")
    official_txt.write_text(
        "Inception Score: 281.7\n"
        "FID: 2.31\n"
        "sFID: 5.0\n"
        "Precision: 0.8\n"
        "Recall: 0.6\n",
        encoding="utf-8",
    )
    (official_dir / "baseline_eval_report.json").write_text(
        __import__("json").dumps(
            {
                "protocol": "official_lth14_pth_ema_eval_only",
                "status": "samples_and_npz_completed_eval_pending",
                "parameters": {
                    "num_images": 50_000,
                    "class_num": 1_000,
                    "balanced_images_per_class": 50,
                    "num_ar_steps": 256,
                    "num_sampling_steps": 100,
                    "cfg_scale": 2.9,
                    "cfg_schedule": "linear",
                    "temperature": 1.0,
                    "seed": 0,
                },
                "artifacts": {
                    "weights": {
                        "model": {
                            "checkpoint_state": "model_ema",
                            "sha256": module.MAR_MODEL_SHA256,
                        },
                        "vae": {
                            "checkpoint_state": "model",
                            "sha256": module.MAR_VAE_SHA256,
                        },
                        "official_repo": {"commit": module.MAR_REPO_COMMIT},
                    },
                    "sample_npz": {
                        "shape": [50_000, 256, 256, 3],
                        "dtype": "uint8",
                        "key": "arr_0",
                    },
                },
                "progress": {
                    "completed_images": 50_000,
                    "expected_images": 50_000,
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        module,
        "count_pngs",
        lambda path: 50_000 if path == official_dir else 0,
    )
    monkeypatch.setattr(module, "adm_npz_is_valid", lambda path: path == official_npz)
    rows = module.build_rows(tmp_path)
    mar = next(row for row in rows if row["alias"] == "mar")

    assert mar["status"] == "completed_eval_only_50k"
    assert mar["source_kind"] == "official_lth14_pth_model_ema"
    assert mar["protocol"] == "official LTH14 PTH model_ema eval-only"
    assert mar["npz"] == str(official_npz)


def test_parse_metric_txt_requires_numeric_values(tmp_path: Path) -> None:
    module = load_module()
    path = tmp_path / "metrics.txt"
    path.write_text("FID: 2.5\nRecall: bad\nIgnored: 7\n", encoding="utf-8")

    assert module.parse_metric_txt(path) == {"fid": 2.5}


def test_complete_metric_set_requires_all_five_finite_values() -> None:
    module = load_module()
    complete = {
        "fid": 2.5,
        "sfid": 5.0,
        "inception_score": 280.0,
        "precision": 0.8,
        "recall": 0.6,
    }

    assert module.complete_metric_set(complete)
    assert not module.complete_metric_set({"fid": 2.5})
    assert not module.complete_metric_set({**complete, "fid": float("nan")})


def test_npz_header_validation_does_not_require_loading_payload(tmp_path: Path) -> None:
    module = load_module()
    path = tmp_path / "samples.npz"
    np.savez(path, arr_0=np.zeros((2, 4, 4, 3), dtype=np.uint8))

    assert module.adm_npz_is_valid(path, expected_shape=(2, 4, 4, 3))
    assert not module.adm_npz_is_valid(path, expected_shape=(3, 4, 4, 3))


def test_incomplete_50k_rows_requires_each_official_method() -> None:
    module = load_module()
    rows = [
        {"method": "D-AR", "status": "completed_eval_only_50k"},
        {"method": "MAR", "status": "official_ema_sampling_or_eval_pending"},
        {"method": "ReTok", "status": "completed_eval_only_50k"},
    ]

    assert module.incomplete_50k_rows(rows) == [
        "MAR=official_ema_sampling_or_eval_pending"
    ]
