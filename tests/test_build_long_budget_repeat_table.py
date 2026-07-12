import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "build_long_budget_repeat_table.py"


def load_module():
    spec = importlib.util.spec_from_file_location("build_long_budget_repeat_table", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_report(
    path: Path,
    dataset: str,
    seed: int,
    mse: float,
    auc: float,
    method: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "data": {"dataset": dataset},
            "runtime": {"seed": seed, "steps": 20_000},
            "model": {"token_count": 8},
            "loss": {
                "denoise_path_prefix_weight": 1.5 if method == "cofitok" else 0.0
            },
        },
        "prefix_summary": {
            "prefix_mse_to_clean": [1.0, mse],
            "prefix_mse_to_denoise_path_auc": auc,
            "energy_effective_token_count": 6.5,
            "diagnostics": {"zero_token_component_energy_ratio": 0.0},
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_quality_report(
    path: Path,
    dataset: str,
    seed: int,
    mse: float,
    auc: float,
    method: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "data": {"dataset": dataset},
            "runtime": {"seed": seed, "steps": 20_000},
            "model": {"token_count": 8},
            "loss": {
                "denoise_path_prefix_weight": 1.5 if method == "cofitok" else 0.0
            },
        },
        "checkpoint": str(path.parent / "checkpoint_final.pt"),
        "evaluation": {
            "image_count": 1_024,
            "fixed_timestep": 500,
            "prefix_budgets": list(range(1, 9)),
            "component_order": "ordered",
            "random_order_seed": 0,
        },
        "metrics_by_prefix": {"8": {"mse": mse}},
        "curve_auc": {"denoise_path_mse": auc},
        "component_energy": {"energy_effective_token_count": 6.5},
        "diagnostics": {"zero_token_component_energy_ratio": 0.0},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_uses_paired_seeds(monkeypatch, tmp_path: Path) -> None:
    module = load_module()
    specs = []
    for dataset in ["a", "b"]:
        for method in ["endpoint_only_factorized", "cofitok"]:
            reports = []
            quality_reports = []
            for seed in [103, 139]:
                relative = f"{dataset}_{method}_{seed}/report.json"
                quality_relative = f"{dataset}_{method}_{seed}/quality_report.json"
                reports.append(relative)
                quality_reports.append(quality_relative)
                mse = 1.0 if method == "endpoint_only_factorized" else 1.02
                auc = 1.0 if method == "endpoint_only_factorized" else 0.04
                write_report(tmp_path / relative, dataset, seed, mse, auc, method)
                write_quality_report(
                    tmp_path / quality_relative,
                    dataset,
                    seed,
                    mse,
                    auc,
                    method,
                )
            specs.append(
                {
                    "dataset": dataset,
                    "dataset_display": dataset,
                    "method": method,
                    "method_display": method,
                    "reports": reports,
                    "quality_reports": quality_reports,
                }
            )
    monkeypatch.setattr(module, "REPORT_SPECS", specs)

    payload = module.build(tmp_path)

    assert payload["summary"]["pair_count"] == 4
    assert payload["summary"]["path_auc_better_pairs"] == 4
    assert payload["summary"]["final_mse_better_pairs"] == 0
    assert payload["summary"]["mean_path_auc_relative_reduction"] == 0.96
    assert round(payload["summary"]["mean_final_mse_relative_change"], 6) == 0.02
