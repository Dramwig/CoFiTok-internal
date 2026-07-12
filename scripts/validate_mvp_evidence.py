from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DATASETS = {"cifar10", "tiny_imagenet_200", "imagenet_1k_64x64_hf"}
MAIN_DATASETS = {"tiny_imagenet_200", "imagenet_1k_64x64_hf"}
ORDERS = {"ordered", "random", "reverse"}


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    evidence: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate that CoFiTok MVP evidence covers the core idea requirements.")
    parser.add_argument(
        "--summary",
        default="artifacts/reports/summary_2026-07-08/experiment_summary.json",
        help="Experiment summary JSON produced by summarize_experiments.py.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _rows(summary: dict[str, Any], key: str) -> list[dict[str, Any]]:
    rows = summary.get(key, [])
    if not isinstance(rows, list):
        raise AssertionError(f"summary[{key!r}] is not a list")
    return rows


def _require(condition: bool, name: str, evidence: str) -> CheckResult:
    if not condition:
        raise AssertionError(f"{name}: {evidence}")
    return CheckResult(name=name, status="ok", evidence=evidence)


def _find_rows(rows: Iterable[dict[str, Any]], **criteria: Any) -> list[dict[str, Any]]:
    matched = []
    for row in rows:
        ok = True
        for key, value in criteria.items():
            if row.get(key) != value:
                ok = False
                break
        if ok:
            matched.append(row)
    return matched


def _one_row(rows: Iterable[dict[str, Any]], **criteria: Any) -> dict[str, Any]:
    matched = _find_rows(rows, **criteria)
    if not matched:
        raise AssertionError(f"missing row matching {criteria}")
    return matched[0]


def _has_number(row: dict[str, Any], key: str) -> bool:
    return isinstance(row.get(key), (int, float))


def check_dataset_coverage(summary: dict[str, Any]) -> CheckResult:
    train_datasets = {str(row.get("dataset")) for row in _rows(summary, "train")}
    missing = DATASETS - train_datasets
    return _require(not missing, "dataset_coverage", f"train datasets include {sorted(DATASETS)}")


def check_main_20k_tradeoff(summary: dict[str, Any]) -> CheckResult:
    train = _rows(summary, "train")
    for dataset in sorted(MAIN_DATASETS):
        epsilon = _one_row(train, dataset=dataset, variant="epsilon_only", token_count=8, steps=20000)
        light = _one_row(train, dataset=dataset, variant="light_denoise_path", token_count=8, steps=20000)
        for row in (epsilon, light):
            for key in ["final_clean_mse", "path_auc", "effective_tokens", "zero_token_ratio"]:
                if not _has_number(row, key):
                    raise AssertionError(f"missing numeric {key} for {dataset}/{row.get('variant')}")
        if light["path_auc"] >= epsilon["path_auc"]:
            raise AssertionError(f"{dataset}: light path_auc should be lower than epsilon-only")
        if abs(float(light["final_clean_mse"]) - float(epsilon["final_clean_mse"])) > 0.02:
            raise AssertionError(f"{dataset}: 20k endpoint MSE gap is too large for MVP claim")
    return CheckResult("main_20k_tradeoff", "ok", "Tiny/ImageNet 20k epsilon vs K8 light rows exist and support prefix tradeoff")


def check_order_scaling(summary: dict[str, Any]) -> CheckResult:
    order_rows = _rows(summary, "order_eval")
    for token_count in [4, 8, 16]:
        for seed in [103, 139]:
            rows_by_order = {
                order: _one_row(
                    order_rows,
                    dataset="imagenet_1k_64x64_hf",
                    variant="light_denoise_path",
                    token_count=token_count,
                    seed=seed,
                    component_order=order,
                )
                for order in ORDERS
            }
            ordered = float(rows_by_order["ordered"]["path_auc"])
            random = float(rows_by_order["random"]["path_auc"])
            reverse = float(rows_by_order["reverse"]["path_auc"])
            if random <= ordered:
                raise AssertionError(f"K{token_count} seed {seed}: random order does not worsen path AUC")
            if reverse <= ordered + 0.1:
                raise AssertionError(f"K{token_count} seed {seed}: reverse order does not strongly worsen path AUC")
            if float(rows_by_order["ordered"]["zero_token_ratio"]) != 0.0:
                raise AssertionError(f"K{token_count} seed {seed}: restricted ordered zero ratio is nonzero")
    return CheckResult("order_scaling", "ok", "K4/K8/K16 ordered/random/reverse order evals exist for seeds 103 and 139")


def check_deep_and_simultaneous(summary: dict[str, Any]) -> CheckResult:
    train = _rows(summary, "train")
    order_rows = _rows(summary, "order_eval")
    for seed in [103, 139]:
        deep = _one_row(
            train,
            dataset="imagenet_1k_64x64_hf",
            variant="deep_synthesis_ablation",
            token_count=8,
            seed=seed,
        )
        if float(deep.get("zero_token_ratio", 0.0)) <= 0.01:
            raise AssertionError(f"deep S_k seed {seed}: zero-token ratio is not a degeneration signal")

        sim_orders = {
            order: _one_row(
                order_rows,
                dataset="imagenet_1k_64x64_hf",
                variant="simultaneous_predictor",
                token_count=8,
                seed=seed,
                component_order=order,
            )
            for order in ORDERS
        }
        ordered = float(sim_orders["ordered"]["path_auc"])
        if float(sim_orders["random"]["path_auc"]) <= ordered or float(sim_orders["reverse"]["path_auc"]) <= ordered:
            raise AssertionError(f"simultaneous seed {seed}: order ablation does not worsen path AUC")
    return CheckResult("deep_and_simultaneous", "ok", "Deep S_k and simultaneous ablations have seed repeats and diagnostics")


def check_quality_seed_slice(summary: dict[str, Any]) -> CheckResult:
    quality = _rows(summary, "quality")
    required = [
        ("epsilon_only", 8),
        ("light_denoise_path", 4),
        ("light_denoise_path", 8),
        ("light_denoise_path", 16),
        ("simultaneous_predictor", 8),
        ("deep_synthesis_ablation", 8),
    ]
    for variant, token_count in required:
        row = _one_row(
            quality,
            dataset="imagenet_1k_64x64_hf",
            variant=variant,
            token_count=token_count,
            seed=103,
        )
        for key in ["final_mse", "final_psnr_db", "final_inception_frechet", "final_lpips_alex", "mse_auc"]:
            if not _has_number(row, key):
                raise AssertionError(f"quality seed slice missing numeric {key} for {variant}/K{token_count}")
        if row.get("lpips_available") is not True or row.get("inception_available") is not True:
            raise AssertionError(f"quality seed slice optional metrics unavailable for {variant}/K{token_count}")
    return CheckResult("quality_seed_slice", "ok", "Seed-103 ImageNet quality slice covers endpoint and prefix metrics")


def check_sampling_and_generated_quality(summary: dict[str, Any]) -> CheckResult:
    sampling = _rows(summary, "sampling")
    generated = _rows(summary, "generated_quality")
    sampling_datasets = {str(row.get("dataset")) for row in sampling if row.get("variant") == "light_denoise_path"}
    missing_sampling = DATASETS - sampling_datasets
    if missing_sampling:
        raise AssertionError(f"missing light-denoise sampling rows for {sorted(missing_sampling)}")

    for dataset in sorted(MAIN_DATASETS):
        for variant in ["epsilon_only", "light_denoise_path"]:
            row = _one_row(
                generated,
                dataset=dataset,
                variant=variant,
                token_count=8,
                steps=20000,
                sample_image_count=1024,
                real_image_count=4096,
            )
            if not _has_number(row, "inception_frechet") or not _has_number(row, "lowres_frechet_proxy"):
                raise AssertionError(f"generated quality metrics missing for {dataset}/{variant}")
    return CheckResult("sampling_and_generated_quality", "ok", "Sampling smoke and 1024-sample generated quality rows exist")


def validate_evidence(summary_path: Path) -> dict[str, Any]:
    summary = _read_json(summary_path)
    checks = [
        check_dataset_coverage(summary),
        check_main_20k_tradeoff(summary),
        check_order_scaling(summary),
        check_deep_and_simultaneous(summary),
        check_quality_seed_slice(summary),
        check_sampling_and_generated_quality(summary),
    ]
    return {
        "summary": summary_path.as_posix(),
        "status": "ok",
        "check_count": len(checks),
        "checks": [check.__dict__ for check in checks],
    }


def main() -> None:
    args = parse_args()
    result = validate_evidence(Path(args.summary))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
