from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


MAIN_DATASETS = ("tiny_imagenet_200", "imagenet_1k_64x64_hf")
MAIN_VARIANTS = ("epsilon_only", "light_denoise_path")
ORDER_VARIANT = "light_denoise_path"
LOSS_ABLATION_VARIANTS = ("no_prefix_loss_ablation", "clean_monotonic_ablation")
ORDERS = ("ordered", "random", "reverse")


@dataclass(frozen=True)
class ReadinessCriteria:
    min_main_train_seeds: int = 2
    main_steps: int = 20_000
    multiscale_steps: int = 10_000
    token_count: int = 8
    min_quality_images: int = 1024
    min_generated_samples: int = 2048
    min_real_images: int = 8192
    min_sample_steps: int = 50


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    evidence: dict[str, Any]
    missing: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate whether the CoFiTok summary has enough evidence for "
            "publication-facing claims, beyond the MVP smoke gates."
        )
    )
    parser.add_argument(
        "--summary",
        default="artifacts/reports/summary_2026-07-08/experiment_summary.json",
        help="Experiment summary JSON produced by summarize_experiments.py.",
    )
    parser.add_argument("--min-main-train-seeds", type=int, default=2)
    parser.add_argument("--min-quality-images", type=int, default=1024)
    parser.add_argument("--min-generated-samples", type=int, default=2048)
    parser.add_argument("--min-real-images", type=int, default=8192)
    parser.add_argument("--min-sample-steps", type=int, default=50)
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="Exit nonzero when any publication-readiness criterion is missing.",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional path for writing the full readiness result JSON.",
    )
    parser.add_argument(
        "--output-md",
        default="",
        help="Optional path for writing a compact Markdown gap report.",
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


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _at_least(value: Any, threshold: int | float) -> bool:
    return _number(value) and float(value) >= float(threshold)


def _equals_number(value: Any, expected: int | float) -> bool:
    return _number(value) and float(value) == float(expected)


def _predictor_type(row: dict[str, Any]) -> str:
    return str(row.get("predictor_type") or "tiny_conv")


def _is_multiscale(row: dict[str, Any]) -> bool:
    return (
        _predictor_type(row) == "multiscale_unet"
        or "multiscale" in str(row.get("config_name", "")).lower()
        or "multiscale" in str(row.get("report_dir", "")).lower()
    )


def _matches(row: dict[str, Any], **criteria: Any) -> bool:
    for key, expected in criteria.items():
        if row.get(key) != expected:
            return False
    return True


def _select(rows: Iterable[dict[str, Any]], **criteria: Any) -> list[dict[str, Any]]:
    return [row for row in rows if _matches(row, **criteria)]


def _best_count(rows: Iterable[dict[str, Any]], key: str) -> int | None:
    values = [int(row[key]) for row in rows if _number(row.get(key))]
    return max(values) if values else None


def _metric_ready(row: dict[str, Any], metric: str) -> bool:
    return _number(row.get(metric))


def _optional_metric_ready(row: dict[str, Any], available_key: str, metric: str) -> bool:
    return row.get(available_key) is True and _metric_ready(row, metric)


def _ok(name: str, evidence: dict[str, Any], missing: list[str]) -> CheckResult:
    return CheckResult(
        name=name,
        status="ok" if not missing else "missing",
        evidence=evidence,
        missing=missing,
    )


def check_main_train_seed_coverage(summary: dict[str, Any], criteria: ReadinessCriteria) -> CheckResult:
    train = _rows(summary, "train")
    evidence: dict[str, Any] = {}
    missing = []
    for dataset in MAIN_DATASETS:
        for variant in MAIN_VARIANTS:
            rows = [
                row
                for row in train
                if row.get("dataset") == dataset
                and row.get("variant") == variant
                and row.get("token_count") == criteria.token_count
                and row.get("steps") == criteria.main_steps
                and _predictor_type(row) == "tiny_conv"
            ]
            seeds = sorted({row.get("seed") for row in rows if row.get("seed") is not None})
            key = f"{dataset}/{variant}"
            evidence[key] = {"seed_count": len(seeds), "seeds": seeds}
            if len(seeds) < criteria.min_main_train_seeds:
                missing.append(
                    f"{key}: need {criteria.min_main_train_seeds} train seeds at {criteria.main_steps} steps, found {len(seeds)}"
                )
    return _ok("main_train_seed_coverage", evidence, missing)


def check_quality_scale(summary: dict[str, Any], criteria: ReadinessCriteria) -> CheckResult:
    quality = _rows(summary, "quality")
    evidence: dict[str, Any] = {}
    missing = []
    for dataset in MAIN_DATASETS:
        for variant in MAIN_VARIANTS:
            rows = [
                row
                for row in quality
                if row.get("dataset") == dataset
                and row.get("variant") == variant
                and row.get("token_count") == criteria.token_count
                and row.get("steps") == criteria.main_steps
                and _predictor_type(row) == "tiny_conv"
                and _at_least(row.get("image_count"), criteria.min_quality_images)
                and _optional_metric_ready(row, "inception_available", "final_inception_frechet")
                and _optional_metric_ready(row, "lpips_available", "final_lpips_alex")
            ]
            key = f"{dataset}/{variant}"
            evidence[key] = {
                "matching_rows": len(rows),
                "max_image_count": _best_count(_select(quality, dataset=dataset, variant=variant), "image_count"),
            }
            if not rows:
                missing.append(
                    f"{key}: need >= {criteria.min_quality_images} quality images with Inception and LPIPS"
                )
    return _ok("quality_scale", evidence, missing)


def check_20k_order_diagnostics(summary: dict[str, Any], criteria: ReadinessCriteria) -> CheckResult:
    order_rows = _rows(summary, "order_eval")
    evidence: dict[str, Any] = {}
    missing = []
    for dataset in MAIN_DATASETS:
        dataset_rows = [
            row
            for row in order_rows
            if row.get("dataset") == dataset
            and row.get("variant") == ORDER_VARIANT
            and row.get("token_count") == criteria.token_count
            and row.get("steps") == criteria.main_steps
            and _predictor_type(row) == "tiny_conv"
        ]
        complete_seed: int | None = None
        auc_by_order: dict[str, float] = {}
        for seed in sorted({row.get("seed") for row in dataset_rows if row.get("seed") is not None}):
            rows_by_order = {
                order: [row for row in dataset_rows if row.get("seed") == seed and row.get("component_order") == order]
                for order in ORDERS
            }
            if all(rows_by_order.values()):
                candidate = {order: float(rows_by_order[order][0]["path_auc"]) for order in ORDERS}
                if candidate["random"] > candidate["ordered"] and candidate["reverse"] > candidate["ordered"]:
                    complete_seed = int(seed)
                    auc_by_order = candidate
                    break
        evidence[dataset] = {"complete_seed": complete_seed, "path_auc": auc_by_order}
        if complete_seed is None:
            missing.append(
                f"{dataset}: need ordered/random/reverse 20k K8 order evals where random and reverse worsen path AUC"
            )
    return _ok("main_20k_order_diagnostics", evidence, missing)


def check_sampling_scale(summary: dict[str, Any], criteria: ReadinessCriteria) -> CheckResult:
    sampling = _rows(summary, "sampling")
    evidence: dict[str, Any] = {}
    missing = []
    for dataset in MAIN_DATASETS:
        for variant in MAIN_VARIANTS:
            rows = [
                row
                for row in sampling
                if row.get("dataset") == dataset
                and row.get("variant") == variant
                and row.get("token_count") == criteria.token_count
                and row.get("steps") == criteria.main_steps
                and _predictor_type(row) == "tiny_conv"
                and _at_least(row.get("num_samples"), criteria.min_generated_samples)
                and _at_least(row.get("sample_steps"), criteria.min_sample_steps)
            ]
            key = f"{dataset}/{variant}"
            evidence[key] = {
                "matching_rows": len(rows),
                "max_num_samples": _best_count(_select(sampling, dataset=dataset, variant=variant), "num_samples"),
                "max_sample_steps": _best_count(_select(sampling, dataset=dataset, variant=variant), "sample_steps"),
            }
            if not rows:
                missing.append(
                    f"{key}: need >= {criteria.min_generated_samples} DDIM samples with >= {criteria.min_sample_steps} steps"
                )
    return _ok("sampling_scale", evidence, missing)


def check_generated_quality_scale(summary: dict[str, Any], criteria: ReadinessCriteria) -> CheckResult:
    generated = _rows(summary, "generated_quality")
    evidence: dict[str, Any] = {}
    missing = []
    for dataset in MAIN_DATASETS:
        for variant in MAIN_VARIANTS:
            rows = [
                row
                for row in generated
                if row.get("dataset") == dataset
                and row.get("variant") == variant
                and row.get("token_count") == criteria.token_count
                and row.get("steps") == criteria.main_steps
                and _at_least(row.get("sample_image_count"), criteria.min_generated_samples)
                and _at_least(row.get("real_image_count"), criteria.min_real_images)
                and _optional_metric_ready(row, "inception_available", "inception_frechet")
                and _metric_ready(row, "lowres_frechet_proxy")
            ]
            key = f"{dataset}/{variant}"
            evidence[key] = {
                "matching_rows": len(rows),
                "max_sample_image_count": _best_count(_select(generated, dataset=dataset, variant=variant), "sample_image_count"),
                "max_real_image_count": _best_count(_select(generated, dataset=dataset, variant=variant), "real_image_count"),
            }
            if not rows:
                missing.append(
                    f"{key}: need generated quality with >= {criteria.min_generated_samples} samples, "
                    f">= {criteria.min_real_images} real images, and Inception enabled"
                )
    return _ok("generated_quality_scale", evidence, missing)


def check_multiscale_pilots(summary: dict[str, Any], criteria: ReadinessCriteria) -> CheckResult:
    train = _rows(summary, "train")
    quality = _rows(summary, "quality")
    sampling = _rows(summary, "sampling")
    generated = _rows(summary, "generated_quality")
    evidence: dict[str, Any] = {}
    missing = []
    for dataset in MAIN_DATASETS:
        train_rows = [
            row
            for row in train
            if row.get("dataset") == dataset
            and row.get("variant") == ORDER_VARIANT
            and row.get("token_count") == criteria.token_count
            and _at_least(row.get("steps"), criteria.multiscale_steps)
            and _is_multiscale(row)
            and row.get("synthesis_mode", "restricted") == "restricted"
            and _equals_number(row.get("zero_token_ratio"), 0.0)
            and _metric_ready(row, "final_clean_mse")
            and _metric_ready(row, "path_auc")
        ]
        quality_rows = [
            row
            for row in quality
            if row.get("dataset") == dataset
            and row.get("variant") == ORDER_VARIANT
            and row.get("token_count") == criteria.token_count
            and _at_least(row.get("steps"), criteria.multiscale_steps)
            and _is_multiscale(row)
            and _at_least(row.get("image_count"), criteria.min_quality_images)
            and _optional_metric_ready(row, "inception_available", "final_inception_frechet")
            and _optional_metric_ready(row, "lpips_available", "final_lpips_alex")
        ]
        sampling_rows = [
            row
            for row in sampling
            if row.get("dataset") == dataset
            and row.get("variant") == ORDER_VARIANT
            and row.get("token_count") == criteria.token_count
            and _at_least(row.get("steps"), criteria.multiscale_steps)
            and _is_multiscale(row)
            and _at_least(row.get("num_samples"), criteria.min_generated_samples)
            and _at_least(row.get("sample_steps"), criteria.min_sample_steps)
        ]
        generated_rows = [
            row
            for row in generated
            if row.get("dataset") == dataset
            and row.get("variant") == ORDER_VARIANT
            and row.get("token_count") == criteria.token_count
            and _at_least(row.get("steps"), criteria.multiscale_steps)
            and _is_multiscale(row)
            and _at_least(row.get("sample_image_count"), criteria.min_generated_samples)
            and _at_least(row.get("real_image_count"), criteria.min_real_images)
            and _optional_metric_ready(row, "inception_available", "inception_frechet")
        ]
        evidence[dataset] = {
            "train_rows": len(train_rows),
            "quality_rows": len(quality_rows),
            "sampling_rows": len(sampling_rows),
            "generated_quality_rows": len(generated_rows),
        }
        if not train_rows:
            missing.append(f"{dataset}: missing trained restricted multiscale_unet pilot")
        if not quality_rows:
            missing.append(f"{dataset}: missing multiscale_unet quality slice")
        if not sampling_rows:
            missing.append(f"{dataset}: missing multiscale_unet DDIM sample set")
        if not generated_rows:
            missing.append(f"{dataset}: missing multiscale_unet generated-quality evaluation")
    return _ok("multiscale_pilots", evidence, missing)


def check_loss_ablation_pilots(summary: dict[str, Any], criteria: ReadinessCriteria) -> CheckResult:
    train = _rows(summary, "train")
    quality = _rows(summary, "quality")
    order_rows = _rows(summary, "order_eval")
    evidence: dict[str, Any] = {}
    missing = []
    for dataset in MAIN_DATASETS:
        for variant in LOSS_ABLATION_VARIANTS:
            train_rows = [
                row
                for row in train
                if row.get("dataset") == dataset
                and row.get("variant") == variant
                and row.get("token_count") == criteria.token_count
                and _at_least(row.get("steps"), criteria.multiscale_steps)
                and _predictor_type(row) == "tiny_conv"
                and row.get("synthesis_mode", "restricted") == "restricted"
                and _equals_number(row.get("zero_token_ratio"), 0.0)
                and _metric_ready(row, "final_clean_mse")
                and _metric_ready(row, "path_auc")
            ]
            quality_rows = [
                row
                for row in quality
                if row.get("dataset") == dataset
                and row.get("variant") == variant
                and row.get("token_count") == criteria.token_count
                and _at_least(row.get("steps"), criteria.multiscale_steps)
                and _predictor_type(row) == "tiny_conv"
                and _at_least(row.get("image_count"), criteria.min_quality_images)
                and _optional_metric_ready(row, "inception_available", "final_inception_frechet")
                and _optional_metric_ready(row, "lpips_available", "final_lpips_alex")
            ]
            orders = {
                str(row.get("component_order"))
                for row in order_rows
                if row.get("dataset") == dataset
                and row.get("variant") == variant
                and row.get("token_count") == criteria.token_count
                and _at_least(row.get("steps"), criteria.multiscale_steps)
                and _predictor_type(row) == "tiny_conv"
                and _metric_ready(row, "path_auc")
            }
            key = f"{dataset}/{variant}"
            evidence[key] = {
                "train_rows": len(train_rows),
                "quality_rows": len(quality_rows),
                "orders": sorted(orders),
            }
            if not train_rows:
                missing.append(f"{key}: missing trained restricted loss-ablation pilot")
            if not quality_rows:
                missing.append(f"{key}: missing loss-ablation quality slice")
            if not set(ORDERS).issubset(orders):
                missing.append(f"{key}: missing ordered/random/reverse loss-ablation order diagnostics")

        no_monotonic_reference = [
            row
            for row in train
            if row.get("dataset") == dataset
            and row.get("variant") == ORDER_VARIANT
            and row.get("token_count") == criteria.token_count
            and _at_least(row.get("steps"), criteria.main_steps)
            and _predictor_type(row) == "tiny_conv"
        ]
        evidence[f"{dataset}/no_monotonic_reference"] = {"train_rows": len(no_monotonic_reference)}
        if not no_monotonic_reference:
            missing.append(f"{dataset}: missing light denoise-path no-monotonic reference")
    return _ok("loss_ablation_pilots", evidence, missing)


def validate_readiness(summary_path: Path, criteria: ReadinessCriteria | None = None) -> dict[str, Any]:
    criteria = criteria or ReadinessCriteria()
    summary = _read_json(summary_path)
    checks = [
        check_main_train_seed_coverage(summary, criteria),
        check_quality_scale(summary, criteria),
        check_20k_order_diagnostics(summary, criteria),
        check_sampling_scale(summary, criteria),
        check_generated_quality_scale(summary, criteria),
        check_multiscale_pilots(summary, criteria),
        check_loss_ablation_pilots(summary, criteria),
    ]
    missing = [check for check in checks if check.status != "ok"]
    return {
        "summary": summary_path.as_posix(),
        "status": "ready" if not missing else "not_ready",
        "criteria": asdict(criteria),
        "check_count": len(checks),
        "ready_count": len(checks) - len(missing),
        "missing_count": len(missing),
        "checks": [asdict(check) for check in checks],
    }


def _inline_evidence(value: Any) -> str:
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            if isinstance(item, (str, int, float, bool)) or item is None:
                parts.append(f"{key}={item}")
            elif isinstance(item, list):
                parts.append(f"{key}={len(item)} items")
            elif isinstance(item, dict):
                parts.append(f"{key}={len(item)} keys")
        return "<br>".join(parts) or "see JSON"
    if isinstance(value, list):
        return f"{len(value)} items"
    return str(value)


def render_markdown(result: dict[str, Any]) -> str:
    criteria = result["criteria"]
    lines = [
        "# CoFiTok Publication-Readiness Gap Report",
        "",
        f"Status: `{result['status']}`",
        "",
        "This gate is intentionally stricter than the MVP evidence gate. It tracks whether current local and remote summaries support publication-facing claims.",
        "",
        "## Criteria",
        "",
        "| criterion | value |",
        "|---|---:|",
    ]
    for key in sorted(criteria):
        lines.append(f"| `{key}` | {criteria[key]} |")

    lines.extend(
        [
            "",
            "## Checks",
            "",
            "| check | status | missing | evidence |",
            "|---|---|---|---|",
        ]
    )
    for check in result["checks"]:
        missing = "<br>".join(check["missing"]) if check["missing"] else "none"
        evidence_bits = []
        for key, value in check["evidence"].items():
            evidence_bits.append(f"{key}: {_inline_evidence(value)}")
        evidence = "<br>".join(evidence_bits[:8]) or "see JSON"
        lines.append(f"| `{check['name']}` | `{check['status']}` | {missing} | {evidence} |")

    if result["status"] != "ready":
        lines.extend(
            [
                "",
                "## Next Action",
                "",
                "Run `artifacts/runbooks/next_validation_queue_2026-07-08.sh` on `pro6000`, regenerate `artifacts/reports/summary_2026-07-08/experiment_summary.json`, then rerun this gate with `--require-ready`.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Next Action",
                "",
                "The publication-readiness gate is satisfied. Freeze the summary, figures, and paper-facing tables before changing claims.",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    criteria = ReadinessCriteria(
        min_main_train_seeds=args.min_main_train_seeds,
        min_quality_images=args.min_quality_images,
        min_generated_samples=args.min_generated_samples,
        min_real_images=args.min_real_images,
        min_sample_steps=args.min_sample_steps,
    )
    result = validate_readiness(Path(args.summary), criteria)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_md:
        Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_md).write_text(render_markdown(result), encoding="utf-8")
    if args.require_ready and result["status"] != "ready":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
