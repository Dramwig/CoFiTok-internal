from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DATASETS = [
    {"dataset": "cifar10", "steps": 3000, "token_count": 8, "protocol": "32x32", "quality_count": 512, "sample_count": 1024, "real_count": 4096},
    {"dataset": "tiny_imagenet_200", "steps": 5000, "token_count": 8, "protocol": "64x64", "quality_count": 512, "sample_count": 1024, "real_count": 4096},
    {"dataset": "imagenet_1k_64x64_hf", "steps": 5000, "token_count": 8, "protocol": "64x64 HF", "quality_count": 512, "sample_count": 1024, "real_count": 4096},
    {"dataset": "downsampled_imagenet_64", "steps": 5000, "token_count": 8, "protocol": "64x64 strict", "quality_count": 512, "sample_count": 1024, "real_count": 4096},
    {"dataset": "ffhq_64", "steps": 5000, "token_count": 8, "protocol": "64x64", "quality_count": 512, "sample_count": 1024, "real_count": 4096},
    {"dataset": "afhqv2_64", "steps": 5000, "token_count": 8, "protocol": "64x64", "quality_count": 512, "sample_count": 1024, "real_count": 4096},
    {"dataset": "imagenet_256_10pct", "steps": 5000, "token_count": 4, "protocol": "256x256 10pct", "quality_count": 256, "sample_count": 512, "real_count": 2048},
    {"dataset": "imagenet_256", "steps": 5000, "token_count": 4, "protocol": "256x256 full", "quality_count": 256, "sample_count": 512, "real_count": 2048},
]

INTERNAL_METHODS = [
    ("cofitok_light", "CoFiTok", "light_denoise_path", ["train", "quality", "generated_quality"]),
    ("same_backbone_dense", "Direct dense epsilon", "dense_monolithic", ["train", "quality", "generated_quality"]),
    ("endpoint_only_factorized", "Endpoint-only factorized", "epsilon_only", ["train", "quality", "generated_quality"]),
    ("channel_mask", "Legacy clean-prefix objective", "channel_mask", ["train", "quality"]),
    ("cofitok_no_prefix_loss", "No denoise-path prefix term", "no_prefix_loss_ablation", ["train", "quality"]),
    # Keep the historical alias for matrix compatibility; the variant adds this term.
    ("cofitok_no_monotonic_loss", "Add clean-space monotonic term", "clean_monotonic_ablation", ["train", "quality"]),
    ("cofitok_simultaneous", "Simultaneous", "simultaneous_predictor", ["train", "quality"]),
    ("cofitok_deep_synthesis", "Deep S_k", "deep_synthesis_ablation", ["train", "quality"]),
]

EXTERNAL_METHODS = [
    ("improved_diffusion", "Improved DDPM"),
    ("edm", "EDM"),
    ("d_ar", "D-AR"),
]
EXTERNAL_NFE = {"improved_diffusion": 50, "edm": 79}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the current P0 method x dataset comparison table.")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument(
        "--dar-feasibility",
        type=Path,
        default=Path("artifacts/reports/baselines/d_ar/adapter_feasibility_2026-07-09.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--require-complete-primary",
        action="store_true",
        help="Fail unless all internal, Improved DDPM, and EDM cells are complete.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def as_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _canonical_config(row: dict[str, Any]) -> bool:
    name = str(row.get("config_name", "")).lower()
    return "_seed" not in name and "multiscale" not in name


def select_internal(
    rows: list[dict[str, Any]],
    dataset: str,
    variant: str,
    steps: int,
    token_count: int,
    *,
    section: str = "train",
    quality_count: int | None = None,
    sample_count: int | None = None,
    real_count: int | None = None,
) -> dict[str, Any]:
    expected_token_count = 1 if variant == "dense_monolithic" else token_count
    candidates = [
        row
        for row in rows
        if row.get("dataset") == dataset
        and row.get("variant") == variant
        and as_int(row.get("steps")) == steps
        and as_int(row.get("token_count")) in {None, expected_token_count}
        and _canonical_config(row)
    ]
    if section == "quality":
        candidates = [
            row
            for row in candidates
            if as_int(row.get("image_count")) == quality_count
            and as_int(row.get("fixed_timestep")) == 500
        ]
        if variant in {"light_denoise_path", "epsilon_only"}:
            expected_budgets = list(range(1, expected_token_count + 1))
            candidates = [
                row
                for row in candidates
                if row.get("prefix_budgets") == expected_budgets
                and row.get("denoise_path_mse_auc") is not None
                and row.get("component_order") == "ordered"
            ]
    elif section == "generated_quality":
        candidates = [
            row
            for row in candidates
            if as_int(row.get("sample_image_count")) == sample_count
            and as_int(row.get("real_image_count")) == real_count
            and as_int(row.get("sample_steps")) == 50
            and as_int(row.get("prefix_budget")) == expected_token_count
        ]
    if not candidates:
        return {}
    candidates.sort(key=lambda row: str(row.get("report_dir", "")))
    return candidates[-1]


def select_baseline(
    rows: list[dict[str, Any]],
    dataset: str,
    baseline: str,
    *,
    steps: int,
    sample_count: int,
    real_count: int,
) -> dict[str, Any]:
    candidates = [
        row
        for row in rows
        if row.get("dataset") == dataset
        and row.get("baseline") == baseline
        and row.get("status") == "completed"
        and row.get("eval_only") is not True
        and as_int(row.get("train_steps")) == steps
        and as_int(row.get("sample_image_count")) == sample_count
        and as_int(row.get("real_image_count")) == real_count
        and as_int(row.get("sampler_nfe")) == EXTERNAL_NFE.get(baseline)
    ]
    if not candidates:
        return {}
    candidates.sort(key=lambda row: str(row.get("run_name", "")))
    return candidates[-1]


def status(required: list[str], payload_by_section: dict[str, dict[str, Any]]) -> str:
    present = [section for section in required if payload_by_section.get(section)]
    if len(present) == len(required):
        return "completed"
    if present:
        return "partial"
    return "missing"


def evidence(required: list[str], payload_by_section: dict[str, dict[str, Any]]) -> str:
    return ",".join(section for section in required if payload_by_section.get(section))


def fmt(value: Any, digits: int = 4) -> str:
    if value in {None, ""}:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    if isinstance(value, int):
        return str(value)
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def build_rows(summary: dict[str, Any], baseline_summary: dict[str, Any], dar_feasibility: dict[str, Any] | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    train_rows = summary.get("train", [])
    quality_rows = summary.get("quality", [])
    generated_rows = summary.get("generated_quality", [])
    baseline_rows = baseline_summary.get("rows", [])

    for spec in DATASETS:
        dataset = str(spec["dataset"])
        target_steps = int(spec["steps"])
        token_count = int(spec["token_count"])
        protocol = str(spec["protocol"])
        quality_count = int(spec["quality_count"])
        sample_count = int(spec["sample_count"])
        real_count = int(spec["real_count"])

        for method, display_name, variant, required in INTERNAL_METHODS:
            train = select_internal(
                train_rows, dataset, variant, target_steps, token_count, section="train"
            )
            quality = select_internal(
                quality_rows,
                dataset,
                variant,
                target_steps,
                token_count,
                section="quality",
                quality_count=quality_count,
            )
            generated = select_internal(
                generated_rows,
                dataset,
                variant,
                target_steps,
                token_count,
                section="generated_quality",
                sample_count=sample_count,
                real_count=real_count,
            )
            payloads = {"train": train, "quality": quality, "generated_quality": generated}
            train_batch_size = as_int(train.get("train_batch_size"))
            rows.append(
                {
                    "dataset": dataset,
                    "protocol": protocol,
                    "train_steps": target_steps,
                    "train_batch_size": train_batch_size,
                    "nominal_train_images": (
                        target_steps * train_batch_size if train_batch_size is not None else None
                    ),
                    "method": method,
                    "display_name": display_name,
                    "status": status(required, payloads),
                    "evidence": evidence(required, payloads),
                    "sample_nfe": 50 if generated else None,
                    "quality_image_count": quality.get("image_count"),
                    "sample_image_count": generated.get("sample_image_count"),
                    "real_image_count": generated.get("real_image_count"),
                    "sample_lowres_frechet": generated.get("lowres_frechet_proxy"),
                    "sample_inception_frechet": generated.get("inception_frechet"),
                    "denoise_psnr": quality.get("final_psnr_db"),
                    "denoise_inception_frechet": quality.get("final_inception_frechet"),
                    "denoise_lpips": quality.get("final_lpips_alex"),
                    "path_auc": (
                        quality.get("denoise_path_mse_auc")
                        if variant in {"light_denoise_path", "epsilon_only"}
                        else None
                    ),
                    "effective_tokens": (
                        quality.get("energy_effective_token_count")
                        if quality.get("energy_effective_token_count") is not None
                        else train.get("effective_tokens")
                    ),
                    "zero_token_ratio": train.get("zero_token_ratio"),
                    "parameter_count": train.get("parameter_count"),
                    "train_report": train.get("report_path", train.get("report_dir", "")),
                    "quality_report": quality.get("report_path", quality.get("report_dir", "")),
                    "generated_report": generated.get("report_path", generated.get("report_dir", "")),
                }
            )

        for baseline, display_name in EXTERNAL_METHODS:
            row = select_baseline(
                baseline_rows,
                dataset,
                baseline,
                steps=target_steps,
                sample_count=sample_count,
                real_count=real_count,
            )
            if baseline == "d_ar":
                row_status = "protocol_blocked" if dar_feasibility else "missing"
                row_evidence = "feasibility_report" if dar_feasibility else ""
            else:
                row_status = "completed" if row else "missing"
                row_evidence = "baseline_train,baseline_eval" if row else ""
            train_batch_size = as_int(row.get("batch_size"))
            rows.append(
                {
                    "dataset": dataset,
                    "protocol": protocol,
                    "train_steps": target_steps,
                    "train_batch_size": train_batch_size,
                    "nominal_train_images": (
                        target_steps * train_batch_size if train_batch_size is not None else None
                    ),
                    "method": baseline,
                    "display_name": display_name,
                    "status": row_status,
                    "evidence": row_evidence,
                    "sample_nfe": row.get("sampler_nfe"),
                    "quality_image_count": None,
                    "sample_image_count": row.get("sample_image_count"),
                    "real_image_count": row.get("real_image_count"),
                    "sample_lowres_frechet": row.get("lowres_frechet_proxy"),
                    "sample_inception_frechet": row.get("inception_frechet"),
                    "denoise_psnr": None,
                    "denoise_inception_frechet": None,
                    "denoise_lpips": None,
                    "path_auc": None,
                    "effective_tokens": None,
                    "zero_token_ratio": None,
                    "parameter_count": row.get("parameter_count"),
                    "train_report": row.get("train_report", ""),
                    "quality_report": "",
                    "generated_report": row.get("eval_report", ""),
                }
            )
    return rows


def factorization_contract_violations(rows: list[dict[str, Any]]) -> list[str]:
    indexed = {(row["dataset"], row["method"]): row for row in rows}
    violations: list[str] = []
    for spec in DATASETS:
        dataset = str(spec["dataset"])
        cofitok = indexed[(dataset, "cofitok_light")]
        dense = indexed[(dataset, "same_backbone_dense")]
        endpoint = indexed[(dataset, "endpoint_only_factorized")]
        counts = {
            "cofitok": as_int(cofitok.get("parameter_count")),
            "dense": as_int(dense.get("parameter_count")),
            "endpoint": as_int(endpoint.get("parameter_count")),
        }
        if any(value is None for value in counts.values()):
            violations.append(f"{dataset}: missing factorization-control parameter count")
        else:
            if counts["endpoint"] != counts["cofitok"]:
                violations.append(
                    f"{dataset}: endpoint-only params {counts['endpoint']} != CoFiTok {counts['cofitok']}"
                )
            relative_gap = abs(counts["dense"] / counts["cofitok"] - 1.0)
            if relative_gap > 0.02:
                violations.append(
                    f"{dataset}: direct-dense parameter gap {relative_gap:.2%} exceeds 2%"
                )

        batches = {
            as_int(cofitok.get("train_batch_size")),
            as_int(dense.get("train_batch_size")),
            as_int(endpoint.get("train_batch_size")),
        }
        if None in batches or len(batches) != 1:
            violations.append(f"{dataset}: factorization-control train batches differ")
    return violations


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    headers = [
        "dataset",
        "protocol",
        "method",
        "status",
        "steps",
        "batch",
        "nominal train imgs",
        "NFE",
        "quality n",
        "sample/real n",
        "sample lowres",
        "sample Inception",
        "denoise PSNR",
        "denoise Inception",
        "LPIPS",
        "path AUC",
        "eff K",
        "zero ratio",
        "params",
        "evidence",
    ]
    lines = [
        "# P0 Current Comparison",
        "",
        "This table audits the locked matched-dataset/optimizer-step P0 evidence. Batch size, nominal images seen, parameter count, and NFE remain explicit because external methods are not compute matched. Lower Frechet-style metrics are better; this is not a formal SOTA FID table.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        values = [
            str(row["dataset"]),
            str(row["protocol"]),
            str(row["display_name"]),
            str(row["status"]),
            fmt(row.get("train_steps"), 0),
            fmt(row.get("train_batch_size"), 0),
            fmt(row.get("nominal_train_images"), 0),
            fmt(row.get("sample_nfe"), 0),
            fmt(row.get("quality_image_count"), 0),
            f"{fmt(row.get('sample_image_count'), 0)}/{fmt(row.get('real_image_count'), 0)}",
            fmt(row.get("sample_lowres_frechet")),
            fmt(row.get("sample_inception_frechet")),
            fmt(row.get("denoise_psnr"), 3),
            fmt(row.get("denoise_inception_frechet")),
            fmt(row.get("denoise_lpips"), 4),
            fmt(row.get("path_auc")),
            fmt(row.get("effective_tokens"), 3),
            fmt(row.get("zero_token_ratio")),
            fmt(row.get("parameter_count"), 0),
            str(row.get("evidence", "")),
        ]
        lines.append("| " + " | ".join(values) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    summary = read_json(args.summary)
    baseline_summary = read_json(args.baseline_summary)
    dar_feasibility = read_json(args.dar_feasibility) if args.dar_feasibility.exists() else None
    rows = build_rows(summary, baseline_summary, dar_feasibility)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "p0_paper_table.json").write_text(
        json.dumps({"row_count": len(rows), "rows": rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_dir / "p0_paper_table.csv", rows)
    write_markdown(args.output_dir / "p0_paper_table.md", rows)
    if args.require_complete_primary:
        primary_methods = {method[0] for method in INTERNAL_METHODS} | {
            "improved_diffusion",
            "edm",
        }
        incomplete = [
            f"{row['dataset']}:{row['method']}={row['status']}"
            for row in rows
            if row["method"] in primary_methods and row["status"] != "completed"
        ]
        if incomplete:
            raise RuntimeError(
                "P0 primary evidence contract is incomplete: " + ", ".join(incomplete)
            )
        fairness_violations = factorization_contract_violations(rows)
        if fairness_violations:
            raise RuntimeError(
                "P0 factorization fairness contract failed: "
                + ", ".join(fairness_violations)
            )
    print(f"wrote {args.output_dir / 'p0_paper_table.md'}")


if __name__ == "__main__":
    main()
