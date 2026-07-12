from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DATASETS = [
    ("cifar10", 3000),
    ("tiny_imagenet_200", 5000),
    ("imagenet_1k_64x64_hf", 5000),
    ("downsampled_imagenet_64", 5000),
    ("ffhq_64", 5000),
    ("afhqv2_64", 5000),
]

INTERNAL_METHODS = [
    ("cofitok_light", "CoFiTok", "light_denoise_path", ["train", "quality", "generated_quality"]),
    ("same_backbone_dense", "Dense epsilon", "epsilon_only", ["train", "quality", "generated_quality"]),
    ("channel_mask", "Legacy clean-prefix objective", "channel_mask", ["train", "quality"]),
    ("cofitok_no_prefix_loss", "No prefix loss", "no_prefix_loss_ablation", ["train", "quality"]),
    ("cofitok_clean_monotonic", "Add clean-space monotonic term", "clean_monotonic_ablation", ["train", "quality"]),
    ("cofitok_simultaneous", "Simultaneous components", "simultaneous_predictor", ["train", "quality"]),
    ("cofitok_deep_synthesis", "Deep S_k", "deep_synthesis_ablation", ["train", "quality"]),
]

EXTERNAL_METHODS = [
    ("improved_diffusion", "Improved DDPM"),
    ("edm", "EDM"),
    ("d_ar", "D-AR"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build current P0 64x64 method x dataset comparison table.")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--dar-feasibility", type=Path, default=Path("artifacts/reports/baselines/d_ar/adapter_feasibility_2026-07-09.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def select_internal(rows: list[dict[str, Any]], dataset: str, variant: str, steps: int) -> dict[str, Any]:
    candidates = [
        row for row in rows
        if row.get("dataset") == dataset
        and row.get("variant") == variant
        and row.get("steps") == steps
        and (row.get("token_count") in {None, 8})
    ]
    if not candidates:
        return {}
    candidates.sort(key=lambda row: str(row.get("report_dir", "")))
    return candidates[-1]


def select_baseline(rows: list[dict[str, Any]], dataset: str, baseline: str) -> dict[str, Any]:
    candidates = [row for row in rows if row.get("dataset") == dataset and row.get("baseline") == baseline]
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
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    if value is None:
        return ""
    return str(value)


def build_rows(summary: dict[str, Any], baseline_summary: dict[str, Any], dar_feasibility: dict[str, Any] | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    train_rows = summary.get("train", [])
    quality_rows = summary.get("quality", [])
    generated_rows = summary.get("generated_quality", [])
    baseline_rows = baseline_summary.get("rows", [])

    for dataset, target_steps in DATASETS:
        for method, display_name, variant, required in INTERNAL_METHODS:
            train = select_internal(train_rows, dataset, variant, target_steps)
            quality = select_internal(quality_rows, dataset, variant, target_steps)
            generated = select_internal(generated_rows, dataset, variant, target_steps)
            payloads = {"train": train, "quality": quality, "generated_quality": generated}
            rows.append(
                {
                    "dataset": dataset,
                    "protocol_steps": target_steps,
                    "method": method,
                    "display_name": display_name,
                    "status": status(required, payloads),
                    "evidence": evidence(required, payloads),
                    "sample_nfe": 50 if generated else None,
                    "sample_lowres_frechet": generated.get("lowres_frechet_proxy"),
                    "sample_inception_frechet": generated.get("inception_frechet"),
                    "denoise_psnr": quality.get("final_psnr_db"),
                    "path_auc": train.get("path_auc"),
                    "effective_tokens": train.get("effective_tokens"),
                    "zero_token_ratio": train.get("zero_token_ratio"),
                }
            )
        for baseline, display_name in EXTERNAL_METHODS:
            row = select_baseline(baseline_rows, dataset, baseline)
            if baseline == "d_ar":
                row_status = "protocol_blocked" if dar_feasibility else "missing"
                row_evidence = "feasibility_report" if dar_feasibility else ""
            else:
                row_status = "completed" if row else "missing"
                row_evidence = "baseline_train,baseline_eval" if row else ""
            rows.append(
                {
                    "dataset": dataset,
                    "protocol_steps": target_steps,
                    "method": baseline,
                    "display_name": display_name,
                    "status": row_status,
                    "evidence": row_evidence,
                    "sample_nfe": row.get("sampler_nfe"),
                    "sample_lowres_frechet": row.get("lowres_frechet_proxy"),
                    "sample_inception_frechet": row.get("inception_frechet"),
                    "denoise_psnr": None,
                    "path_auc": None,
                    "effective_tokens": None,
                    "zero_token_ratio": None,
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    headers = ["dataset", "method", "status", "steps", "NFE", "sample lowres", "sample Inception", "denoise PSNR", "path AUC", "eff K", "zero ratio", "evidence"]
    lines = [
        "# P0 64x64 Comparison",
        "",
        "CIFAR10 uses the existing 32x32 3k smoke protocol. Other rows use 64x64 5k protocol. Lower Frechet-style metrics are better.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["dataset"]),
                    str(row["display_name"]),
                    str(row["status"]),
                    fmt(row.get("protocol_steps"), 0),
                    fmt(row.get("sample_nfe"), 0),
                    fmt(row.get("sample_lowres_frechet")),
                    fmt(row.get("sample_inception_frechet")),
                    fmt(row.get("denoise_psnr"), 3),
                    fmt(row.get("path_auc")),
                    fmt(row.get("effective_tokens"), 3),
                    fmt(row.get("zero_token_ratio")),
                    str(row.get("evidence", "")),
                ]
            )
            + " |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    summary = read_json(args.summary)
    baseline_summary = read_json(args.baseline_summary)
    dar_feasibility = read_json(args.dar_feasibility) if args.dar_feasibility.exists() else None
    rows = build_rows(summary, baseline_summary, dar_feasibility)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "p0_64_paper_table.json").write_text(
        json.dumps({"row_count": len(rows), "rows": rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_dir / "p0_64_paper_table.csv", rows)
    write_markdown(args.output_dir / "p0_64_paper_table.md", rows)
    print(f"wrote {args.output_dir / 'p0_64_paper_table.md'}")


if __name__ == "__main__":
    main()
