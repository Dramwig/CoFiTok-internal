from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DATASETS = ["tiny_imagenet_200", "downsampled_imagenet_64", "ffhq_64", "afhqv2_64"]

INTERNAL_METHODS = [
    ("cofitok_light", "CoFiTok", "ours", "light_denoise_path", ["train", "quality", "generated_quality"]),
    ("same_backbone_dense", "Dense epsilon", "internal_baseline", "epsilon_only", ["train", "quality", "generated_quality"]),
    ("channel_mask", "Legacy clean-prefix objective", "internal_baseline", "channel_mask", ["train", "quality"]),
    ("cofitok_no_prefix_loss", "No prefix loss", "internal_ablation", "no_prefix_loss_ablation", ["train", "quality"]),
    (
        "cofitok_clean_monotonic",
        "Add clean-space monotonic term",
        "internal_ablation",
        "clean_monotonic_ablation",
        ["train", "quality"],
    ),
    (
        "cofitok_simultaneous",
        "Simultaneous components",
        "internal_ablation",
        "simultaneous_predictor",
        ["train", "quality"],
    ),
    ("cofitok_deep_synthesis", "Deep S_k", "internal_ablation", "deep_synthesis_ablation", ["train", "quality"]),
]

EXTERNAL_METHODS = [
    ("improved_diffusion", "Improved DDPM", "external_baseline"),
    ("edm", "EDM", "external_baseline"),
    ("d_ar", "D-AR", "external_baseline"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paper-facing formal64 comparison table.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("artifacts/reports/summary_2026-07-09_p0_formal64/experiment_summary.json"),
    )
    parser.add_argument(
        "--baseline-summary",
        type=Path,
        default=Path("artifacts/reports/baselines/summary_2026-07-09_plus_edm/baseline_summary.json"),
    )
    parser.add_argument(
        "--dar-feasibility",
        type=Path,
        default=Path("artifacts/reports/baselines/d_ar/adapter_feasibility_2026-07-09.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _select_internal(rows: list[dict[str, Any]], dataset: str, variant: str) -> dict[str, Any]:
    candidates = [
        row for row in rows
        if row.get("dataset") == dataset
        and row.get("variant") == variant
        and (row.get("steps") in {None, 5000})
        and (row.get("token_count") in {None, 8})
    ]
    if not candidates:
        return {}
    candidates.sort(key=lambda row: str(row.get("report_dir", "")))
    return candidates[-1]


def _select_baseline(rows: list[dict[str, Any]], dataset: str, baseline: str) -> dict[str, Any]:
    candidates = [row for row in rows if row.get("dataset") == dataset and row.get("baseline") == baseline]
    if not candidates:
        return {}
    candidates.sort(key=lambda row: str(row.get("run_name", "")))
    return candidates[-1]


def _status(required: list[str], payload_by_section: dict[str, dict[str, Any]]) -> str:
    present = [section for section in required if payload_by_section.get(section)]
    if len(present) == len(required):
        return "completed"
    if present:
        return "partial"
    return "missing"


def _evidence(required: list[str], payload_by_section: dict[str, dict[str, Any]]) -> str:
    return ",".join(section for section in required if payload_by_section.get(section))


def _fmt(value: Any, digits: int = 4) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    if value is None:
        return ""
    return str(value)


def build_rows(
    summary: dict[str, Any],
    baseline_summary: dict[str, Any],
    dar_feasibility: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    train_rows = summary.get("train", [])
    quality_rows = summary.get("quality", [])
    generated_rows = summary.get("generated_quality", [])
    baseline_rows = baseline_summary.get("rows", [])

    for dataset in DATASETS:
        for method_alias, display_name, method_type, variant, required in INTERNAL_METHODS:
            train = _select_internal(train_rows, dataset, variant)
            quality = _select_internal(quality_rows, dataset, variant)
            generated = _select_internal(generated_rows, dataset, variant)
            payload_by_section = {
                "train": train,
                "quality": quality,
                "generated_quality": generated,
            }
            rows.append(
                {
                    "dataset": dataset,
                    "method": method_alias,
                    "display_name": display_name,
                    "method_type": method_type,
                    "status": _status(required, payload_by_section),
                    "required": ",".join(required),
                    "train_steps": train.get("steps") or quality.get("steps") or generated.get("steps"),
                    "sample_nfe": 50 if generated else None,
                    "sample_lowres_frechet": generated.get("lowres_frechet_proxy"),
                    "sample_inception_frechet": generated.get("inception_frechet"),
                    "denoise_psnr": quality.get("final_psnr_db"),
                    "denoise_lpips": quality.get("final_lpips_alex"),
                    "path_auc": train.get("path_auc"),
                    "clean_auc": train.get("clean_auc"),
                    "effective_tokens": train.get("effective_tokens"),
                    "zero_token_ratio": train.get("zero_token_ratio"),
                    "shuffled_final_ratio": train.get("shuffled_final_ratio"),
                    "evidence": _evidence(required, payload_by_section),
                    "note": "",
                }
            )
        for baseline, display_name, method_type in EXTERNAL_METHODS:
            row = _select_baseline(baseline_rows, dataset, baseline)
            if baseline == "d_ar":
                dar_status = "protocol_blocked" if dar_feasibility else "missing"
                note = (dar_feasibility or {}).get("decision", "")
                evidence = "feasibility_report" if dar_feasibility else ""
            else:
                dar_status = "completed" if row else "missing"
                note = ""
                evidence = "baseline_train,baseline_eval" if row else ""
            rows.append(
                {
                    "dataset": dataset,
                    "method": baseline,
                    "display_name": display_name,
                    "method_type": method_type,
                    "status": dar_status,
                    "required": "baseline_train,baseline_eval",
                    "train_steps": row.get("train_steps"),
                    "sample_nfe": row.get("sampler_nfe"),
                    "sample_lowres_frechet": row.get("lowres_frechet_proxy"),
                    "sample_inception_frechet": row.get("inception_frechet"),
                    "denoise_psnr": None,
                    "denoise_lpips": None,
                    "path_auc": None,
                    "clean_auc": None,
                    "effective_tokens": None,
                    "zero_token_ratio": None,
                    "shuffled_final_ratio": None,
                    "evidence": evidence,
                    "note": note,
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
    headers = [
        "dataset",
        "method",
        "status",
        "steps",
        "NFE",
        "sample lowres",
        "sample Inception",
        "denoise PSNR",
        "path AUC",
        "eff K",
        "zero ratio",
        "evidence",
    ]
    lines = [
        "# Formal64 Paper Comparison",
        "",
        "Lower Frechet-style metrics are better. Prefix/path metrics apply only to CoFiTok-style internal methods.",
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
                    _fmt(row.get("train_steps"), 0),
                    _fmt(row.get("sample_nfe"), 0),
                    _fmt(row.get("sample_lowres_frechet")),
                    _fmt(row.get("sample_inception_frechet")),
                    _fmt(row.get("denoise_psnr"), 3),
                    _fmt(row.get("path_auc")),
                    _fmt(row.get("effective_tokens"), 3),
                    _fmt(row.get("zero_token_ratio")),
                    str(row.get("evidence", "")),
                ]
            )
            + " |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    summary = _read_json(args.summary)
    baseline_summary = _read_json(args.baseline_summary)
    dar_feasibility = _read_json(args.dar_feasibility) if args.dar_feasibility.is_file() else None
    rows = build_rows(summary, baseline_summary, dar_feasibility)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "formal64_paper_table.json").write_text(
        json.dumps({"row_count": len(rows), "rows": rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_dir / "formal64_paper_table.csv", rows)
    write_markdown(args.output_dir / "formal64_paper_table.md", rows)
    print(f"wrote {args.output_dir / 'formal64_paper_table.md'}")


if __name__ == "__main__":
    main()
