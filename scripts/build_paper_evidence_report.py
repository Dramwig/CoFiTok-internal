from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


P0_GENERATION_METHODS = {
    "cofitok_light": "CoFiTok",
    "same_backbone_dense": "Direct dense epsilon",
    "improved_diffusion": "Improved DDPM",
    "edm": "EDM",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a paper-facing evidence and claim-support report.")
    parser.add_argument("--p0-table", type=Path, required=True)
    parser.add_argument("--p1-table", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument(
        "--official-related",
        type=Path,
        default=Path("artifacts/reports/baselines/official_related_methods_2026-07-10/official_related_methods_table.json"),
    )
    parser.add_argument(
        "--long-budget",
        type=Path,
        default=Path(
            "artifacts/reports/long_budget_repeat_2026-07-11/"
            "long_budget_repeat_table.json"
        ),
    )
    parser.add_argument(
        "--imagenet256-confirmatory",
        type=Path,
        default=Path(
            "artifacts/reports/imagenet256_confirmatory_2026-07-11/"
            "imagenet256_confirmatory_report.json"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _number(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: Any, digits: int = 4) -> str:
    number = _number(value)
    if number is None:
        return "" if value is None else str(value)
    return f"{number:.{digits}f}"


def _rank(values: dict[str, float | None], method: str) -> int | None:
    valid = [(name, value) for name, value in values.items() if value is not None]
    if method not in {name for name, _ in valid}:
        return None
    ordered = sorted(valid, key=lambda item: item[1])
    for idx, (name, _) in enumerate(ordered, start=1):
        if name == method:
            return idx
    return None


def _index_p0(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(str(row["dataset"]), str(row["method"])): row for row in rows}


def build_generation_rows(p0_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index = _index_p0(p0_rows)
    datasets = []
    for row in p0_rows:
        dataset = str(row["dataset"])
        if dataset not in datasets:
            datasets.append(dataset)
    output: list[dict[str, Any]] = []
    for dataset in datasets:
        method_rows = {method: index.get((dataset, method), {}) for method in P0_GENERATION_METHODS}
        lowres = {
            method: _number(row.get("sample_lowres_frechet"))
            for method, row in method_rows.items()
        }
        inception = {
            method: _number(row.get("sample_inception_frechet"))
            for method, row in method_rows.items()
        }
        lowres_winner = min((item for item in lowres.items() if item[1] is not None), key=lambda item: item[1])[0]
        inception_winner = min((item for item in inception.items() if item[1] is not None), key=lambda item: item[1])[0]
        output.append(
            {
                "dataset": dataset,
                "cofitok_lowres": lowres.get("cofitok_light"),
                "dense_lowres": lowres.get("same_backbone_dense"),
                "improved_ddpm_lowres": lowres.get("improved_diffusion"),
                "edm_lowres": lowres.get("edm"),
                "lowres_winner": P0_GENERATION_METHODS[lowres_winner],
                "cofitok_lowres_rank": _rank(lowres, "cofitok_light"),
                "cofitok_inception": inception.get("cofitok_light"),
                "dense_inception": inception.get("same_backbone_dense"),
                "improved_ddpm_inception": inception.get("improved_diffusion"),
                "edm_inception": inception.get("edm"),
                "inception_winner": P0_GENERATION_METHODS[inception_winner],
                "cofitok_inception_rank": _rank(inception, "cofitok_light"),
            }
        )
    return output


def build_diagnostic_rows(p0_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index = _index_p0(p0_rows)
    datasets = []
    for row in p0_rows:
        dataset = str(row["dataset"])
        if dataset not in datasets:
            datasets.append(dataset)
    output: list[dict[str, Any]] = []
    for dataset in datasets:
        cofitok = index.get((dataset, "cofitok_light"), {})
        dense = index.get((dataset, "same_backbone_dense"), {})
        endpoint_only = index.get((dataset, "endpoint_only_factorized"), {})
        channel = index.get((dataset, "channel_mask"), {})
        deep = index.get((dataset, "cofitok_deep_synthesis"), {})
        output.append(
            {
                "dataset": dataset,
                "cofitok_psnr": _number(cofitok.get("denoise_psnr")),
                "dense_psnr": _number(dense.get("denoise_psnr")),
                "channel_mask_psnr": _number(channel.get("denoise_psnr")),
                "deep_synthesis_psnr": _number(deep.get("denoise_psnr")),
                "cofitok_path_auc": _number(cofitok.get("path_auc")),
                "endpoint_only_path_auc": _number(endpoint_only.get("path_auc")),
                "cofitok_effective_tokens": _number(cofitok.get("effective_tokens")),
                "endpoint_only_effective_tokens": _number(endpoint_only.get("effective_tokens")),
                "cofitok_zero_ratio": _number(cofitok.get("zero_token_ratio")),
                "deep_synthesis_zero_ratio": _number(deep.get("zero_token_ratio")),
            }
        )
    return output


def summarize_generation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lowres_wins = Counter(row["lowres_winner"] for row in rows)
    inception_wins = Counter(row["inception_winner"] for row in rows)
    lowres_rank_avg = sum(row["cofitok_lowres_rank"] for row in rows) / len(rows)
    inception_rank_avg = sum(row["cofitok_inception_rank"] for row in rows) / len(rows)
    cofitok_best_lowres = sum(1 for row in rows if row["cofitok_lowres_rank"] == 1)
    cofitok_best_inception = sum(1 for row in rows if row["cofitok_inception_rank"] == 1)
    return {
        "dataset_count": len(rows),
        "lowres_wins": dict(lowres_wins),
        "inception_wins": dict(inception_wins),
        "cofitok_best_lowres_count": cofitok_best_lowres,
        "cofitok_best_inception_count": cofitok_best_inception,
        "cofitok_average_lowres_rank": lowres_rank_avg,
        "cofitok_average_inception_rank": inception_rank_avg,
    }


def summarize_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    path_auc_better_than_endpoint_only = sum(
        1
        for row in rows
        if row["cofitok_path_auc"] is not None
        and row["endpoint_only_path_auc"] is not None
        and row["cofitok_path_auc"] < row["endpoint_only_path_auc"]
    )
    channel_mask_gap = [
        row["cofitok_psnr"] - row["channel_mask_psnr"]
        for row in rows
        if row["cofitok_psnr"] is not None and row["channel_mask_psnr"] is not None
    ]
    deep_zero_nonzero = sum(
        1
        for row in rows
        if row["deep_synthesis_zero_ratio"] is not None and row["deep_synthesis_zero_ratio"] > 0.0
    )
    cofitok_zero_count = sum(
        1
        for row in rows
        if row["cofitok_zero_ratio"] is not None
        and abs(row["cofitok_zero_ratio"]) <= 1e-12
    )
    return {
        "dataset_count": len(rows),
        "cofitok_path_auc_better_than_endpoint_only": path_auc_better_than_endpoint_only,
        "mean_cofitok_minus_channel_mask_psnr": sum(channel_mask_gap) / len(channel_mask_gap),
        "deep_synthesis_nonzero_zero_ratio_count": deep_zero_nonzero,
        "cofitok_zero_ratio_zero_count": cofitok_zero_count,
    }


def summarize_official_related(rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = sorted(
        str(row.get("method"))
        for row in rows
        if str(row.get("status", "")).startswith("completed_eval_only_50k")
    )
    pending = sorted(
        str(row.get("method"))
        for row in rows
        if row.get("method")
        and str(row.get("status", "")) in {
            "sampling_or_eval_pending",
            "official_ema_sampling_or_eval_pending",
            "community_non_ema_audit_completed_official_ema_pending",
            "pilot128_metrics_completed_50k_running",
        }
    )
    smoke_only = sorted(
        str(row.get("method"))
        for row in rows
        if row.get("method") and row.get("method") not in completed + pending
    )
    parts = []
    if completed:
        parts.append(f"completed official 50K: {', '.join(completed)}")
    if pending:
        parts.append(f"running/eval pending: {', '.join(pending)}")
    if smoke_only:
        parts.append(f"smoke/probe only: {', '.join(smoke_only)}")
    text = "; ".join(parts) if parts else "no official related-method rows"
    return {
        "completed_50k": completed,
        "pending": pending,
        "smoke_only": smoke_only,
        "text": text + ". All rows remain secondary and outside P0 matched-dataset/step training.",
    }


def summarize_imagenet256_confirmatory(payload: dict[str, Any]) -> dict[str, Any]:
    if not payload:
        return {
            "available": False,
            "overall_pass": False,
            "gate_pass_count": 0,
            "gate_count": 0,
            "mean_endpoint_relative_change": None,
            "text": "ImageNet-256 20k confirmatory report is pending.",
        }
    decision = payload.get("decision", {})
    gates = decision.get("gates", {})
    passed = sum(bool(value) for value in gates.values())
    overall = bool(decision.get("overall_pass"))
    endpoint = _number(decision.get("mean_endpoint_relative_change"))
    endpoint_text = "n/a" if endpoint is None else f"{100 * endpoint:+.2f}%"
    exhaustive_passes = decision.get("exhaustive_order_pair_passes", [])
    ranks = [int(row["ordered_rank_of_24"]) for row in payload.get("rows", [])]
    rank_text = "/".join(str(rank) for rank in ranks) if ranks else "n/a"
    return {
        "available": True,
        "overall_pass": overall,
        "gate_pass_count": passed,
        "gate_count": len(gates),
        "mean_endpoint_relative_change": endpoint,
        "exhaustive_order_pair_passes": exhaustive_passes,
        "ordered_ranks_of_24": ranks,
        "text": (
            f"ImageNet-256 K4 20k confirmatory gates: {passed}/{len(gates)} passed; "
            f"overall {'PASS' if overall else 'FAIL'}; mean endpoint-MSE change {endpoint_text}; "
            f"ordered ranks among all 24 permutations {rank_text}."
        ),
    }


def scoped_claim_pack_ready(
    diagnostic_summary: dict[str, Any],
    long_budget: dict[str, Any],
    confirmatory_summary: dict[str, Any],
    related_summary: dict[str, Any],
) -> bool:
    repeat = long_budget.get("summary", {})
    dataset_count = int(diagnostic_summary.get("dataset_count", 0))
    pair_count = int(repeat.get("pair_count", 0))
    return all(
        (
            dataset_count >= 8,
            int(
                diagnostic_summary.get(
                    "cofitok_path_auc_better_than_endpoint_only", 0
                )
            )
            == dataset_count,
            pair_count >= 4,
            int(repeat.get("path_auc_better_pairs", 0)) == pair_count,
            float(repeat.get("mean_final_mse_relative_change", 1.0)) <= 0.05,
            int(diagnostic_summary.get("cofitok_zero_ratio_zero_count", 0))
            == dataset_count,
            bool(confirmatory_summary.get("overall_pass")),
            set(related_summary.get("completed_50k", []))
            == {"D-AR", "MAR", "ReTok"},
        )
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def _write_markdown(
    path: Path,
    matrix: dict[str, Any],
    generation_rows: list[dict[str, Any]],
    diagnostic_rows: list[dict[str, Any]],
    p1_rows: list[dict[str, Any]],
    official_rows: list[dict[str, Any]],
    generation_summary: dict[str, Any],
    diagnostic_summary: dict[str, Any],
    long_budget: dict[str, Any],
    imagenet256_confirmatory: dict[str, Any],
    confirmatory_summary: dict[str, Any],
    related_summary: dict[str, Any],
) -> None:
    matrix_counts = matrix.get("status_counts", {})
    generation_table = [
        [
            row["dataset"],
            _fmt(row["cofitok_lowres"]),
            _fmt(row["dense_lowres"]),
            _fmt(row["improved_ddpm_lowres"]),
            _fmt(row["edm_lowres"]),
            row["lowres_winner"],
            row["cofitok_lowres_rank"],
            _fmt(row["cofitok_inception"]),
            _fmt(row["dense_inception"]),
            _fmt(row["improved_ddpm_inception"]),
            _fmt(row["edm_inception"]),
            row["inception_winner"],
            row["cofitok_inception_rank"],
        ]
        for row in generation_rows
    ]
    diagnostic_table = [
        [
            row["dataset"],
            _fmt(row["cofitok_psnr"], 3),
            _fmt(row["dense_psnr"], 3),
            _fmt(row["channel_mask_psnr"], 3),
            _fmt(row["cofitok_path_auc"]),
            _fmt(row["endpoint_only_path_auc"]),
            _fmt(row["cofitok_effective_tokens"], 3),
            _fmt(row["endpoint_only_effective_tokens"], 3),
            _fmt(row["cofitok_zero_ratio"]),
            _fmt(row["deep_synthesis_zero_ratio"]),
        ]
        for row in diagnostic_rows
    ]
    p1_table = [
        [
            row["dataset"],
            row["baseline"],
            row["image_count"],
            row["source_resolution"],
            row["eval_image_size"],
            row["nfe"],
            _fmt(row["reconstruction_psnr_db"]),
            _fmt(row["lowres_frechet_proxy"]),
        ]
        for row in p1_rows
    ]
    official_table = [
        [
            row["method"],
            row["dataset"],
            row["protocol"],
            row["status"],
            row["sample_count"],
            _fmt(row.get("fid")),
            _fmt(row.get("sfid")),
            _fmt(row.get("inception_score")),
            _fmt(row.get("precision")),
            _fmt(row.get("recall")),
        ]
        for row in official_rows
    ]
    long_budget_table = [
        [
            row["dataset_display"],
            row["method_display"],
            row["seed_count"],
            f"{row['final_mse']['mean']:.4f} +/- {row['final_mse']['std']:.4f}",
            f"{row['path_auc']['mean']:.4f} +/- {row['path_auc']['std']:.4f}",
            f"{row['effective_k']['mean']:.3f} +/- {row['effective_k']['std']:.3f}",
        ]
        for row in long_budget.get("aggregates", [])
    ]
    repeat_summary = long_budget.get("summary", {})
    confirmatory_table = [
        [
            row["seed"],
            _fmt(row["endpoint_only_path_auc"], 5),
            _fmt(row["cofitok_ordered_path_auc"], 5),
            _fmt(row["cofitok_random_path_auc"], 5),
            _fmt(row["cofitok_reverse_path_auc"], 5),
            _fmt(row["exhaustive_nonidentity_mean_path_auc"], 5),
            row["ordered_rank_of_24"],
            _fmt(row["nonidentity_delta_ci_low"], 5),
            _fmt(row["dense_monolithic_endpoint_mse"], 5),
            _fmt(row["cofitok_endpoint_mse"], 5),
            _fmt(row["cofitok_zero_ratio"]),
        ]
        for row in imagenet256_confirmatory.get("rows", [])
    ]
    body = "\n\n".join(
        [
            "# Paper Evidence Report",
            "This report is generated from current artifacts. Lower Frechet-style values are better.",
            "## Completion State",
            _markdown_table(
                ["status", "count"],
                [[key, matrix_counts[key]] for key in sorted(matrix_counts)],
            ),
            "Interpretation: P0 train/eval evidence is complete for the locked matched-dataset/optimizer-step protocols. Batch size, nominal images seen, parameter count, and NFE remain explicit because this is not a compute-matched SOTA table. P1 tokenizer rows and official related-method rows are eval-only secondary evidence.",
            "## P0 Generation Comparison",
            _markdown_table(
                [
                    "dataset",
                    "CoFiTok lowres",
                    "Direct dense lowres",
                    "DDPM lowres",
                    "EDM lowres",
                    "lowres winner",
                    "CoFiTok rank",
                    "CoFiTok Inc",
                    "Direct dense Inc",
                    "DDPM Inc",
                    "EDM Inc",
                    "Inc winner",
                    "CoFiTok rank",
                ],
                generation_table,
            ),
            "## CoFiTok Diagnostic Evidence",
            _markdown_table(
                [
                    "dataset",
                    "CoFiTok PSNR",
                    "Direct dense PSNR",
                    "Channel PSNR",
                    "CoFiTok path AUC",
                    "Endpoint-only path AUC",
                    "CoFiTok eff K",
                    "Endpoint-only eff K",
                    "CoFiTok zero",
                    "Deep-S zero",
                ],
                diagnostic_table,
            ),
            "## Matched 20k Two-Seed Evidence",
            _markdown_table(
                ["dataset", "method", "seeds", "endpoint x0 MSE (t=500)", "path AUC", "effective K"],
                long_budget_table,
            ),
            "## ImageNet-256 20k Confirmatory Scaling",
            _markdown_table(
                [
                    "seed",
                    "endpoint-only path AUC",
                    "ordered",
                    "random",
                    "reverse",
                    "all non-ID mean",
                    "ordered rank/24",
                    "non-ID delta CI low",
                    "direct-dense endpoint MSE",
                    "CoFiTok endpoint MSE",
                    "zero ratio",
                ],
                confirmatory_table,
            ),
            "## P1 Tokenizer Reconstruction",
            _markdown_table(
                ["dataset", "baseline", "images", "src res", "eval res", "NFE", "PSNR", "lowres Frechet"],
                p1_table,
            ),
            "## Official Related-Method Eval-Only Rows",
            _markdown_table(
                ["method", "dataset", "protocol", "status", "samples", "FID", "sFID", "IS", "precision", "recall"],
                official_table,
            ),
            "## Machine Summary",
            _markdown_table(
                ["claim", "evidence"],
                [
                    [
                        "Generation quality",
                        (
                            f"CoFiTok is best on {generation_summary['cofitok_best_lowres_count']}/"
                            f"{generation_summary['dataset_count']} lowres rows and "
                            f"{generation_summary['cofitok_best_inception_count']}/"
                            f"{generation_summary['dataset_count']} Inception rows; average ranks "
                            f"{generation_summary['cofitok_average_lowres_rank']:.2f} lowres and "
                            f"{generation_summary['cofitok_average_inception_rank']:.2f} Inception."
                        ),
                    ],
                    [
                        "Prefix/factorization behavior",
                        (
                            f"CoFiTok path AUC is lower than the endpoint-only factorized control on "
                            f"{diagnostic_summary['cofitok_path_auc_better_than_endpoint_only']}/"
                            f"{diagnostic_summary['dataset_count']} datasets; mean PSNR gap over channel-mask "
                            f"ablation is {diagnostic_summary['mean_cofitok_minus_channel_mask_psnr']:.3f} dB."
                        ),
                    ],
                    [
                        "Repeated long-budget prefix control",
                        (
                            f"CoFiTok path AUC is lower in "
                            f"{repeat_summary.get('path_auc_better_pairs', 0)}/"
                            f"{repeat_summary.get('pair_count', 0)} paired dataset-seed runs, "
                            f"with mean relative reduction "
                            f"{100 * repeat_summary.get('mean_path_auc_relative_reduction', 0.0):.2f}%; "
                            f"mean endpoint-MSE change is "
                            f"{100 * repeat_summary.get('mean_final_mse_relative_change', 0.0):+.2f}%."
                        ),
                    ],
                    [
                        "ImageNet-256 confirmatory scaling",
                        confirmatory_summary["text"],
                    ],
                    [
                        "Restricted synthesis safety",
                        (
                            f"CoFiTok zero-token ratio remains zero in current rows, while deep-S_k has "
                            f"nonzero zero-token ratio on {diagnostic_summary['deep_synthesis_nonzero_zero_ratio_count']}/"
                            f"{diagnostic_summary['dataset_count']} datasets."
                        ),
                    ],
                    [
                        "Related-method coverage",
                        related_summary["text"],
                    ],
                    [
                        "Top-tier claim stance",
                        "Support is strongest for a scoped method paper about ordered restricted dense-noise factorization and prefix-controllable denoising. Current evidence does not support claiming broad unconditional generation SOTA.",
                    ],
                ],
            ),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body + "\n", encoding="utf-8")


def _write_claim_pack(
    path: Path,
    generation_summary: dict[str, Any],
    diagnostic_summary: dict[str, Any],
    long_budget: dict[str, Any],
    confirmatory_summary: dict[str, Any],
    related_summary: dict[str, Any],
) -> None:
    repeat = long_budget.get("summary", {})
    ready = scoped_claim_pack_ready(
        diagnostic_summary, long_budget, confirmatory_summary, related_summary
    )
    decision = (
        "The current evidence clears the core empirical gates for a scoped method "
        "submission about ordered, restricted dense-noise factorization with "
        "prefix-controllable denoising."
        if ready
        else "The current evidence does not yet clear every preregistered core gate; "
        "the scoped submission claim remains pending."
    )
    body = f"""# CoFiTok Paper Claim Pack

## Decision

{decision} It does not support a broad generation-quality or
visual-tokenizer-superiority claim.

## Supported Evidence

- Broad short-budget coverage: CoFiTok path AUC is lower than the
  endpoint-only factorized control on {diagnostic_summary['cofitok_path_auc_better_than_endpoint_only']}/{diagnostic_summary['dataset_count']} datasets.
- Matched 20k repeats: path AUC is lower in
  {repeat.get('path_auc_better_pairs', 0)}/{repeat.get('pair_count', 0)} paired dataset-seed runs, with a mean relative
  reduction of {100 * repeat.get('mean_path_auc_relative_reduction', 0.0):.2f}%.
- The repeated structural gain has an explicit endpoint cost: mean endpoint
  x0-MSE change at t=500 is {100 * repeat.get('mean_final_mse_relative_change', 0.0):+.2f}%, with endpoint wins in
  {repeat.get('final_mse_better_pairs', 0)}/{repeat.get('pair_count', 0)} pairs.
- ImageNet-256 scaling state: {confirmatory_summary['text']}
- CoFiTok exceeds the channel-mask control by
  {diagnostic_summary['mean_cofitok_minus_channel_mask_psnr']:.3f} dB PSNR on average.
- Restricted CoFiTok keeps zero-token ratio at zero; deep-$S_k$ has nonzero
  leakage on {diagnostic_summary['deep_synthesis_nonzero_zero_ratio_count']}/{diagnostic_summary['dataset_count']} datasets.
- Related-method state: {related_summary['text']}

## Claims To Avoid

- CoFiTok is not the best lowres Frechet method on the current broad table
  ({generation_summary['cofitok_best_lowres_count']}/{generation_summary['dataset_count']} wins).
- CoFiTok is best on only {generation_summary['cofitok_best_inception_count']}/{generation_summary['dataset_count']} Inception-style rows.
- Do not claim superiority over EDM, direct dense epsilon, FlexTok, TiTok, D-AR, MAR,
  or ReTok outside their matched task and protocol.
- Do not count official pretrained eval-only rows as matched-dataset/step retraining.

## Reviewer-Risk Boundary

The positive result is mechanistic and interface-level: meaningful denoising
prefixes arise from factorizing dense noise prediction through restricted
token-only synthesis. The main residual risks are small-model/short-budget
scaling, mixed sample quality, and the absence of matched-dataset/step all-dataset
training for architecture-incompatible nearest methods. These must remain
limitations rather than being hidden by the matrix completion counts.
"""
    path.write_text(body, encoding="utf-8")


def main() -> None:
    args = parse_args()
    p0_payload = _read_json(args.p0_table)
    p1_payload = _read_json(args.p1_table)
    matrix_payload = _read_json(args.matrix)
    official_payload = _read_json(args.official_related) if args.official_related.exists() else {"rows": []}
    long_budget_payload = _read_json(args.long_budget) if args.long_budget.exists() else {"aggregates": [], "summary": {}}
    imagenet256_confirmatory_payload = (
        _read_json(args.imagenet256_confirmatory)
        if args.imagenet256_confirmatory.exists()
        else {}
    )
    p0_rows = p0_payload["rows"]
    p1_rows = p1_payload["rows"]
    official_rows = official_payload.get("rows", [])
    related_summary = summarize_official_related(official_rows)
    confirmatory_summary = summarize_imagenet256_confirmatory(
        imagenet256_confirmatory_payload
    )
    generation_rows = build_generation_rows(p0_rows)
    diagnostic_rows = build_diagnostic_rows(p0_rows)
    generation_summary = summarize_generation(generation_rows)
    diagnostic_summary = summarize_diagnostics(diagnostic_rows)
    core_claim_ready = scoped_claim_pack_ready(
        diagnostic_summary,
        long_budget_payload,
        confirmatory_summary,
        related_summary,
    )
    supported_claims = [
        "ordered restricted dense-noise factorization",
        "prefix-controllable partial denoising",
        "diagnostic separation from channel-mask/deep-synthesis failures",
        "matched 20k two-seed prefix-control evidence",
        "secondary official-checkpoint related-method coverage",
    ]
    if confirmatory_summary.get("overall_pass"):
        supported_claims.append("ImageNet-256 20k confirmatory scaling")
    payload = {
        "inputs": {
            "p0_table": args.p0_table.as_posix(),
            "p1_table": args.p1_table.as_posix(),
            "matrix": args.matrix.as_posix(),
            "official_related": args.official_related.as_posix() if args.official_related.exists() else "",
            "long_budget": args.long_budget.as_posix() if args.long_budget.exists() else "",
            "imagenet256_confirmatory": (
                args.imagenet256_confirmatory.as_posix()
                if args.imagenet256_confirmatory.exists()
                else ""
            ),
        },
        "generation_summary": generation_summary,
        "diagnostic_summary": diagnostic_summary,
        "generation_rows": generation_rows,
        "diagnostic_rows": diagnostic_rows,
        "p1_rows": p1_rows,
        "official_related_rows": official_rows,
        "official_related_summary": related_summary,
        "long_budget_repeat": long_budget_payload,
        "imagenet256_confirmatory": imagenet256_confirmatory_payload,
        "imagenet256_confirmatory_summary": confirmatory_summary,
        "claim_stance": {
            "core_empirical_gates_ready": core_claim_ready,
            "supported": supported_claims,
            "pending": (
                []
                if core_claim_ready
                else ["one or more preregistered core evidence gates"]
            ),
            "not_supported": [
                "broad unconditional generation SOTA",
                "general visual-tokenizer superiority over decoder-based tokenizers",
            ],
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "paper_evidence_report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(args.output_dir / "p0_generation_ranking.csv", generation_rows)
    _write_csv(args.output_dir / "cofitok_diagnostic_summary.csv", diagnostic_rows)
    _write_csv(args.output_dir / "p1_tokenizer_reconstruction_rows.csv", p1_rows)
    _write_csv(args.output_dir / "long_budget_repeat_rows.csv", long_budget_payload.get("aggregates", []))
    _write_markdown(
        args.output_dir / "paper_evidence_report.md",
        matrix_payload,
        generation_rows,
        diagnostic_rows,
        p1_rows,
        official_rows,
        generation_summary,
        diagnostic_summary,
        long_budget_payload,
        imagenet256_confirmatory_payload,
        confirmatory_summary,
        related_summary,
    )
    _write_claim_pack(
        args.output_dir / "paper_claim_pack.md",
        generation_summary,
        diagnostic_summary,
        long_budget_payload,
        confirmatory_summary,
        related_summary,
    )
    print(f"wrote {args.output_dir / 'paper_evidence_report.md'}")


if __name__ == "__main__":
    main()

