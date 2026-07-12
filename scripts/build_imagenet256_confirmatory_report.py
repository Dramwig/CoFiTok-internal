#!/usr/bin/env python
"""Build the preregistered ImageNet-256 20k confirmatory report."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import statistics
from pathlib import Path
from typing import Any


SEEDS = (103, 139)
RUN_TAG = "2026-07-11_imagenet256_20k_repeat"
CONFIG_TEMPLATES = {
    "cofitok": "train_imagenet256_k4_denoisepath_p150_light_20k_seed{seed}_cuda",
    "endpoint_only": "train_imagenet256_k4_epsilononly_p150eval_20k_seed{seed}_cuda",
    "dense_monolithic": "train_imagenet256_k4_densehead_p150eval_20k_seed{seed}_cuda",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("/root/autodl-tmp/CoFiTok/checkpoints"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def quality_path(root: Path, method: str, seed: int, order: str) -> Path:
    return (
        root
        / f"quality_path_imagenet256_{method}_20k_seed{seed}_{order}_1024_t500_{RUN_TAG}"
        / "quality_report.json"
    )


def generation_path(root: Path, method: str, seed: int) -> Path:
    path_method = "dense" if method == "endpoint_only_factorized" else method
    return (
        root
        / f"generated_quality_stream_imagenet256_{path_method}_20k_seed{seed}_4096_ddim50_{RUN_TAG}"
        / "generated_quality_report.json"
    )


def order_permutation_path(root: Path, seed: int) -> Path:
    return (
        root
        / f"order_permutations_imagenet256_cofitok_20k_seed{seed}_1024_t500_{RUN_TAG}"
        / "order_permutation_report.json"
    )


def expected_config_name(method: str, seed: int) -> str:
    return CONFIG_TEMPLATES[method].format(seed=seed)


def validate_method_identity(
    config: dict[str, Any], checkpoint: str, method: str, seed: int
) -> None:
    expected_name = expected_config_name(method, seed)
    if config.get("name") != expected_name:
        raise ValueError(f"config name mismatch for {method}: expected {expected_name}")
    if Path(checkpoint).name != "checkpoint_final.pt" or Path(checkpoint).parent.name != (
        expected_name + "_" + RUN_TAG
    ):
        raise ValueError(f"checkpoint identity mismatch for {method}/seed{seed}")

    model = config["model"]
    loss = config["loss"]
    denoise_path_weight = float(loss.get("denoise_path_prefix_weight", 0.0))
    if method == "cofitok":
        valid = (
            int(model["token_count"]) == 4
            and model["synthesis_mode"] == "restricted"
            and bool(model["predictor_use_feedback"])
            and denoise_path_weight > 0.0
        )
    elif method == "endpoint_only":
        valid = (
            int(model["token_count"]) == 4
            and model["synthesis_mode"] == "restricted"
            and bool(model["predictor_use_feedback"])
            and denoise_path_weight == 0.0
        )
    else:
        valid = (
            int(model["token_count"]) == 1
            and model["synthesis_mode"] == "dense_identity"
            and not bool(model["predictor_use_feedback"])
            and denoise_path_weight == 0.0
        )
    if not valid:
        raise ValueError(f"model/loss identity mismatch for {method}/seed{seed}")


def validate_quality(
    payload: dict[str, Any],
    *,
    seed: int,
    order: str,
    expected_token_count: int,
    expected_budgets: list[int],
    expected_method: str,
) -> None:
    config = payload["config"]
    evaluation = payload["evaluation"]
    if config["data"]["dataset"] != "imagenet_256":
        raise ValueError("quality report is not imagenet_256")
    if int(config["runtime"]["seed"]) != seed:
        raise ValueError(f"quality report seed mismatch: expected {seed}")
    if int(config["runtime"]["steps"]) != 20_000:
        raise ValueError("quality report is not a 20k run")
    if int(config["model"]["token_count"]) != expected_token_count:
        raise ValueError(f"quality report token count is not {expected_token_count}")
    validate_method_identity(
        config, str(payload.get("checkpoint", "")), expected_method, seed
    )
    if int(evaluation["image_count"]) != 1_024 or int(evaluation["fixed_timestep"]) != 500:
        raise ValueError("quality report must use 1,024 images at t=500")
    if evaluation["prefix_budgets"] != expected_budgets:
        raise ValueError(f"quality report prefix budgets are not {expected_budgets}")
    if evaluation["component_order"] != order:
        raise ValueError(f"quality report order mismatch: expected {order}")
    expected_random_seed = 1 if order == "random" else 0
    if int(evaluation.get("random_order_seed", -1)) != expected_random_seed:
        raise ValueError(
            f"quality report random-order seed mismatch: expected {expected_random_seed}"
        )


def extract_quality(payload: dict[str, Any], endpoint_budget: int) -> dict[str, float]:
    endpoint = payload["metrics_by_prefix"][str(endpoint_budget)]
    return {
        "path_auc": float(payload["curve_auc"]["denoise_path_mse"]),
        "clean_auc": float(payload["curve_auc"]["mse"]),
        "endpoint_mse": float(endpoint["mse"]),
        "endpoint_psnr": float(endpoint["psnr_db"]),
        "zero_ratio": float(payload["diagnostics"]["zero_token_component_energy_ratio"]),
        "endpoint_epsilon_mse": float(payload["diagnostics"]["endpoint_epsilon_sum_mse"]),
    }


def evaluate_gates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    path_pairs = [
        row["cofitok_ordered_path_auc"] < row["endpoint_only_path_auc"]
        for row in rows
    ]
    endpoint_relative_changes = [
        row["cofitok_endpoint_mse"] / row["dense_monolithic_endpoint_mse"] - 1.0
        for row in rows
    ]
    mean_endpoint_relative_change = statistics.mean(endpoint_relative_changes)
    random_order_pairs = [
        row["cofitok_random_path_auc"] > row["cofitok_ordered_path_auc"] for row in rows
    ]
    reverse_order_pairs = [
        row["cofitok_reverse_path_auc"] > row["cofitok_ordered_path_auc"] for row in rows
    ]
    endpoint_order_relative_spreads = []
    for row in rows:
        endpoints = [
            row["cofitok_ordered_endpoint_mse"],
            row["cofitok_random_endpoint_mse"],
            row["cofitok_reverse_endpoint_mse"],
        ]
        endpoint_order_relative_spreads.append(
            (max(endpoints) - min(endpoints)) / max(row["cofitok_ordered_endpoint_mse"], 1e-30)
        )
    zero_pairs = [abs(row["cofitok_zero_ratio"]) <= 1e-12 for row in rows]
    exhaustive_pairs = [
        row["nonidentity_delta_ci_low"] > 0.0 and row["reverse_delta_ci_low"] > 0.0
        for row in rows
    ]
    exhaustive_endpoint_pairs = [
        row["exhaustive_endpoint_epsilon_sum_mse_max"] <= 1e-12 for row in rows
    ]

    gates = {
        "path_auc_lower_than_endpoint_only_both_seeds": all(path_pairs),
        "mean_endpoint_mse_within_plus_5pct_of_dense_monolithic": (
            mean_endpoint_relative_change <= 0.05
        ),
        "random_and_reverse_worse_both_seeds": all(random_order_pairs)
        and all(reverse_order_pairs),
        "exhaustive_nonidentity_and_reverse_ci_positive_both_seeds": all(exhaustive_pairs),
        "all_24_permutation_endpoint_sums_preserved": all(exhaustive_endpoint_pairs),
        "endpoint_sum_preserved_across_orders": max(endpoint_order_relative_spreads) <= 1e-5,
        "zero_token_ratio_zero_both_seeds": all(zero_pairs),
    }
    return {
        "gates": gates,
        "overall_pass": all(gates.values()),
        "path_auc_pair_passes": path_pairs,
        "random_order_pair_passes": random_order_pairs,
        "reverse_order_pair_passes": reverse_order_pairs,
        "zero_ratio_pair_passes": zero_pairs,
        "exhaustive_order_pair_passes": exhaustive_pairs,
        "exhaustive_endpoint_pair_passes": exhaustive_endpoint_pairs,
        "endpoint_relative_changes": endpoint_relative_changes,
        "mean_endpoint_relative_change": mean_endpoint_relative_change,
        "endpoint_order_relative_spreads": endpoint_order_relative_spreads,
    }


def build(checkpoint_root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    generation_rows: list[dict[str, Any]] = []
    source_reports: list[str] = []
    for seed in SEEDS:
        endpoint_path = quality_path(checkpoint_root, "endpoint_only", seed, "ordered")
        endpoint_payload = read_json(endpoint_path)
        validate_quality(
            endpoint_payload,
            seed=seed,
            order="ordered",
            expected_token_count=4,
            expected_budgets=[1, 2, 3, 4],
            expected_method="endpoint_only",
        )
        endpoint_only = extract_quality(endpoint_payload, endpoint_budget=4)

        dense_path = quality_path(checkpoint_root, "dense_monolithic", seed, "ordered")
        dense_payload = read_json(dense_path)
        validate_quality(
            dense_payload,
            seed=seed,
            order="ordered",
            expected_token_count=1,
            expected_budgets=[1],
            expected_method="dense_monolithic",
        )
        dense_monolithic = extract_quality(dense_payload, endpoint_budget=1)

        cofitok_by_order: dict[str, dict[str, float]] = {}
        for order in ("ordered", "random", "reverse"):
            path = quality_path(checkpoint_root, "cofitok", seed, order)
            payload = read_json(path)
            validate_quality(
                payload,
                seed=seed,
                order=order,
                expected_token_count=4,
                expected_budgets=[1, 2, 3, 4],
                expected_method="cofitok",
            )
            cofitok_by_order[order] = extract_quality(payload, endpoint_budget=4)
            source_reports.append(str(path))
        source_reports.extend([str(endpoint_path), str(dense_path)])

        ordered = cofitok_by_order["ordered"]
        random = cofitok_by_order["random"]
        reverse = cofitok_by_order["reverse"]
        permutation_path = order_permutation_path(checkpoint_root, seed)
        permutation_payload = read_json(permutation_path)
        permutation_config = permutation_payload["config"]
        permutation_eval = permutation_payload["evaluation"]
        permutation_summary = permutation_payload["summary"]
        if (
            permutation_config["data"]["dataset"] != "imagenet_256"
            or int(permutation_config["runtime"]["seed"]) != seed
            or int(permutation_config["runtime"]["steps"]) != 20_000
            or int(permutation_eval["image_count"]) != 1_024
            or int(permutation_eval["fixed_timestep"]) != 500
            or int(permutation_eval["token_count"]) != 4
            or int(permutation_eval["permutation_count"]) != 24
            or int(permutation_eval["bootstrap_repetitions"]) != 10_000
            or int(permutation_eval["bootstrap_seed"]) != 0
        ):
            raise ValueError(f"invalid exhaustive order report: {permutation_path}")
        validate_method_identity(
            permutation_config,
            str(permutation_payload.get("checkpoint", "")),
            "cofitok",
            seed,
        )
        permutations = permutation_payload.get("permutations", [])
        expected_orders = set(itertools.permutations(range(4)))
        actual_orders = {tuple(row.get("order", [])) for row in permutations}
        actual_ranks = {int(row.get("rank", 0)) for row in permutations}
        if (
            len(permutations) != 24
            or actual_orders != expected_orders
            or actual_ranks != set(range(1, 25))
        ):
            raise ValueError(f"invalid exhaustive permutation coverage: {permutation_path}")
        nonidentity_bootstrap = permutation_summary[
            "nonidentity_minus_ordered_paired_bootstrap"
        ]
        reverse_bootstrap = permutation_summary["reverse_minus_ordered_paired_bootstrap"]
        rows.append(
            {
                "seed": seed,
                "endpoint_only_path_auc": endpoint_only["path_auc"],
                "cofitok_ordered_path_auc": ordered["path_auc"],
                "cofitok_random_path_auc": random["path_auc"],
                "cofitok_reverse_path_auc": reverse["path_auc"],
                "dense_monolithic_endpoint_mse": dense_monolithic["endpoint_mse"],
                "cofitok_endpoint_mse": ordered["endpoint_mse"],
                "cofitok_ordered_endpoint_mse": ordered["endpoint_mse"],
                "cofitok_random_endpoint_mse": random["endpoint_mse"],
                "cofitok_reverse_endpoint_mse": reverse["endpoint_mse"],
                "cofitok_zero_ratio": ordered["zero_ratio"],
                "dense_monolithic_zero_ratio": dense_monolithic["zero_ratio"],
                "exhaustive_nonidentity_mean_path_auc": float(
                    permutation_summary["nonidentity_mean_path_auc"]
                ),
                "ordered_rank_of_24": int(permutation_summary["ordered_rank"]),
                "fraction_nonidentity_worse": float(
                    permutation_summary["fraction_nonidentity_worse"]
                ),
                "nonidentity_delta_ci_low": float(nonidentity_bootstrap["ci_low"]),
                "nonidentity_delta_ci_high": float(nonidentity_bootstrap["ci_high"]),
                "reverse_delta_ci_low": float(reverse_bootstrap["ci_low"]),
                "reverse_delta_ci_high": float(reverse_bootstrap["ci_high"]),
                "exhaustive_endpoint_epsilon_sum_mse_max": float(
                    permutation_summary["endpoint_epsilon_sum_mse_max"]
                ),
            }
        )
        source_reports.append(str(permutation_path))

        for method in ("endpoint_only_factorized", "dense_monolithic", "cofitok"):
            path = generation_path(checkpoint_root, method, seed)
            generation = read_json(path)
            if generation["dataset"] != "imagenet_256":
                raise ValueError(f"generation report is not imagenet_256: {path}")
            evaluation = generation["evaluation"]
            sampling = generation["sampling"]
            expected_prefix = 1 if method == "dense_monolithic" else 4
            method_key = "endpoint_only" if method == "endpoint_only_factorized" else method
            expected_name = expected_config_name(method_key, seed)
            checkpoint = Path(str(generation.get("checkpoint", "")))
            if (
                int(evaluation["sample_image_count"]) != 4_096
                or int(evaluation["real_image_count"]) != 10_000
                or int(evaluation["seed"]) != seed
                or int(sampling["sample_steps"]) != 50
                or int(sampling["prefix_budget"]) != expected_prefix
                or generation.get("config_name") != expected_name
                or checkpoint.name != "checkpoint_final.pt"
                or checkpoint.parent.name != expected_name + "_" + RUN_TAG
            ):
                raise ValueError(f"invalid confirmatory generation protocol: {path}")
            generation_rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "sample_count": int(evaluation["sample_image_count"]),
                    "real_count": int(evaluation["real_image_count"]),
                    "lowres_frechet_proxy": float(
                        generation["metrics"]["lowres_frechet_proxy"]
                    ),
                    "inception_frechet": float(generation["metrics"]["inception_frechet"]),
                    "report": str(path),
                }
            )
            source_reports.append(str(path))

    decision = evaluate_gates(rows)
    return {
        "schema_version": 1,
        "protocol": "imagenet256_20k_two_seed_cofitok_endpoint_only_and_dense_monolithic",
        "dataset": "imagenet_256",
        "resolution": 256,
        "seeds": list(SEEDS),
        "quality_image_count": 1_024,
        "fixed_timestep": 500,
        "generation_sample_count": 4_096,
        "rows": rows,
        "generation_rows": generation_rows,
        "decision": decision,
        "source_reports": source_reports,
        "claim_scope": (
            "Tests ordered denoising-path factorization and endpoint preservation; "
            "generation Frechet-style metrics are exploratory."
        ),
    }


def format_value(value: float) -> str:
    return f"{value:.5f}"


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# ImageNet-256 20k Confirmatory Report",
        "",
        "All confirmatory path metrics use 1,024 validation images at t=500.",
        "",
        "| seed | endpoint-only AUC | ordered | random | reverse | all non-ID mean | rank/24 | dense endpoint MSE | CoFiTok endpoint MSE | delta |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["rows"]:
        delta = (
            row["cofitok_endpoint_mse"] / row["dense_monolithic_endpoint_mse"] - 1.0
        )
        lines.append(
            "| {seed} | {dense_auc} | {ordered} | {random} | {reverse} | {all_mean} | {rank}/24 | {dense_mse} | {cofitok_mse} | {delta:+.2%} |".format(
                seed=row["seed"],
                dense_auc=format_value(row["endpoint_only_path_auc"]),
                ordered=format_value(row["cofitok_ordered_path_auc"]),
                random=format_value(row["cofitok_random_path_auc"]),
                reverse=format_value(row["cofitok_reverse_path_auc"]),
                all_mean=format_value(row["exhaustive_nonidentity_mean_path_auc"]),
                rank=row["ordered_rank_of_24"],
                dense_mse=format_value(row["dense_monolithic_endpoint_mse"]),
                cofitok_mse=format_value(row["cofitok_endpoint_mse"]),
                delta=delta,
            )
        )

    decision = payload["decision"]
    lines.extend(["", "## Preregistered decision", ""])
    for name, passed in decision["gates"].items():
        lines.append(f"- `{name}`: {'PASS' if passed else 'FAIL'}")
    lines.extend(
        [
            "",
            f"Overall confirmatory result: **{'PASS' if decision['overall_pass'] else 'FAIL'}**.",
            f"Mean paired endpoint-MSE change: {decision['mean_endpoint_relative_change']:+.2%}.",
            "",
            "## Exploratory generation controls",
            "",
            "| seed | method | samples | lowres Frechet | Inception Frechet |",
            "| ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for row in payload["generation_rows"]:
        lines.append(
            f"| {row['seed']} | {row['method']} | {row['sample_count']} | "
            f"{row['lowres_frechet_proxy']:.4f} | {row['inception_frechet']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Generation metrics are exploratory and must not be used to claim broad generation-quality superiority.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(payload: dict[str, Any], path: Path) -> None:
    fieldnames = list(payload["rows"][0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(payload["rows"])


def main() -> None:
    args = parse_args()
    payload = build(args.checkpoint_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "imagenet256_confirmatory_report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(payload, args.output_dir / "imagenet256_confirmatory_report.md")
    write_csv(payload, args.output_dir / "imagenet256_confirmatory_rows.csv")
    print(args.output_dir)


if __name__ == "__main__":
    main()
