from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


RUN_PATHS: dict[str, dict[str, str]] = {
    "tiny_epsilononly_20k_seed2": {
        "train": "train_tiny_imagenet_k8_epsilononly_p150eval_20k_seed2_2026-07-08/report.json",
        "quality": "quality_tiny_epsilononly_20k_seed2_1024_t500_lpips_inception_2026-07-08/quality_report.json",
        "sample": "generated_tiny_epsilononly_20k_seed2_2048_ddim50_2026-07-08/sample_report.json",
        "generated_quality": "generated_quality_tiny_epsilononly_20k_seed2_2048_ddim50_2026-07-08/generated_quality_report.json",
    },
    "tiny_k8_light_20k_seed2": {
        "train": "train_tiny_imagenet_k8_denoisepath_p150_light_20k_seed2_2026-07-08/report.json",
        "quality": "quality_tiny_k8_light_20k_seed2_1024_t500_lpips_inception_2026-07-08/quality_report.json",
        "order_ordered": "order_tiny_k8_light_20k_seed2_ordered_2026-07-08/report.json",
        "order_random": "order_tiny_k8_light_20k_seed2_random_2026-07-08/report.json",
        "order_reverse": "order_tiny_k8_light_20k_seed2_reverse_2026-07-08/report.json",
        "sample": "generated_tiny_k8_light_20k_seed2_2048_ddim50_2026-07-08/sample_report.json",
        "generated_quality": "generated_quality_tiny_k8_light_20k_seed2_2048_ddim50_2026-07-08/generated_quality_report.json",
    },
    "imagenet_hf_epsilononly_20k_seed2": {
        "train": "train_imagenet_1k_64x64_hf_k8_epsilononly_p150eval_20k_seed2_2026-07-08/report.json",
        "quality": "quality_imagenet_hf_epsilononly_20k_seed2_1024_t500_lpips_inception_2026-07-08/quality_report.json",
        "sample": "generated_imagenet_hf_epsilononly_20k_seed2_2048_ddim50_2026-07-08/sample_report.json",
        "generated_quality": "generated_quality_imagenet_hf_epsilononly_20k_seed2_2048_ddim50_2026-07-08/generated_quality_report.json",
    },
    "imagenet_hf_k8_light_20k_seed2": {
        "train": "train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_20k_seed2_2026-07-08/report.json",
        "quality": "quality_imagenet_hf_k8_light_20k_seed2_1024_t500_lpips_inception_2026-07-08/quality_report.json",
        "order_ordered": "order_imagenet_hf_k8_light_20k_seed2_ordered_2026-07-08/report.json",
        "order_random": "order_imagenet_hf_k8_light_20k_seed2_random_2026-07-08/report.json",
        "order_reverse": "order_imagenet_hf_k8_light_20k_seed2_reverse_2026-07-08/report.json",
        "sample": "generated_imagenet_hf_k8_light_20k_seed2_2048_ddim50_2026-07-08/sample_report.json",
        "generated_quality": "generated_quality_imagenet_hf_k8_light_20k_seed2_2048_ddim50_2026-07-08/generated_quality_report.json",
    },
    "tiny_k8_light_multiscale_10k": {
        "train": "train_tiny_imagenet_k8_denoisepath_p150_light_multiscale_10k_2026-07-08/report.json",
        "quality": "quality_tiny_k8_light_multiscale_10k_1024_t500_lpips_inception_2026-07-08/quality_report.json",
        "order_ordered": "order_tiny_k8_light_multiscale_10k_ordered_2026-07-08/report.json",
        "order_random": "order_tiny_k8_light_multiscale_10k_random_2026-07-08/report.json",
        "order_reverse": "order_tiny_k8_light_multiscale_10k_reverse_2026-07-08/report.json",
        "sample": "generated_tiny_k8_light_multiscale_10k_2048_ddim50_2026-07-08/sample_report.json",
        "generated_quality": "generated_quality_tiny_k8_light_multiscale_10k_2048_ddim50_2026-07-08/generated_quality_report.json",
    },
    "imagenet_hf_k8_light_multiscale_10k": {
        "train": "train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_multiscale_10k_2026-07-08/report.json",
        "quality": "quality_imagenet_hf_k8_light_multiscale_10k_1024_t500_lpips_inception_2026-07-08/quality_report.json",
        "order_ordered": "order_imagenet_hf_k8_light_multiscale_10k_ordered_2026-07-08/report.json",
        "order_random": "order_imagenet_hf_k8_light_multiscale_10k_random_2026-07-08/report.json",
        "order_reverse": "order_imagenet_hf_k8_light_multiscale_10k_reverse_2026-07-08/report.json",
        "sample": "generated_imagenet_hf_k8_light_multiscale_10k_2048_ddim50_2026-07-08/sample_report.json",
        "generated_quality": "generated_quality_imagenet_hf_k8_light_multiscale_10k_2048_ddim50_2026-07-08/generated_quality_report.json",
    },
    "tiny_k8_light_nopathprefix_10k": {
        "train": "train_tiny_imagenet_k8_denoisepath_p150_light_nopathprefix_10k_2026-07-08/report.json",
        "quality": "quality_tiny_k8_light_nopathprefix_10k_1024_t500_lpips_inception_2026-07-08/quality_report.json",
        "order_ordered": "order_tiny_k8_light_nopathprefix_10k_ordered_2026-07-08/report.json",
        "order_random": "order_tiny_k8_light_nopathprefix_10k_random_2026-07-08/report.json",
        "order_reverse": "order_tiny_k8_light_nopathprefix_10k_reverse_2026-07-08/report.json",
    },
    "tiny_k8_light_cleanmono_10k": {
        "train": "train_tiny_imagenet_k8_denoisepath_p150_light_cleanmono_10k_2026-07-08/report.json",
        "quality": "quality_tiny_k8_light_cleanmono_10k_1024_t500_lpips_inception_2026-07-08/quality_report.json",
        "order_ordered": "order_tiny_k8_light_cleanmono_10k_ordered_2026-07-08/report.json",
        "order_random": "order_tiny_k8_light_cleanmono_10k_random_2026-07-08/report.json",
        "order_reverse": "order_tiny_k8_light_cleanmono_10k_reverse_2026-07-08/report.json",
    },
    "imagenet_hf_k8_light_nopathprefix_10k": {
        "train": "train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_nopathprefix_10k_2026-07-08/report.json",
        "quality": "quality_imagenet_hf_k8_light_nopathprefix_10k_1024_t500_lpips_inception_2026-07-08/quality_report.json",
        "order_ordered": "order_imagenet_hf_k8_light_nopathprefix_10k_ordered_2026-07-08/report.json",
        "order_random": "order_imagenet_hf_k8_light_nopathprefix_10k_random_2026-07-08/report.json",
        "order_reverse": "order_imagenet_hf_k8_light_nopathprefix_10k_reverse_2026-07-08/report.json",
    },
    "imagenet_hf_k8_light_cleanmono_10k": {
        "train": "train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_cleanmono_10k_2026-07-08/report.json",
        "quality": "quality_imagenet_hf_k8_light_cleanmono_10k_1024_t500_lpips_inception_2026-07-08/quality_report.json",
        "order_ordered": "order_imagenet_hf_k8_light_cleanmono_10k_ordered_2026-07-08/report.json",
        "order_random": "order_imagenet_hf_k8_light_cleanmono_10k_random_2026-07-08/report.json",
        "order_reverse": "order_imagenet_hf_k8_light_cleanmono_10k_reverse_2026-07-08/report.json",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize locally synced reports from the remote validation queue.")
    parser.add_argument("--reports-dir", default="artifacts/reports")
    parser.add_argument("--output-dir", default="artifacts/reports/remote_queue_progress_2026-07-08")
    parser.add_argument("--date", default="2026-07-08")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _final_prefix(metrics_by_prefix: dict[str, Any] | None) -> dict[str, Any]:
    if not metrics_by_prefix:
        return {}
    key = sorted(metrics_by_prefix, key=lambda value: int(value))[-1]
    value = metrics_by_prefix[key]
    return value if isinstance(value, dict) else {}


def _last(values: Any) -> Any:
    if isinstance(values, list) and values:
        return values[-1]
    return None


def summarize_run(paths: dict[str, Path]) -> dict[str, Any]:
    entry: dict[str, Any] = {}
    train_path = paths.get("train")
    if train_path and train_path.exists():
        train = _read_json(train_path)
        config = train.get("config", {})
        prefix = train.get("prefix_summary") or {}
        diagnostics = prefix.get("diagnostics") or {}
        entry.update(
            {
                "dataset": config.get("data", {}).get("dataset"),
                "seed": config.get("runtime", {}).get("seed"),
                "steps": config.get("runtime", {}).get("steps"),
                "token_count": config.get("model", {}).get("token_count"),
                "synthesis_mode": config.get("model", {}).get("synthesis_mode"),
                "predictor_type": config.get("model", {}).get("predictor_type"),
                "loss_total_final": train.get("final_losses", {}).get("total"),
                "loss_epsilon_final": train.get("final_losses", {}).get("epsilon"),
                "train_prefix_auc_clean_mse": prefix.get("prefix_mse_to_clean_auc"),
                "train_prefix_auc_denoise_path": prefix.get("prefix_mse_to_denoise_path_auc"),
                "train_final_clean_mse": _last(prefix.get("prefix_mse_to_clean")),
                "zero_token_energy_ratio": diagnostics.get("zero_token_component_energy_ratio"),
                "random_token_energy_ratio": diagnostics.get("random_token_component_energy_ratio"),
                "shuffled_final_mse_ratio": prefix.get("shuffled_final_mse_ratio"),
                "energy_effective_token_count": prefix.get("energy_effective_token_count"),
                "tail_energy_ratio": prefix.get("tail_energy_ratio"),
            }
        )

    quality_path = paths.get("quality")
    if quality_path and quality_path.exists():
        quality = _read_json(quality_path)
        final = _final_prefix(quality.get("metrics_by_prefix"))
        entry.update(
            {
                "quality_image_count": quality.get("evaluation", {}).get("image_count"),
                "quality_prefix_auc_clean_mse": quality.get("curve_auc", {}).get("mse"),
                "quality_final_clean_mse": final.get("mse"),
                "quality_final_lpips_alex": final.get("lpips_alex"),
                "quality_final_inception_frechet": final.get("inception_frechet"),
            }
        )

    sample_path = paths.get("sample")
    if sample_path and sample_path.exists():
        sample = _read_json(sample_path)
        entry["sample_count"] = sample.get("sampling", {}).get("num_samples")
        entry["sample_steps"] = sample.get("sampling", {}).get("sample_steps")

    generated_quality_path = paths.get("generated_quality")
    if generated_quality_path and generated_quality_path.exists():
        generated_quality = _read_json(generated_quality_path)
        entry.update(
            {
                "generated_quality_sample_count": generated_quality.get("evaluation", {}).get("sample_image_count"),
                "generated_quality_real_count": generated_quality.get("evaluation", {}).get("real_image_count"),
                "generated_inception_frechet": generated_quality.get("metrics", {}).get("inception_frechet"),
                "generated_lowres_frechet_proxy": generated_quality.get("metrics", {}).get("lowres_frechet_proxy"),
            }
        )

    for order_key in ("order_ordered", "order_random", "order_reverse"):
        order_path = paths.get(order_key)
        if order_path and order_path.exists():
            order_report = _read_json(order_path)
            prefix = order_report.get("prefix_summary") or {}
            entry[f"{order_key}_prefix_auc_clean_mse"] = prefix.get("prefix_mse_to_clean_auc")
            entry[f"{order_key}_prefix_auc_denoise_path"] = prefix.get("prefix_mse_to_denoise_path_auc")
            entry[f"{order_key}_final_clean_mse"] = _last(prefix.get("prefix_mse_to_clean"))

    return entry


def is_complete(paths: dict[str, Path]) -> bool:
    return all(path.exists() for path in paths.values())


def build_progress_summary(reports_dir: Path, date: str) -> dict[str, Any]:
    runs: dict[str, Any] = {}
    completed_labels = []
    for label, relative_paths in RUN_PATHS.items():
        paths = {key: reports_dir / relative for key, relative in relative_paths.items()}
        runs[label] = summarize_run(paths)
        if is_complete(paths):
            completed_labels.append(label)
    return {
        "date": date,
        "completed_labels": completed_labels,
        "runs": runs,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def render_markdown(summary: dict[str, Any]) -> str:
    headers = [
        "label",
        "dataset",
        "seed",
        "steps",
        "quality_mse",
        "quality_lpips",
        "gen_inception",
        "path_auc",
        "zero_ratio",
        "shuffle_ratio",
        "tail_ratio",
    ]
    lines = [
        "# Remote Queue Progress Summary",
        "",
        f"Completed labels: {len(summary['completed_labels'])}",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for label, entry in summary["runs"].items():
        row = [
            label,
            entry.get("dataset"),
            entry.get("seed"),
            entry.get("steps"),
            entry.get("quality_final_clean_mse"),
            entry.get("quality_final_lpips_alex"),
            entry.get("generated_inception_frechet"),
            entry.get("quality_prefix_auc_clean_mse"),
            entry.get("zero_token_energy_ratio"),
            entry.get("shuffled_final_mse_ratio"),
            entry.get("tail_energy_ratio"),
        ]
        lines.append("| " + " | ".join(_fmt(value) for value in row) + " |")
    return "\n".join(lines) + "\n"


def write_progress_summary(summary: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "completed_seed2_summary.json"
    markdown_path = output_dir / "completed_seed2_summary.md"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(summary), encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}


def main() -> None:
    args = parse_args()
    summary = build_progress_summary(Path(args.reports_dir), args.date)
    outputs = write_progress_summary(summary, Path(args.output_dir))
    print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
