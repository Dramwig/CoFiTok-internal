from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any


if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))
    sys.path.insert(0, str(project_root))


_PARAMETER_COUNT_CACHE: dict[str, int] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate CoFiTok experiment reports into reusable tables.")
    parser.add_argument(
        "--reports-root",
        default="/root/autodl-tmp/CoFiTok/checkpoints",
        help="Directory containing report subdirectories.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for summary JSON/CSV/Markdown files.")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _get(mapping: dict[str, Any], path: str, default: Any = None) -> Any:
    value: Any = mapping
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _last(values: Any, default: Any = None) -> Any:
    if isinstance(values, list) and values:
        return values[-1]
    return default


def _parameter_count(payload: dict[str, Any]) -> int | None:
    reported = _get(payload, "runtime.parameter_count")
    if reported is not None:
        return int(reported)
    model_payload = _get(payload, "config.model")
    if not isinstance(model_payload, dict):
        return None
    cache_key = json.dumps(model_payload, sort_keys=True)
    if cache_key in _PARAMETER_COUNT_CACHE:
        return _PARAMETER_COUNT_CACHE[cache_key]
    try:
        from cofitok.configs import ModelConfig
        from cofitok.models import CoFiTokTiny

        known = ModelConfig.__dataclass_fields__
        model_config = ModelConfig(
            **{key: value for key, value in model_payload.items() if key in known}
        )
        count = sum(parameter.numel() for parameter in CoFiTokTiny(model_config).parameters())
    except (ImportError, KeyError, RuntimeError, TypeError, ValueError):
        return None
    _PARAMETER_COUNT_CACHE[cache_key] = count
    return count


def infer_variant(config: dict[str, Any]) -> str:
    name = str(config.get("name", "")).lower()
    model = config.get("model", {})
    loss = config.get("loss", {})
    if (
        model.get("synthesis_mode") in {"dense", "dense_identity", "monolithic_dense"}
        or "densehead" in name
        or "dense_monolithic" in name
    ):
        return "dense_monolithic"
    if model.get("synthesis_mode", "restricted") != "restricted" or "deepsk" in name:
        return "deep_synthesis_ablation"
    if model.get("predictor_use_feedback", True) is False or "simultaneous" in name:
        return "simultaneous_predictor"
    if "nopathprefix" in name or "no_path_prefix" in name:
        return "no_prefix_loss_ablation"
    if "cleanmono" in name or "clean_mono" in name:
        return "clean_monotonic_ablation"
    if "decor" in name or loss.get("component_decorrelation_weight", 0.0) > 0.0:
        return "component_decorrelation_ablation"
    if "channelmask" in name:
        return "channel_mask"
    if "epsilononly" in name:
        return "epsilon_only"
    if loss.get("denoise_path_prefix_weight", 0.0) > 0.0 or loss.get("denoise_path_component_weight", 0.0) > 0.0:
        if "light" in name:
            return "light_denoise_path"
        return "full_denoise_path"
    if loss.get("epsilon_band_prefix_weight", 0.0) > 0.0 or loss.get("epsilon_band_component_weight", 0.0) > 0.0:
        return "epsilon_band"
    if loss.get("prefix_weight", 0.0) > 0.0 or loss.get("monotonic_weight", 0.0) > 0.0:
        return "prefix_monotonic"
    return "other"


def _base_row(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    config = payload.get("config", {})
    return {
        "report_dir": path.parent.name,
        "report_path": path.as_posix(),
        "config_name": config.get("name"),
        "dataset": _get(config, "data.dataset"),
        "image_size": _get(config, "data.image_size"),
        "variant": infer_variant(config),
        "token_count": _get(config, "model.token_count"),
        "token_channels": _get(config, "model.token_channels"),
        "predictor_type": _get(config, "model.predictor_type", "tiny_conv"),
        "predictor_multiscale_levels": _get(config, "model.predictor_multiscale_levels", 2),
        "synthesis_mode": _get(config, "model.synthesis_mode", "restricted"),
        "predictor_use_feedback": _get(config, "model.predictor_use_feedback", True),
        "seed": _get(config, "runtime.seed"),
        "steps": _get(config, "runtime.steps"),
        "train_batch_size": _get(config, "data.batch_size"),
        "parameter_count": _parameter_count(payload),
    }


def collect_train_rows(reports_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(reports_root.glob("*/report.json")):
        payload = _read_json(path)
        if "final_losses" not in payload or "prefix_summary" not in payload:
            continue
        prefix = payload["prefix_summary"]
        diagnostics = prefix.get("diagnostics", {})
        row = {
            **_base_row(path, payload),
            "report_type": "train",
            "final_clean_mse": _last(prefix.get("prefix_mse_to_clean")),
            "path_auc": prefix.get("prefix_mse_to_denoise_path_auc"),
            "clean_auc": prefix.get("prefix_mse_to_clean_auc"),
            "effective_tokens": prefix.get("energy_effective_token_count"),
            "late_half_ratio": prefix.get("late_half_energy_ratio"),
            "active_tail_tokens": prefix.get("active_tail_tokens"),
            "zero_token_ratio": diagnostics.get("zero_token_component_energy_ratio"),
            "random_token_ratio": diagnostics.get("random_token_component_energy_ratio"),
            "shuffled_final_ratio": prefix.get("shuffled_final_mse_ratio"),
            "final_epsilon_loss": _get(payload, "final_losses.epsilon"),
            "final_component_decorrelation_loss": _get(payload, "final_losses.component_decorrelation"),
            "final_component_decorrelation_effective_weight": _get(
                payload,
                "final_losses.component_decorrelation_effective_weight",
            ),
            "final_total_loss": _get(payload, "final_losses.total"),
            "elapsed_seconds": _get(payload, "runtime.elapsed_seconds"),
            "checkpoint": _get(payload, "artifacts.checkpoint"),
        }
        rows.append(row)
    return rows


def collect_order_rows(reports_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(reports_root.glob("*/report.json")):
        payload = _read_json(path)
        if "checkpoint" not in payload or "prefix_summary" not in payload or "final_losses" in payload:
            continue
        prefix = payload["prefix_summary"]
        diagnostics = prefix.get("diagnostics", {})
        row = {
            **_base_row(path, payload),
            "report_type": "order_eval",
            "component_order": prefix.get("component_order"),
            "component_order_indices": prefix.get("component_order_indices"),
            "final_clean_mse": _last(prefix.get("prefix_mse_to_clean")),
            "path_auc": prefix.get("prefix_mse_to_denoise_path_auc"),
            "clean_auc": prefix.get("prefix_mse_to_clean_auc"),
            "effective_tokens": prefix.get("energy_effective_token_count"),
            "zero_token_ratio": diagnostics.get("zero_token_component_energy_ratio"),
            "random_token_ratio": diagnostics.get("random_token_component_energy_ratio"),
            "shuffled_final_ratio": prefix.get("shuffled_final_mse_ratio"),
            "checkpoint": payload.get("checkpoint"),
        }
        rows.append(row)
    return rows


def collect_quality_rows(reports_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(reports_root.glob("*/quality_report.json")):
        payload = _read_json(path)
        budgets = payload.get("evaluation", {}).get("prefix_budgets", [])
        if not budgets:
            continue
        final_budget = str(budgets[-1])
        final_metrics = payload.get("metrics_by_prefix", {}).get(final_budget, {})
        row = {
            **_base_row(path, payload),
            "report_type": "quality",
            "split": _get(payload, "evaluation.split"),
            "image_count": _get(payload, "evaluation.image_count"),
            "fixed_timestep": _get(payload, "evaluation.fixed_timestep"),
            "prefix_budgets": budgets,
            "component_order": _get(payload, "evaluation.component_order"),
            "component_order_indices": _get(
                payload, "evaluation.component_order_indices"
            ),
            "random_order_seed": _get(payload, "evaluation.random_order_seed"),
            "final_budget": int(final_budget),
            "final_mse": final_metrics.get("mse"),
            "final_psnr_db": final_metrics.get("psnr_db"),
            "final_lowres_frechet_proxy": final_metrics.get("lowres_frechet_proxy"),
            "final_inception_frechet": final_metrics.get("inception_frechet"),
            "final_lpips_alex": final_metrics.get("lpips_alex"),
            "mse_auc": _get(payload, "curve_auc.mse"),
            "psnr_auc": _get(payload, "curve_auc.psnr_db"),
            "lowres_frechet_proxy_auc": _get(payload, "curve_auc.lowres_frechet_proxy"),
            "denoise_path_mse_auc": _get(payload, "curve_auc.denoise_path_mse"),
            "inception_frechet_auc": _get(payload, "curve_auc.inception_frechet"),
            "lpips_alex_auc": _get(payload, "curve_auc.lpips_alex"),
            "component_mean_abs_cosine": _get(payload, "component_correlation.mean_abs_cosine"),
            "component_max_abs_cosine": _get(payload, "component_correlation.max_abs_cosine"),
            "component_energy_ratios": _get(payload, "component_energy.ratios"),
            "energy_effective_token_count": _get(
                payload, "component_energy.energy_effective_token_count"
            ),
            "zero_token_ratio": _get(
                payload, "diagnostics.zero_token_component_energy_ratio"
            ),
            "endpoint_epsilon_sum_mse": _get(
                payload, "diagnostics.endpoint_epsilon_sum_mse"
            ),
            "checkpoint": payload.get("checkpoint"),
            "lpips_available": _get(payload, "metric_notes.lpips.available"),
            "inception_available": _get(payload, "metric_notes.inception_frechet.available"),
        }
        rows.append(row)
    return rows


def collect_sample_rows(reports_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(reports_root.glob("*/sample_report.json")):
        payload = _read_json(path)
        if "sampling" not in payload:
            continue
        artifacts = payload.get("artifacts", {})
        rows.append(
            {
                **_base_row(path, payload),
                "report_type": "sampling",
                "checkpoint": payload.get("checkpoint"),
                "num_samples": _get(payload, "sampling.num_samples"),
                "batch_size": _get(payload, "sampling.batch_size"),
                "sample_steps": _get(payload, "sampling.sample_steps"),
                "actual_timestep_count": _get(payload, "sampling.actual_timestep_count"),
                "first_timestep": _get(payload, "sampling.first_timestep"),
                "last_timestep": _get(payload, "sampling.last_timestep"),
                "prefix_budgets": _get(payload, "sampling.prefix_budgets"),
                "eta": _get(payload, "sampling.eta"),
                "clip_x0": _get(payload, "sampling.clip_x0"),
                "sampling_seed": _get(payload, "sampling.seed"),
                "elapsed_seconds": _get(payload, "runtime.elapsed_seconds"),
                "artifact_count": len(artifacts),
                "artifacts": artifacts,
            }
        )
    return rows


def _infer_token_count_from_name(name: str) -> int | None:
    match = re.search(r"_k(\d+)(?:_|$)", name.lower())
    return int(match.group(1)) if match else None


def _infer_steps_from_name(name: str) -> int | None:
    matches = re.findall(r"_(\d+)k(?:_|$)", name.lower())
    return int(matches[-1]) * 1000 if matches else None


def _infer_predictor_type_from_generated_name(config_name: str, report_dir: str) -> str:
    combined = f"{config_name} {report_dir}".lower()
    if "multiscale" in combined:
        return "multiscale_unet"
    return "tiny_conv"


def collect_generated_quality_rows(reports_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(reports_root.glob("*/generated_quality_report.json")):
        payload = _read_json(path)
        config_name = str(payload.get("config_name", ""))
        report_dir = path.parent.name
        dataset = payload.get("dataset")
        metrics = payload.get("metrics", {})
        evaluation = payload.get("evaluation", {})
        sampling = payload.get("sampling", {})
        variant = infer_variant(
            {
                "name": config_name,
                "model": {},
                "loss": {
                    "denoise_path_prefix_weight": (
                        1.0 if "denoisepath" in config_name.lower() else 0.0
                    )
                },
            }
        )
        rows.append(
            {
                "report_dir": report_dir,
                "report_path": path.as_posix(),
                "report_type": "generated_quality",
                "config_name": config_name,
                "dataset": dataset,
                "variant": variant,
                "token_count": (
                    1 if variant == "dense_monolithic" else _infer_token_count_from_name(config_name)
                ),
                "steps": _infer_steps_from_name(config_name),
                "predictor_type": _infer_predictor_type_from_generated_name(config_name, report_dir),
                "split": evaluation.get("split"),
                "real_image_count": evaluation.get("real_image_count"),
                "sample_image_count": evaluation.get("sample_image_count"),
                "available_sample_images": evaluation.get("available_sample_images"),
                "feature_size": evaluation.get("feature_size"),
                "sample_steps": sampling.get("sample_steps"),
                "prefix_budget": sampling.get("prefix_budget"),
                "sampling_batch_size": sampling.get("batch_size"),
                "sampling_eta": sampling.get("eta"),
                "sampling_seed": evaluation.get("seed"),
                "lowres_frechet_proxy": metrics.get("lowres_frechet_proxy"),
                "inception_frechet": metrics.get("inception_frechet"),
                "inception_available": _get(payload, "metric_notes.inception_frechet.available"),
                "samples_dir": payload.get("samples_dir"),
                "elapsed_seconds": _get(payload, "runtime.elapsed_seconds"),
            }
        )
    return rows


def _official_fid_report_paths(reports_root: Path) -> list[Path]:
    patterns = [
        "official_fid_export_*/official_fid_report/official_fid_report.json",
        "official_fid_protocol_*/official_fid_export_*/official_fid_report.json",
    ]
    seen: set[Path] = set()
    paths: list[Path] = []
    for pattern in patterns:
        for path in sorted(reports_root.glob(pattern)):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            paths.append(path)
    return paths


def _official_fid_manifest_path(report_path: Path) -> Path:
    if report_path.parent.name == "official_fid_report":
        return report_path.parent.parent / "official_fid_export_manifest.json"
    return report_path.parent / "official_fid_export_manifest.json"


def _official_fid_export_name(report_path: Path) -> str:
    if report_path.parent.name == "official_fid_report":
        return report_path.parent.parent.name
    return report_path.parent.name


def collect_official_fid_rows(reports_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in _official_fid_report_paths(reports_root):
        payload = _read_json(path)
        manifest_path = _official_fid_manifest_path(path)
        manifest = _read_json(manifest_path) if manifest_path.is_file() else {}
        config = manifest.get("config", {})
        config_name = str(config.get("name") or manifest.get("config_name") or "")
        rows.append(
            {
                "report_dir": _official_fid_export_name(path),
                "report_type": "official_fid",
                "config_name": config_name,
                "dataset": _get(config, "data.dataset"),
                "variant": infer_variant(config if config else {"name": config_name, "model": {}, "loss": {}}),
                "token_count": _get(config, "model.token_count") or _infer_token_count_from_name(config_name),
                "steps": _get(config, "runtime.steps") or _infer_steps_from_name(config_name),
                "predictor_type": _get(config, "model.predictor_type")
                or _infer_predictor_type_from_generated_name(config_name, _official_fid_export_name(path)),
                "split": _get(manifest, "sampling.split") or _get(config, "data.split"),
                "real_image_count": _get(payload, "counts.real_image_count"),
                "generated_image_count": _get(payload, "counts.generated_image_count"),
                "official_fid": _get(payload, "metrics.fid"),
                "metric": payload.get("metric"),
                "status": payload.get("status"),
                "implementation_package": _get(payload, "implementation.package"),
                "implementation_available": _get(payload, "implementation.available"),
                "batch_size": _get(payload, "parameters.batch_size"),
                "device": _get(payload, "parameters.device"),
                "dims": _get(payload, "parameters.dims"),
                "sample_steps": _get(manifest, "sampling.sample_steps"),
                "prefix_budget": _get(manifest, "sampling.prefix_budget"),
                "elapsed_seconds": _get(payload, "runtime.elapsed_seconds"),
                "report_path": path.as_posix(),
                "manifest_path": manifest_path.as_posix() if manifest_path.is_file() else "",
                "real_dir": _get(payload, "paths.real_dir"),
                "generated_dir": _get(payload, "paths.generated_dir"),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in row.items()})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _format_float(value: Any, digits: int = 4) -> str:
    if isinstance(value, (float, int)):
        return f"{float(value):.{digits}f}"
    return ""


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(item) for item in row) + " |" for row in rows)
    return "\n".join(lines)


def _curated_quality_table(quality_rows: list[dict[str, Any]]) -> str:
    wanted = {
        ("cifar10", "channel_mask"),
        ("cifar10", "epsilon_only"),
        ("cifar10", "light_denoise_path"),
        ("tiny_imagenet_200", "channel_mask"),
        ("tiny_imagenet_200", "epsilon_only"),
        ("tiny_imagenet_200", "light_denoise_path"),
        ("imagenet_1k_64x64_hf", "channel_mask"),
        ("imagenet_1k_64x64_hf", "epsilon_only"),
        ("imagenet_1k_64x64_hf", "light_denoise_path"),
    }
    selected_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in quality_rows:
        key = (str(row.get("dataset")), str(row.get("variant")))
        if key not in wanted or row.get("token_count") != 8:
            continue
        previous = selected_by_key.get(key)
        if previous is None:
            selected_by_key[key] = row
            continue
        row_score = int(row.get("lpips_available") is True) + int(row.get("inception_available") is True)
        previous_score = int(previous.get("lpips_available") is True) + int(previous.get("inception_available") is True)
        if row_score > previous_score:
            selected_by_key[key] = row
    selected = sorted(selected_by_key.values(), key=lambda row: (str(row["dataset"]), str(row["variant"])))
    rows = [
        [
            row["dataset"],
            row["variant"],
            row.get("steps"),
            _format_float(row.get("final_mse")),
            _format_float(row.get("final_psnr_db"), 3),
            _format_float(row.get("final_lowres_frechet_proxy")),
            _format_float(row.get("final_inception_frechet")),
            _format_float(row.get("final_lpips_alex")),
            _format_float(row.get("mse_auc")),
            _format_float(row.get("lowres_frechet_proxy_auc")),
            _format_float(row.get("inception_frechet_auc")),
            _format_float(row.get("lpips_alex_auc")),
        ]
        for row in selected
    ]
    return _markdown_table(
        [
            "dataset",
            "variant",
            "steps",
            "final MSE",
            "final PSNR",
            "final proxy",
            "final Inception",
            "final LPIPS",
            "MSE AUC",
            "proxy AUC",
            "Inception AUC",
            "LPIPS AUC",
        ],
        rows,
    )


def _curated_imagenet_table(train_rows: list[dict[str, Any]]) -> str:
    selected = [
        row for row in train_rows
        if row.get("dataset") == "imagenet_1k_64x64_hf"
        and row.get("token_count") == 8
        and row.get("variant") in {
            "channel_mask",
            "epsilon_only",
            "full_denoise_path",
            "light_denoise_path",
            "simultaneous_predictor",
            "deep_synthesis_ablation",
        }
    ]
    preferred_order = {
        "channel_mask": 0,
        "epsilon_only": 1,
        "full_denoise_path": 2,
        "light_denoise_path": 3,
        "simultaneous_predictor": 4,
        "deep_synthesis_ablation": 5,
    }
    selected.sort(key=lambda row: (preferred_order.get(str(row["variant"]), 99), str(row["report_dir"])))
    rows = [
        [
            row["variant"],
            row.get("seed"),
            _format_float(row.get("final_clean_mse")),
            _format_float(row.get("path_auc")),
            _format_float(row.get("effective_tokens"), 3),
            _format_float(row.get("zero_token_ratio")),
            _format_float(row.get("shuffled_final_ratio"), 1),
        ]
        for row in selected
    ]
    return _markdown_table(
        ["variant", "seed", "final MSE", "path AUC", "effective K", "zero ratio", "shuffle ratio"],
        rows,
    )


def _curated_order_table(order_rows: list[dict[str, Any]]) -> str:
    selected = [
        row for row in order_rows
        if row.get("dataset") == "imagenet_1k_64x64_hf"
        and row.get("variant") == "light_denoise_path"
        and row.get("token_count") in {4, 8, 16}
        and row.get("seed") == 139
    ]
    selected.sort(key=lambda row: (int(row.get("token_count") or 0), str(row.get("component_order"))))
    rows = [
        [
            row.get("token_count"),
            row.get("component_order"),
            _format_float(row.get("final_clean_mse")),
            _format_float(row.get("path_auc")),
            _format_float(row.get("zero_token_ratio")),
        ]
        for row in selected
    ]
    return _markdown_table(["K", "order", "final MSE", "path AUC", "zero ratio"], rows)


def _curated_sampling_table(sample_rows: list[dict[str, Any]]) -> str:
    selected = [
        row for row in sample_rows
        if row.get("variant") in {"epsilon_only", "light_denoise_path"}
        and row.get("token_count") == 8
        and row.get("sample_steps") == 20
    ]
    selected.sort(
        key=lambda row: (
            str(row.get("dataset")),
            int(row.get("steps") or 0),
            int(row.get("sample_steps") or 0),
            int(row.get("num_samples") or 0),
        )
    )
    rows = [
        [
            row.get("dataset"),
            row.get("variant"),
            row.get("steps"),
            row.get("num_samples"),
            row.get("sample_steps"),
            row.get("prefix_budgets"),
            row.get("eta"),
            row.get("sampling_seed"),
            row.get("artifact_count"),
        ]
        for row in selected
    ]
    return _markdown_table(
        ["dataset", "variant", "train steps", "samples", "DDIM steps", "prefix budgets", "eta", "seed", "artifact count"],
        rows,
    )


def _curated_generated_quality_table(generated_quality_rows: list[dict[str, Any]]) -> str:
    selected = [
        row for row in generated_quality_rows
        if row.get("variant") in {"epsilon_only", "light_denoise_path"}
        and row.get("token_count") == 8
    ]
    selected.sort(
        key=lambda row: (
            str(row.get("dataset")),
            int(row.get("steps") or 0),
            int(row.get("sample_image_count") or 0),
        )
    )
    rows = [
        [
            row.get("dataset"),
            row.get("variant"),
            row.get("predictor_type"),
            row.get("steps"),
            row.get("sample_image_count"),
            row.get("real_image_count"),
            _format_float(row.get("lowres_frechet_proxy")),
            _format_float(row.get("inception_frechet")),
        ]
        for row in selected
    ]
    return _markdown_table(
        ["dataset", "variant", "backbone", "train steps", "samples", "real images", "sample proxy", "sample Inception"],
        rows,
    )


def _curated_official_fid_table(official_fid_rows: list[dict[str, Any]]) -> str:
    selected = [
        row for row in official_fid_rows
        if row.get("variant") in {"epsilon_only", "light_denoise_path"}
        and row.get("token_count") == 8
    ]
    selected.sort(
        key=lambda row: (
            str(row.get("dataset")),
            str(row.get("variant")),
            int(row.get("generated_image_count") or 0),
        )
    )
    rows = [
        [
            row.get("dataset"),
            row.get("variant"),
            row.get("predictor_type"),
            row.get("steps"),
            row.get("generated_image_count"),
            row.get("real_image_count"),
            row.get("sample_steps"),
            row.get("prefix_budget"),
            _format_float(row.get("official_fid")),
            row.get("implementation_package"),
            row.get("status"),
        ]
        for row in selected
    ]
    return _markdown_table(
        [
            "dataset",
            "variant",
            "backbone",
            "train steps",
            "generated",
            "real",
            "DDIM steps",
            "prefix budget",
            "official FID",
            "implementation",
            "status",
        ],
        rows,
    )


def write_markdown_summary(
    path: Path,
    train_rows: list[dict[str, Any]],
    order_rows: list[dict[str, Any]],
    quality_rows: list[dict[str, Any]],
    sample_rows: list[dict[str, Any]],
    generated_quality_rows: list[dict[str, Any]],
    official_fid_rows: list[dict[str, Any]],
) -> None:
    body = "\n\n".join(
        [
            "# CoFiTok Experiment Summary",
            "Generated from report JSON files. The low-res Frechet proxy is not formal Inception FID.",
            "## ImageNet-64 HF Training Matrix",
            _curated_imagenet_table(train_rows),
            "## ImageNet-64 HF Order Matrix",
            _curated_order_table(order_rows),
            "## Cross-Dataset Quality Matrix",
            _curated_quality_table(quality_rows),
            "## Prefix-Aware Sampling Smoke Matrix",
            _curated_sampling_table(sample_rows),
            "## Generated Sample Quality Smoke Matrix",
            _curated_generated_quality_table(generated_quality_rows),
            "## Official FID Matrix",
            _curated_official_fid_table(official_fid_rows),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    reports_root = Path(args.reports_root)
    output_dir = Path(args.output_dir)
    train_rows = collect_train_rows(reports_root)
    order_rows = collect_order_rows(reports_root)
    quality_rows = collect_quality_rows(reports_root)
    sample_rows = collect_sample_rows(reports_root)
    generated_quality_rows = collect_generated_quality_rows(reports_root)
    official_fid_rows = collect_official_fid_rows(reports_root)
    payload = {
        "reports_root": str(reports_root),
        "counts": {
            "train": len(train_rows),
            "order_eval": len(order_rows),
            "quality": len(quality_rows),
            "sampling": len(sample_rows),
            "generated_quality": len(generated_quality_rows),
            "official_fid": len(official_fid_rows),
        },
        "train": train_rows,
        "order_eval": order_rows,
        "quality": quality_rows,
        "sampling": sample_rows,
        "generated_quality": generated_quality_rows,
        "official_fid": official_fid_rows,
    }
    _write_json(output_dir / "experiment_summary.json", payload)
    _write_csv(output_dir / "train_summary.csv", train_rows)
    _write_csv(output_dir / "order_eval_summary.csv", order_rows)
    _write_csv(output_dir / "quality_summary.csv", quality_rows)
    _write_csv(output_dir / "sample_summary.csv", sample_rows)
    _write_csv(output_dir / "generated_quality_summary.csv", generated_quality_rows)
    _write_csv(output_dir / "official_fid_summary.csv", official_fid_rows)
    write_markdown_summary(
        output_dir / "core_summary.md",
        train_rows,
        order_rows,
        quality_rows,
        sample_rows,
        generated_quality_rows,
        official_fid_rows,
    )
    print(f"wrote {output_dir / 'experiment_summary.json'}")
    print(f"wrote {output_dir / 'core_summary.md'}")


if __name__ == "__main__":
    main()
