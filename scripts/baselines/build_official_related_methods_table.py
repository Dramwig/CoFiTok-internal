#!/usr/bin/env python
"""Build the secondary official-checkpoint related-method table."""

from __future__ import annotations

import argparse
import json
import math
import zipfile
from pathlib import Path
from typing import Any

import numpy as np


DAR_SAMPLE_STEM = (
    "GPT-L-D-AR-L-360K-size-256-size-256-VQ-16-topk-0-topp-1.0-"
    "temperature-1.0-cfg-1.2,8.0-seed-0-None"
)
RETOK_SAMPLE_STEM = (
    "GPT-XL-size-256-size-256-topk-0-topp-1.0-temperature-1.0-"
    "cfg-1.5-step-seed-0"
)
MAR_COMMUNITY_SAMPLE_STEM = (
    "MAR-B-hf-size-256-ariter-256-diffsteps-100-cfg-2.9-linear-"
    "temp-1.0-seed-0"
)
MAR_OFFICIAL_SAMPLE_STEM = (
    "MAR-B-official-pth-ema-size-256-ariter-256-diffsteps-100-cfg-2.9-"
    "linear-temp-1.0-seed-0"
)
# Backward-compatible name for the community-conversion sample stem.
MAR_SAMPLE_STEM = MAR_COMMUNITY_SAMPLE_STEM
REQUIRED_EVAL_METRICS = {
    "fid",
    "sfid",
    "inception_score",
    "precision",
    "recall",
}
MAR_MODEL_SHA256 = "7e970a33bc90353e2fabe3498ed1f2d194dd8d17cd387665f80b2984dfca538c"
MAR_VAE_SHA256 = "34ce001bcfffb7af67ec8af1e683a30d7bd45760855ddc7deedc1330f2cfd38f"
MAR_REPO_COMMIT = "c6d53f7fa6427634b5850ebed771b7c2d19ea21f"


def parse_metric_txt(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    aliases = {
        "Inception Score": "inception_score",
        "FID": "fid",
        "sFID": "sfid",
        "Precision": "precision",
        "Recall": "recall",
    }
    metrics: dict[str, float] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in raw_line:
            continue
        key, value = raw_line.split(":", 1)
        key = key.strip()
        if key not in aliases:
            continue
        try:
            metrics[aliases[key]] = float(value.strip())
        except ValueError:
            continue
    return metrics


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def count_pngs(path: Path) -> int:
    return sum(1 for _ in path.glob("*.png")) if path.exists() else 0


def complete_metric_set(metrics: dict[str, float]) -> bool:
    return REQUIRED_EVAL_METRICS <= metrics.keys() and all(
        math.isfinite(metrics[key]) for key in REQUIRED_EVAL_METRICS
    )


def npz_array_metadata(path: Path, key: str = "arr_0") -> dict[str, Any]:
    try:
        with zipfile.ZipFile(path) as archive:
            with archive.open(f"{key}.npy") as handle:
                version = np.lib.format.read_magic(handle)
                if version == (1, 0):
                    shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(
                        handle
                    )
                elif version == (2, 0):
                    shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(
                        handle
                    )
                else:
                    return {}
    except (FileNotFoundError, KeyError, OSError, ValueError, zipfile.BadZipFile):
        return {}
    return {
        "shape": list(shape),
        "fortran_order": bool(fortran_order),
        "dtype": str(dtype),
        "key": key,
    }


def adm_npz_is_valid(
    path: Path, expected_shape: tuple[int, ...] = (50_000, 256, 256, 3)
) -> bool:
    metadata = npz_array_metadata(path)
    return (
        metadata.get("shape") == list(expected_shape)
        and metadata.get("dtype") == "uint8"
        and metadata.get("fortran_order") is False
    )


def inspect_sample_run(root: Path, stem: str) -> dict[str, Any]:
    sample_dir = root / stem
    npz = root / f"{stem}.npz"
    metrics_txt = root / f"{stem}.txt"
    generation_report = sample_dir / "baseline_eval_report.json"
    report = read_json(generation_report)
    png_count = count_pngs(sample_dir)
    metrics = parse_metric_txt(metrics_txt)
    return {
        "sample_dir": sample_dir,
        "npz": npz,
        "metrics_txt": metrics_txt,
        "generation_report": generation_report,
        "report": report,
        "png_count": png_count,
        "metrics": metrics,
        "complete": png_count >= 50_000
        and adm_npz_is_valid(npz)
        and complete_metric_set(metrics),
        "partial": png_count > 0
        or report.get("status")
        in {
            "running",
            "running_or_partial",
            "samples_completed",
            "samples_and_npz_completed_eval_pending",
        },
    }


def official_mar_ema_provenance_is_valid(run: dict[str, Any]) -> bool:
    if not run["complete"]:
        return False
    report = run["report"]
    parameters = report.get("parameters", {})
    artifacts = report.get("artifacts", {})
    weights = artifacts.get("weights", {})
    model = weights.get("model", {})
    vae = weights.get("vae", {})
    repo = weights.get("official_repo", {})
    packed = artifacts.get("sample_npz", {})
    progress = report.get("progress", {})
    return all(
        (
            report.get("protocol") == "official_lth14_pth_ema_eval_only",
            report.get("status") == "samples_and_npz_completed_eval_pending",
            int(parameters.get("num_images", 0)) == 50_000,
            int(parameters.get("class_num", 0)) == 1_000,
            int(parameters.get("balanced_images_per_class", 0)) == 50,
            int(parameters.get("num_ar_steps", 0)) == 256,
            int(parameters.get("num_sampling_steps", 0)) == 100,
            float(parameters.get("cfg_scale", 0.0)) == 2.9,
            parameters.get("cfg_schedule") == "linear",
            float(parameters.get("temperature", 0.0)) == 1.0,
            int(parameters.get("seed", -1)) == 0,
            model.get("checkpoint_state") == "model_ema",
            model.get("sha256") == MAR_MODEL_SHA256,
            vae.get("checkpoint_state") == "model",
            vae.get("sha256") == MAR_VAE_SHA256,
            repo.get("commit") == MAR_REPO_COMMIT,
            packed.get("shape") == [50_000, 256, 256, 3],
            packed.get("dtype") == "uint8",
            packed.get("key") == "arr_0",
            int(progress.get("completed_images", 0)) == 50_000,
            int(progress.get("expected_images", 0)) == 50_000,
        )
    )


def build_rows(project_root: Path) -> list[dict[str, Any]]:
    internal = project_root / "CoFiTok-internal"
    samples_root = (
        project_root
        / "checkpoints/baselines/d_ar/official_imagenet256_eval_only/samples"
    )
    dar_sample_dir = samples_root / DAR_SAMPLE_STEM
    dar_npz = samples_root / f"{DAR_SAMPLE_STEM}.npz"
    dar_txt = samples_root / f"{DAR_SAMPLE_STEM}.txt"
    dar_png_count = count_pngs(dar_sample_dir)
    dar_metrics = parse_metric_txt(dar_txt)
    dar_status = "running_or_partial"
    dar_sample_count = 0
    if (
        dar_png_count >= 50_000
        and adm_npz_is_valid(dar_npz)
        and complete_metric_set(dar_metrics)
    ):
        dar_status = "completed_eval_only_50k"
        dar_sample_count = 50000
    elif dar_png_count > 0:
        dar_status = "sampling_or_eval_pending"
        dar_sample_count = dar_png_count

    mar_smoke_report = read_json(
        internal
        / "artifacts/reports/baselines/mar/official_hf_imagenet256_smoke_1_2026-07-10/baseline_eval_report.json"
    )
    mar_probe_report = read_json(
        internal
        / "artifacts/reports/baselines/mar/official_imagenet256_probe_2026-07-10/baseline_eval_report.json"
    )
    mar_community_root = (
        project_root
        / "checkpoints/baselines/mar/official_imagenet256_eval_only/samples_50k_hf"
    )
    mar_official_root = (
        project_root
        / "checkpoints/baselines/mar/official_imagenet256_eval_only/samples_50k_official_pth"
    )
    mar_community = inspect_sample_run(mar_community_root, MAR_COMMUNITY_SAMPLE_STEM)
    mar_official = inspect_sample_run(mar_official_root, MAR_OFFICIAL_SAMPLE_STEM)
    mar_npz = ""
    mar_txt = ""
    mar_generation_report = ""
    mar_metrics: dict[str, float] = {}

    if official_mar_ema_provenance_is_valid(mar_official):
        mar_status = "completed_eval_only_50k"
        mar_dataset = "imagenet_256"
        mar_resolution = 256
        mar_sample_count = 50_000
        mar_png_count = mar_official["png_count"]
        mar_npz = str(mar_official["npz"])
        mar_txt = str(mar_official["metrics_txt"])
        mar_generation_report = str(mar_official["generation_report"])
        mar_metrics = mar_official["metrics"]
        mar_protocol = "official LTH14 PTH model_ema eval-only"
        mar_source_kind = "official_lth14_pth_model_ema"
        mar_notes = (
            "Pinned LTH14 MAR-B model_ema 50K eval-only completed. "
            "Do not merge into P0 same-budget table."
        )
    elif mar_official["partial"]:
        mar_status = "official_ema_sampling_or_eval_pending"
        mar_dataset = "imagenet_256"
        mar_resolution = 256
        mar_png_count = mar_official["png_count"]
        mar_sample_count = max(
            mar_png_count,
            int(mar_official["report"].get("progress", {}).get("completed_images", 0)),
        )
        mar_npz = str(mar_official["npz"]) if mar_official["npz"].exists() else ""
        mar_txt = (
            str(mar_official["metrics_txt"])
            if mar_official["metrics_txt"].exists()
            else ""
        )
        mar_generation_report = str(mar_official["generation_report"])
        mar_metrics = mar_official["metrics"]
        mar_protocol = "official LTH14 PTH model_ema eval-only"
        mar_source_kind = "official_lth14_pth_model_ema"
        mar_notes = "Pinned LTH14 MAR-B model_ema 50K sampling or evaluation is pending."
    elif mar_community["complete"]:
        mar_status = "community_non_ema_audit_completed_official_ema_pending"
        mar_dataset = "imagenet_256"
        mar_resolution = 256
        mar_sample_count = 50_000
        mar_png_count = mar_community["png_count"]
        mar_npz = str(mar_community["npz"])
        mar_txt = str(mar_community["metrics_txt"])
        mar_generation_report = str(mar_community["generation_report"])
        mar_metrics = mar_community["metrics"]
        mar_protocol = "community non-EMA conversion audit"
        mar_source_kind = "community_hf_safetensors_non_ema"
        mar_notes = (
            "Community safetensors exactly match checkpoint['model'], not model_ema; "
            "keep metrics as an audit only while the official EMA run is pending."
        )
    elif mar_smoke_report.get("status") == "completed":
        mar_status = "hf_safetensors_smoke_completed_50k_not_run"
        mar_dataset = mar_smoke_report.get("dataset", "imagenet_256")
        mar_resolution = mar_smoke_report.get("resolution", 256)
        mar_sample_count = mar_smoke_report.get("parameters", {}).get("num_images", 1)
        mar_png_count = 0
        mar_protocol = "community conversion smoke"
        mar_source_kind = "community_hf_safetensors_non_ema"
        mar_notes = (
            "Community safetensors load/generation smoke passed; no official EMA 50K metrics."
        )
    else:
        mar_status = mar_probe_report.get("status", "blocked_weight_acquisition")
        mar_dataset = mar_probe_report.get("dataset_alias", "imagenet_256")
        mar_resolution = mar_probe_report.get("resolution", 256)
        mar_sample_count = mar_probe_report.get("sample_count", 0)
        mar_png_count = 0
        mar_protocol = "official LTH14 PTH model_ema eval-only"
        mar_source_kind = "official_lth14_pth_model_ema"
        mar_notes = "Dependencies ready; official weights blocked."

    retok_generation_report = read_json(
        internal
        / "artifacts/reports/baselines/retok/official_generation_smoke_2_2026-07-10/baseline_eval_report.json"
    )
    retok_reconstruction_report = read_json(
        internal
        / "artifacts/reports/baselines/retok/official_reconstruction_smoke_4_2026-07-10/baseline_eval_report.json"
    )
    retok_config = read_json(
        internal
        / "configs/baselines/retok/official_imagenet256_eval_only_2026-07-10.json"
    )
    retok_50k_root = (
        project_root
        / "checkpoints/baselines/retok/official_imagenet256_eval_only/samples_50k"
    )
    retok_50k_dir = retok_50k_root / RETOK_SAMPLE_STEM
    retok_50k_npz = retok_50k_root / f"{RETOK_SAMPLE_STEM}.npz"
    retok_50k_txt = retok_50k_root / f"{RETOK_SAMPLE_STEM}.txt"
    retok_50k_png_count = count_pngs(retok_50k_dir)
    retok_50k_metrics = parse_metric_txt(retok_50k_txt)

    retok_pilot_root = (
        project_root
        / "checkpoints/baselines/retok/official_imagenet256_eval_only/samples_pilot128"
    )
    retok_pilot_npz = retok_pilot_root / f"{RETOK_SAMPLE_STEM}.npz"
    retok_pilot_txt = retok_pilot_root / f"{RETOK_SAMPLE_STEM}.txt"
    retok_pilot_metrics = parse_metric_txt(retok_pilot_txt)

    retok_npz = ""
    retok_txt = ""
    retok_metrics: dict[str, float] = {}
    if (
        retok_50k_png_count >= 50_000
        and adm_npz_is_valid(retok_50k_npz)
        and complete_metric_set(retok_50k_metrics)
    ):
        retok_status = "completed_eval_only_50k"
        retok_dataset = "imagenet_256"
        retok_resolution = 256
        retok_sample_count = 50000
        retok_npz = str(retok_50k_npz)
        retok_txt = str(retok_50k_txt)
        retok_metrics = retok_50k_metrics
        retok_notes = (
            "Official GPT-XL + VQ 50K eval-only completed. Do not merge into P0 same-budget table."
        )
    elif retok_pilot_npz.exists() and retok_pilot_metrics:
        retok_status = "pilot128_metrics_completed_50k_running"
        retok_dataset = "imagenet_256"
        retok_resolution = 256
        retok_sample_count = 128
        retok_npz = str(retok_pilot_npz)
        retok_txt = str(retok_pilot_txt)
        retok_metrics = retok_pilot_metrics
        retok_notes = (
            "Official GPT-XL + VQ 128-sample evaluator pilot passed; 50K eval-only is running/pending."
        )
    elif retok_generation_report.get("status") == "completed":
        retok_status = "official_gpt_vq_generation_smoke_passed_50k_not_run"
        retok_dataset = retok_generation_report.get("dataset", "imagenet_256")
        retok_resolution = retok_generation_report.get("resolution", 256)
        retok_sample_count = retok_generation_report.get("parameters", {}).get("num_fid_samples", 2)
        retok_notes = (
            "Official GPT-XL + VQ generation smoke passed; no 50K official metrics."
        )
    elif retok_reconstruction_report.get("status") == "completed":
        retok_status = "vq_reconstruction_smoke_completed_gpt_generation_pending"
        retok_dataset = retok_reconstruction_report.get("dataset", "imagenet_256")
        retok_resolution = retok_reconstruction_report.get("resolution", 256)
        retok_sample_count = retok_reconstruction_report.get("parameters", {}).get("num_images", 4)
        retok_notes = (
            "Official VQ checkpoint reconstruction smoke passed; GPT generation/50K metrics pending."
        )
    else:
        retok_status = retok_config.get("matrix_status", "protocol_blocked")
        retok_dataset = retok_config.get("dataset_alias", "imagenet_256")
        retok_resolution = retok_config.get("resolution", 256)
        retok_sample_count = 0
        retok_notes = "Official weights require separate acquisition."

    return [
        {
            "method": "D-AR",
            "alias": "d_ar",
            "dataset": "imagenet_256",
            "resolution": 256,
            "protocol": "official pretrained eval-only",
            "status": dar_status,
            "sample_count": dar_sample_count,
            "png_count": dar_png_count,
            "npz": str(dar_npz),
            "metrics_txt": str(dar_txt),
            "inception_score": dar_metrics.get("inception_score"),
            "fid": dar_metrics.get("fid"),
            "sfid": dar_metrics.get("sfid"),
            "precision": dar_metrics.get("precision"),
            "recall": dar_metrics.get("recall"),
            "paper_table_role": "secondary related-method only",
            "notes": "Do not merge into P0 same-budget table.",
        },
        {
            "method": "MAR",
            "alias": "mar",
            "dataset": mar_dataset,
            "resolution": mar_resolution,
            "protocol": mar_protocol,
            "status": mar_status,
            "sample_count": mar_sample_count,
            "png_count": mar_png_count,
            "npz": mar_npz,
            "metrics_txt": mar_txt,
            "generation_report": mar_generation_report,
            "source_kind": mar_source_kind,
            "inception_score": mar_metrics.get("inception_score"),
            "fid": mar_metrics.get("fid"),
            "sfid": mar_metrics.get("sfid"),
            "precision": mar_metrics.get("precision"),
            "recall": mar_metrics.get("recall"),
            "paper_table_role": "secondary related-method only",
            "notes": mar_notes,
        },
        {
            "method": "ReTok",
            "alias": "retok",
            "dataset": retok_dataset,
            "resolution": retok_resolution,
            "protocol": "official pretrained eval-only",
            "status": retok_status,
            "sample_count": retok_sample_count,
            "png_count": retok_50k_png_count,
            "npz": retok_npz,
            "metrics_txt": retok_txt,
            "inception_score": retok_metrics.get("inception_score"),
            "fid": retok_metrics.get("fid"),
            "sfid": retok_metrics.get("sfid"),
            "precision": retok_metrics.get("precision"),
            "recall": retok_metrics.get("recall"),
            "paper_table_role": "secondary related-method only",
            "notes": retok_notes,
        },
    ]


def write_markdown(rows: list[dict[str, Any]], output: Path) -> None:
    lines = [
        "# Official Related-Method Table",
        "",
        "This table is separate from the P0 same-dataset, same-budget generation table.",
        "",
        "| method | dataset | protocol | status | samples | FID | sFID | IS | precision | recall | role |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {method} | {dataset} | {protocol} | {status} | {samples} | {fid} | {sfid} | {iscore} | {precision} | {recall} | {role} |".format(
                method=row["method"],
                dataset=row["dataset"],
                protocol=row["protocol"],
                status=row["status"],
                samples=fmt(row["sample_count"], 0),
                fid=fmt(row["fid"]),
                sfid=fmt(row["sfid"]),
                iscore=fmt(row["inception_score"]),
                precision=fmt(row["precision"]),
                recall=fmt(row["recall"]),
                role=row["paper_table_role"],
            )
        )
    lines.extend(
        [
            "",
            "Interpretation rule: completed rows here may be cited only as official-checkpoint related-method evidence. They are not fair retraining baselines for the all-dataset P0 matrix.",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def write_dar_report(project_root: Path, rows: list[dict[str, Any]]) -> None:
    dar = next((row for row in rows if row["alias"] == "d_ar"), None)
    if not dar or dar["status"] != "completed_eval_only_50k":
        return
    internal = project_root / "CoFiTok-internal"
    out_dir = (
        internal
        / "artifacts/reports/baselines/d_ar/official_imagenet256_50k_2026-07-10"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "baseline": "d_ar",
        "protocol": "official_imagenet256_eval_only_50k",
        "status": "completed_eval_only",
        "created_at": "2026-07-10",
        "dataset_alias": "imagenet_256",
        "resolution": 256,
        "conditioning": "ImageNet class-conditional",
        "repo_dir": str(project_root / "baselines/repos/d_ar"),
        "weights": {
            "tokenizer": str(
                project_root
                / "checkpoints/baselines/d_ar/official_imagenet256_eval_only/weights/D-AR-tokenizer_v1.pt"
            ),
            "gpt": str(
                project_root
                / "checkpoints/baselines/d_ar/official_imagenet256_eval_only/weights/D-AR-L-360K.pt"
            ),
        },
        "sample_npz": dar["npz"],
        "sample_shape": [50000, 256, 256, 3],
        "sample_count": 50000,
        "png_count": dar.get("png_count", dar["sample_count"]),
        "reference_npz": str(
            project_root
            / "checkpoints/baselines/official_refs/VIRTUAL_imagenet256_labeled.npz"
        ),
        "metrics": {
            "inception_score": dar["inception_score"],
            "fid": dar["fid"],
            "sfid": dar["sfid"],
            "precision": dar["precision"],
            "recall": dar["recall"],
        },
        "limitations": [
            "Official ImageNet-256 pretrained eval-only, not same-dataset retraining.",
            "Use only as a secondary related-method row.",
            "Do not merge into the P0 all-dataset same-budget generation table.",
        ],
    }
    (out_dir / "baseline_eval_report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_mar_report(project_root: Path, rows: list[dict[str, Any]]) -> None:
    mar = next((row for row in rows if row["alias"] == "mar"), None)
    if not mar or mar["status"] != "completed_eval_only_50k":
        return
    if mar.get("source_kind") != "official_lth14_pth_model_ema":
        return
    internal = project_root / "CoFiTok-internal"
    source_report = Path(mar["generation_report"])
    payload = read_json(source_report)
    payload.update(
        {
            "schema_version": 1,
            "report_type": "baseline_eval",
            "baseline": "mar",
            "dataset": "imagenet_256",
            "resolution": 256,
            "status": "completed",
            "protocol": "official_imagenet256_lth14_pth_ema_eval_only_50k",
            "paper_table_role": "secondary related-method only",
            "sample_count": 50000,
            "png_count": mar.get("png_count", 50000),
            "reference_npz": str(
                project_root
                / "checkpoints/baselines/official_refs/VIRTUAL_imagenet256_labeled.npz"
            ),
            "metrics": {
                "inception_score": mar["inception_score"],
                "fid": mar["fid"],
                "sfid": mar["sfid"],
                "precision": mar["precision"],
                "recall": mar["recall"],
            },
            "limitations": [
                "Official ImageNet-256 pretrained eval-only, not same-dataset retraining.",
                "Uses checkpoint['model_ema'] from the pinned LTH14 MAR-B PTH.",
                "Use only as a secondary related-method row.",
                "Do not merge into the P0 all-dataset same-budget generation table.",
            ],
        }
    )
    artifacts = payload.setdefault("artifacts", {})
    artifacts["npz"] = mar["npz"]
    artifacts["metrics_txt"] = mar["metrics_txt"]
    artifacts["generation_report"] = str(source_report)

    out_dir = internal / "artifacts/reports/baselines/mar/official_pth_50k_2026-07-11"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "baseline_eval_report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_mar_community_audit(project_root: Path) -> None:
    root = (
        project_root
        / "checkpoints/baselines/mar/official_imagenet256_eval_only/samples_50k_hf"
    )
    run = inspect_sample_run(root, MAR_COMMUNITY_SAMPLE_STEM)
    if not run["complete"]:
        return
    payload = read_json(run["generation_report"])
    payload.update(
        {
            "schema_version": 1,
            "report_type": "baseline_eval_audit",
            "baseline": "mar",
            "dataset": "imagenet_256",
            "resolution": 256,
            "status": "completed_audit_only",
            "protocol": "community_hf_safetensors_non_ema_eval_only_50k",
            "paper_table_role": "conversion audit only",
            "sample_count": 50_000,
            "png_count": run["png_count"],
            "metrics": run["metrics"],
            "limitations": [
                "The community model safetensors exactly match checkpoint['model'], not model_ema.",
                "This row does not reproduce the official LTH14 MAR-B checkpoint protocol.",
                "Do not use this row as the paper's MAR comparison.",
            ],
        }
    )
    artifacts = payload.setdefault("artifacts", {})
    artifacts["npz"] = str(run["npz"])
    artifacts["metrics_txt"] = str(run["metrics_txt"])
    artifacts["generation_report"] = str(run["generation_report"])
    artifacts["conversion_audit"] = str(
        project_root
        / "checkpoints/baselines/mar/official_imagenet256_eval_only/official_pth/community_conversion_audit.json"
    )
    out_dir = (
        project_root
        / "CoFiTok-internal/artifacts/reports/baselines/mar/community_non_ema_50k_2026-07-11"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "baseline_eval_report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_retok_report(project_root: Path, rows: list[dict[str, Any]]) -> None:
    retok = next((row for row in rows if row["alias"] == "retok"), None)
    if not retok or retok["status"] != "completed_eval_only_50k":
        return
    internal = project_root / "CoFiTok-internal"
    out_dir = internal / "artifacts/reports/baselines/retok/official_50k_2026-07-10"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "report_type": "baseline_eval",
        "baseline": "retok",
        "dataset": "imagenet_256",
        "resolution": 256,
        "status": "completed",
        "protocol": "official_imagenet256_eval_only",
        "paper_table_role": "secondary related-method only",
        "parameters": {
            "eval_only": True,
            "checkpoint_family": "official ReTok GPT-XL + VQ_SB256",
            "sample_count": 50000,
            "sampler": "official sample_c2i_search_cfg.sh",
            "per_proc_batch_size": 64,
        },
        "artifacts": {
            "npz": retok["npz"],
            "metrics_txt": retok["metrics_txt"],
        },
        "sample_count": 50000,
        "png_count": retok.get("png_count", retok["sample_count"]),
        "reference_npz": str(
            project_root
            / "checkpoints/baselines/official_refs/VIRTUAL_imagenet256_labeled.npz"
        ),
        "metrics": {
            "inception_score": retok["inception_score"],
            "fid": retok["fid"],
            "sfid": retok["sfid"],
            "precision": retok["precision"],
            "recall": retok["recall"],
        },
        "limitations": [
            "Official ImageNet-256 pretrained eval-only, not same-dataset retraining.",
            "Use only as a secondary related-method row.",
            "Do not merge into the P0 all-dataset same-budget generation table.",
        ],
    }
    (out_dir / "baseline_eval_report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def incomplete_50k_rows(rows: list[dict[str, Any]]) -> list[str]:
    expected = {"D-AR", "MAR", "ReTok"}
    indexed = {str(row.get("method")): row for row in rows}
    incomplete = []
    for method in sorted(expected):
        status = indexed.get(method, {}).get("status", "missing")
        if status != "completed_eval_only_50k":
            incomplete.append(f"{method}={status}")
    return incomplete


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default="/root/autodl-tmp/CoFiTok")
    parser.add_argument(
        "--output-dir",
        default="artifacts/reports/baselines/official_related_methods_2026-07-10",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write a progress table instead of failing when any official 50K row is pending.",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root)
    internal = project_root / "CoFiTok-internal"
    output_dir = internal / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = build_rows(project_root)
    payload = {"schema_version": 1, "rows": rows}
    (output_dir / "official_related_methods_table.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_markdown(rows, output_dir / "official_related_methods_table.md")
    write_dar_report(project_root, rows)
    write_mar_community_audit(project_root)
    write_mar_report(project_root, rows)
    write_retok_report(project_root, rows)
    incomplete = incomplete_50k_rows(rows)
    if incomplete and not args.allow_incomplete:
        raise RuntimeError("official 50K table is incomplete: " + ", ".join(incomplete))
    print(output_dir)


if __name__ == "__main__":
    main()


