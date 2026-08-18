from __future__ import annotations

import argparse
import gc
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from PIL import Image
from torchvision import transforms

from cofitok.configs import ExperimentConfig, config_from_dict
from cofitok.diffusion import DiffusionSchedule
from cofitok.inference_replay import (
    file_identity,
    read_json_object,
    reject_symlink_chain,
)
from cofitok.models import CoFiTokTiny
from cofitok.output_lock import exclusive_output_lock
from cofitok.reporting import git_provenance, write_json_report
from cofitok.training.checkpointing import (
    checkpoint_integrity_path,
    verify_training_checkpoint,
)
from cofitok.utils.seed import seed_everything


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_FILENAME = "conditioning_sensitivity_manifest.json"
REPORT_FILENAME = "conditioning_sensitivity_report.json"
MANIFEST_SCHEMA_VERSION = 1
REPORT_SCHEMA_VERSION = 1
MANIFEST_ROLE = "generation_conditioning_sensitivity_manifest"
REPORT_ROLE = "generation_conditioning_sensitivity_report"

PARAMETER_AUDIT_NAMES = (
    "predictor.class_embed.weight",
    "predictor.encoder.0.blocks.0.embedding.1.weight",
    "predictor.encoder.0.blocks.0.in_conv.weight",
    "predictor.input_proj.weight",
    "predictor.token_heads.0.weight",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure how strongly a class-conditional generation checkpoint "
            "responds to correct, wrong, and null ImageNet labels. The audit is "
            "CPU-only and also compares learned raw/EMA parameters with exact "
            "seeded initialization."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--weights", choices=["ema", "model"], default="ema")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--start-label", type=int, default=0)
    parser.add_argument("--wrong-label-offset", type=int, default=500)
    parser.add_argument("--timesteps", type=int, nargs="+", default=[100, 500, 900])
    parser.add_argument("--noise-seed", type=int, default=102030)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--dataset-root",
        help=(
            "Optional dataset alias root. Defaults to "
            "<checkpoint data.root>/<checkpoint data.dataset>."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Revalidate and reuse an exactly bound completed report.",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.num_samples < 1:
        raise ValueError("num_samples must be positive")
    if args.start_label < 0:
        raise ValueError("start_label must be non-negative")
    if args.wrong_label_offset < 1:
        raise ValueError("wrong_label_offset must be positive")
    if args.noise_seed < 0:
        raise ValueError("noise_seed must be non-negative")
    if args.threads < 1:
        raise ValueError("threads must be positive")
    if not args.timesteps:
        raise ValueError("at least one timestep is required")
    if len(set(args.timesteps)) != len(args.timesteps):
        raise ValueError("timesteps must be unique")


def _checkpoint_identity(
    path: str | Path,
) -> tuple[Path, Path, dict[str, Any]]:
    checkpoint = reject_symlink_chain(path, name="conditioning checkpoint")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    integrity_path = reject_symlink_chain(
        checkpoint_integrity_path(checkpoint),
        name="conditioning checkpoint integrity manifest",
    )
    integrity = verify_training_checkpoint(checkpoint)
    return checkpoint, integrity_path, {
        "path": checkpoint.resolve().as_posix(),
        "bytes": int(integrity["checkpoint_bytes"]),
        "sha256": str(integrity["checkpoint_sha256"]),
        "format_version": int(integrity["checkpoint_format_version"]),
        "step": int(integrity["step"]),
        "integrity_manifest": file_identity(integrity_path),
        "git": (
            {
                "revision": integrity["git_revision"],
                "branch": integrity["git_branch"],
                "dirty": integrity["git_dirty"],
            }
            if "git_revision" in integrity
            else None
        ),
        "dataset_identity_sha256": integrity.get("dataset_identity_sha256"),
        "runtime_environment_sha256": integrity.get(
            "runtime_environment_sha256"
        ),
    }


def _load_checkpoint_payload(
    checkpoint: Path,
    identity: Mapping[str, Any],
) -> tuple[dict[str, Any], ExperimentConfig]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint payload must be a mapping")
    if (
        int(payload.get("format_version", -1)) != int(identity["format_version"])
        or int(payload.get("step", -1)) != int(identity["step"])
    ):
        raise ValueError("Checkpoint payload differs from integrity metadata")
    for key in ("config", "model", "ema"):
        if key not in payload:
            raise ValueError(f"Checkpoint payload is missing {key}")
    config = config_from_dict(payload["config"])
    if not config.data.class_conditional or config.model.num_classes < 2:
        raise ValueError("Conditioning sensitivity requires a class-conditional model")
    if config.model.predictor_type != "scalable_unet":
        raise ValueError("Conditioning sensitivity currently requires scalable_unet")
    return payload, config


def _unwrap_label_map(payload: Mapping[str, Any]) -> list[str]:
    candidate: Any = payload.get("label_to_wnid", payload)
    if isinstance(candidate, list):
        mapping = [str(value) for value in candidate]
    elif isinstance(candidate, Mapping):
        try:
            indices = sorted(int(key) for key in candidate)
        except (TypeError, ValueError) as error:
            raise ValueError("label_to_wnid keys must be integer-like") from error
        if indices != list(range(len(indices))):
            raise ValueError("label_to_wnid indices must be contiguous from zero")
        mapping = [str(candidate[str(index)]) for index in indices]
    else:
        raise ValueError("label_to_wnid must be a list or mapping")
    if not mapping or any(not value for value in mapping):
        raise ValueError("label_to_wnid contains empty entries")
    return mapping


def _image_candidates(directory: Path) -> list[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )


def select_samples(
    *,
    dataset_root: str | Path,
    num_classes: int,
    start_label: int,
    num_samples: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = reject_symlink_chain(dataset_root, name="conditioning dataset root")
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    label_map_path = reject_symlink_chain(
        root / "metadata" / "label_to_wnid.json",
        name="conditioning label map",
    )
    label_payload = read_json_object(label_map_path, name="conditioning label map")
    label_to_wnid = _unwrap_label_map(label_payload)
    if len(label_to_wnid) != num_classes:
        raise ValueError(
            "label_to_wnid count differs from checkpoint num_classes: "
            f"{len(label_to_wnid)} != {num_classes}"
        )
    if start_label + num_samples > num_classes:
        raise ValueError("requested label range exceeds checkpoint num_classes")

    validation_root = reject_symlink_chain(
        root / "extracted" / "val",
        name="conditioning validation root",
    )
    if not validation_root.is_dir():
        validation_root = reject_symlink_chain(
            root / "extracted" / "validation",
            name="conditioning validation root",
        )
    if not validation_root.is_dir():
        raise FileNotFoundError("ImageNet validation root is missing")

    samples = []
    for label in range(start_label, start_label + num_samples):
        wnid = label_to_wnid[label]
        class_dir = reject_symlink_chain(
            validation_root / wnid,
            name=f"conditioning class directory {wnid}",
        )
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Validation class directory is missing: {class_dir}")
        candidates = _image_candidates(class_dir)
        if not candidates:
            raise FileNotFoundError(f"No validation images found for {wnid}")
        image = reject_symlink_chain(
            candidates[0],
            name=f"conditioning sample image {wnid}",
        )
        samples.append(
            {
                "label": label,
                "wnid": wnid,
                "image": file_identity(image),
            }
        )
    return file_identity(label_map_path), samples


def _request(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "weights": str(args.weights),
        "num_samples": int(args.num_samples),
        "start_label": int(args.start_label),
        "wrong_label_offset": int(args.wrong_label_offset),
        "timesteps": [int(value) for value in args.timesteps],
        "noise_seed": int(args.noise_seed),
        "threads": int(args.threads),
    }


def _prepare_output(output_dir: str | Path, *, resume: bool) -> tuple[Path, Path, Path]:
    output = reject_symlink_chain(output_dir, name="conditioning output directory")
    if output.exists() and not output.is_dir():
        raise ValueError(f"Conditioning output is not a directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    output = output.resolve()
    manifest_path = output / MANIFEST_FILENAME
    report_path = output / REPORT_FILENAME
    allowed = {MANIFEST_FILENAME, REPORT_FILENAME}
    unexpected = sorted(path.name for path in output.iterdir() if path.name not in allowed)
    if unexpected:
        raise ValueError(
            "Conditioning output contains unexpected files: " + ", ".join(unexpected)
        )
    if not resume and (manifest_path.exists() or report_path.exists()):
        raise FileExistsError(
            "Conditioning evidence already exists; pass --resume to validate it"
        )
    if report_path.exists() and not manifest_path.exists():
        raise ValueError("Conditioning report exists without its manifest")
    return output, manifest_path, report_path


def _manifest(
    *,
    git: dict[str, Any],
    checkpoint: dict[str, Any],
    request: dict[str, Any],
    dataset: dict[str, Any],
    output_dir: Path,
    report_path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "role": MANIFEST_ROLE,
        "git": git,
        "checkpoint": checkpoint,
        "request": request,
        "dataset": dataset,
        "output_dir": output_dir.as_posix(),
        "report": report_path.as_posix(),
    }


def _validate_completed_report(
    report: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    manifest_identity: Mapping[str, Any],
) -> None:
    if (
        report.get("schema_version") != REPORT_SCHEMA_VERSION
        or report.get("role") != REPORT_ROLE
        or report.get("status") != "completed"
        or report.get("manifest") != manifest_identity
        or report.get("git") != manifest.get("git")
        or report.get("checkpoint") != manifest.get("checkpoint")
        or report.get("request") != manifest.get("request")
        or report.get("dataset") != manifest.get("dataset")
    ):
        raise ValueError("Completed conditioning report binding differs")
    runtime = report.get("runtime")
    timesteps = report.get("timesteps")
    parameter_audit = report.get("parameter_update_audit")
    if (
        not isinstance(runtime, Mapping)
        or runtime.get("device") != "cpu"
        or not math.isfinite(float(runtime.get("elapsed_seconds", -1.0)))
        or float(runtime["elapsed_seconds"]) <= 0.0
        or not isinstance(timesteps, list)
        or len(timesteps) != len(manifest["request"]["timesteps"])
        or not isinstance(parameter_audit, Mapping)
    ):
        raise ValueError("Completed conditioning report is malformed")


def _tensor_statistics(value: torch.Tensor) -> dict[str, Any]:
    tensor = value.detach().float().cpu()
    return {
        "mean": float(tensor.mean()),
        "std": float(tensor.std(unbiased=False)),
        "rms": float(tensor.square().mean().sqrt()),
        "norm": float(tensor.norm()),
    }


def parameter_update_statistics(
    initial: torch.Tensor,
    current: torch.Tensor,
) -> dict[str, Any]:
    initial_value = initial.detach().float().cpu()
    current_value = current.detach().float().cpu()
    if initial_value.shape != current_value.shape:
        raise ValueError("Parameter shapes differ in update audit")
    difference = current_value - initial_value
    initial_rms = float(initial_value.square().mean().sqrt())
    current_rms = float(current_value.square().mean().sqrt())
    update_rms = float(difference.square().mean().sqrt())
    flat_initial = initial_value.flatten()
    flat_current = current_value.flatten()
    denominator = float(flat_initial.norm() * flat_current.norm())
    cosine = (
        float(torch.dot(flat_initial, flat_current) / denominator)
        if denominator > 0.0
        else None
    )
    return {
        "shape": list(initial_value.shape),
        "initial_rms": initial_rms,
        "current_rms": current_rms,
        "update_rms": update_rms,
        "relative_update_rms": (
            update_rms / initial_rms if initial_rms > 0.0 else None
        ),
        "initial_zero": initial_rms == 0.0,
        "cosine_initial_current": cosine,
    }


def _selected_parameter_audit(
    *,
    initial_state: Mapping[str, torch.Tensor],
    raw_state: Mapping[str, torch.Tensor],
    ema_state: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    missing = [
        name
        for name in PARAMETER_AUDIT_NAMES
        if name not in initial_state or name not in raw_state or name not in ema_state
    ]
    if missing:
        raise ValueError("Parameter update audit is missing: " + ", ".join(missing))
    return {
        "raw": {
            name: parameter_update_statistics(initial_state[name], raw_state[name])
            for name in PARAMETER_AUDIT_NAMES
        },
        "ema": {
            name: parameter_update_statistics(initial_state[name], ema_state[name])
            for name in PARAMETER_AUDIT_NAMES
        },
    }


def _class_embedding_audit(
    *,
    initial_state: Mapping[str, torch.Tensor],
    raw_state: Mapping[str, torch.Tensor],
    ema_state: Mapping[str, torch.Tensor],
    selected_labels: Sequence[int],
    null_label: int,
) -> dict[str, Any]:
    name = "predictor.class_embed.weight"
    rows = sorted(set([*selected_labels, null_label]))
    result: dict[str, Any] = {}
    for state_name, state in (("raw", raw_state), ("ema", ema_state)):
        weight = state[name].detach().float().cpu()
        result[state_name] = {
            "shape": list(weight.shape),
            "all_class_coordinate_std_mean": float(
                weight[:null_label].std(dim=0, unbiased=False).mean()
            ),
            "selected_rows": {
                str(label): {
                    **_tensor_statistics(weight[label]),
                    "update": parameter_update_statistics(
                        initial_state[name][label], weight[label]
                    ),
                }
                for label in rows
            },
        }
    return result


def _load_image(path: str | Path, image_size: int) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    with Image.open(path) as image:
        return transform(image.convert("RGB")).unsqueeze(0)


def _condition_row(
    *,
    epsilon: torch.Tensor,
    predicted_x0: torch.Tensor,
    noise: torch.Tensor,
    clean: torch.Tensor,
) -> dict[str, float]:
    return {
        "epsilon_mse_to_noise": float(
            torch.nn.functional.mse_loss(epsilon.float(), noise.float())
        ),
        "x0_mse_to_clean": float(
            torch.nn.functional.mse_loss(predicted_x0.float(), clean.float())
        ),
        "epsilon_rms": float(epsilon.float().square().mean().sqrt()),
    }


def _paired_metrics(
    *,
    correct: torch.Tensor,
    wrong: torch.Tensor,
    null: torch.Tensor,
    condition_rows: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    correct_rms = float(correct.float().square().mean().sqrt())

    def relative_delta(other: torch.Tensor) -> float:
        return float((correct.float() - other.float()).square().mean().sqrt()) / max(
            correct_rms, 1e-12
        )

    correct_mse = float(condition_rows["correct"]["epsilon_mse_to_noise"])
    wrong_mse = float(condition_rows["wrong"]["epsilon_mse_to_noise"])
    null_mse = float(condition_rows["null"]["epsilon_mse_to_noise"])
    return {
        "relative_delta_to_correct_rms": {
            "correct_vs_wrong": relative_delta(wrong),
            "correct_vs_null": relative_delta(null),
            "wrong_vs_null": float(
                (wrong.float() - null.float()).square().mean().sqrt()
            )
            / max(correct_rms, 1e-12),
        },
        "correct_relative_mse_improvement": {
            "versus_wrong": (wrong_mse - correct_mse) / max(wrong_mse, 1e-12),
            "versus_null": (null_mse - correct_mse) / max(null_mse, 1e-12),
        },
        "correct_better": {
            "than_wrong": correct_mse < wrong_mse,
            "than_null": correct_mse < null_mse,
        },
    }


@torch.no_grad()
def evaluate_sensitivity(
    *,
    model: CoFiTokTiny,
    config: ExperimentConfig,
    samples: Sequence[Mapping[str, Any]],
    timesteps: Sequence[int],
    wrong_label_offset: int,
    noise_seed: int,
) -> list[dict[str, Any]]:
    schedule = DiffusionSchedule(config.diffusion, device="cpu")
    rows = []
    model.eval()
    for sample_index, sample in enumerate(samples):
        correct_label = int(sample["label"])
        wrong_label = (correct_label + wrong_label_offset) % config.model.num_classes
        if wrong_label == correct_label:
            raise ValueError("wrong label resolved to the correct label")
        clean = _load_image(sample["image"]["path"], config.model.image_size)
        generator = torch.Generator(device="cpu").manual_seed(noise_seed + sample_index)
        noise = torch.randn(clean.shape, generator=generator)
        for timestep in timesteps:
            timestep_tensor = torch.tensor([timestep], dtype=torch.long)
            noisy = schedule.add_noise(clean, noise, timestep_tensor)
            batch_noisy = noisy.repeat(3, 1, 1, 1)
            batch_timestep = timestep_tensor.repeat(3)
            labels = torch.tensor(
                [correct_label, wrong_label, config.model.num_classes],
                dtype=torch.long,
            )
            started = time.monotonic()
            output = model(batch_noisy, batch_timestep, class_labels=labels).epsilon.float()
            forward_seconds = time.monotonic() - started
            predicted_x0 = schedule.predict_x0_from_epsilon(
                batch_noisy,
                output,
                batch_timestep,
            )
            condition_rows = {
                name: _condition_row(
                    epsilon=output[index : index + 1],
                    predicted_x0=predicted_x0[index : index + 1],
                    noise=noise,
                    clean=clean,
                )
                for index, name in enumerate(("correct", "wrong", "null"))
            }
            rows.append(
                {
                    "sample_index": sample_index,
                    "correct_label": correct_label,
                    "correct_wnid": sample["wnid"],
                    "wrong_label": wrong_label,
                    "null_label": config.model.num_classes,
                    "noise_seed": noise_seed + sample_index,
                    "timestep": timestep,
                    "forward_seconds": forward_seconds,
                    "conditions": condition_rows,
                    **_paired_metrics(
                        correct=output[0:1],
                        wrong=output[1:2],
                        null=output[2:3],
                        condition_rows=condition_rows,
                    ),
                }
            )
    return rows


def _summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("Cannot summarize an empty sequence")
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "population_std": statistics.pstdev(values),
    }


def summarize_timestep_rows(
    rows: Sequence[Mapping[str, Any]],
    timesteps: Sequence[int],
) -> list[dict[str, Any]]:
    result = []
    for timestep in timesteps:
        selected = [row for row in rows if int(row["timestep"]) == timestep]
        if not selected:
            raise ValueError(f"No sensitivity rows for timestep {timestep}")
        conditions = {}
        for condition in ("correct", "wrong", "null"):
            conditions[condition] = {
                metric: _summary(
                    [float(row["conditions"][condition][metric]) for row in selected]
                )
                for metric in (
                    "epsilon_mse_to_noise",
                    "x0_mse_to_clean",
                    "epsilon_rms",
                )
            }
        result.append(
            {
                "timestep": timestep,
                "sample_count": len(selected),
                "conditions": conditions,
                "relative_delta_to_correct_rms": {
                    comparison: _summary(
                        [
                            float(
                                row["relative_delta_to_correct_rms"][comparison]
                            )
                            for row in selected
                        ]
                    )
                    for comparison in (
                        "correct_vs_wrong",
                        "correct_vs_null",
                        "wrong_vs_null",
                    )
                },
                "correct_relative_mse_improvement": {
                    comparison: _summary(
                        [
                            float(
                                row["correct_relative_mse_improvement"][comparison]
                            )
                            for row in selected
                        ]
                    )
                    for comparison in ("versus_wrong", "versus_null")
                },
                "correct_better_count": {
                    "than_wrong": sum(
                        bool(row["correct_better"]["than_wrong"]) for row in selected
                    ),
                    "than_null": sum(
                        bool(row["correct_better"]["than_null"]) for row in selected
                    ),
                },
            }
        )
    return result


def main() -> None:
    args = parse_args()
    _validate_args(args)
    output_hint = reject_symlink_chain(
        args.output_dir,
        name="conditioning output directory",
    )
    with exclusive_output_lock(output_hint, role=REPORT_ROLE):
        output_dir, manifest_path, report_path = _prepare_output(
            args.output_dir,
            resume=args.resume,
        )
        started = time.monotonic()
        torch.set_num_threads(args.threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

        checkpoint_path, _integrity_path, checkpoint_identity = _checkpoint_identity(
            args.checkpoint
        )
        payload, config = _load_checkpoint_payload(
            checkpoint_path,
            checkpoint_identity,
        )
        if any(
            timestep < 0 or timestep >= config.diffusion.num_train_timesteps
            for timestep in args.timesteps
        ):
            raise ValueError("requested timestep is outside the diffusion schedule")
        if args.start_label + args.num_samples > config.model.num_classes:
            raise ValueError("requested sample labels exceed checkpoint num_classes")

        dataset_root = Path(args.dataset_root) if args.dataset_root else (
            Path(config.data.root) / config.data.dataset
        )
        label_map_identity, samples = select_samples(
            dataset_root=dataset_root,
            num_classes=config.model.num_classes,
            start_label=args.start_label,
            num_samples=args.num_samples,
        )
        dataset = {
            "alias": config.data.dataset,
            "root": reject_symlink_chain(
                dataset_root,
                name="conditioning dataset root",
            ).resolve().as_posix(),
            "label_map": label_map_identity,
            "samples": samples,
        }
        git = git_provenance(PROJECT_ROOT)
        request = _request(args)
        expected_manifest = _manifest(
            git=git,
            checkpoint=checkpoint_identity,
            request=request,
            dataset=dataset,
            output_dir=output_dir,
            report_path=report_path,
        )
        if manifest_path.exists():
            existing_manifest = read_json_object(
                manifest_path,
                name="conditioning sensitivity manifest",
            )
            if existing_manifest != expected_manifest:
                raise ValueError("Conditioning resume manifest differs from request")
        else:
            write_json_report(manifest_path, expected_manifest)
        manifest_identity = file_identity(manifest_path)
        if report_path.exists():
            if not args.resume:
                raise FileExistsError(
                    "Conditioning report already exists; pass --resume to validate it"
                )
            report = read_json_object(
                report_path,
                name="conditioning sensitivity report",
            )
            _validate_completed_report(
                report,
                manifest=expected_manifest,
                manifest_identity=manifest_identity,
            )
            print(report_path.as_posix())
            return

        seed_everything(config.runtime.seed)
        model = CoFiTokTiny(config.model)
        initial_state = {
            name: tensor.detach().clone()
            for name, tensor in model.state_dict().items()
            if name in PARAMETER_AUDIT_NAMES
        }
        raw_state = payload["model"]
        ema_payload = payload["ema"]
        if not isinstance(raw_state, Mapping) or not isinstance(ema_payload, Mapping):
            raise ValueError("Checkpoint model or EMA state is malformed")
        ema_state = ema_payload.get("shadow")
        if not isinstance(ema_state, Mapping):
            raise ValueError("Checkpoint EMA shadow state is malformed")
        parameter_audit = _selected_parameter_audit(
            initial_state=initial_state,
            raw_state=raw_state,
            ema_state=ema_state,
        )
        selected_labels = [int(sample["label"]) for sample in samples]
        parameter_audit.update(
            {
                "seed": config.runtime.seed,
                "selected_parameters": list(PARAMETER_AUDIT_NAMES),
                "class_embedding": _class_embedding_audit(
                    initial_state=initial_state,
                    raw_state=raw_state,
                    ema_state=ema_state,
                    selected_labels=selected_labels,
                    null_label=config.model.num_classes,
                ),
            }
        )

        selected_state = ema_state if args.weights == "ema" else raw_state
        model.load_state_dict(selected_state, strict=True)
        model.eval()
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        del payload, raw_state, ema_payload, ema_state, selected_state, initial_state
        gc.collect()

        rows = evaluate_sensitivity(
            model=model,
            config=config,
            samples=samples,
            timesteps=args.timesteps,
            wrong_label_offset=args.wrong_label_offset,
            noise_seed=args.noise_seed,
        )
        elapsed_seconds = time.monotonic() - started
        report = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "role": REPORT_ROLE,
            "status": "completed",
            "manifest": manifest_identity,
            "git": git,
            "checkpoint": checkpoint_identity,
            "request": request,
            "dataset": dataset,
            "model": {
                "name": config.name,
                "predictor_type": config.model.predictor_type,
                "synthesis_mode": config.model.synthesis_mode,
                "token_count": config.model.token_count,
                "num_classes": config.model.num_classes,
                "parameter_count": parameter_count,
            },
            "weights": args.weights,
            "parameter_update_audit": parameter_audit,
            "sample_rows": rows,
            "timesteps": summarize_timestep_rows(rows, args.timesteps),
            "runtime": {
                "device": "cpu",
                "threads": args.threads,
                "elapsed_seconds": elapsed_seconds,
                "torch_version": torch.__version__,
            },
            "claim_boundary": {
                "diagnostic_only": True,
                "generates_new_samples": False,
                "authorizes_training": False,
                "authorizes_sampling": False,
                "replaces_formal_quality_gate": False,
            },
        }
        write_json_report(report_path, report)
        _validate_completed_report(
            report,
            manifest=expected_manifest,
            manifest_identity=manifest_identity,
        )
        print(report_path.as_posix())


if __name__ == "__main__":
    main()
