from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, read_image
from torchvision.models import ResNet50_Weights, resnet50

from cofitok.diffusion import select_sampling_timesteps
from cofitok.generation import conditioning_ranking_posttraining_sampling as posttraining
from cofitok.environment import capture_runtime_environment, runtime_environment_sha256
from cofitok.inference_replay import read_json_object, reject_symlink_chain
from cofitok.output_lock import exclusive_output_lock
from cofitok.reporting import git_provenance, write_json_report
from scripts.evaluate_generation_class_fidelity import (
    DEFAULT_CLASSIFIER_CHECKPOINT,
    classifier_identity,
)
from scripts.evaluate_generation_metrics import find_images, validate_sampling_provenance


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_SCHEMA_VERSION = 1
REPORT_ROLE = "generation_conditioning_ranking_paired_sample_class_fidelity"
REPORT_FILENAME = "paired_class_fidelity_report.json"
EXPECTED_NUM_SAMPLES = 5_000
EXPECTED_SAMPLE_STEPS = 50
EXPECTED_SEED = 406_020
EXPECTED_CHECKPOINT_STEP = 1_000
EXPECTED_GUIDANCE_SCALE = 1.5
EXPECTED_GUIDANCE_RESCALE = 0.0
LEGACY_SAMPLING_STAGE = "conditioning_ranking_four_arm_sampling5k_v1"
MIN_MEAN_TARGET_LOG_PROBABILITY_DELTA = 0.02
MIN_TARGET_PROBABILITY_RATIO = 1.02
MAX_PREDICTED_CLASS_FRACTION_REGRESSION = 0.05
SIGNIFICANCE_LEVEL = 0.05
METHOD_PREFIX_BUDGETS = {"cofitok": 8, "dense_identity": 1}
CLAIM_BOUNDARY = {
    "diagnostic_only": True,
    "formal_quality_gate": False,
    "authorizes_training": False,
    "authorizes_sampling": False,
    "authorizes_checkpoint_promotion": False,
    "authorizes_full_training": False,
    "authorizes_release": False,
    "cofitok_specific_advantage_claim_allowed": False,
}


def _sampling_contract(stage: str) -> dict[str, Any]:
    if stage == LEGACY_SAMPLING_STAGE:
        return {
            "stage": LEGACY_SAMPLING_STAGE,
            "num_samples": EXPECTED_NUM_SAMPLES,
            "start_index": 0,
            "sample_steps": EXPECTED_SAMPLE_STEPS,
            "seed": EXPECTED_SEED,
            "checkpoint_step": EXPECTED_CHECKPOINT_STEP,
            "method_prefix_budgets": METHOD_PREFIX_BUDGETS,
        }
    if stage == posttraining.STAGE:
        return {
            "stage": posttraining.STAGE,
            "num_samples": posttraining.SAMPLING_PROTOCOL["num_samples_per_arm"],
            "start_index": posttraining.SAMPLING_PROTOCOL["start_index"],
            "sample_steps": posttraining.SAMPLING_PROTOCOL["sample_steps"],
            "seed": posttraining.SAMPLING_PROTOCOL["seed"],
            "checkpoint_step": posttraining.EXPECTED_CHECKPOINT_STEP,
            "method_prefix_budgets": posttraining.METHOD_PREFIX_BUDGETS,
        }
    raise ValueError("paired sampling stage is unsupported")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare control and ranked generated samples with a fixed, "
            "source-bound ImageNet classifier and paired per-index statistics."
        )
    )
    parser.add_argument("--method", choices=sorted(METHOD_PREFIX_BUDGETS), required=True)
    parser.add_argument("--control-generated-dir", required=True)
    parser.add_argument("--control-sampling-report", required=True)
    parser.add_argument("--ranked-generated-dir", required=True)
    parser.add_argument("--ranked-sampling-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--classifier-checkpoint",
        default=DEFAULT_CLASSIFIER_CHECKPOINT,
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--min-samples", type=int, default=EXPECTED_NUM_SAMPLES)
    parser.add_argument(
        "--sampling-stage",
        choices=[LEGACY_SAMPLING_STAGE, posttraining.STAGE],
        default=LEGACY_SAMPLING_STAGE,
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def one_sided_sign_test_pvalue(positive: int, negative: int) -> float:
    if positive < 0 or negative < 0 or positive + negative < 1:
        raise ValueError("sign-test counts must include at least one non-tied pair")
    trials = positive + negative
    return sum(math.comb(trials, value) for value in range(positive, trials + 1)) / (
        2**trials
    )


def _arm_metrics(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    if arm not in {"control", "ranked"}:
        raise ValueError("paired class-fidelity arm is unsupported")
    sample_count = len(rows)
    if sample_count < 1:
        raise ValueError("paired class-fidelity rows are empty")
    requested = [0] * 1_000
    predicted = [0] * 1_000
    top1_correct = 0
    top5_correct = 0
    probability_sum = 0.0
    log_probability_sum = 0.0
    for row in rows:
        target = int(row["requested_class"])
        values = row[arm]
        if not isinstance(values, Mapping):
            raise ValueError("paired class-fidelity arm row is malformed")
        prediction = int(values["predicted_class"])
        probability = float(values["target_probability"])
        log_probability = float(values["target_log_probability"])
        if (
            not 0 <= target < 1_000
            or not 0 <= prediction < 1_000
            or not math.isfinite(probability)
            or not 0.0 < probability <= 1.0
            or not math.isfinite(log_probability)
            or log_probability > 1e-12
            or type(values["top1_correct"]) is not bool
            or type(values["top5_correct"]) is not bool
        ):
            raise ValueError("paired class-fidelity row value is invalid")
        requested[target] += 1
        predicted[prediction] += 1
        top1_correct += int(bool(values["top1_correct"]))
        top5_correct += int(bool(values["top5_correct"]))
        probability_sum += probability
        log_probability_sum += log_probability
    if top1_correct > top5_correct:
        raise ValueError("paired class-fidelity Top-1 exceeds Top-5")
    nonzero = [count / sample_count for count in predicted if count > 0]
    entropy = -sum(value * math.log(value) for value in nonzero)
    return {
        "sample_count": sample_count,
        "num_classes": 1_000,
        "top1_correct": top1_correct,
        "top5_correct": top5_correct,
        "top1_accuracy": top1_correct / sample_count,
        "top5_accuracy": top5_correct / sample_count,
        "mean_target_probability": probability_sum / sample_count,
        "target_negative_log_likelihood": -log_probability_sum / sample_count,
        "requested_class_count": sum(count > 0 for count in requested),
        "requested_count_min": min(requested),
        "requested_count_max": max(requested),
        "predicted_class_count": sum(count > 0 for count in predicted),
        "predicted_class_fraction": sum(count > 0 for count in predicted) / 1_000,
        "predicted_class_entropy": entropy,
        "normalized_predicted_class_entropy": entropy / math.log(1_000),
    }


def paired_class_fidelity_summary(
    rows: list[dict[str, Any]],
    *,
    expected_start_index: int = 0,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("paired class-fidelity rows are empty")
    expected_indices = list(
        range(expected_start_index, expected_start_index + len(rows))
    )
    actual_indices = [int(row.get("sample_index", -1)) for row in rows]
    if actual_indices != expected_indices:
        raise ValueError("paired class-fidelity sample indices are not exact and ordered")
    deltas: list[float] = []
    for row in rows:
        control = float(row["control"]["target_log_probability"])
        ranked = float(row["ranked"]["target_log_probability"])
        reported = float(row["ranked_minus_control_target_log_probability"])
        delta = ranked - control
        if (
            not math.isfinite(control)
            or not math.isfinite(ranked)
            or not math.isfinite(reported)
            or not math.isclose(reported, delta, rel_tol=0.0, abs_tol=1e-12)
        ):
            raise ValueError("paired target-log-probability delta is invalid")
        deltas.append(delta)
    positive = sum(delta > 0.0 for delta in deltas)
    negative = sum(delta < 0.0 for delta in deltas)
    ties = len(deltas) - positive - negative
    sign_pvalue = (
        one_sided_sign_test_pvalue(positive, negative)
        if positive + negative > 0
        else 1.0
    )
    control_metrics = _arm_metrics(rows, "control")
    ranked_metrics = _arm_metrics(rows, "ranked")
    control_probability = float(control_metrics["mean_target_probability"])
    if control_probability <= 0.0:
        raise ValueError("control mean target probability must be positive")
    probability_ratio = (
        float(ranked_metrics["mean_target_probability"]) / control_probability
    )
    mean_delta = sum(deltas) / len(deltas)
    gates = {
        "mean_target_log_probability_delta": (
            mean_delta >= MIN_MEAN_TARGET_LOG_PROBABILITY_DELTA
        ),
        "paired_target_log_probability_significant": sign_pvalue < SIGNIFICANCE_LEVEL,
        "target_probability_ratio": probability_ratio >= MIN_TARGET_PROBABILITY_RATIO,
        "top1_not_regressed": (
            float(ranked_metrics["top1_accuracy"])
            >= float(control_metrics["top1_accuracy"])
        ),
        "top5_not_regressed": (
            float(ranked_metrics["top5_accuracy"])
            >= float(control_metrics["top5_accuracy"])
        ),
        "predicted_class_fraction_within_tolerance": (
            float(ranked_metrics["predicted_class_fraction"])
            >= float(control_metrics["predicted_class_fraction"])
            - MAX_PREDICTED_CLASS_FRACTION_REGRESSION
        ),
    }
    return {
        "control": control_metrics,
        "ranked": ranked_metrics,
        "ranked_minus_control": {
            "mean_target_log_probability": mean_delta,
            "mean_target_probability": (
                float(ranked_metrics["mean_target_probability"])
                - float(control_metrics["mean_target_probability"])
            ),
            "target_probability_ratio": probability_ratio,
            "top1_accuracy": (
                float(ranked_metrics["top1_accuracy"])
                - float(control_metrics["top1_accuracy"])
            ),
            "top5_accuracy": (
                float(ranked_metrics["top5_accuracy"])
                - float(control_metrics["top5_accuracy"])
            ),
            "predicted_class_fraction": (
                float(ranked_metrics["predicted_class_fraction"])
                - float(control_metrics["predicted_class_fraction"])
            ),
        },
        "paired_sign_test": {
            "positive": positive,
            "negative": negative,
            "ties": ties,
            "one_sided_pvalue": sign_pvalue,
            "alternative": "ranked_target_log_probability_greater_than_control",
        },
        "thresholds": {
            "min_mean_target_log_probability_delta": (
                MIN_MEAN_TARGET_LOG_PROBABILITY_DELTA
            ),
            "min_target_probability_ratio": MIN_TARGET_PROBABILITY_RATIO,
            "max_predicted_class_fraction_regression": (
                MAX_PREDICTED_CLASS_FRACTION_REGRESSION
            ),
            "significance_level": SIGNIFICANCE_LEVEL,
        },
        "gates": gates,
        "class_alignment_pass": all(gates.values()),
    }


def validate_sampling_pair(
    control: Mapping[str, Any],
    ranked: Mapping[str, Any],
    *,
    method: str,
    sampling_stage: str = LEGACY_SAMPLING_STAGE,
) -> dict[str, Any]:
    contract = _sampling_contract(sampling_stage)
    prefix_budgets = contract["method_prefix_budgets"]
    if method not in prefix_budgets:
        raise ValueError("paired sampling method is unsupported")
    expected_budget = prefix_budgets[method]
    expected_sampling = {
        "num_samples": contract["num_samples"],
        "start_index": contract["start_index"],
        "sample_steps": contract["sample_steps"],
        "num_train_timesteps": 1_000,
        "guidance_scale": EXPECTED_GUIDANCE_SCALE,
        "guidance_rescale": EXPECTED_GUIDANCE_RESCALE,
        "cfg_batch_mode": "batched",
        "eta": 0.0,
        "seed": contract["seed"],
        "precision": "bf16",
        "class_schedule": "balanced_modulo",
        "sampler": "ddim",
        "clip_x0": True,
        "protocol_schema": "cofitok_ddim_sampling_v1",
        "prefix_budgets": [expected_budget],
        "image_shape": [3, 256, 256],
        "actual_timesteps": select_sampling_timesteps(
            1_000, contract["sample_steps"]
        ),
    }
    for label, source in (("control", control), ("ranked", ranked)):
        sampling = source.get("sampling")
        if not isinstance(sampling, Mapping) or any(
            sampling.get(field) != value for field, value in expected_sampling.items()
        ):
            raise ValueError(f"{label} paired sampling protocol differs")
        if (
            source.get("weights") != "ema"
            or int(source.get("checkpoint_step", -1))
            != contract["checkpoint_step"]
            or int(source.get("selected_prefix_budget", -1)) != expected_budget
            or not isinstance(source.get("git"), Mapping)
            or source["git"].get("tracked_dirty") is not False
        ):
            raise ValueError(f"{label} paired sampling source differs")
    if control["sampling"] != ranked["sampling"]:
        raise ValueError("control and ranked sampling protocols are not exact matches")
    if control["git"] != ranked["git"]:
        raise ValueError("control and ranked sampling Git identities differ")
    if control.get("runtime_environment") != ranked.get("runtime_environment"):
        raise ValueError("control and ranked sampling environments differ")
    if control.get("runtime_environment_sha256") != ranked.get(
        "runtime_environment_sha256"
    ):
        raise ValueError("control and ranked sampling environment hashes differ")
    if control.get("checkpoint_sha256") == ranked.get("checkpoint_sha256"):
        raise ValueError("control and ranked checkpoints must be distinct")
    result = {
        "method": method,
        "sampling": dict(control["sampling"]),
        "weights": "ema",
        "checkpoint_step": contract["checkpoint_step"],
        "prefix_budget": expected_budget,
        "git": dict(control["git"]),
        "runtime_environment_sha256": control["runtime_environment_sha256"],
        "control": dict(control),
        "ranked": dict(ranked),
    }
    if sampling_stage != LEGACY_SAMPLING_STAGE:
        result["sampling_stage"] = sampling_stage
    return result


class _PairedImageDataset(
    Dataset[tuple[torch.Tensor, torch.Tensor, int, int]]
):
    def __init__(self, control: list[Path], ranked: list[Path]) -> None:
        if [path.name for path in control] != [path.name for path in ranked]:
            raise ValueError("control and ranked generated filenames differ")
        self.control = control
        self.ranked = ranked
        self.transform = ResNet50_Weights.IMAGENET1K_V2.transforms()

    def __len__(self) -> int:
        return len(self.control)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        control_path = self.control[index]
        ranked_path = self.ranked[index]
        sample_index = int(control_path.stem)
        target = sample_index % 1_000
        control = read_image(control_path.as_posix(), mode=ImageReadMode.RGB)
        ranked = read_image(ranked_path.as_posix(), mode=ImageReadMode.RGB)
        return self.transform(control), self.transform(ranked), target, sample_index


def _load_classifier(checkpoint: str | Path, device: torch.device) -> torch.nn.Module:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError("paired class-fidelity classifier checkpoint is malformed")
    model = resnet50(weights=None)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model.to(device)


def calculate_paired_rows(
    *,
    control_images: list[Path],
    ranked_images: list[Path],
    classifier_checkpoint: str | Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    dataset = _PairedImageDataset(control_images, ranked_images)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    model = _load_classifier(classifier_checkpoint, device)
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for control, ranked, targets, sample_indices in loader:
            targets = targets.to(device=device, dtype=torch.long, non_blocking=True)
            control_logits = model(control.to(device, non_blocking=True))
            ranked_logits = model(ranked.to(device, non_blocking=True))
            if not torch.isfinite(control_logits).all() or not torch.isfinite(
                ranked_logits
            ).all():
                raise ValueError("paired classifier logits are not finite")
            row_indices = torch.arange(targets.shape[0], device=device)
            arm_values: dict[str, dict[str, torch.Tensor]] = {}
            for name, logits in (("control", control_logits), ("ranked", ranked_logits)):
                log_probabilities = torch.log_softmax(logits.float(), dim=1)
                top5 = torch.topk(logits, k=5, dim=1).indices
                arm_values[name] = {
                    "log_probability": log_probabilities[row_indices, targets],
                    "probability": log_probabilities[row_indices, targets].exp(),
                    "prediction": top5[:, 0],
                    "top1": top5[:, 0] == targets,
                    "top5": (top5 == targets[:, None]).any(dim=1),
                }
            for offset, sample_index in enumerate(sample_indices.tolist()):
                control_log = float(arm_values["control"]["log_probability"][offset])
                ranked_log = float(arm_values["ranked"]["log_probability"][offset])
                row: dict[str, Any] = {
                    "sample_index": int(sample_index),
                    "requested_class": int(targets[offset].item()),
                    "ranked_minus_control_target_log_probability": (
                        ranked_log - control_log
                    ),
                }
                for name in ("control", "ranked"):
                    values = arm_values[name]
                    row[name] = {
                        "target_log_probability": float(
                            values["log_probability"][offset]
                        ),
                        "target_probability": float(values["probability"][offset]),
                        "predicted_class": int(values["prediction"][offset]),
                        "top1_correct": bool(values["top1"][offset]),
                        "top5_correct": bool(values["top5"][offset]),
                    }
                rows.append(row)
    return rows


def _prepare_output(output_dir: str | Path, *, resume: bool) -> tuple[Path, Path]:
    output = reject_symlink_chain(output_dir, name="paired class-fidelity output")
    if output.exists() and not output.is_dir():
        raise ValueError("paired class-fidelity output is not a directory")
    output.mkdir(parents=True, exist_ok=True)
    output = output.resolve()
    report = output / REPORT_FILENAME
    reject_symlink_chain(report, name="paired class-fidelity report")
    unexpected = sorted(path.name for path in output.iterdir() if path.name != REPORT_FILENAME)
    if unexpected:
        raise ValueError("paired class-fidelity output contains unexpected files")
    if report.exists() and not report.is_file():
        raise ValueError("paired class-fidelity report is not a regular file")
    if report.is_file() and not resume:
        raise FileExistsError("paired class-fidelity report exists; pass --resume")
    return output, report


def _validate_completed_report(report: Mapping[str, Any], *, expected: Mapping[str, Any]) -> None:
    if (
        report.get("schema_version") != REPORT_SCHEMA_VERSION
        or report.get("role") != REPORT_ROLE
        or report.get("status") != "completed"
        or report.get("protocol")
        != "torchvision_imagenet_paired_conditioning_ranking_class_fidelity"
        or report.get("claim_boundary") != CLAIM_BOUNDARY
    ):
        raise ValueError("completed paired class-fidelity report contract differs")
    for field in (
        "git",
        "runtime_environment",
        "runtime_environment_sha256",
        "paths",
        "classifier",
        "sampling_pair",
        "parameters",
    ):
        if report.get(field) != expected.get(field):
            raise ValueError(f"completed paired class-fidelity {field} differs")
    rows = report.get("sample_rows")
    if (
        not isinstance(rows, list)
        or len(rows) != int(expected["parameters"]["sample_count"])
    ):
        raise ValueError("completed paired class-fidelity rows are missing")
    if paired_class_fidelity_summary(
        rows,
        expected_start_index=int(expected["parameters"].get("start_index", 0)),
    ) != report.get("metrics"):
        raise ValueError("completed paired class-fidelity metrics differ from rows")
    runtime = report.get("runtime")
    if (
        not isinstance(runtime, Mapping)
        or not math.isfinite(float(runtime.get("elapsed_seconds", math.nan)))
        or float(runtime["elapsed_seconds"]) <= 0.0
        or runtime.get("torch_version") != torch.__version__
    ):
        raise ValueError("completed paired class-fidelity runtime differs")


def _run(args: argparse.Namespace) -> None:
    contract = _sampling_contract(args.sampling_stage)
    if (
        args.batch_size < 1
        or args.num_workers < 0
        or args.min_samples != contract["num_samples"]
    ):
        raise ValueError("paired class-fidelity batch/workers/sample contract differs")
    control_dir = reject_symlink_chain(
        args.control_generated_dir,
        name="control generated directory",
    )
    ranked_dir = reject_symlink_chain(
        args.ranked_generated_dir,
        name="ranked generated directory",
    )
    control_report = reject_symlink_chain(
        args.control_sampling_report,
        name="control sampling report",
    )
    ranked_report = reject_symlink_chain(
        args.ranked_sampling_report,
        name="ranked sampling report",
    )
    output, report_path = _prepare_output(args.output_dir, resume=args.resume)
    control_images = find_images(control_dir)
    ranked_images = find_images(ranked_dir)
    if (
        len(control_images) != contract["num_samples"]
        or len(ranked_images) != contract["num_samples"]
    ):
        raise ValueError("paired class-fidelity requires exact 5K sample sets")
    if [path.name for path in control_images] != [path.name for path in ranked_images]:
        raise ValueError("control and ranked sample indices differ")
    control_provenance = validate_sampling_provenance(
        control_report,
        control_dir,
        control_images,
    )
    ranked_provenance = validate_sampling_provenance(
        ranked_report,
        ranked_dir,
        ranked_images,
    )
    sampling_pair = validate_sampling_pair(
        control_provenance,
        ranked_provenance,
        method=args.method,
        sampling_stage=args.sampling_stage,
    )
    classifier = classifier_identity(args.classifier_checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    environment = capture_runtime_environment(device, project_root=PROJECT_ROOT)
    environment_sha = runtime_environment_sha256(environment)
    parameters = {
        "method": args.method,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "sample_count": contract["num_samples"],
        "num_classes": 1_000,
        "device": device.type,
        "cpu_requested": bool(args.cpu),
        "paired_unit": "global_sample_index",
    }
    if args.sampling_stage != LEGACY_SAMPLING_STAGE:
        parameters["start_index"] = contract["start_index"]
        parameters["sampling_stage"] = args.sampling_stage
    expected = {
        "git": git_provenance(PROJECT_ROOT),
        "runtime_environment": environment,
        "runtime_environment_sha256": environment_sha,
        "paths": {
            "control_generated_dir": control_dir.resolve().as_posix(),
            "control_sampling_report": control_report.resolve().as_posix(),
            "ranked_generated_dir": ranked_dir.resolve().as_posix(),
            "ranked_sampling_report": ranked_report.resolve().as_posix(),
            "output_dir": output.as_posix(),
            "report": report_path.as_posix(),
        },
        "classifier": classifier,
        "sampling_pair": sampling_pair,
        "parameters": parameters,
    }
    if report_path.is_file():
        existing = read_json_object(report_path, name="paired class-fidelity report")
        _validate_completed_report(existing, expected=expected)
        print(f"reused completed paired class-fidelity report {report_path}")
        return
    started = time.time()
    rows = calculate_paired_rows(
        control_images=control_images,
        ranked_images=ranked_images,
        classifier_checkpoint=classifier["weights_path"],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    metrics = paired_class_fidelity_summary(
        rows,
        expected_start_index=contract["start_index"],
    )
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "role": REPORT_ROLE,
        "status": "completed",
        "protocol": "torchvision_imagenet_paired_conditioning_ranking_class_fidelity",
        **expected,
        "metrics": metrics,
        "sample_rows": rows,
        "runtime": {
            "elapsed_seconds": time.time() - started,
            "torch_version": torch.__version__,
            "torchvision_version": __import__("torchvision").__version__,
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }
    write_json_report(report_path, report)
    print(f"wrote {report_path}")


def main() -> None:
    args = parse_args()
    with exclusive_output_lock(
        args.output_dir,
        role="generation_conditioning_ranking_paired_sample_class_fidelity",
    ):
        _run(args)


if __name__ == "__main__":
    main()
