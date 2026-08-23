from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

import torch
from torchvision.io import ImageReadMode, read_image
from torchvision.utils import make_grid, save_image

from cofitok.generation.quality_bridge import (
    QUALITY_BRIDGE_RESULT_ROLE,
    RESULT_AUTHORIZATION_BOUNDARY,
)
from cofitok.inference_replay import file_identity, read_json_object, reject_symlink_chain
from cofitok.reporting import git_provenance, write_json_report

try:
    from scripts.calibrate_generation_class_fidelity_classifier import (
        CLAIM_BOUNDARY as CALIBRATION_CLAIM_BOUNDARY,
        REPORT_ROLE as CALIBRATION_REPORT_ROLE,
        class_order_sha256,
        load_class_order,
        select_balanced_real_images,
        selection_sha256,
    )
except ModuleNotFoundError:
    from calibrate_generation_class_fidelity_classifier import (
        CLAIM_BOUNDARY as CALIBRATION_CLAIM_BOUNDARY,
        REPORT_ROLE as CALIBRATION_REPORT_ROLE,
        class_order_sha256,
        load_class_order,
        select_balanced_real_images,
        selection_sha256,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_SCHEMA_VERSION = 1
REPORT_ROLE = "generation_requested_class_visual_audit"
REPORT_FILENAME = "requested_class_visual_audit_report.json"
PANEL_PREFIX = "requested_class_panel"
CLAIM_BOUNDARY = {
    "visual_diagnostic_only": True,
    "quantitative_generation_metric": False,
    "replaces_class_fidelity_evaluation": False,
    "replaces_fid_or_distribution_metrics": False,
    "replaces_frozen_promotion_gate": False,
    "training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "promotion_or_release_allowed": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build deterministic real-target/CoFiTok/dense visual panels from "
            "a matched class-conditional sampling window."
        )
    )
    parser.add_argument("--cofitok-sampling-report", required=True)
    parser.add_argument("--dense-sampling-report", required=True)
    parser.add_argument("--cofitok-dir", required=True)
    parser.add_argument("--dense-dir", required=True)
    parser.add_argument("--quality-result", required=True)
    parser.add_argument("--classifier-calibration-report", required=True)
    parser.add_argument("--real-dir", required=True)
    parser.add_argument("--indices", required=True)
    parser.add_argument("--panel-columns", type=int, default=8)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def parse_indices(raw: str) -> list[int]:
    try:
        indices = [int(value.strip()) for value in raw.split(",") if value.strip()]
    except ValueError as error:
        raise ValueError("visual-audit indices must be comma-separated integers") from error
    if (
        not indices
        or any(index < 0 for index in indices)
        or indices != sorted(set(indices))
    ):
        raise ValueError(
            "visual-audit indices must be unique, non-negative, and increasing"
        )
    return indices


def _identity_matches(identity: dict[str, Any], *, name: str) -> None:
    if not isinstance(identity, dict) or not isinstance(identity.get("path"), str):
        raise ValueError(f"{name} identity is malformed")
    observed = file_identity(reject_symlink_chain(identity["path"], name=name))
    if observed != identity:
        raise ValueError(f"{name} identity differs")


def _sampling_source(
    report_path: str | Path,
    *,
    expected_dir: str | Path,
    name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = reject_symlink_chain(report_path, name=f"{name} sampling report")
    report = read_json_object(path, name=f"{name} sampling report")
    sampling = report.get("sampling")
    if (
        report.get("status") != "completed"
        or int(report.get("schema_version", -1)) < 6
        or report.get("weights") != "ema"
        or not isinstance(sampling, dict)
    ):
        raise ValueError(f"{name} sampling report is not an eligible completed report")
    budgets = sampling.get("prefix_budgets")
    if not isinstance(budgets, list) or len(budgets) != 1:
        raise ValueError(f"{name} visual audit requires exactly one prefix budget")
    budget = int(budgets[0])
    key = str(budget)
    output_dirs = report.get("output_dirs")
    sample_sets = report.get("sample_sets")
    if not isinstance(output_dirs, dict) or not isinstance(sample_sets, dict):
        raise ValueError(f"{name} sampling output provenance is malformed")
    directory = reject_symlink_chain(output_dirs.get(key, ""), name=f"{name} samples")
    expected = reject_symlink_chain(expected_dir, name=f"expected {name} samples")
    if directory.resolve() != expected.resolve() or not directory.is_dir():
        raise ValueError(f"{name} sample directory differs from its sampling report")
    sample_set = sample_sets.get(key)
    if not isinstance(sample_set, dict):
        raise ValueError(f"{name} sample-set provenance is malformed")
    count = int(sample_set.get("count", -1))
    start_index = int(sampling.get("start_index", -1))
    num_samples = int(sampling.get("num_samples", -1))
    declared_num_classes = sampling.get("num_classes")
    image_shape = sampling.get("image_shape")
    if (
        count < 1
        or count != num_samples
        or start_index < 0
        or (
            declared_num_classes is not None
            and (
                isinstance(declared_num_classes, bool)
                or not isinstance(declared_num_classes, int)
                or declared_num_classes < 5
            )
        )
        or len(str(sample_set.get("sha256", ""))) != 64
        or image_shape != [3, 256, 256]
        or sampling.get("class_schedule") != "balanced_modulo"
        or sampling.get("sampler") != "ddim"
        or int(report.get("checkpoint_step", -1)) < 1
        or len(str(report.get("checkpoint_sha256", ""))) != 64
        or len(str(report.get("sampling_manifest_sha256", ""))) != 64
    ):
        raise ValueError(f"{name} sampling contract differs")
    source = {
        "report": file_identity(path),
        "directory": directory.resolve().as_posix(),
        "budget": budget,
        "sample_set": {
            "count": count,
            "sha256": sample_set["sha256"],
            "start_index": start_index,
            "stop_index_exclusive": start_index + count,
        },
        "checkpoint": {
            "step": int(report["checkpoint_step"]),
            "sha256": report["checkpoint_sha256"],
        },
        "sampling_manifest_sha256": report["sampling_manifest_sha256"],
        "git": report.get("git"),
        "runtime_environment_sha256": report.get("runtime_environment_sha256"),
    }
    return report, source


def _matched_sampling_protocol(
    cofitok: dict[str, Any],
    dense: dict[str, Any],
) -> dict[str, Any]:
    cofitok_sampling = {
        key: value
        for key, value in cofitok["sampling"].items()
        if key != "prefix_budgets"
    }
    dense_sampling = {
        key: value
        for key, value in dense["sampling"].items()
        if key != "prefix_budgets"
    }
    if cofitok_sampling != dense_sampling:
        raise ValueError("CoFiTok and dense visual sources use different sampling protocols")
    if (
        cofitok.get("checkpoint_step") != dense.get("checkpoint_step")
        or cofitok.get("git") != dense.get("git")
        or cofitok.get("runtime_environment_sha256")
        != dense.get("runtime_environment_sha256")
    ):
        raise ValueError("CoFiTok and dense visual sources are not matched")
    return cofitok_sampling


def _required_dict(
    payload: dict[str, Any],
    *path: str,
    name: str,
) -> dict[str, Any]:
    value: Any = payload
    for key in path:
        if not isinstance(value, dict):
            raise ValueError(f"{name} is malformed")
        value = value.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{name} is malformed")
    return value


def _quality_bridge_class_contract(
    result_path: str | Path,
    *,
    cofitok_sampling_path: str | Path,
    dense_sampling_path: str | Path,
    cofitok_sampling_report: dict[str, Any],
    dense_sampling_report: dict[str, Any],
) -> tuple[int, dict[str, Any]]:
    path = reject_symlink_chain(result_path, name="quality-bridge terminal result")
    report = read_json_object(path, name="quality-bridge terminal result")
    if (
        report.get("schema_version") != 1
        or report.get("status") != "completed"
        or report.get("role") != QUALITY_BRIDGE_RESULT_ROLE
        or report.get("authorization_boundary") != RESULT_AUTHORIZATION_BOUNDARY
    ):
        raise ValueError("quality-bridge terminal result contract differs")

    pair_validation = _required_dict(
        report,
        "training",
        "pair_validation",
        name="quality-bridge training pair validation",
    )
    recipe = _required_dict(
        pair_validation,
        "training_recipe",
        name="quality-bridge training recipe",
    )
    expected_shared = _required_dict(
        recipe,
        "expected_shared",
        name="quality-bridge expected shared training recipe",
    )
    observed = _required_dict(
        recipe,
        "observed",
        name="quality-bridge observed training recipe",
    )
    observed_cofitok = _required_dict(
        observed,
        "cofitok",
        name="quality-bridge observed CoFiTok recipe",
    )
    observed_dense = _required_dict(
        observed,
        "dense_identity",
        name="quality-bridge observed dense recipe",
    )
    terminal = _required_dict(report, "terminal", name="quality-bridge terminal evidence")
    methods = _required_dict(
        terminal,
        "methods",
        name="quality-bridge terminal methods",
    )
    class_fidelity = _required_dict(
        terminal,
        "class_fidelity",
        name="quality-bridge class fidelity",
    )
    classifier = _required_dict(
        class_fidelity,
        "classifier",
        name="quality-bridge class-fidelity classifier",
    )
    class_metrics = _required_dict(
        class_fidelity,
        "metrics",
        name="quality-bridge class-fidelity metrics",
    )
    cofitok_class_metrics = _required_dict(
        class_metrics,
        "cofitok",
        name="quality-bridge CoFiTok class-fidelity metrics",
    )
    dense_class_metrics = _required_dict(
        class_metrics,
        "dense_identity",
        name="quality-bridge dense class-fidelity metrics",
    )
    sampling_contract = _required_dict(
        class_fidelity,
        "sampling_contract",
        name="quality-bridge class-fidelity sampling contract",
    )

    class_counts = [
        expected_shared.get("model.num_classes"),
        observed_cofitok.get("model.num_classes"),
        observed_dense.get("model.num_classes"),
        classifier.get("num_classes"),
        cofitok_class_metrics.get("num_classes"),
        dense_class_metrics.get("num_classes"),
    ]
    if (
        any(isinstance(value, bool) or not isinstance(value, int) for value in class_counts)
        or len(set(class_counts)) != 1
        or int(class_counts[0]) < 5
        or pair_validation.get("status") != "pass"
        or recipe.get("valid") is not True
    ):
        raise ValueError("quality-bridge class-count contract differs")
    num_classes = int(class_counts[0])

    cofitok_path = reject_symlink_chain(
        cofitok_sampling_path,
        name="CoFiTok sampling report",
    )
    dense_path = reject_symlink_chain(
        dense_sampling_path,
        name="dense sampling report",
    )
    expected_methods = {
        "cofitok": (cofitok_path, cofitok_sampling_report),
        "dense_identity": (dense_path, dense_sampling_report),
    }
    for method_name, (sampling_path, sampling_report) in expected_methods.items():
        method = _required_dict(
            methods,
            method_name,
            name=f"quality-bridge terminal {method_name} method",
        )
        budget = int(sampling_report["sampling"]["prefix_budgets"][0])
        sample_set = sampling_report["sample_sets"][str(budget)]
        declared = sampling_report["sampling"].get("num_classes")
        if (
            method.get("sampling_report") != file_identity(sampling_path)
            or method.get("sampling") != sampling_report.get("sampling")
            or int(method.get("selected_prefix_budget", -1)) != budget
            or int(method.get("sample_count", -1)) != int(sample_set["count"])
            or method.get("sample_set_sha256") != sample_set.get("sha256")
            or (declared is not None and int(declared) != num_classes)
        ):
            raise ValueError(
                f"quality-bridge terminal {method_name} sampling binding differs"
            )

    matched_sampling = {
        key: value
        for key, value in cofitok_sampling_report["sampling"].items()
        if key not in {"prefix_budgets", "num_classes"}
    }
    class_sampling = sampling_contract.get("sampling")
    if isinstance(class_sampling, dict):
        class_sampling = {
            key: value
            for key, value in class_sampling.items()
            if key != "num_classes"
        }
    cofitok_budget = int(cofitok_sampling_report["sampling"]["prefix_budgets"][0])
    dense_budget = int(dense_sampling_report["sampling"]["prefix_budgets"][0])
    cofitok_sample_set = cofitok_sampling_report["sample_sets"][str(cofitok_budget)]
    dense_sample_set = dense_sampling_report["sample_sets"][str(dense_budget)]
    if (
        class_sampling != matched_sampling
        or int(sampling_contract.get("sample_count_per_method", -1))
        != int(cofitok_sample_set["count"])
        or int(cofitok_sample_set["count"]) != int(dense_sample_set["count"])
        or int(sampling_contract.get("cofitok_prefix_budget", -1)) != cofitok_budget
        or int(sampling_contract.get("dense_prefix_budget", -1)) != dense_budget
        or sampling_contract.get("cofitok_sample_set_sha256")
        != cofitok_sample_set.get("sha256")
        or sampling_contract.get("dense_sample_set_sha256")
        != dense_sample_set.get("sha256")
    ):
        raise ValueError("quality-bridge class-fidelity sampling binding differs")

    class_sources = _required_dict(
        class_fidelity,
        "sources",
        name="quality-bridge class-fidelity sources",
    )
    cofitok_class_source = _required_dict(
        class_sources,
        "cofitok",
        name="quality-bridge CoFiTok class-fidelity source",
    )
    dense_class_source = _required_dict(
        class_sources,
        "dense_identity",
        name="quality-bridge dense class-fidelity source",
    )
    _identity_matches(
        cofitok_class_source,
        name="quality-bridge CoFiTok class-fidelity source",
    )
    _identity_matches(
        dense_class_source,
        name="quality-bridge dense class-fidelity source",
    )

    return num_classes, {
        "report": file_identity(path),
        "num_classes": num_classes,
        "training_recipe_schema": recipe.get("schema"),
        "class_fidelity_status": class_fidelity.get("status"),
        "cofitok_class_fidelity_source": cofitok_class_source,
        "dense_class_fidelity_source": dense_class_source,
    }


def _calibration_source(
    report_path: str | Path,
    *,
    real_dir: str | Path,
    num_classes: int,
) -> tuple[dict[str, Any], list[str], dict[int, Path], dict[str, Any]]:
    path = reject_symlink_chain(report_path, name="classifier calibration report")
    report = read_json_object(path, name="classifier calibration report")
    if (
        report.get("status") != "completed"
        or report.get("calibration_status") != "pass"
        or report.get("role") != CALIBRATION_REPORT_ROLE
        or report.get("claim_boundary") != CALIBRATION_CLAIM_BOUNDARY
    ):
        raise ValueError("classifier calibration report is not an eligible pass")
    sources = report.get("sources")
    selection = report.get("selection")
    parameters = report.get("parameters")
    paths = report.get("paths")
    if not all(isinstance(value, dict) for value in (sources, selection, parameters, paths)):
        raise ValueError("classifier calibration provenance is malformed")
    for source_name in ("label_to_wnid", "timm_synsets", "torchvision_categories"):
        _identity_matches(sources.get(source_name), name=f"calibration {source_name}")
    real_root = reject_symlink_chain(real_dir, name="real-validation directory")
    if real_root.resolve().as_posix() != paths.get("real_dir"):
        raise ValueError("real-validation directory differs from calibration report")
    if int(parameters.get("num_classes", -1)) != num_classes:
        raise ValueError("calibration and sampling class counts differ")
    class_order = load_class_order(
        sources["label_to_wnid"]["path"],
        sources["timm_synsets"]["path"],
        sources["torchvision_categories"]["path"],
        expected_num_classes=num_classes,
    )
    selected = select_balanced_real_images(
        real_root,
        class_order,
        samples_per_class=int(selection.get("samples_per_class", -1)),
    )
    if (
        int(selection.get("samples_per_class", -1)) != 1
        or int(selection.get("sample_count", -1)) != num_classes
        or selection.get("class_order_sha256") != class_order_sha256(class_order)
        or selection.get("selected_sample_set_sha256")
        != selection_sha256(selected, real_dir=real_root)
    ):
        raise ValueError("classifier calibration selection differs")
    selected_by_class = {class_index: image for image, class_index in selected}
    calibration = {
        "report": file_identity(path),
        "git": report.get("git"),
        "runtime_environment_sha256": report.get("runtime_environment_sha256"),
        "classifier": report.get("classifier"),
        "real_dir": real_root.resolve().as_posix(),
        "class_order_sha256": selection["class_order_sha256"],
        "selected_sample_set_sha256": selection["selected_sample_set_sha256"],
        "top1_accuracy": report["metrics"]["top1_accuracy"],
        "top5_accuracy": report["metrics"]["top5_accuracy"],
    }
    return report, class_order, selected_by_class, calibration


def _load_image(path: Path, *, expected_shape: Sequence[int], name: str) -> torch.Tensor:
    source = reject_symlink_chain(path, name=name)
    if not source.is_file():
        raise FileNotFoundError(f"{name} is missing: {source}")
    image = read_image(source.as_posix(), mode=ImageReadMode.RGB).float() / 255.0
    if list(image.shape) != list(expected_shape) or not torch.isfinite(image).all():
        raise ValueError(f"{name} shape or pixels differ")
    return image


def _selected_file(path: Path) -> dict[str, Any]:
    return file_identity(reject_symlink_chain(path, name="selected visual source"))


def _statistics(images: Sequence[torch.Tensor]) -> dict[str, Any]:
    stacked = torch.stack(list(images))
    return {
        "image_count": len(images),
        "pixel_mean": float(stacked.mean().item()),
        "pixel_std": float(stacked.std().item()),
        "pixel_min": float(stacked.min().item()),
        "pixel_max": float(stacked.max().item()),
    }


def _atomic_panel(
    images: Sequence[torch.Tensor],
    path: Path,
    *,
    columns: int,
    indices: Sequence[int],
) -> dict[str, Any]:
    if not images or columns < 1 or len(images) != 3 * len(indices):
        raise ValueError("requested-class visual panel layout is invalid")
    grid = make_grid(list(images), nrow=columns, padding=2, pad_value=1.0)
    temporary = path.with_name(f".{path.name}.part")
    try:
        save_image(grid, temporary, format="png")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        **file_identity(path),
        "indices": list(indices),
        "row_order": ["real_validation", "cofitok", "dense_identity"],
        "columns": columns,
        "grid_shape": list(grid.shape),
    }


def _validate_completed_report(
    report: dict[str, Any],
    *,
    expected: dict[str, Any],
    output_dir: Path,
) -> None:
    if (
        report.get("schema_version") != REPORT_SCHEMA_VERSION
        or report.get("role") != REPORT_ROLE
        or report.get("status") != "completed"
        or report.get("claim_boundary") != CLAIM_BOUNDARY
    ):
        raise ValueError("requested-class visual-audit report contract differs")
    for field in (
        "git",
        "indices",
        "panel_columns",
        "sampling_protocol",
        "sources",
        "classifier_calibration",
        "selected_files",
        "statistics",
    ):
        if report.get(field) != expected.get(field):
            raise ValueError(f"requested-class visual-audit report {field} differs")
    panels = report.get("panels")
    expected_panel_count = math.ceil(
        len(expected["indices"]) / int(expected["panel_columns"])
    )
    if (
        not isinstance(panels, list)
        or not panels
        or len(panels) != expected_panel_count
    ):
        raise ValueError("requested-class visual-audit panels are missing")
    for panel_index, panel in enumerate(panels):
        if not isinstance(panel, dict):
            raise ValueError("requested-class visual-audit panel is malformed")
        path = Path(str(panel.get("path", ""))).resolve()
        offset = panel_index * int(expected["panel_columns"])
        panel_indices = expected["indices"][
            offset : offset + int(expected["panel_columns"])
        ]
        expected_path = (output_dir / f"{PANEL_PREFIX}_{panel_index:02d}.png").resolve()
        if (
            path != expected_path
            or panel.get("indices") != panel_indices
            or panel.get("row_order")
            != ["real_validation", "cofitok", "dense_identity"]
            or int(panel.get("columns", -1)) != len(panel_indices)
        ):
            raise ValueError("requested-class visual-audit panel layout differs")
        observed = file_identity(reject_symlink_chain(path, name="visual-audit panel"))
        for field in ("path", "bytes", "sha256"):
            if panel.get(field) != observed.get(field):
                raise ValueError("requested-class visual-audit panel identity differs")


def build_requested_class_visual_audit(
    *,
    cofitok_sampling_report: str | Path,
    dense_sampling_report: str | Path,
    cofitok_dir: str | Path,
    dense_dir: str | Path,
    quality_result: str | Path,
    classifier_calibration_report: str | Path,
    real_dir: str | Path,
    indices: Sequence[int],
    panel_columns: int,
    output_dir: str | Path,
    expected_git: dict[str, Any] | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    if not indices or list(indices) != sorted(set(indices)) or panel_columns < 1:
        raise ValueError("requested-class visual-audit selection is invalid")
    output = Path(output_dir).resolve()
    if output.is_symlink():
        raise ValueError("requested-class visual-audit output must not be a symlink")
    report_path = output / REPORT_FILENAME

    git = git_provenance(PROJECT_ROOT)
    if expected_git is not None and git != expected_git:
        raise ValueError(f"requested-class visual-audit Git identity differs: {git}")
    cofitok_report, cofitok_source = _sampling_source(
        cofitok_sampling_report,
        expected_dir=cofitok_dir,
        name="CoFiTok",
    )
    dense_report, dense_source = _sampling_source(
        dense_sampling_report,
        expected_dir=dense_dir,
        name="dense identity",
    )
    sampling_protocol = _matched_sampling_protocol(cofitok_report, dense_report)
    start_index = int(sampling_protocol["start_index"])
    stop_index = start_index + int(sampling_protocol["num_samples"])
    if min(indices) < start_index or max(indices) >= stop_index:
        raise ValueError("requested-class visual indices fall outside the sampling window")

    num_classes, quality_source = _quality_bridge_class_contract(
        quality_result,
        cofitok_sampling_path=cofitok_sampling_report,
        dense_sampling_path=dense_sampling_report,
        cofitok_sampling_report=cofitok_report,
        dense_sampling_report=dense_report,
    )
    _, class_order, real_by_class, calibration = _calibration_source(
        classifier_calibration_report,
        real_dir=real_dir,
        num_classes=num_classes,
    )
    expected_shape = list(sampling_protocol["image_shape"])
    cofitok_images: list[torch.Tensor] = []
    dense_images: list[torch.Tensor] = []
    real_images: list[torch.Tensor] = []
    selected_files: list[dict[str, Any]] = []
    for global_index in indices:
        class_index = global_index % num_classes
        wnid = class_order[class_index]
        real_path = real_by_class[class_index]
        cofitok_path = Path(cofitok_source["directory"]) / f"{global_index:06d}.png"
        dense_path = Path(dense_source["directory"]) / f"{global_index:06d}.png"
        real_images.append(
            _load_image(real_path, expected_shape=expected_shape, name="real visual source")
        )
        cofitok_images.append(
            _load_image(
                cofitok_path,
                expected_shape=expected_shape,
                name="CoFiTok visual source",
            )
        )
        dense_images.append(
            _load_image(
                dense_path,
                expected_shape=expected_shape,
                name="dense visual source",
            )
        )
        selected_files.append(
            {
                "global_index": global_index,
                "requested_class_index": class_index,
                "requested_wnid": wnid,
                "real_validation": _selected_file(real_path),
                "cofitok": _selected_file(cofitok_path),
                "dense_identity": _selected_file(dense_path),
            }
        )

    statistics = {
        "real_validation": _statistics(real_images),
        "cofitok": _statistics(cofitok_images),
        "dense_identity": _statistics(dense_images),
    }
    for cohort in statistics.values():
        if any(not math.isfinite(float(value)) for value in cohort.values()):
            raise ValueError("requested-class visual statistics are not finite")
    expected = {
        "git": git,
        "indices": list(indices),
        "panel_columns": panel_columns,
        "sampling_protocol": sampling_protocol,
        "sources": {
            "cofitok": cofitok_source,
            "dense_identity": dense_source,
            "quality_bridge_result": quality_source,
        },
        "classifier_calibration": calibration,
        "selected_files": selected_files,
        "statistics": statistics,
    }
    if report_path.is_file():
        if not resume:
            raise FileExistsError("visual-audit report exists; pass --resume to validate it")
        report = read_json_object(report_path, name="requested-class visual-audit report")
        _validate_completed_report(report, expected=expected, output_dir=output)
        return report
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("requested-class visual-audit output is not empty")
    output.mkdir(parents=True, exist_ok=True)

    panels = []
    for panel_index, offset in enumerate(range(0, len(indices), panel_columns)):
        panel_indices = list(indices[offset : offset + panel_columns])
        count = len(panel_indices)
        images = [
            *real_images[offset : offset + count],
            *cofitok_images[offset : offset + count],
            *dense_images[offset : offset + count],
        ]
        panels.append(
            _atomic_panel(
                images,
                output / f"{PANEL_PREFIX}_{panel_index:02d}.png",
                columns=count,
                indices=panel_indices,
            )
        )
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "role": REPORT_ROLE,
        "status": "completed",
        "claim_boundary": CLAIM_BOUNDARY,
        **expected,
        "panels": panels,
    }
    _validate_completed_report(report, expected=expected, output_dir=output)
    write_json_report(report_path, report)
    return report


def main() -> None:
    args = parse_args()
    report = build_requested_class_visual_audit(
        cofitok_sampling_report=args.cofitok_sampling_report,
        dense_sampling_report=args.dense_sampling_report,
        cofitok_dir=args.cofitok_dir,
        dense_dir=args.dense_dir,
        quality_result=args.quality_result,
        classifier_calibration_report=args.classifier_calibration_report,
        real_dir=args.real_dir,
        indices=parse_indices(args.indices),
        panel_columns=args.panel_columns,
        output_dir=args.output_dir,
        expected_git={
            "revision": args.expected_revision,
            "branch": args.expected_branch,
            "tracked_dirty": False,
        },
        resume=args.resume,
    )
    print(
        json.dumps(
            {
                "report": (Path(args.output_dir).resolve() / REPORT_FILENAME).as_posix(),
                "status": report["status"],
                "panels": len(report["panels"]),
                "indices": len(report["indices"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
