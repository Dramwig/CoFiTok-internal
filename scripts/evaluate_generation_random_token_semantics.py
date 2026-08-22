from __future__ import annotations

import argparse
import math
import os
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw
from torch import nn
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid

from cofitok.generation import load_generation_model
from cofitok.inference_replay import file_identity, read_json_object, reject_symlink_chain
from cofitok.output_lock import exclusive_output_lock
from cofitok.reporting import file_sha256, write_json_report
from cofitok.token_layout import resolve_token_layout, token_layout_summary
from cofitok.training.runtime import autocast_context


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_SCHEMA_VERSION = 1
REPORT_SCHEMA_VERSION = 1
MANIFEST_ROLE = "generation_random_token_semantic_visual_manifest"
REPORT_ROLE = "generation_random_token_semantic_visual_diagnostic"
MANIFEST_FILENAME = "random_token_semantic_manifest.json"
REPORT_FILENAME = "random_token_semantic_report.json"
COMPONENT_PANEL_FILENAME = "random_token_components_vs_gaussian.png"
PREFIX_PANEL_FILENAME = "random_token_prefixes_vs_gaussian.png"
EXPECTED_FILENAMES = {
    MANIFEST_FILENAME,
    REPORT_FILENAME,
    COMPONENT_PANEL_FILENAME,
    PREFIX_PANEL_FILENAME,
}
GPU_QUERY_COMMAND = [
    "nvidia-smi",
    "--query-compute-apps=pid,process_name,used_memory,gpu_uuid",
    "--format=csv,noheader,nounits",
]
MATCHED_FACTORIZATION_TERMINAL_DETAIL = (
    "matched_factorization_quality_regression_diagnostic_completed"
)
MATCHED_FACTORIZATION_SCOPE = (
    "imagenet256_full_100k_matched_factorization_rollout_diagnostic_v1"
)
MATCHED_FACTORIZATION_ROUTE = {
    "id": "run_matched_factorization_quality_regression_probe",
    "category": "matched_quality_regression",
}
MATCHED_FACTORIZATION_FAILED_CHECKS = {
    "matched_fid_tolerance",
    "matched_precision_tolerance",
    "matched_recall_tolerance",
}
EXPECTED_FACTORIZATION_SUPERVISOR_CHECKOUT = {
    "path": (
        "/root/autodl-tmp/CoFiTok/checkouts/"
        "factorization-decision-rebind-v3-06154f4/CoFiTok-internal"
    ),
    "revision": "06154f41b159c8848000b6aa84f267493c51e3bf",
    "tree": "a4ee6f6543660bb41e9d3a21b87093dfff154131",
    "branch": (
        "analysis/generation-factorization-quality-regression-"
        "decision-rebind-v3-20260822"
    ),
    "tracked_dirty": False,
}
MATCHED_FACTORIZATION_SOURCE_KEYS = {
    "deployment_receipt",
    "quality_bridge_followup_decision",
    "source_binding",
    "execution_authorization",
    "diagnostic_report",
}


def _read_stable_json(path: Path, *, name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    before = file_identity(path)
    payload = read_json_object(path, name=name)
    after = file_identity(path)
    if before != after:
        raise ValueError(f"{name} changed while it was being validated")
    return payload, after


def _current_bound_identity(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"path", "bytes", "sha256"}:
        raise ValueError(f"{name} identity is malformed")
    path = reject_symlink_chain(value["path"], name=name)
    current = file_identity(path)
    if current != value:
        raise ValueError(f"{name} identity differs from the current file")
    return current


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a source-bound random-token synthesis diagnostic for an exact "
            "generation checkpoint. The diagnostic is permanently non-authorizing."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--terminal-quality-result", required=True)
    parser.add_argument("--terminal-system-guard", required=True)
    parser.add_argument("--factorization-supervisor-status", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-images", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--weights", choices=["ema", "model"], default="ema")
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--prefix-budgets", default="1,2,4,8")
    parser.add_argument("--clip-sigma", type=float, default=3.0)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-tree", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-checkpoint-step", type=int, required=True)
    parser.add_argument("--expected-training-revision", required=True)
    parser.add_argument("--expected-training-branch", required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse an exactly bound completed report or rerun its immutable manifest.",
    )
    return parser.parse_args()


def _git(*args: str, root: Path = PROJECT_ROOT) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def git_identity(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    return {
        "revision": _git("rev-parse", "HEAD", root=root),
        "tree": _git("rev-parse", "HEAD^{tree}", root=root),
        "branch": _git("branch", "--show-current", root=root),
        "tracked_dirty": bool(
            _git("status", "--porcelain", "--untracked-files=no", root=root)
        ),
    }


def validate_expected_git(
    actual: dict[str, Any],
    *,
    revision: str,
    tree: str,
    branch: str,
) -> None:
    expected = {
        "revision": revision,
        "tree": tree,
        "branch": branch,
        "tracked_dirty": False,
    }
    if actual != expected:
        raise ValueError(f"Random-token evaluator Git identity differs: {actual!r}")


def expected_checkpoint_reference(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_step: int,
) -> dict[str, Any]:
    if len(expected_sha256) != 64:
        raise ValueError("expected checkpoint SHA256 must contain 64 hexadecimal characters")
    try:
        int(expected_sha256, 16)
    except ValueError as error:
        raise ValueError("expected checkpoint SHA256 is not hexadecimal") from error
    if expected_step < 1:
        raise ValueError("expected checkpoint step must be positive")
    checkpoint = reject_symlink_chain(path, name="random-token checkpoint preflight")
    integrity = reject_symlink_chain(
        Path(f"{checkpoint}.integrity.json"),
        name="random-token checkpoint integrity preflight",
    )
    if not checkpoint.is_file() or not integrity.is_file():
        raise FileNotFoundError("random-token checkpoint preflight evidence is incomplete")
    return {
        "path": checkpoint.resolve().as_posix(),
        "sha256": expected_sha256,
        "step": int(expected_step),
    }


def require_gpu_idle() -> dict[str, Any]:
    try:
        result = subprocess.run(
            GPU_QUERY_COMMAND,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError as error:
        raise RuntimeError("nvidia-smi is unavailable; GPU idleness is unverified") from error
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or "").strip()
        raise RuntimeError(
            f"nvidia-smi GPU-idle query failed: {detail or error.returncode}"
        ) from error
    processes = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        columns = [value.strip() for value in line.split(",", 3)]
        if len(columns) != 4:
            raise RuntimeError(f"nvidia-smi returned a malformed compute row: {line!r}")
        try:
            pid = int(columns[0])
        except ValueError as error:
            raise RuntimeError(f"nvidia-smi returned a malformed compute PID: {line!r}") from error
        processes.append(
            {
                "pid": pid,
                "process_name": columns[1],
                "used_memory_mib": columns[2],
                "gpu_uuid": columns[3],
            }
        )
    if processes:
        raise RuntimeError(
            "GPU is not idle; refusing random-token semantic diagnostic: "
            + ", ".join(str(process["pid"]) for process in processes)
        )
    return {
        "gpu_idle_before_checkpoint_load": True,
        "compute_processes": [],
        "query_command": list(GPU_QUERY_COMMAND),
    }


def parse_prefix_budgets(raw: str, token_count: int) -> list[int]:
    try:
        values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    except ValueError as error:
        raise ValueError("prefix budgets must be comma-separated integers") from error
    budgets = sorted(set(values))
    if not budgets or any(value < 1 or value > token_count for value in budgets):
        raise ValueError(
            f"prefix budgets must be in [1, {token_count}], got {budgets}"
        )
    return budgets


def _checkpoint_identity(loaded: Any) -> dict[str, Any]:
    checkpoint = reject_symlink_chain(
        loaded.checkpoint_path,
        name="random-token checkpoint",
    )
    integrity = reject_symlink_chain(
        loaded.checkpoint_integrity_manifest,
        name="random-token checkpoint integrity manifest",
    )
    if not checkpoint.is_file() or not integrity.is_file():
        raise FileNotFoundError("random-token checkpoint evidence is incomplete")
    if checkpoint.stat().st_size < 1 or len(str(loaded.checkpoint_sha256)) != 64:
        raise ValueError("random-token checkpoint identity is malformed")
    return {
        "path": checkpoint.resolve().as_posix(),
        "bytes": checkpoint.stat().st_size,
        "sha256": str(loaded.checkpoint_sha256),
        "integrity_manifest": file_identity(integrity),
        "step": int(loaded.checkpoint_step),
        "artifact_type": str(loaded.artifact_type),
    }


def _request(args: argparse.Namespace, *, token_count: int) -> dict[str, Any]:
    if args.num_images < 1:
        raise ValueError("num-images must be positive")
    if not math.isfinite(args.clip_sigma) or args.clip_sigma <= 0.0:
        raise ValueError("clip-sigma must be finite and positive")
    return {
        "num_images": int(args.num_images),
        "seed": int(args.seed),
        "weights": str(args.weights),
        "precision": str(args.precision),
        "prefix_budgets": parse_prefix_budgets(args.prefix_budgets, token_count),
        "clip_sigma": float(args.clip_sigma),
        "random_token_distribution": "independent_standard_normal_per_token_scalar",
        "gaussian_control": "independent_pixel_gaussian_per_component_with_matched_per_sample_rms",
    }


def _manifest(
    *,
    git: dict[str, Any],
    checkpoint: dict[str, Any],
    terminal_quality_result: dict[str, Any],
    terminal_system_guard: dict[str, Any],
    factorization_supervisor_status: dict[str, Any],
    runtime_preflight: dict[str, Any],
    request: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "role": MANIFEST_ROLE,
        "git": git,
        "evaluator_source": file_identity(Path(__file__)),
        "checkpoint": checkpoint,
        "terminal_quality_result": terminal_quality_result,
        "terminal_system_guard": terminal_system_guard,
        "factorization_supervisor_status": factorization_supervisor_status,
        "runtime_preflight": runtime_preflight,
        "request": request,
        "output_dir": output_dir.resolve().as_posix(),
        "expected_outputs": sorted(EXPECTED_FILENAMES - {MANIFEST_FILENAME}),
        "authorization_policy": {
            "non_authorizing": True,
            "full_training_launch_allowed": False,
            "full_300k_launch_allowed": False,
            "release_authorization_allowed": False,
            "promotion_gate_substitute": False,
        },
    }


def terminal_quality_result_identity(
    path: str | Path,
    *,
    checkpoint: dict[str, Any],
    expected_training_revision: str,
    expected_training_branch: str,
) -> dict[str, Any]:
    source = reject_symlink_chain(
        path,
        name="random-token terminal quality result",
    )
    if not source.is_file():
        raise FileNotFoundError(f"terminal quality result is missing: {source}")
    result, identity = _read_stable_json(
        source,
        name="random-token terminal quality result",
    )
    if (
        result.get("status") != "completed"
        or result.get("role") != "stability_full_data_quality_bridge_result"
        or result.get("stage") != "stability_quality_bridge"
    ):
        raise ValueError("terminal quality result is not the completed quality bridge")
    if result.get("git") != {
        "revision": expected_training_revision,
        "branch": expected_training_branch,
        "tracked_dirty": False,
    }:
        raise ValueError("terminal quality result training Git identity differs")
    boundary = result.get("authorization_boundary")
    if not isinstance(boundary, dict) or any(
        boundary.get(key) is not False
        for key in (
            "quality_bridge_execution_allowed",
            "full_training_launch_allowed",
            "full_300k_launch_allowed",
            "report_is_promotion_gate",
            "release_authorization_allowed",
        )
    ):
        raise ValueError("terminal quality result authorization boundary differs")
    method = result.get("terminal", {}).get("methods", {}).get("cofitok", {})
    if (
        method.get("checkpoint") != checkpoint["path"]
        or method.get("checkpoint_sha256") != checkpoint["sha256"]
        or int(method.get("checkpoint_step", -1)) != checkpoint["step"]
        or method.get("weights") != "ema"
    ):
        raise ValueError("terminal quality result uses another CoFiTok checkpoint")
    return identity


def terminal_system_guard_identity(
    path: str | Path,
    *,
    terminal_quality_result: dict[str, Any],
) -> dict[str, Any]:
    source = reject_symlink_chain(path, name="random-token terminal system guard")
    if not source.is_file():
        raise FileNotFoundError(f"terminal system guard is missing: {source}")
    guard, identity = _read_stable_json(
        source,
        name="random-token terminal system guard",
    )
    policy = guard.get("claim_policy")
    if (
        guard.get("schema_version") != 1
        or guard.get("role") != "generation_terminal_system_claim_guard"
        or guard.get("status") not in {"pass", "hold"}
        or not isinstance(policy, dict)
        or policy.get("terminal_system_evidence_complete") is not True
        or policy.get("larger_training_launch_allowed") is not False
        or policy.get("inference_export_authorization_allowed") is not False
        or policy.get("release_authorization_allowed") is not False
        or policy.get("broad_generation_superiority_claim_allowed") is not False
        or policy.get("sota_claim_allowed") is not False
        or guard.get("sources", {}).get("quality_bridge_result")
        != terminal_quality_result
    ):
        raise ValueError("terminal system guard contract differs")
    return identity


def factorization_supervisor_status_identity(path: str | Path) -> dict[str, Any]:
    source = reject_symlink_chain(
        path,
        name="random-token factorization supervisor status",
    )
    if not source.is_file():
        raise FileNotFoundError(f"factorization supervisor status is missing: {source}")
    status, identity = _read_stable_json(
        source,
        name="random-token factorization supervisor status",
    )
    boundary = status.get("authorization_boundary")
    child_pid = status.get("child_pid")
    sources = status.get("sources")
    if (
        status.get("schema_version") != 1
        or status.get("role") != "generation_factorization_quality_regression_supervisor"
        or status.get("status") != "completed"
        or status.get("detail") != MATCHED_FACTORIZATION_TERMINAL_DETAIL
        or isinstance(child_pid, bool)
        or not isinstance(child_pid, int)
        or child_pid < 1
        or not isinstance(sources, dict)
        or not MATCHED_FACTORIZATION_SOURCE_KEYS.issubset(sources)
        or not isinstance(boundary, dict)
        or boundary.get("training_launch_allowed") is not False
        or boundary.get("checkpoint_promotion_allowed") is not False
        or boundary.get("followup_experiment_launch_allowed") is not False
        or boundary.get("full_300k_launch_allowed") is not False
        or boundary.get("release_authorization_allowed") is not False
    ):
        raise ValueError("factorization supervisor terminal contract differs")

    bound = {
        name: _current_bound_identity(sources[name], name=f"factorization {name}")
        for name in MATCHED_FACTORIZATION_SOURCE_KEYS
    }
    deployment, deployment_identity = _read_stable_json(
        Path(bound["deployment_receipt"]["path"]),
        name="factorization supervisor deployment receipt",
    )
    if (
        deployment_identity != bound["deployment_receipt"]
        or deployment.get("schema_version") != 1
        or deployment.get("role")
        != "generation_factorization_quality_regression_supervisor_deployment"
        or deployment.get("status") != "pass"
        or deployment.get("control", {}).get("checkout")
        != EXPECTED_FACTORIZATION_SUPERVISOR_CHECKOUT
        or deployment.get("targets", {}).get("status_output")
        != source.resolve().as_posix()
    ):
        raise ValueError("factorization supervisor deployment binding differs")
    if status.get("project") != EXPECTED_FACTORIZATION_SUPERVISOR_CHECKOUT["path"]:
        raise ValueError("factorization supervisor project binding differs")

    source_binding, source_binding_identity = _read_stable_json(
        Path(bound["source_binding"]["path"]),
        name="factorization source binding",
    )
    selected_route = source_binding.get("selected_route")
    if (
        source_binding_identity != bound["source_binding"]
        or source_binding.get("schema_version") != 1
        or source_binding.get("role")
        != "generation_factorization_quality_regression_source_binding"
        or source_binding.get("status") != "verified"
        or source_binding.get("scope") != MATCHED_FACTORIZATION_SCOPE
        or not isinstance(selected_route, dict)
        or selected_route.get("id") != MATCHED_FACTORIZATION_ROUTE["id"]
        or selected_route.get("category")
        != MATCHED_FACTORIZATION_ROUTE["category"]
        or not isinstance(selected_route.get("failed_checks"), list)
        or not selected_route["failed_checks"]
        or not set(selected_route["failed_checks"]).issubset(
            MATCHED_FACTORIZATION_FAILED_CHECKS
        )
        or source_binding.get("sources", {}).get("followup_decision")
        != bound["quality_bridge_followup_decision"]
    ):
        raise ValueError("factorization matched-route source binding differs")

    authorization, authorization_identity = _read_stable_json(
        Path(bound["execution_authorization"]["path"]),
        name="factorization execution authorization",
    )
    if (
        authorization_identity != bound["execution_authorization"]
        or authorization.get("schema_version") != 1
        or authorization.get("role")
        != "generation_factorization_quality_regression_execution_authorization"
        or authorization.get("status") != "authorized"
        or authorization.get("scope") != MATCHED_FACTORIZATION_SCOPE
        or authorization.get("source_binding") != bound["source_binding"]
        or authorization.get("git")
        != {
            "revision": EXPECTED_FACTORIZATION_SUPERVISOR_CHECKOUT["revision"],
            "branch": EXPECTED_FACTORIZATION_SUPERVISOR_CHECKOUT["branch"],
            "tracked_dirty": False,
        }
        or authorization.get("authorization_boundary", {}).get(
            "training_launch_allowed"
        )
        is not False
        or authorization.get("authorization_boundary", {}).get(
            "full_300k_launch_allowed"
        )
        is not False
    ):
        raise ValueError("factorization execution authorization binding differs")

    diagnostic, diagnostic_identity = _read_stable_json(
        Path(bound["diagnostic_report"]["path"]),
        name="factorization diagnostic report",
    )
    if (
        diagnostic_identity != bound["diagnostic_report"]
        or diagnostic.get("schema_version") != 1
        or diagnostic.get("role")
        != "generation_factorization_quality_regression_diagnostic"
        or diagnostic.get("status") != "completed"
        or diagnostic.get("scope") != MATCHED_FACTORIZATION_SCOPE
        or diagnostic.get("sources", {}).get("execution_authorization")
        != bound["execution_authorization"]
        or diagnostic.get("claim_boundary", {}).get("training_launch_allowed")
        is not False
        or diagnostic.get("claim_boundary", {}).get(
            "quality_advantage_claim_allowed"
        )
        is not False
    ):
        raise ValueError("factorization diagnostic binding differs")
    return identity


def validate_loaded_target(
    loaded: Any,
    *,
    requested_weights: str,
    expected_checkpoint_sha256: str,
    expected_checkpoint_step: int,
) -> None:
    if requested_weights != "ema" or loaded.weights != "ema":
        raise ValueError("random-token semantic diagnostic requires EMA checkpoint weights")
    if len(expected_checkpoint_sha256) != 64:
        raise ValueError("expected checkpoint SHA256 must contain 64 hexadecimal characters")
    try:
        int(expected_checkpoint_sha256, 16)
    except ValueError as error:
        raise ValueError("expected checkpoint SHA256 is not hexadecimal") from error
    if expected_checkpoint_step < 1:
        raise ValueError("expected checkpoint step must be positive")
    if (
        loaded.checkpoint_sha256 != expected_checkpoint_sha256
        or loaded.checkpoint_step != expected_checkpoint_step
    ):
        raise ValueError(
            "random-token checkpoint differs from the exact requested SHA256/step"
        )
    synthesis_mode = str(loaded.config.model.synthesis_mode)
    if synthesis_mode in {"dense", "dense_identity", "monolithic_dense"}:
        raise ValueError("random-token semantic diagnostic requires factorized CoFiTok")


def prepare_output_directory(
    output_dir: str | Path,
    *,
    resume: bool,
) -> tuple[Path, Path, Path, Path, Path]:
    output = reject_symlink_chain(
        output_dir,
        name="random-token diagnostic output directory",
    )
    if output.exists() and not output.is_dir():
        raise ValueError(f"random-token output is not a directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    output = output.resolve()
    unexpected = sorted(
        path.name for path in output.iterdir() if path.name not in EXPECTED_FILENAMES
    )
    if unexpected:
        raise ValueError(
            "random-token output contains unexpected files: " + ", ".join(unexpected)
        )
    manifest = output / MANIFEST_FILENAME
    report = output / REPORT_FILENAME
    component_panel = output / COMPONENT_PANEL_FILENAME
    prefix_panel = output / PREFIX_PANEL_FILENAME
    if not resume and any(
        path.exists() for path in (manifest, report, component_panel, prefix_panel)
    ):
        raise FileExistsError(
            "random-token output already contains evidence; pass --resume for exact replay"
        )
    return output, manifest, report, component_panel, prefix_panel


def _random_tokens(
    *,
    model_config: Any,
    batch_size: int,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[list[torch.Tensor], dict[str, Any]]:
    layout = resolve_token_layout(
        image_size=model_config.image_size,
        image_channels=model_config.image_channels,
        token_count=model_config.token_count,
        token_channels=model_config.token_channels,
        token_channel_schedule=model_config.token_channel_schedule,
        token_spatial_strides=model_config.token_spatial_strides,
    )
    tokens = [
        torch.randn(
            (batch_size, channels, spatial_size, spatial_size),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        for channels, spatial_size in zip(layout.channels, layout.spatial_sizes)
    ]
    return tokens, token_layout_summary(layout)


def _synthesize(
    synthesis: nn.Module,
    tokens: list[torch.Tensor],
    *,
    device: torch.device,
    precision: str,
) -> list[torch.Tensor]:
    with autocast_context(device, precision):
        return [component.float() for component in synthesis(tokens)]


def _prefix_sums(components: list[torch.Tensor]) -> list[torch.Tensor]:
    if not components:
        raise ValueError("random-token synthesis produced no components")
    running = torch.zeros_like(components[0])
    prefixes = []
    for component in components:
        running = running + component
        prefixes.append(running)
    return prefixes


def _matched_gaussian_controls(
    components: list[torch.Tensor],
    *,
    generator: torch.Generator,
) -> list[torch.Tensor]:
    controls = []
    for component in components:
        control = torch.randn(
            component.shape,
            generator=generator,
            device=component.device,
            dtype=torch.float32,
        )
        component_rms = component.square().flatten(1).mean(dim=1).sqrt().view(-1, 1, 1, 1)
        control_rms = control.square().flatten(1).mean(dim=1).sqrt().view(-1, 1, 1, 1)
        controls.append(control * (component_rms / control_rms.clamp_min(1e-12)))
    return controls


def _lag_one_correlation(tensor: torch.Tensor) -> dict[str, float]:
    centered = tensor.float() - tensor.float().mean(dim=(-2, -1), keepdim=True)

    def corr(first: torch.Tensor, second: torch.Tensor) -> float:
        numerator = (first * second).mean()
        denominator = first.square().mean().sqrt() * second.square().mean().sqrt()
        return float((numerator / denominator.clamp_min(1e-12)).item())

    return {
        "horizontal": corr(centered[..., :, :-1], centered[..., :, 1:]),
        "vertical": corr(centered[..., :-1, :], centered[..., 1:, :]),
    }


def _tensor_statistics(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.float()
    mean = value.mean()
    centered = value - mean
    variance = centered.square().mean()
    rms = value.square().mean().sqrt()
    kurtosis = centered.pow(4).mean() / variance.square().clamp_min(1e-12)
    return {
        "shape": list(value.shape),
        "mean": float(mean.item()),
        "std": float(value.std(unbiased=False).item()),
        "rms": float(rms.item()),
        "min": float(value.min().item()),
        "max": float(value.max().item()),
        "kurtosis": float(kurtosis.item()),
        "lag_one_correlation": _lag_one_correlation(value),
    }


def _synthesis_architecture(synthesis: nn.Module) -> dict[str, Any]:
    parameters = list(synthesis.named_parameters())
    buffers = list(synthesis.named_buffers())
    module_types = Counter(type(module).__name__ for module in synthesis.modules())
    nonlinear_types = (
        nn.ReLU,
        nn.SiLU,
        nn.GELU,
        nn.ELU,
        nn.LeakyReLU,
        nn.PReLU,
        nn.Sigmoid,
        nn.Tanh,
        nn.Softmax,
        nn.MultiheadAttention,
    )
    return {
        "module_type": type(synthesis).__name__,
        "module_type_counts": dict(sorted(module_types.items())),
        "parameter_count": sum(parameter.numel() for _, parameter in parameters),
        "trainable_parameter_count": sum(
            parameter.numel() for _, parameter in parameters if parameter.requires_grad
        ),
        "parameter_names": [name for name, _ in parameters],
        "bias_parameter_names": [name for name, _ in parameters if name.endswith("bias")],
        "buffer_count": sum(buffer.numel() for _, buffer in buffers),
        "buffer_names": [name for name, _ in buffers],
        "nonlinear_or_attention_modules": [
            type(module).__name__
            for module in synthesis.modules()
            if isinstance(module, nonlinear_types)
        ],
    }


def _numerical_contract(
    synthesis: nn.Module,
    first: list[torch.Tensor],
    second: list[torch.Tensor],
) -> dict[str, float]:
    alpha = 0.375
    beta = -0.625
    first_components = [value.float() for value in synthesis(first)]
    second_components = [value.float() for value in synthesis(second)]
    combined = [alpha * a + beta * b for a, b in zip(first, second)]
    combined_components = [value.float() for value in synthesis(combined)]
    expected = [alpha * a + beta * b for a, b in zip(first_components, second_components)]
    linearity_max_abs = max(
        float((actual - target).abs().max().item())
        for actual, target in zip(combined_components, expected)
    )
    zero_components = [value.float() for value in synthesis([torch.zeros_like(x) for x in first])]
    zero_max_abs = max(float(value.abs().max().item()) for value in zero_components)
    off_token_max_abs = 0.0
    for active_index in range(len(first)):
        isolated = [torch.zeros_like(value) for value in first]
        isolated[active_index] = first[active_index]
        isolated_components = [value.float() for value in synthesis(isolated)]
        off_token_max_abs = max(
            off_token_max_abs,
            max(
                (
                    float(component.abs().max().item())
                    if index != active_index
                    else 0.0
                )
                for index, component in enumerate(isolated_components)
            ),
        )
    return {
        "linearity_max_abs": linearity_max_abs,
        "zero_token_max_abs": zero_max_abs,
        "cross_token_leakage_max_abs": off_token_max_abs,
    }


def _display_row(tensor: torch.Tensor, *, clip_sigma: float) -> tuple[torch.Tensor, float]:
    rms = float(tensor.float().square().mean().sqrt().item())
    scale = max(rms * clip_sigma, 1e-12)
    display = (0.5 + tensor.float() / (2.0 * scale)).clamp(0.0, 1.0)
    return display.cpu(), scale


def _atomic_labeled_panel(
    rows: list[tuple[str, torch.Tensor]],
    path: Path,
    *,
    clip_sigma: float,
) -> tuple[dict[str, Any], dict[str, float]]:
    if not rows:
        raise ValueError("random-token visual panel has no rows")
    row_images = []
    scales: dict[str, float] = {}
    image_count = rows[0][1].shape[0]
    if image_count < 1 or any(row.shape[0] != image_count for _, row in rows):
        raise ValueError("random-token visual rows have inconsistent batch sizes")
    for label, row in rows:
        display, scale = _display_row(row, clip_sigma=clip_sigma)
        scales[label] = scale
        row_images.append(
            to_pil_image(make_grid(display, nrow=image_count, padding=2, pad_value=1.0))
        )
    label_width = 250
    row_gap = 8
    title_height = 42
    width = label_width + max(image.width for image in row_images)
    height = title_height + sum(image.height for image in row_images) + row_gap * (len(rows) - 1)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (12, 12),
        "Random-token S_k output (neutral=0; symmetric RMS clipping)",
        fill=(20, 20, 20),
    )
    y = title_height
    for (label, _), image in zip(rows, row_images):
        draw.text((12, y + 8), label, fill=(20, 20, 20))
        canvas.paste(image.convert("RGB"), (label_width, y))
        y += image.height + row_gap
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.part")
    try:
        canvas.save(temporary, format="PNG")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return (
        {
            "path": path.resolve().as_posix(),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
            "row_count": len(rows),
            "images_per_row": image_count,
            "width": canvas.width,
            "height": canvas.height,
        },
        scales,
    )


@torch.no_grad()
def evaluate_random_token_synthesis(
    *,
    synthesis: nn.Module,
    model_config: Any,
    device: torch.device,
    num_images: int,
    seed: int,
    precision: str,
    prefix_budgets: list[int],
    clip_sigma: float,
    component_panel_path: Path,
    prefix_panel_path: Path,
) -> dict[str, Any]:
    generator = torch.Generator(device=device).manual_seed(seed)
    control_generator = torch.Generator(device=device).manual_seed(seed + 1)
    second_generator = torch.Generator(device=device).manual_seed(seed + 2)
    tokens, layout = _random_tokens(
        model_config=model_config,
        batch_size=num_images,
        device=device,
        generator=generator,
    )
    second_tokens, _ = _random_tokens(
        model_config=model_config,
        batch_size=num_images,
        device=device,
        generator=second_generator,
    )
    components = _synthesize(
        synthesis,
        tokens,
        device=device,
        precision=precision,
    )
    controls = _matched_gaussian_controls(components, generator=control_generator)
    prefixes = _prefix_sums(components)
    control_prefixes = _prefix_sums(controls)
    component_rows = [
        (f"S_{index}(random z_{index})", component)
        for index, component in enumerate(components, start=1)
    ] + [
        (f"matched pixel Gaussian {index}", control)
        for index, control in enumerate(controls, start=1)
    ]
    prefix_rows = [
        (f"random-token cumulative m={budget}", prefixes[budget - 1])
        for budget in prefix_budgets
    ] + [
        (f"Gaussian-control cumulative m={budget}", control_prefixes[budget - 1])
        for budget in prefix_budgets
    ]
    component_panel, component_scales = _atomic_labeled_panel(
        component_rows,
        component_panel_path,
        clip_sigma=clip_sigma,
    )
    prefix_panel, prefix_scales = _atomic_labeled_panel(
        prefix_rows,
        prefix_panel_path,
        clip_sigma=clip_sigma,
    )
    numerical = _numerical_contract(synthesis, tokens, second_tokens)
    return {
        "token_layout": layout,
        "synthesis_architecture": _synthesis_architecture(synthesis),
        "numerical_contract": numerical,
        "random_components": [
            _tensor_statistics(component) for component in components
        ],
        "matched_gaussian_components": [
            _tensor_statistics(control) for control in controls
        ],
        "random_prefixes": {
            str(budget): _tensor_statistics(prefixes[budget - 1])
            for budget in prefix_budgets
        },
        "matched_gaussian_prefixes": {
            str(budget): _tensor_statistics(control_prefixes[budget - 1])
            for budget in prefix_budgets
        },
        "visualization": {
            "mapping": "display=clip(0.5 + value/(2*clip_sigma*RMS), 0, 1)",
            "clip_sigma": clip_sigma,
            "component_row_scales": component_scales,
            "prefix_row_scales": prefix_scales,
        },
        "artifacts": {
            "components_vs_gaussian": component_panel,
            "prefixes_vs_gaussian": prefix_panel,
        },
    }


def validate_completed_report(
    report: dict[str, Any],
    *,
    manifest_identity: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    if (
        report.get("schema_version") != REPORT_SCHEMA_VERSION
        or report.get("role") != REPORT_ROLE
        or report.get("status") != "completed"
        or report.get("manifest") != manifest_identity
        or report.get("git") != manifest.get("git")
        or report.get("evaluator_source") != manifest.get("evaluator_source")
        or report.get("checkpoint") != manifest.get("checkpoint")
        or report.get("terminal_quality_result")
        != manifest.get("terminal_quality_result")
        or report.get("terminal_system_guard") != manifest.get("terminal_system_guard")
        or report.get("factorization_supervisor_status")
        != manifest.get("factorization_supervisor_status")
        or report.get("runtime_preflight") != manifest.get("runtime_preflight")
        or report.get("request") != manifest.get("request")
        or report.get("authorization_policy") != manifest.get("authorization_policy")
    ):
        raise ValueError("random-token completed report binding differs")
    for name in (
        "evaluator_source",
        "terminal_quality_result",
        "terminal_system_guard",
        "factorization_supervisor_status",
    ):
        claimed = manifest.get(name)
        if not isinstance(claimed, dict) or file_identity(
            str(claimed.get("path", ""))
        ) != claimed:
            raise ValueError(f"random-token completed {name} identity differs")
    artifacts = report.get("diagnostic", {}).get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {
        "components_vs_gaussian",
        "prefixes_vs_gaussian",
    }:
        raise ValueError("random-token completed report artifact set differs")
    expected_names = {COMPONENT_PANEL_FILENAME, PREFIX_PANEL_FILENAME}
    actual_names = set()
    for artifact in artifacts.values():
        if not isinstance(artifact, dict):
            raise ValueError("random-token artifact identity is malformed")
        path = reject_symlink_chain(
            str(artifact.get("path", "")),
            name="random-token completed artifact",
        )
        actual_names.add(path.name)
        if (
            not path.is_file()
            or path.parent.resolve().as_posix() != manifest["output_dir"]
            or int(artifact.get("bytes", -1)) != path.stat().st_size
            or str(artifact.get("sha256", "")) != file_sha256(path)
        ):
            raise ValueError("random-token completed artifact identity differs")
    if actual_names != expected_names:
        raise ValueError("random-token completed artifact filenames differ")


def _run(args: argparse.Namespace) -> None:
    (
        output_dir,
        manifest_path,
        report_path,
        component_panel,
        prefix_panel,
    ) = prepare_output_directory(args.output_dir, resume=args.resume)
    git = git_identity(PROJECT_ROOT)
    validate_expected_git(
        git,
        revision=args.expected_revision,
        tree=args.expected_tree,
        branch=args.expected_branch,
    )
    preflight_checkpoint = expected_checkpoint_reference(
        args.checkpoint,
        expected_sha256=args.expected_checkpoint_sha256,
        expected_step=args.expected_checkpoint_step,
    )
    preflight_quality_result = terminal_quality_result_identity(
        args.terminal_quality_result,
        checkpoint=preflight_checkpoint,
        expected_training_revision=args.expected_training_revision,
        expected_training_branch=args.expected_training_branch,
    )
    preflight_terminal_guard = terminal_system_guard_identity(
        args.terminal_system_guard,
        terminal_quality_result=preflight_quality_result,
    )
    preflight_factorization_status = factorization_supervisor_status_identity(
        args.factorization_supervisor_status
    )
    runtime_preflight = require_gpu_idle()
    loaded = load_generation_model(args.checkpoint, weights=args.weights)
    validate_loaded_target(
        loaded,
        requested_weights=args.weights,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        expected_checkpoint_step=args.expected_checkpoint_step,
    )
    checkpoint = _checkpoint_identity(loaded)
    quality_result = terminal_quality_result_identity(
        args.terminal_quality_result,
        checkpoint=checkpoint,
        expected_training_revision=args.expected_training_revision,
        expected_training_branch=args.expected_training_branch,
    )
    terminal_guard = terminal_system_guard_identity(
        args.terminal_system_guard,
        terminal_quality_result=quality_result,
    )
    factorization_status = factorization_supervisor_status_identity(
        args.factorization_supervisor_status
    )
    if (
        quality_result != preflight_quality_result
        or terminal_guard != preflight_terminal_guard
        or factorization_status != preflight_factorization_status
    ):
        raise ValueError("random-token terminal evidence changed during checkpoint load")
    if git_identity(PROJECT_ROOT) != git:
        raise ValueError("random-token evaluator Git identity changed during checkpoint load")
    request = _request(args, token_count=loaded.config.model.token_count)
    expected_manifest = _manifest(
        git=git,
        checkpoint=checkpoint,
        terminal_quality_result=quality_result,
        terminal_system_guard=terminal_guard,
        factorization_supervisor_status=factorization_status,
        runtime_preflight=runtime_preflight,
        request=request,
        output_dir=output_dir,
    )
    if manifest_path.is_file():
        existing_manifest = read_json_object(
            manifest_path,
            name="random-token semantic manifest",
        )
        if existing_manifest != expected_manifest:
            raise ValueError("random-token resume manifest does not match the request")
    else:
        write_json_report(manifest_path, expected_manifest)
    manifest_identity = file_identity(manifest_path)
    if report_path.is_file():
        existing_report = read_json_object(
            report_path,
            name="random-token semantic report",
        )
        validate_completed_report(
            existing_report,
            manifest_identity=manifest_identity,
            manifest=expected_manifest,
        )
        print(f"reused completed random-token diagnostic {report_path}")
        return

    start = time.time()
    diagnostic = evaluate_random_token_synthesis(
        synthesis=loaded.model.synthesis,
        model_config=loaded.config.model,
        device=loaded.device,
        num_images=request["num_images"],
        seed=request["seed"],
        precision=request["precision"],
        prefix_budgets=request["prefix_budgets"],
        clip_sigma=request["clip_sigma"],
        component_panel_path=component_panel,
        prefix_panel_path=prefix_panel,
    )
    architecture = diagnostic["synthesis_architecture"]
    numerical = diagnostic["numerical_contract"]
    if git_identity(PROJECT_ROOT) != git:
        raise ValueError("random-token evaluator Git identity changed during evaluation")
    for name, claimed in (
        ("evaluator source", expected_manifest["evaluator_source"]),
        ("terminal quality result", quality_result),
        ("terminal system guard", terminal_guard),
        ("factorization supervisor status", factorization_status),
    ):
        if file_identity(str(claimed["path"])) != claimed:
            raise ValueError(f"random-token {name} changed during evaluation")
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "role": REPORT_ROLE,
        "status": "completed",
        "git": git,
        "evaluator_source": expected_manifest["evaluator_source"],
        "manifest": manifest_identity,
        "checkpoint": checkpoint,
        "terminal_quality_result": quality_result,
        "terminal_system_guard": terminal_guard,
        "factorization_supervisor_status": factorization_status,
        "runtime_preflight": runtime_preflight,
        "request": request,
        "weights": loaded.weights,
        "authorization_policy": expected_manifest["authorization_policy"],
        "claim_policy": {
            "supports": [
                "human inspection of random-token synthesis outputs",
                (
                    "exact-checkpoint verification of linearity, zero preservation, "
                    "and per-token isolation"
                ),
                "static inspection of synthesis parameters, biases, nonlinearities, and attention",
            ],
            "does_not_support": [
                "proof that every possible random token lacks semantic structure",
                "generation-quality superiority",
                "promotion, release, full training, or full-300K authorization",
            ],
            "semantic_absence_proven": False,
            "generation_quality_advantage_proven": False,
        },
        "checks": {
            "ema_weights_used": loaded.weights == "ema",
            "checkpoint_step_matches_request": (
                loaded.checkpoint_step == args.expected_checkpoint_step
            ),
            "synthesis_has_no_trainable_parameters": architecture["trainable_parameter_count"] == 0,
            "synthesis_has_no_bias_parameters": not architecture["bias_parameter_names"],
            "synthesis_has_no_nonlinearity_or_attention": not architecture[
                "nonlinear_or_attention_modules"
            ],
            "zero_token_exact": numerical["zero_token_max_abs"] == 0.0,
            "cross_token_leakage_exact": numerical["cross_token_leakage_max_abs"] == 0.0,
            "linearity_numerically_close": numerical["linearity_max_abs"] <= 1e-6,
        },
        "diagnostic": diagnostic,
        "runtime": {
            "elapsed_seconds": time.time() - start,
            "device": str(loaded.device),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
    }
    write_json_report(report_path, report)
    print(f"wrote {report_path}")
    print(f"wrote {component_panel}")
    print(f"wrote {prefix_panel}")


def main() -> None:
    args = parse_args()
    with exclusive_output_lock(
        args.output_dir,
        role=REPORT_ROLE,
    ):
        _run(args)


if __name__ == "__main__":
    main()
