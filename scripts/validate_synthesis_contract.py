from __future__ import annotations

import argparse
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from cofitok.configs import ExperimentConfig, load_config


FORBIDDEN_RESTRICTED_MODULE_NAMES = {
    "BatchNorm1d",
    "BatchNorm2d",
    "BatchNorm3d",
    "GELU",
    "GroupNorm",
    "LayerNorm",
    "LeakyReLU",
    "MultiheadAttention",
    "PReLU",
    "ReLU",
    "SELU",
    "SiLU",
    "Softmax",
    "Tanh",
}


@dataclass(frozen=True)
class ContractResult:
    config: str
    name: str
    dataset: str
    synthesis_mode: str
    status: str
    evidence: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the CoFiTok restricted S_k contract for experiment configs.")
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Config path to validate. Can be repeated. If omitted, --config-glob is used.",
    )
    parser.add_argument(
        "--config-glob",
        default="configs/*.json",
        help="Glob used when --config is omitted.",
    )
    parser.add_argument(
        "--strict-ablation",
        action="store_true",
        help="Fail configs whose synthesis mode is not restricted instead of recording them as ablations.",
    )
    parser.add_argument(
        "--static-only",
        action="store_true",
        help="Run config/source-level checks only. Remote validation should omit this for full dynamic checks.",
    )
    parser.add_argument("--token-spatial-size", type=int, default=8)
    return parser.parse_args()


def _config_paths(configs: list[str], config_glob: str) -> list[Path]:
    if configs:
        paths = [Path(path) for path in configs]
    else:
        paths = sorted(Path().glob(config_glob))
    if not paths:
        raise FileNotFoundError(f"No configs matched: {configs or [config_glob]}")
    return paths


def _signature_has_only_payload_arg(callable_obj: Any, payload_arg: str) -> bool:
    signature = inspect.signature(callable_obj)
    parameters = list(signature.parameters.values())
    return len(parameters) == 1 and parameters[0].name == payload_arg


def _all_conv_bias_free(module: Any) -> bool:
    from torch import nn

    return all(layer.bias is None for layer in module.modules() if isinstance(layer, nn.Conv2d))


def _forbidden_modules(module: Any) -> list[str]:
    names = []
    for child in module.modules():
        child_name = child.__class__.__name__
        if child_name in FORBIDDEN_RESTRICTED_MODULE_NAMES:
            names.append(child_name)
    return names


def _contains_suspicious_constant_name(module: Any) -> list[str]:
    suspicious = []
    for name, _ in list(module.named_parameters()) + list(module.named_buffers()):
        lowered = name.lower()
        if "constant" in lowered or "prior" in lowered or "semantic" in lowered:
            suspicious.append(name)
    return suspicious


def _zero_token_max_abs(model: Any, config: ExperimentConfig, spatial_size: int) -> float:
    import torch

    tokens = [
        torch.zeros(2, config.model.token_channels, spatial_size, spatial_size)
        for _ in range(config.model.token_count)
    ]
    with torch.no_grad():
        components = model.synthesis(tokens)
    if not components:
        raise AssertionError("synthesis bank returned no components")
    return max(float(component.abs().max().item()) for component in components)


def _static_source_contract() -> dict[str, Any]:
    source_path = Path(__file__).resolve().parents[1] / "src" / "cofitok" / "models" / "synthesis.py"
    source = source_path.read_text(encoding="utf-8")
    required_snippets = [
        "class RestrictedSynthesis(nn.Module):",
        "def forward(self, token: torch.Tensor) -> torch.Tensor:",
        "bias=False",
        "self.proj(token)",
        "self.local(",
    ]
    missing = [snippet for snippet in required_snippets if snippet not in source]
    if missing:
        raise AssertionError(f"Static synthesis source contract missing snippets: {missing}")
    restricted_block = source.split("class RestrictedSynthesis(nn.Module):", 1)[1].split("class DeepSynthesis", 1)[0]
    forbidden = [name for name in FORBIDDEN_RESTRICTED_MODULE_NAMES if f"nn.{name}" in restricted_block]
    if forbidden:
        raise AssertionError(f"Static restricted S_k source includes forbidden modules: {forbidden}")
    if "bias=True" in restricted_block:
        raise AssertionError("Static restricted S_k source includes bias=True")
    return {
        "mode": "static",
        "source": source_path.as_posix(),
        "required_snippet_count": len(required_snippets),
        "forbidden_source_module_count": 0,
    }


def _static_config_contract(config: ExperimentConfig) -> dict[str, Any]:
    if config.model.token_count < 1:
        raise AssertionError("token_count must be >= 1")
    if config.model.token_channels < 1:
        raise AssertionError("token_channels must be >= 1")
    if config.model.synthesis_kernel_size < 1 or config.model.synthesis_kernel_size % 2 == 0:
        raise AssertionError("restricted synthesis kernel must be a positive odd integer")
    active = config.model.synthesis_active_token_channels
    if active and len(active) != config.model.token_count:
        raise AssertionError("active token channel schedule must match token_count")
    if any(value < 1 or value > config.model.token_channels for value in active):
        raise AssertionError("active token channel values must stay inside token_channels")
    strides = config.model.synthesis_token_strides
    if strides and len(strides) != config.model.token_count:
        raise AssertionError("token stride schedule must match token_count")
    if any(value < 1 for value in strides):
        raise AssertionError("token stride values must be >= 1")
    return {
        "token_count": config.model.token_count,
        "token_channels": config.model.token_channels,
        "kernel_size": config.model.synthesis_kernel_size,
        "active_token_channels": active,
        "token_strides": strides,
    }


def _dynamic_restricted_contract(config: ExperimentConfig, path: Path, spatial_size: int) -> dict[str, Any]:
    from cofitok.models import CoFiTokTiny
    from cofitok.models.synthesis import RestrictedSynthesis, RestrictedSynthesisBank

    model = CoFiTokTiny(config.model)
    evidence: dict[str, Any] = {
        "mode": "dynamic",
        "token_count": config.model.token_count,
        "token_channels": config.model.token_channels,
        "signature": "token-list only",
    }

    if not isinstance(model.synthesis, RestrictedSynthesisBank):
        raise AssertionError(f"{path}: restricted config did not build RestrictedSynthesisBank")
    if not _signature_has_only_payload_arg(model.synthesis.forward, "tokens"):
        raise AssertionError(f"{path}: synthesis bank forward must accept only tokens")

    synthesizer_count = 0
    for synthesizer in model.synthesis.synthesizers:
        synthesizer_count += 1
        if not isinstance(synthesizer, RestrictedSynthesis):
            raise AssertionError(f"{path}: restricted bank contains non-RestrictedSynthesis module")
        if not _signature_has_only_payload_arg(synthesizer.forward, "token"):
            raise AssertionError(f"{path}: restricted S_k forward must accept only token")

    if synthesizer_count != config.model.token_count:
        raise AssertionError(f"{path}: expected {config.model.token_count} synthesizers, got {synthesizer_count}")

    forbidden = _forbidden_modules(model.synthesis)
    if forbidden:
        raise AssertionError(f"{path}: restricted S_k contains forbidden modules {forbidden}")
    if not _all_conv_bias_free(model.synthesis):
        raise AssertionError(f"{path}: restricted S_k contains a Conv2d bias")
    suspicious_constants = _contains_suspicious_constant_name(model.synthesis)
    if suspicious_constants:
        raise AssertionError(f"{path}: restricted S_k has suspicious constant/prior names {suspicious_constants}")

    zero_max_abs = _zero_token_max_abs(model, config, spatial_size)
    if zero_max_abs != 0.0:
        raise AssertionError(f"{path}: S_k(0) max abs is {zero_max_abs}, expected exact zero")

    evidence.update(
        {
            "synthesizer_count": synthesizer_count,
            "conv_bias_free": True,
            "forbidden_module_count": 0,
            "suspicious_constant_count": 0,
            "zero_token_max_abs": zero_max_abs,
        }
    )
    return evidence


def validate_restricted_config(path: Path, spatial_size: int = 8, static_only: bool = False) -> ContractResult:
    config = load_config(path)

    if config.model.synthesis_mode != "restricted":
        return ContractResult(
            config=path.as_posix(),
            name=config.name,
            dataset=config.data.dataset,
            synthesis_mode=config.model.synthesis_mode,
            status="ablation",
            evidence={"reason": "non-restricted synthesis mode is an explicit ablation, not a valid default S_k"},
        )

    if static_only:
        evidence = {**_static_source_contract(), **_static_config_contract(config)}
    else:
        evidence = _dynamic_restricted_contract(config, path, spatial_size)
    return ContractResult(
        config=path.as_posix(),
        name=config.name,
        dataset=config.data.dataset,
        synthesis_mode=config.model.synthesis_mode,
        status="ok",
        evidence=evidence,
    )


def validate_configs(
    paths: Iterable[Path],
    spatial_size: int = 8,
    strict_ablation: bool = False,
    static_only: bool = False,
) -> dict[str, Any]:
    results = [
        validate_restricted_config(path, spatial_size=spatial_size, static_only=static_only)
        for path in paths
    ]
    ablations = [result for result in results if result.status == "ablation"]
    if strict_ablation and ablations:
        names = [result.config for result in ablations]
        raise AssertionError(f"Found non-restricted ablation configs under strict mode: {names}")
    ok = [result for result in results if result.status == "ok"]
    if not ok:
        raise AssertionError("No restricted configs were validated")
    return {
        "status": "ok",
        "checked_count": len(results),
        "restricted_ok_count": len(ok),
        "ablation_count": len(ablations),
        "validation_mode": "static" if static_only else "dynamic",
        "results": [result.__dict__ for result in results],
    }


def main() -> None:
    args = parse_args()
    paths = _config_paths(args.config, args.config_glob)
    result = validate_configs(
        paths,
        spatial_size=args.token_spatial_size,
        strict_ablation=args.strict_ablation,
        static_only=args.static_only,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
