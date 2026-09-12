"""Non-updating gradient measurements for separately authorized frozen probes.

This module has no checkpoint loader, optimizer, sampler, output writer or
authorization builder. A source-bound runner must guard its use on real data.
The unit is a single *selected* image, not an original training minibatch.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping

import torch

from cofitok.configs import ExperimentConfig
from cofitok.diffusion import DiffusionSchedule
from cofitok.training.losses import compute_losses
from cofitok.training.rollout import (
    consistency_weight_scale,
    ema_teacher_consistency_loss,
    rollout_consistency_loss,
)
from cofitok.training.runtime import autocast_context


GROUPS = ("class_embedding", "conditioning_projections", "output_heads", "shared_trunk")
TERMS = ("epsilon", "rollout", "ema_teacher", "other_auxiliary")


def parameter_group(name: str) -> str:
    if name.startswith("predictor.class_embed."):
        return "class_embedding"
    if name.startswith("predictor.time_embed.") or ".embedding." in name:
        return "conditioning_projections"
    if name.startswith(("predictor.token_heads.", "predictor.feedback.")):
        return "output_heads"
    if name.startswith("predictor."):
        return "shared_trunk"
    raise ValueError(f"unexpected trainable parameter outside predictor: {name}")


def gradient_statistics(names, gradients, reference, *, reference_label="epsilon") -> dict:
    if not (len(names) == len(gradients) == len(reference)):
        raise ValueError("gradient structures differ")
    sums = {group: dict(squared_norm=0.0, reference_squared_norm=0.0,
                       dot=0.0, parameter_tensors=0, used_parameter_tensors=0,
                       nonzero_elements=0) for group in (*GROUPS, "all")}
    for name, grad, ref in zip(names, gradients, reference):
        if grad is not None and ref is not None and grad.shape != ref.shape:
            raise ValueError("gradient shapes differ")
        for value in (grad, ref):
            if value is not None and not torch.isfinite(value).all():
                raise ValueError("nonfinite gradient")
        g2 = float(grad.double().square().sum()) if grad is not None else 0.0
        r2 = float(ref.double().square().sum()) if ref is not None else 0.0
        dot = float((grad.double() * ref.double()).sum()) if grad is not None and ref is not None else 0.0
        nonzero = int(torch.count_nonzero(grad)) if grad is not None else 0
        for group in (parameter_group(name), "all"):
            row = sums[group]
            row["squared_norm"] += g2
            row["reference_squared_norm"] += r2
            row["dot"] += dot
            row["parameter_tensors"] += 1
            row["used_parameter_tensors"] += grad is not None
            row["nonzero_elements"] += nonzero
    for row in sums.values():
        norm, ref_norm = math.sqrt(row["squared_norm"]), math.sqrt(row["reference_squared_norm"])
        row["norm"] = norm
        row["reference_norm"] = ref_norm
        row[f"norm_ratio_to_{reference_label}"] = norm / ref_norm if ref_norm else None
        row[f"cosine_to_{reference_label}"] = min(1.0, max(-1.0, row["dot"] / (norm * ref_norm))) if norm and ref_norm else None
        row["state"] = "unused" if not row["used_parameter_tensors"] else ("zero" if norm == 0 else "nonzero")
    return sums


def _gradients(loss, parameters):
    if not loss.requires_grad:
        return (None,) * len(parameters)
    return torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)


def _tensor_versions(named_tensors):
    return [(name, id(tensor), tensor._version) for name, tensor in named_tensors]


def _weighted_terms(losses, config, teacher_scale, rollout_scale):
    terms = {name: losses.total.new_zeros(()) for name in TERMS}
    components = {}
    ignored = {"total", "sampled_budget", "rollout_consistency_scale", "ema_teacher_consistency_scale"}
    for field in dataclasses.fields(losses):
        name = field.name
        if name in ignored:
            continue
        weight = float(getattr(config, name + "_weight"))
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"invalid loss weight: {name}")
        group = "other_auxiliary"
        if name == "epsilon":
            group = "epsilon"
        elif name == "ema_teacher_consistency":
            group, weight = "ema_teacher", weight * teacher_scale
        elif name == "rollout_consistency":
            group, weight = "rollout", weight * rollout_scale
        weighted = getattr(losses, name) * weight
        terms[group] = terms[group] + weighted
        components[name] = {"effective_weight": weight, "weighted_loss": float(weighted.detach())}
    if not torch.allclose(sum(terms.values()).detach(), losses.total.detach(), atol=1e-6, rtol=1e-5):
        raise ValueError("weighted decomposition differs from production loss")
    return terms, components


def measure_selected_example(
    model: torch.nn.Module,
    *,
    ema_state: Mapping[str, torch.Tensor],
    config: ExperimentConfig,
    clean: torch.Tensor,
    label: int,
    timestep: int,
    checkpoint_step: int,
    noise_seed: int,
    dropout_seed: int,
) -> dict:
    """Use production forwards/losses, autograd.grad only, with reversible hooks.

    Checkpoint/data provenance, exact source identity, GPU admission and execution
    authorization are caller responsibilities. Calling this API grants none.
    """
    if torch.is_inference_mode_enabled():
        raise ValueError("gradient probe cannot run in inference mode")
    if clean.ndim != 4 or clean.shape[0] != 1:
        raise ValueError("probe requires exactly one selected image")
    if clean.requires_grad or clean.dtype != torch.float32:
        raise ValueError("clean image must be detached float32")
    if not torch.isfinite(clean).all() or clean.min() < -1 or clean.max() > 1:
        raise ValueError("clean image must be finite in [-1,1]")
    if type(label) is not int or not 0 <= label < config.model.num_classes:
        raise ValueError("invalid class label")
    if type(timestep) is not int or not 0 <= timestep < config.diffusion.num_train_timesteps:
        raise ValueError("invalid timestep")
    if type(checkpoint_step) is not int or checkpoint_step < 0:
        raise ValueError("invalid checkpoint step")
    if any(type(seed) is not int or not 0 <= seed < 2**63 for seed in (noise_seed, dropout_seed)):
        raise ValueError("invalid probe seed")
    if config.model.synthesis_mode not in {"fixed_basis", "dense_identity"}:
        raise ValueError("frozen probe requires restricted fixed or dense synthesis")
    if model.config != config.model:
        raise ValueError("model config differs from frozen config")
    if ema_state.keys() != model.state_dict().keys():
        raise ValueError("EMA structure differs")
    if any(value.requires_grad for value in ema_state.values()):
        raise ValueError("EMA teacher must be detached")
    named = list(model.named_parameters())
    if not named or any(not parameter.requires_grad for _, parameter in named):
        raise ValueError("all frozen predictor parameters must expose gradients")
    names, parameters = zip(*named)
    for name in names:
        parameter_group(name)
    device = next(model.parameters()).device
    if clean.device != device:
        raise ValueError("clean image and model devices differ")
    if config.runtime.precision not in {"fp32", "bf16"}:
        raise ValueError("probe supports only fp32 or bf16")
    if config.runtime.precision == "bf16" and device.type != "cuda":
        raise ValueError("bf16 probe cannot silently fall back to CPU fp32")
    if any(value.device != device for value in ema_state.values()):
        raise ValueError("EMA and model devices differ")
    versions = _tensor_versions([*model.named_parameters(), *model.named_buffers()])
    ema_versions = _tensor_versions(ema_state.items())
    old_grads = [parameter.grad for parameter in parameters]
    modes = [(module, module.training) for module in model.modules()]
    phase = ["main"]
    routes, rollout_outputs, handles = [], [], []
    schedule = DiffusionSchedule(config.diffusion, device)
    timesteps = torch.tensor([timestep], dtype=torch.long, device=device)
    labels = torch.tensor([label], dtype=torch.long, device=device)
    noise = torch.randn(clean.shape, generator=torch.Generator(device=device).manual_seed(noise_seed),
                        device=device, dtype=clean.dtype)
    noisy = schedule.add_noise(clean, noise, timesteps)

    def label_hook(module, args):
        del module
        routes.append({"phase": phase[0], "effective_labels": args[0].detach().cpu().tolist()})

    def output_hook(module, args, kwargs, output):
        del module, kwargs
        if phase[0] != "rollout":
            return
        with torch.no_grad():
            raw = schedule.predict_x0_from_epsilon(args[0].float(), output.epsilon.float(), args[1])
        rollout_outputs.append((output.epsilon, {
            "timestep": int(args[1][0]),
            "strictly_saturated_fraction": float((raw.abs() > 1).float().mean()),
            "clamp_boundary_fraction": float((raw.abs() == 1).float().mean()),
            "raw_x0_rms": float(raw.double().square().mean().sqrt()),
        }))

    device_index = device.index if device.index is not None else (torch.cuda.current_device() if device.type == "cuda" else None)
    teacher_scale = consistency_weight_scale(checkpoint_step,
        start_step=config.loss.ema_teacher_consistency_start_step,
        warmup_steps=config.loss.ema_teacher_consistency_warmup_steps)
    rollout_scale = consistency_weight_scale(checkpoint_step,
        start_step=config.loss.rollout_consistency_start_step,
        warmup_steps=config.loss.rollout_consistency_warmup_steps)
    result = None
    try:
        handles.append(model.predictor.class_embed.register_forward_pre_hook(label_hook))
        handles.append(model.register_forward_hook(output_hook, with_kwargs=True))
        model.train()
        with torch.random.fork_rng(devices=[device_index] if device.type == "cuda" else []), torch.enable_grad():
            torch.random.default_generator.manual_seed(dropout_seed)
            if device.type == "cuda":
                torch.cuda.default_generators[device_index].manual_seed(dropout_seed)
            with autocast_context(device, config.runtime.precision):
                output = model(noisy, timesteps, class_labels=labels)
                teacher = rollout = None
                if config.loss.ema_teacher_consistency_weight > 0 and teacher_scale > 0:
                    phase[0] = "ema_teacher"
                    teacher = ema_teacher_consistency_loss(model, ema_state=ema_state,
                        student_epsilon=output.epsilon, noisy_images=noisy, timesteps=timesteps,
                        class_labels=labels, batch_fraction=config.loss.ema_teacher_consistency_batch_fraction)
                if config.loss.rollout_consistency_weight > 0 and rollout_scale > 0:
                    phase[0] = "rollout"
                    rollout = rollout_consistency_loss(model, first_epsilon=output.epsilon,
                        schedule=schedule, noisy_images=noisy, clean_images=clean, timesteps=timesteps,
                        class_labels=labels, timestep_delta=config.loss.rollout_consistency_timestep_delta,
                        unroll_steps=config.loss.rollout_consistency_unroll_steps,
                        batch_fraction=config.loss.rollout_consistency_batch_fraction,
                        clip_x0=config.loss.rollout_consistency_clip_x0, mode=config.loss.rollout_consistency_mode)
                phase[0] = "loss"
                losses = compute_losses(config.loss, output, schedule, noisy, clean, noise, timesteps,
                    zero_components=model.synthesis.zero_components_like(output.tokens) if config.loss.zero_token_weight else None,
                    rollout_consistency=rollout, rollout_consistency_scale=rollout_scale,
                    ema_teacher_consistency=teacher, ema_teacher_consistency_scale=teacher_scale)
                terms, components = _weighted_terms(losses, config.loss, teacher_scale, rollout_scale)
            if not torch.isfinite(losses.total):
                raise ValueError("nonfinite probe loss")
            phase[0] = "backward_recomputation"
            reference = _gradients(terms["epsilon"], parameters)
            accumulated = [torch.zeros_like(p) for p in parameters]
            measured = {}
            for name, term in terms.items():
                gradients = reference if name == "epsilon" else _gradients(term, parameters)
                measured[name] = {"weighted_loss": float(term.detach()),
                                  "gradient": gradient_statistics(names, gradients, reference)}
                for total, grad in zip(accumulated, gradients):
                    if grad is not None:
                        total.add_(grad.detach())
            combined = _gradients(losses.total, parameters)
            measured["total"] = {"weighted_loss": float(losses.total.detach()),
                                 "gradient": gradient_statistics(names, combined, reference)}
            residuals = [summed - (grad if grad is not None else 0) for summed, grad in zip(accumulated, combined)]
            closure = gradient_statistics(names, residuals, combined, reference_label="total")
            rollout_rows = []
            for epsilon, row in rollout_outputs:
                derivative = _gradients(terms["rollout"], (epsilon,))[0]
                if derivative is not None and not torch.isfinite(derivative).all():
                    raise ValueError("nonfinite rollout epsilon derivative")
                row["weighted_loss_epsilon_gradient_norm"] = float(derivative.double().norm()) if derivative is not None else None
                row["weighted_loss_epsilon_gradient_nonzero_fraction"] = float((derivative != 0).float().mean()) if derivative is not None else None
                rollout_rows.append(row)
            batch = config.data.batch_size
            result = {
                "measurement_unit": "single_selected_image_not_original_minibatch",
                "requested_label": label, "timestep": timestep, "checkpoint_step": checkpoint_step,
                "noise_seed": noise_seed, "dropout_seed": dropout_seed,
                "precision": config.runtime.precision, "terms": measured,
                "weighted_components": components, "conditioning_routes": routes,
                "rollout_steps": rollout_rows, "gradient_sum_rounding_residual": closure,
                "normalization": {
                    "probe_batch_size": 1, "original_batch_size": batch,
                    "original_teacher_selected_count": max(1, math.ceil(batch * config.loss.ema_teacher_consistency_batch_fraction)),
                    "original_rollout_selected_count_if_all_timesteps_valid": max(1, math.ceil(batch * config.loss.rollout_consistency_batch_fraction)),
                    "probe_selected_count_when_active_and_valid": 1,
                    "is_original_minibatch_gradient": False,
                },
                "parameter_update_performed": False,
                "authorization_granted": False,
            }
    finally:
        for handle in handles:
            handle.remove()
        for module, mode in modes:
            module.training = mode
        if _tensor_versions([*model.named_parameters(), *model.named_buffers()]) != versions:
            raise RuntimeError("model tensors changed during frozen probe")
        if _tensor_versions(ema_state.items()) != ema_versions:
            raise RuntimeError("EMA tensors changed during frozen probe")
        if any(parameter.grad is not old for parameter, old in zip(parameters, old_grads)):
            raise RuntimeError("parameter .grad buffers changed during probe")
    return result
