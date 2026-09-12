"""Independent scalar replay and image-level readout; never recomputes gradients.

This validator does not import the measurement core or reuse its arithmetic.
Physical rows plus source bindings are evidence of a source-bound assay, not
independent reconstruction of full parameter-gradient vectors.
"""
from __future__ import annotations

import math
import statistics

from cofitok.generation.frozen_gradient_stage import METHODS, PROTOCOL, TIMESTEPS, invocation_row

GROUPS = ("class_embedding", "conditioning_projections", "output_heads", "shared_trunk")
TERMS = ("epsilon", "rollout", "ema_teacher", "other_auxiliary", "total")
COMPONENTS = ("epsilon", "prefix", "monotonic", "zero_token", "energy_budget", "residual_component",
              "component_decorrelation", "tail_floor", "sampled_prefix", "sampled_component", "group_residual",
              "epsilon_band_prefix", "epsilon_band_component", "denoise_path_prefix", "denoise_path_component",
              "denoise_path_energy", "low_snr_high_frequency", "rollout_consistency", "ema_teacher_consistency")


def number(x):
    if type(x) not in (int, float) or not math.isfinite(x):
        raise ValueError("nonfinite or nonnumeric measurement")
    return float(x)


def close(actual, expected):
    if expected is None:
        if actual is not None:
            raise ValueError("undefined statistic was replaced")
    elif not math.isclose(number(actual), expected, rel_tol=1e-8, abs_tol=1e-12):
        raise ValueError("scalar arithmetic replay differs")


def validate_statistics(groups, reference="epsilon"):
    if set(groups) != set(GROUPS) | {"all"}:
        raise ValueError("parameter group coverage differs")
    for row in groups.values():
        a, b, dot = (number(row[k]) for k in ("squared_norm", "reference_squared_norm", "dot"))
        if a < 0 or b < 0:
            raise ValueError("negative squared norm")
        norm, ref = math.sqrt(a), math.sqrt(b)
        if abs(dot) > norm * ref * (1 + 1e-8) + 1e-14:
            raise ValueError("gradient dot violates Cauchy-Schwarz")
        close(row["norm"], norm)
        close(row["reference_norm"], ref)
        close(row[f"norm_ratio_to_{reference}"], norm / ref if ref else None)
        close(row[f"cosine_to_{reference}"], min(1., max(-1., dot / (norm * ref))) if norm and ref else None)
        for k in ("parameter_tensors", "used_parameter_tensors", "nonzero_elements"):
            if type(row[k]) is not int or row[k] < 0:
                raise ValueError("invalid gradient counts")
        if row["used_parameter_tensors"] > row["parameter_tensors"]:
            raise ValueError("used count exceeds parameter count")
        state = "unused" if row["used_parameter_tensors"] == 0 else ("zero" if a == 0 else "nonzero")
        if row["state"] != state or (row["nonzero_elements"] > 0) != (a > 0):
            raise ValueError("zero/unused/nonzero gradient state differs")
        if state == "unused" and a != 0:
            raise ValueError("unused gradient has a norm")
    for key in ("squared_norm", "reference_squared_norm", "dot", "parameter_tensors",
                "used_parameter_tensors", "nonzero_elements"):
        close(groups["all"][key], sum(groups[g][key] for g in GROUPS))


def validate_row(row, *, method, index, timestep, catalog):
    sample = catalog["selection"]["samples"][index]
    expected = {**invocation_row(index, timestep), "method": method,
                "image": sample["image"], "requested_label": sample["label"],
                "checkpoint_step": 100000, "precision": "bf16",
                "measurement_unit": PROTOCOL["unit"],
                "parameter_update_performed": False, "authorization_granted": False}
    for key, value in expected.items():
        if row.get(key) != value or type(row.get(key)) is not type(value):
            raise ValueError(f"measurement row contract differs: {key}")
    if row["normalization"] != {
        "probe_batch_size": 1, "original_batch_size": 64, "original_teacher_selected_count": 4,
        "original_rollout_selected_count_if_all_timesteps_valid": 8,
        "probe_selected_count_when_active_and_valid": 1, "is_original_minibatch_gradient": False,
    }:
        raise ValueError("selected-example normalization differs")
    if set(row["terms"]) != set(TERMS):
        raise ValueError("term coverage differs")
    for term in TERMS:
        if number(row["terms"][term]["weighted_loss"]) < 0:
            raise ValueError("negative weighted loss")
        validate_statistics(row["terms"][term]["gradient"])
        for group in (*GROUPS, "all"):
            close(row["terms"][term]["gradient"][group]["reference_squared_norm"],
                  row["terms"]["epsilon"]["gradient"][group]["squared_norm"])
    # Scalar summation is FP32, not exact decimal arithmetic.
    total = row["terms"]["total"]["weighted_loss"]
    if not math.isclose(sum(row["terms"][t]["weighted_loss"] for t in TERMS[:-1]),
                        total, rel_tol=1e-5, abs_tol=1e-6):
        raise ValueError("loss decomposition sum differs")
    config = catalog["methods"][method]["config"]["loss"]
    if set(row["weighted_components"]) != set(COMPONENTS):
        raise ValueError("loss component coverage differs")
    sums = {name: 0.0 for name in TERMS[:-1]}
    for name, component in row["weighted_components"].items():
        close(component["effective_weight"], config[name + "_weight"])
        if number(component["weighted_loss"]) < 0:
            raise ValueError("negative component")
        target = {"epsilon": "epsilon", "rollout_consistency": "rollout",
                  "ema_teacher_consistency": "ema_teacher"}.get(name, "other_auxiliary")
        sums[target] += component["weighted_loss"]
    for term, scalar in sums.items():
        if not math.isclose(scalar, row["terms"][term]["weighted_loss"], rel_tol=1e-5, abs_tol=1e-6):
            raise ValueError("component-to-term assignment differs")
    for group in (*GROUPS, "all"):
        epsilon = row["terms"]["epsilon"]["gradient"][group]
        close(epsilon["dot"], epsilon["squared_norm"])
    if not math.isclose(sum(c["weighted_loss"] for c in row["weighted_components"].values()),
                        total, rel_tol=1e-5, abs_tol=1e-6):
        raise ValueError("component decomposition sum differs")
    validate_statistics(row["gradient_sum_rounding_residual"], "total")
    for group in (*GROUPS, "all"):
        close(row["gradient_sum_rounding_residual"][group]["reference_squared_norm"],
              row["terms"]["total"]["gradient"][group]["squared_norm"])
    routes = row["conditioning_routes"]
    if [r["phase"] for r in routes] != ["main", "ema_teacher", "rollout", "rollout"]:
        raise ValueError("actual conditioning route coverage differs")
    for route in routes:
        if route["effective_labels"] not in ([sample["label"]], [1000]):
            raise ValueError("unexpected effective conditioning label")
    if routes[1]["effective_labels"] != [sample["label"]]:
        raise ValueError("EMA teacher must use the true label")
    if [s["timestep"] for s in row["rollout_steps"]] != [timestep - 10, timestep - 20]:
        raise ValueError("rollout step coverage differs")
    for step in row["rollout_steps"]:
        for key in ("strictly_saturated_fraction", "clamp_boundary_fraction",
                    "weighted_loss_epsilon_gradient_nonzero_fraction"):
            if not 0 <= number(step[key]) <= 1:
                raise ValueError("invalid rollout fraction")
        if (number(step["raw_x0_rms"]) < 0
                or number(step["weighted_loss_epsilon_gradient_norm"]) < 0):
            raise ValueError("invalid rollout norm")
    if number(row["elapsed_seconds"]) <= 0 or type(row["peak_vram_bytes"]) is not int or row["peak_vram_bytes"] <= 0:
        raise ValueError("missing real measurement cost")


def _summary(values):
    defined = [v for v in values if v is not None]
    return {"defined_count": len(defined), "undefined_count": len(values) - len(defined),
            "mean": statistics.fmean(defined) if defined else None,
            "median": statistics.median(defined) if defined else None}


def readout(rows, catalog):
    if len(rows) != 256:
        raise ValueError("incomplete diagnostic rows")
    ordered = [(m, i, t) for m in METHODS for i in range(32) for t in TIMESTEPS]
    for row, (method, index, timestep) in zip(rows, ordered):
        validate_row(row, method=method, index=index, timestep=timestep, catalog=catalog)
    for offset in (0, 128):
        expected_counts = {g: rows[offset]["terms"]["epsilon"]["gradient"][g]["parameter_tensors"]
                           for g in (*GROUPS, "all")}
        if any(row["terms"][t]["gradient"][g]["parameter_tensors"] != expected_counts[g]
               for row in rows[offset:offset+128] for t in TERMS for g in expected_counts):
            raise ValueError("parameter group membership changed across measurements")
    by_method, nominations = {}, {}
    rule = PROTOCOL["nomination"]
    for method_index, method in enumerate(METHODS):
        images = []
        for index in range(32):
            measurements = rows[method_index * 128 + index * 4:method_index * 128 + index * 4 + 4]
            terms = {}
            for term in TERMS:
                terms[term] = {}
                for group in (*GROUPS, "all"):
                    stats = [r["terms"][term]["gradient"][group] for r in measurements]
                    opposed = sum(s["cosine_to_epsilon"] is not None
                                  and s["cosine_to_epsilon"] <= rule["cosine_at_most"]
                                  and s["norm_ratio_to_epsilon"] >= rule["norm_ratio_at_least"] for s in stats)
                    terms[term][group] = {
                        "cosine": _summary([s["cosine_to_epsilon"] for s in stats]),
                        "norm_ratio": _summary([s["norm_ratio_to_epsilon"] for s in stats]),
                        "opposed_substantial_timesteps": opposed,
                    }
            saturation_count = sum(all(s["strictly_saturated_fraction"] >= rule["saturation_fraction_at_least"]
                                       for s in r["rollout_steps"]) for r in measurements)
            images.append({"image_index": index, "terms": terms,
                           "main_null_label_timesteps": sum(r["conditioning_routes"][0]["effective_labels"] == [1000] for r in measurements),
                           "both_rollout_steps_saturated_timesteps": saturation_count})
        by_method[method] = images
        nominations[method] = {f"{term}.{group}": sum(
            image["terms"][term][group]["opposed_substantial_timesteps"] >= rule["qualifying_timesteps_per_image"]
            for image in images) for term in rule["terms"] for group in rule["groups"]}
        nominations[method]["rollout_saturation"] = sum(
            image["both_rollout_steps_saturated_timesteps"] >= rule["qualifying_timesteps_per_image"] for image in images)
    shared = [name for name in nominations[METHODS[0]] if all(
        nominations[method][name] >= rule["qualifying_images_per_method"] for method in METHODS)]
    paired = {f"{term}.{group}": _summary([
        (a["terms"][term][group]["cosine"]["mean"] - b["terms"][term][group]["cosine"]["mean"])
        if a["terms"][term][group]["cosine"]["mean"] is not None and b["terms"][term][group]["cosine"]["mean"] is not None
        else None for a, b in zip(by_method[METHODS[0]], by_method[METHODS[1]])])
        for term in TERMS for group in (*GROUPS, "all")}
    return {"status": "diagnostic_only", "independent_unit_count": 32,
            "all_image_summaries": by_method, "paired_image_cosine_delta_cofitok_minus_dense": paired,
            "nomination_counts": nominations, "shared_hypotheses_meeting_frozen_rule": shared,
            "gradient_vector_recomputation_performed": False, "scalar_arithmetic_replayed": True,
            "full_minibatch_gradient_claim": False, "causal_explanation_proven": False,
            "generation_quality_improvement_proven": False, "training_authorized": False,
            "teacher_cannot_solely_explain_pre_activation_10k_collapse": True}
