from __future__ import annotations

from typing import Any

from cofitok.generation_pair import generation_pair_contract


DEFAULT_HIGH_FREQUENCY_TIMESTEPS = (595, 394, 192, 91)


def _ratio(numerator: float, denominator: float, *, name: str) -> float:
    if denominator <= 0.0:
        raise ValueError(f"{name} denominator must be positive")
    return numerator / denominator


def _free_step_by_timestep(report: dict[str, Any], timestep: int) -> dict[str, Any]:
    for step in report["free_sampling_rollout"]["steps"]:
        if int(step["timestep"]) == timestep:
            return step
    raise ValueError(f"free rollout does not contain timestep {timestep}")


def _validate_sources(
    *,
    cofitok_training: dict[str, Any],
    dense_training: dict[str, Any],
    cofitok_checkpoint: dict[str, Any],
    dense_checkpoint: dict[str, Any],
    cofitok_rollout: dict[str, Any],
    dense_rollout: dict[str, Any],
) -> dict[str, Any]:
    for name, report in (
        ("cofitok checkpoint", cofitok_checkpoint),
        ("dense checkpoint", dense_checkpoint),
        ("cofitok rollout", cofitok_rollout),
        ("dense rollout", dense_rollout),
    ):
        if report.get("status") != "completed":
            raise ValueError(f"{name} report is not completed")
        if report.get("weights") != "model":
            raise ValueError(f"{name} report must evaluate raw model weights")

    for name, report in (
        ("cofitok training", cofitok_training),
        ("dense training", dense_training),
    ):
        if report.get("training_complete") is not True:
            raise ValueError(f"{name} report is not complete")
        if report.get("completed_steps") != report.get("target_steps"):
            raise ValueError(f"{name} report did not reach its target step")

    for name, checkpoint, rollout in (
        ("cofitok", cofitok_checkpoint, cofitok_rollout),
        ("dense", dense_checkpoint, dense_rollout),
    ):
        if checkpoint.get("checkpoint_sha256") != rollout.get("checkpoint_sha256"):
            raise ValueError(f"{name} checkpoint and rollout reports use different checkpoints")
        if checkpoint.get("checkpoint_step") != rollout.get("checkpoint_step"):
            raise ValueError(f"{name} checkpoint and rollout reports use different steps")

    cofitok_revision = cofitok_training.get("git", {}).get("revision")
    dense_revision = dense_training.get("git", {}).get("revision")
    if not cofitok_revision or cofitok_revision != dense_revision:
        raise ValueError("training reports do not share one non-empty Git revision")
    for name, training, checkpoint, rollout in (
        ("cofitok", cofitok_training, cofitok_checkpoint, cofitok_rollout),
        ("dense", dense_training, dense_checkpoint, dense_rollout),
    ):
        if checkpoint.get("checkpoint_step") != training.get("completed_steps"):
            raise ValueError(f"{name} evaluation does not use the final training step")
        if checkpoint.get("git", {}).get("revision") != cofitok_revision:
            raise ValueError(f"{name} checkpoint report uses a different Git revision")
        if rollout.get("git", {}).get("revision") != cofitok_revision:
            raise ValueError(f"{name} rollout report uses a different Git revision")
        training_config = training.get("config")
        if not isinstance(training_config, dict):
            raise ValueError(f"{name} training report does not embed its config")
        if checkpoint.get("config") != training_config:
            raise ValueError(f"{name} checkpoint report config does not match training")
        if rollout.get("config") != training_config:
            raise ValueError(f"{name} rollout report config does not match training")

    if cofitok_rollout.get("protocol") != dense_rollout.get("protocol"):
        raise ValueError("CoFiTok and dense rollout protocols do not match")
    rollout_protocol = cofitok_rollout.get("protocol")
    if not isinstance(rollout_protocol, dict):
        raise ValueError("rollout reports do not embed a protocol")
    if int(rollout_protocol.get("num_images", 0)) < 1:
        raise ValueError("rollout protocol num_images must be positive")

    cofitok_checkpoint_metrics = cofitok_checkpoint.get("metrics", {})
    dense_checkpoint_metrics = dense_checkpoint.get("metrics", {})
    for field in ("evaluated_images", "timestep"):
        if cofitok_checkpoint_metrics.get(field) != dense_checkpoint_metrics.get(field):
            raise ValueError(f"checkpoint evaluation field {field} does not match")

    pair_contract = generation_pair_contract(
        cofitok_training["config"],
        dense_training["config"],
    )
    if not pair_contract["valid"]:
        raise ValueError(
            "training reports fail the generation pair contract: "
            + "; ".join(pair_contract["issues"])
        )
    return pair_contract


def build_stability_qualification(
    *,
    cofitok_training: dict[str, Any],
    dense_training: dict[str, Any],
    cofitok_checkpoint: dict[str, Any],
    dense_checkpoint: dict[str, Any],
    cofitok_rollout: dict[str, Any],
    dense_rollout: dict[str, Any],
    high_frequency_timesteps: tuple[int, ...] = DEFAULT_HIGH_FREQUENCY_TIMESTEPS,
    max_tail_two_energy_ratio: float = 0.65,
    max_single_token_energy_ratio: float = 0.35,
    max_relative_regression: float = 0.05,
    max_high_frequency_ratio: float = 1.5,
    min_shuffle_mismatch_ratio: float = 2.0,
    max_zero_token_abs: float = 1e-8,
) -> dict[str, Any]:
    pair_contract = _validate_sources(
        cofitok_training=cofitok_training,
        dense_training=dense_training,
        cofitok_checkpoint=cofitok_checkpoint,
        dense_checkpoint=dense_checkpoint,
        cofitok_rollout=cofitok_rollout,
        dense_rollout=dense_rollout,
    )

    cofitok_metrics = cofitok_checkpoint["metrics"]
    dense_metrics = dense_checkpoint["metrics"]
    component_ratios = [float(value) for value in cofitok_metrics["component_energy_ratio"]]
    if len(component_ratios) < 2:
        raise ValueError("CoFiTok checkpoint must contain at least two component ratios")

    tail_two = sum(component_ratios[-2:])
    max_token = max(component_ratios)
    ordered_rank = int(cofitok_metrics["ordered_rank_by_path_auc"])
    cofitok_endpoint = float(cofitok_metrics["orders"]["ordered"]["endpoint_clean_mse"])
    dense_endpoint = float(dense_metrics["orders"]["ordered"]["endpoint_clean_mse"])
    endpoint_ratio = _ratio(cofitok_endpoint, dense_endpoint, name="endpoint")

    cofitok_validation = float(
        cofitok_training["final_metrics"]["validation_epsilon_mse"]
    )
    dense_validation = float(dense_training["final_metrics"]["validation_epsilon_mse"])
    validation_ratio = _ratio(cofitok_validation, dense_validation, name="validation")

    high_frequency_rows = []
    for timestep in high_frequency_timesteps:
        cofitok_step = _free_step_by_timestep(cofitok_rollout, timestep)
        dense_step = _free_step_by_timestep(dense_rollout, timestep)
        cofitok_value = float(cofitok_step["predicted_x0_high_frequency_ratio"])
        dense_value = float(dense_step["predicted_x0_high_frequency_ratio"])
        high_frequency_rows.append(
            {
                "timestep": timestep,
                "cofitok": cofitok_value,
                "dense": dense_value,
                "ratio": _ratio(
                    cofitok_value,
                    dense_value,
                    name=f"predicted-x0 high-frequency timestep {timestep}",
                ),
            }
        )
    peak_high_frequency_ratio = max(row["ratio"] for row in high_frequency_rows)

    cofitok_reconstruction = cofitok_rollout["reconstruction_rollout"]["summary"]
    dense_reconstruction = dense_rollout["reconstruction_rollout"]["summary"]
    cofitok_final_reconstruction = float(
        cofitok_reconstruction["final_clipped_x0_mse"]
    )
    dense_final_reconstruction = float(dense_reconstruction["final_clipped_x0_mse"])
    reconstruction_ratio = _ratio(
        cofitok_final_reconstruction,
        dense_final_reconstruction,
        name="reconstruction",
    )
    cofitok_amplification = float(
        cofitok_reconstruction["final_to_best_x0_mse_amplification"]
    )
    dense_amplification = float(
        dense_reconstruction["final_to_best_x0_mse_amplification"]
    )

    zero_token = float(cofitok_metrics["zero_token_max_abs"])
    shuffle_ratio = float(cofitok_metrics["shuffled_to_ordered_endpoint_ratio"])
    gates = {
        "tail_two_energy": {
            "value": tail_two,
            "maximum": max_tail_two_energy_ratio,
            "passed": tail_two <= max_tail_two_energy_ratio,
        },
        "single_token_energy": {
            "value": max_token,
            "maximum": max_single_token_energy_ratio,
            "passed": max_token <= max_single_token_energy_ratio,
        },
        "ordered_rank": {
            "value": ordered_rank,
            "required": 1,
            "passed": ordered_rank == 1,
        },
        "endpoint_regression": {
            "ratio": endpoint_ratio,
            "maximum_ratio": 1.0 + max_relative_regression,
            "passed": endpoint_ratio <= 1.0 + max_relative_regression,
        },
        "validation_regression": {
            "ratio": validation_ratio,
            "maximum_ratio": 1.0 + max_relative_regression,
            "passed": validation_ratio <= 1.0 + max_relative_regression,
        },
        "predicted_x0_high_frequency": {
            "peak_ratio": peak_high_frequency_ratio,
            "maximum_ratio": max_high_frequency_ratio,
            "passed": peak_high_frequency_ratio <= max_high_frequency_ratio,
        },
        "reconstruction_regression": {
            "ratio": reconstruction_ratio,
            "maximum_ratio": 1.0 + max_relative_regression,
            "passed": reconstruction_ratio <= 1.0 + max_relative_regression,
        },
        "zero_token": {
            "value": zero_token,
            "maximum": max_zero_token_abs,
            "passed": zero_token <= max_zero_token_abs,
        },
        "shuffle_mismatch": {
            "value": shuffle_ratio,
            "minimum": min_shuffle_mismatch_ratio,
            "passed": shuffle_ratio >= min_shuffle_mismatch_ratio,
        },
    }
    passed = all(bool(gate["passed"]) for gate in gates.values())
    return {
        "schema_version": 2,
        "status": "pass" if passed else "fail",
        "protocol": {
            "weights": "model",
            "checkpoint_step": int(cofitok_checkpoint["checkpoint_step"]),
            "checkpoint_evaluated_images": int(
                cofitok_checkpoint["metrics"]["evaluated_images"]
            ),
            "checkpoint_timestep": int(cofitok_checkpoint["metrics"]["timestep"]),
            "high_frequency_timesteps": list(high_frequency_timesteps),
            "rollout": cofitok_rollout["protocol"],
        },
        "identity": {
            "git_revision": cofitok_training["git"]["revision"],
            "cofitok_checkpoint_sha256": cofitok_checkpoint["checkpoint_sha256"],
            "dense_checkpoint_sha256": dense_checkpoint["checkpoint_sha256"],
        },
        "metrics": {
            "component_energy_ratio": component_ratios,
            "tail_two_energy_ratio": tail_two,
            "max_single_token_energy_ratio": max_token,
            "ordered_rank_by_path_auc": ordered_rank,
            "cofitok_endpoint_clean_mse": cofitok_endpoint,
            "dense_endpoint_clean_mse": dense_endpoint,
            "endpoint_ratio": endpoint_ratio,
            "cofitok_validation_epsilon_mse": cofitok_validation,
            "dense_validation_epsilon_mse": dense_validation,
            "validation_ratio": validation_ratio,
            "predicted_x0_high_frequency": high_frequency_rows,
            "peak_predicted_x0_high_frequency_ratio": peak_high_frequency_ratio,
            "cofitok_final_reconstruction_x0_mse": cofitok_final_reconstruction,
            "dense_final_reconstruction_x0_mse": dense_final_reconstruction,
            "reconstruction_ratio": reconstruction_ratio,
            "cofitok_reconstruction_amplification": cofitok_amplification,
            "dense_reconstruction_amplification": dense_amplification,
            "zero_token_max_abs": zero_token,
            "shuffle_to_ordered_endpoint_ratio": shuffle_ratio,
        },
        "gates": gates,
        "pair_contract": pair_contract,
    }
