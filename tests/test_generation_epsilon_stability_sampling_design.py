from __future__ import annotations

import pytest

from cofitok.generation import (
    EPSILON_STABILITY_SAMPLING_DESIGN_SCHEMA,
    build_epsilon_stability_sampling_design,
    materialize_epsilon_stability_case_protocol,
    sampling_protocol_contract,
)


def _cases() -> dict[str, dict]:
    design = build_epsilon_stability_sampling_design()
    return {case["case_id"]: case for case in design["cases"]}


def test_sampling_design_is_matched_complete_and_non_authorizing() -> None:
    design = build_epsilon_stability_sampling_design()

    assert design["schema"] == EPSILON_STABILITY_SAMPLING_DESIGN_SCHEMA
    assert design["status"] == "design_only"
    assert design["execution_ready"] is False
    assert design["common_sampling_contract"]["total_case_count"] == 8
    assert (
        design["common_sampling_contract"]["total_generated_image_count"]
        == 16_000
    )
    assert design["matched_methods"]["cofitok"]["prefix_budget"] == 8
    assert design["matched_methods"]["dense_identity"]["prefix_budget"] == 1
    assert all(
        value is False for value in design["authorization_boundary"].values()
    )
    assert "terminal_route_receipt_not_bound" in design["blocking_reasons"]
    assert "separate_execution_authorization_identity" in design[
        "required_execution_bindings"
    ]


def test_sampling_design_has_explicit_nested_causal_comparisons() -> None:
    cases = _cases()

    assert cases["start975_unit_hard_clip"]["parent_case_id"] == (
        "legacy_terminal_hard_clip"
    )
    assert cases["start975_unit_hard_clip"]["incremental_control"] == (
        "nonterminal_start"
    )
    assert cases["start975_sigma_hard_clip"]["parent_case_id"] == (
        "start975_unit_hard_clip"
    )
    assert cases["terminal_dynamic_threshold_recompute"]["parent_case_id"] == (
        "terminal_dynamic_threshold"
    )
    assert cases["terminal_hard_clip_recompute"]["parent_case_id"] == (
        "legacy_terminal_hard_clip"
    )
    assert cases[
        "start975_sigma_dynamic_threshold_recompute"
    ]["attribution_role"] == "combined_recovery_upper_bound"


def test_sampling_design_separates_sampling_repair_from_min_snr_training() -> None:
    design = build_epsilon_stability_sampling_design()
    boundary = design["causal_boundary"]

    assert boundary["sampling_only"] is True
    assert boundary["checkpoint_weights_immutable"] is True
    assert boundary["min_snr_training_tested"] is False
    assert boundary["fresh_training_allowed"] is False
    assert boundary["independent_matched_10000_confirmation_required"] is True
    assert all(
        "min_snr" not in str(case).lower() for case in design["cases"]
    )


@pytest.mark.parametrize(
    ("case_id", "formal_issue"),
    [
        ("start975_unit_hard_clip", "formal_start_timestep"),
        ("start975_sigma_hard_clip", "formal_scale_initial_noise_by_sigma"),
        ("terminal_dynamic_threshold", "formal_x0_constraint"),
        (
            "terminal_dynamic_threshold_recompute",
            "formal_recompute_epsilon_after_x0_constraint",
        ),
        (
            "terminal_hard_clip_recompute",
            "formal_recompute_epsilon_after_x0_constraint",
        ),
        (
            "start975_sigma_dynamic_threshold_recompute",
            "formal_recompute_epsilon_after_x0_constraint",
        ),
    ],
)
def test_repair_cases_are_valid_diagnostics_but_rejected_by_formal_gate(
    case_id: str,
    formal_issue: str,
) -> None:
    design = build_epsilon_stability_sampling_design()
    sampling = materialize_epsilon_stability_case_protocol(
        design,
        case_id,
        "cofitok",
        seed=4049,
        start_index=0,
    )

    assert sampling_protocol_contract(sampling)["valid"] is True
    formal = sampling_protocol_contract(
        sampling,
        stage="scaling",
        expected_num_train_timesteps=1000,
    )
    assert formal["valid"] is False
    assert "formal_num_samples" in formal["issues"]
    assert formal_issue in formal["issues"]


def test_materialization_requires_explicit_fresh_stream_values() -> None:
    design = build_epsilon_stability_sampling_design()

    with pytest.raises(ValueError, match="seed"):
        materialize_epsilon_stability_case_protocol(
            design,
            "legacy_terminal_hard_clip",
            "cofitok",
            seed=-1,
            start_index=0,
        )
    with pytest.raises(ValueError, match="start_index"):
        materialize_epsilon_stability_case_protocol(
            design,
            "legacy_terminal_hard_clip",
            "cofitok",
            seed=4049,
            start_index=-1,
        )
    with pytest.raises(ValueError, match="evaluator compatibility"):
        materialize_epsilon_stability_case_protocol(
            design,
            "legacy_terminal_hard_clip",
            "cofitok",
            seed=4049,
            start_index=20_000,
        )
    with pytest.raises(ValueError, match="unknown"):
        materialize_epsilon_stability_case_protocol(
            design,
            "missing_case",
            "cofitok",
            seed=4049,
            start_index=0,
        )
    with pytest.raises(ValueError, match="unknown.*method"):
        materialize_epsilon_stability_case_protocol(
            design,
            "legacy_terminal_hard_clip",
            "missing_method",
            seed=4049,
            start_index=0,
        )


def test_materialized_method_protocols_differ_only_in_prefix_budget() -> None:
    design = build_epsilon_stability_sampling_design()
    cofitok = materialize_epsilon_stability_case_protocol(
        design,
        "start975_sigma_dynamic_threshold_recompute",
        "cofitok",
        seed=4049,
        start_index=0,
    )
    dense = materialize_epsilon_stability_case_protocol(
        design,
        "start975_sigma_dynamic_threshold_recompute",
        "dense_identity",
        seed=4049,
        start_index=0,
    )

    assert cofitok["prefix_budgets"] == [8]
    assert dense["prefix_budgets"] == [1]
    assert {**cofitok, "prefix_budgets": None} == {
        **dense,
        "prefix_budgets": None,
    }
