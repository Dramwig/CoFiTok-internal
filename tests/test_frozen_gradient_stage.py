"""Small CPU tests only; fabricated approvals exist only in in-memory fixtures."""
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from cofitok.generation import frozen_gradient_stage as stage
from cofitok.generation.frozen_gradient_readout import COMPONENTS, readout
from scripts import frozen_loss_gradient_diagnostic as runner


@pytest.fixture
def approval_inputs():
    ident = {"path": "/test/preparation.json", "bytes": 20, "sha256": "a" * 64}
    prepared = {"checkouts": {"evaluator": {"root": "/test/evaluator", "revision": "a" * 40}}}
    approval = {"schema": "user_created_frozen_loss_gradient_approval_v1", "preparation": ident,
                "evaluator": prepared["checkouts"]["evaluator"], "protocol": copy.deepcopy(stage.PROTOCOL),
                "output_root": str(stage.OUTPUT), "permissions": copy.deepcopy(stage.DENIED),
                "user_authorized_frozen_gradient_diagnostic": True,
                "user_instruction": "Synthetic fixture only; not real user consent",
                "approved_at_utc": "2000-01-01T00:00:00Z"}
    return approval, ident, prepared


def test_approval_does_not_grant_training_and_binds_exact_preparation(approval_inputs):
    approval, ident, prepared = approval_inputs
    result = stage.authorization_payload(prepared, ident, approval, {"sha256": "b" * 64})
    assert result["frozen_gradient_execution_allowed"] is True
    assert not any(result["permissions"].values())
    ident["sha256"] = "c" * 64
    with pytest.raises(ValueError, match="exact preparation"):
        stage.validate_approval(approval, dict(ident, sha256="d" * 64), prepared)


@pytest.mark.parametrize("mutation", ["prior_stage", "bool_as_int", "training", "budget", "extra", "no_instruction"])
def test_rejects_prior_broad_changed_or_expansive_approval(approval_inputs, mutation):
    approval, ident, prepared = approval_inputs
    if mutation == "prior_stage":
        approval["schema"] = "cofitok_generation_class_support_contingency_stage_approval_v1"
    elif mutation == "bool_as_int":
        approval["permissions"]["training_allowed"] = 0
    elif mutation == "training":
        approval["permissions"]["training_allowed"] = True
    elif mutation == "budget":
        approval["protocol"]["measurement_rows"] = 512
    elif mutation == "extra":
        approval["full_300k_allowed"] = True
    else:
        approval["user_instruction"] = " "
    with pytest.raises(ValueError):
        stage.validate_approval(approval, ident, prepared)


def test_missing_admission_prevents_runtime_load_and_output(monkeypatch, tmp_path):
    calls = []
    def refused(*args, **kwargs):
        raise ValueError("no separate approval")
    monkeypatch.setattr(runner, "_args_admission", refused)
    monkeypatch.setattr(runner, "runtime_check", lambda *a: calls.append("runtime"))
    monkeypatch.setattr(runner, "load_frozen_pair_member", lambda *a: calls.append("load"))
    monkeypatch.setattr(runner, "OUTPUT", tmp_path / "uncreated")
    with pytest.raises(ValueError, match="no separate"):
        runner.run_assay(SimpleNamespace())
    assert calls == []
    assert not (tmp_path / "uncreated").exists()


def test_gpu_ownership_rejects_unrelated_and_requires_physical_self(monkeypatch):
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **kw: "999999, GPU-test, 100\n")
    with pytest.raises(ValueError, match="ownership conflict"):
        runner.gpu_ownership(allow_self=True)
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **kw: "")
    with pytest.raises(ValueError, match="not physically observed"):
        runner.gpu_ownership(allow_self=True, require_self=True)
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **kw: f"{os.getpid()}, GPU-test, 100\n")
    assert runner.gpu_ownership(allow_self=True, require_self=True)[0]["pid"] == os.getpid()
    with pytest.raises(ValueError, match="ownership conflict"):
        runner.gpu_ownership()


def test_exclusive_outputs_preserve_existing_evidence(monkeypatch, tmp_path):
    monkeypatch.setattr(stage, "OUTPUT", tmp_path)
    path = tmp_path / "result.json"
    path.write_text("locked")
    with pytest.raises(FileExistsError):
        stage.exclusive_json(path, {"replacement": True})
    assert path.read_text() == "locked"
    with pytest.raises(ValueError, match="dedicated"):
        stage.exclusive_json(tmp_path / "unapproved" / "x.json", {})


def test_seed_protocol_pairs_methods_but_not_images_or_timesteps():
    rows = [stage.invocation_row(i, t) for i in range(32) for t in stage.TIMESTEPS]
    assert len({r["noise_seed"] for r in rows}) == 128
    assert len({r["dropout_seed"] for r in rows}) == 128
    with pytest.raises(ValueError):
        stage.invocation_row(True, 100)


def test_source_mismatch_prevents_checkpoint_deserialization(monkeypatch):
    import torch
    called = []
    def mismatch(identity):
        raise ValueError("physical source differs")
    monkeypatch.setattr(runner, "rehash", mismatch)
    monkeypatch.setattr(torch, "load", lambda *a, **kw: called.append("load"))
    with pytest.raises(ValueError, match="physical source"):
        runner.load_frozen_pair_member({"checkpoint": {}}, {})
    assert not called


def stats(norm=1., cosine=1.):
    # Synthetic sufficient statistics, not a real checkpoint measurement.
    def one(n):
        return {"squared_norm": n*n, "reference_squared_norm": 1., "dot": n*cosine,
                "norm": n, "reference_norm": 1., "norm_ratio_to_epsilon": n,
                "cosine_to_epsilon": cosine if n else None, "parameter_tensors": 1,
                "used_parameter_tensors": 1, "nonzero_elements": 1 if n else 0,
                "state": "nonzero" if n else "zero"}
    result = {g: one(norm) for g in ("class_embedding", "conditioning_projections", "output_heads", "shared_trunk")}
    result["all"] = {"squared_norm": 4*norm*norm, "reference_squared_norm": 4., "dot": 4*norm*cosine,
                     "norm": 2*norm, "reference_norm": 2., "norm_ratio_to_epsilon": norm,
                     "cosine_to_epsilon": cosine if norm else None, "parameter_tensors": 4,
                     "used_parameter_tensors": 4, "nonzero_elements": 4 if norm else 0,
                     "state": "nonzero" if norm else "zero"}
    return result


@pytest.fixture
def synthetic_rows():
    catalog = {"selection": {"samples": [{"image": {"path": f"/test/{i}.jpg", "bytes": 1, "sha256": f"{i:064x}"},
                                         "label": i} for i in range(32)]},
               "methods": {m: {"config": {"loss": {name+"_weight": 1. for name in COMPONENTS}}}
                           for m in stage.METHODS}}
    rows = []
    for method in stage.METHODS:
        for index in range(32):
            for timestep in stage.TIMESTEPS:
                closure = stats(0.)
                for value in closure.values():
                    value["norm_ratio_to_total"] = value.pop("norm_ratio_to_epsilon")
                    value["cosine_to_total"] = value.pop("cosine_to_epsilon")
                rows.append({**stage.invocation_row(index, timestep), "method": method,
                    "image": catalog["selection"]["samples"][index]["image"], "requested_label": index,
                    "checkpoint_step": 100000, "precision": "bf16", "measurement_unit": stage.PROTOCOL["unit"],
                    "parameter_update_performed": False, "authorization_granted": False,
                    "normalization": {"probe_batch_size": 1, "original_batch_size": 64,
                        "original_teacher_selected_count": 4, "original_rollout_selected_count_if_all_timesteps_valid": 8,
                        "probe_selected_count_when_active_and_valid": 1, "is_original_minibatch_gradient": False},
                    "terms": {name: {"gradient": stats(), "weighted_loss": 1. if name == "total" else .25}
                              for name in ("epsilon", "rollout", "ema_teacher", "other_auxiliary", "total")},
                    "weighted_components": {name: {"effective_weight": 1., "weighted_loss": .25
                        if name in ("epsilon", "rollout_consistency", "ema_teacher_consistency", "prefix") else 0.}
                        for name in COMPONENTS},
                    "gradient_sum_rounding_residual": closure,
                    "conditioning_routes": [{"phase": phase, "effective_labels": [index]}
                                            for phase in ("main", "ema_teacher", "rollout", "rollout")],
                    "rollout_steps": [{"timestep": timestep-d, "strictly_saturated_fraction": .25,
                        "clamp_boundary_fraction": 0., "raw_x0_rms": 1.,
                        "weighted_loss_epsilon_gradient_norm": .5,
                        "weighted_loss_epsilon_gradient_nonzero_fraction": .75} for d in (10, 20)],
                    "elapsed_seconds": 1., "peak_vram_bytes": 100})
    return rows, catalog


def test_image_level_readout_does_not_count_timesteps_as_independent(synthetic_rows):
    rows, catalog = synthetic_rows
    result = readout(rows, catalog)
    assert result["independent_unit_count"] == 32
    assert len(result["all_image_summaries"]["cofitok"]) == 32
    assert result["shared_hypotheses_meeting_frozen_rule"] == []
    assert result["paired_image_cosine_delta_cofitok_minus_dense"]["rollout.class_embedding"]["mean"] == 0
    assert result["gradient_vector_recomputation_performed"] is False


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "seed", "norm", "cosine", "nonfinite", "null_teacher", "batch", "rollout"])
def test_readout_rejects_incomplete_or_tampered_evidence(synthetic_rows, mutation):
    rows, catalog = synthetic_rows
    if mutation == "missing": rows.pop()
    elif mutation == "duplicate": rows[1] = rows[0]
    elif mutation == "seed": rows[0]["noise_seed"] += 1
    elif mutation == "norm": rows[0]["terms"]["epsilon"]["gradient"]["all"]["norm"] = 5
    elif mutation == "cosine": rows[0]["terms"]["rollout"]["gradient"]["all"]["cosine_to_epsilon"] = -.5
    elif mutation == "nonfinite": rows[0]["terms"]["epsilon"]["weighted_loss"] = float("nan")
    elif mutation == "null_teacher": rows[0]["conditioning_routes"][1]["effective_labels"] = [1000]
    elif mutation == "batch": rows[0]["normalization"]["is_original_minibatch_gradient"] = True
    else: rows[0]["rollout_steps"] = []
    with pytest.raises(ValueError):
        readout(rows, catalog)


def test_nomination_requires_same_route_in_both_methods(synthetic_rows):
    rows, catalog = synthetic_rows
    for row in rows[:128]:
        row["terms"]["rollout"]["gradient"] = stats(.5, -.2)
    result = readout(rows, catalog)
    assert result["nomination_counts"]["cofitok"]["rollout.class_embedding"] == 32
    assert result["shared_hypotheses_meeting_frozen_rule"] == []
    for row in rows[128:]:
        row["terms"]["rollout"]["gradient"] = stats(.5, -.2)
    result = readout(rows, catalog)
    assert "rollout.class_embedding" in result["shared_hypotheses_meeting_frozen_rule"]
    assert result["training_authorized"] is False
    assert result["causal_explanation_proven"] is False
