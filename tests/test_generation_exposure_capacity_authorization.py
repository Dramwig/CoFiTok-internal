from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

import cofitok.generation.exposure_capacity as preparation_module
import cofitok.generation.exposure_capacity_authorization as authorization
import cofitok.generation.exposure_capacity_gate as gate
from scripts.exposure_capacity_result_cli import layout


def _identity(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {"path": path.resolve().as_posix(), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _source_run(root: Path, name: str, *, payload: bytes) -> tuple[Path, dict[str, object]]:
    run = root / name
    run.mkdir()
    checkpoint = run / "checkpoint_step_00100000.pt"
    checkpoint.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    sidecar = checkpoint.with_name(f"{checkpoint.name}.integrity.json")
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "checkpoint": checkpoint.name,
                "checkpoint_bytes": len(payload),
                "checkpoint_sha256": digest,
                "checkpoint_format_version": 1,
                "step": 100_000,
            }
        ),
        encoding="utf-8",
    )
    latest = run / "latest.json"
    latest.write_text(
        json.dumps(
            {
                "checkpoint": checkpoint.name,
                "step": 100_000,
                "checkpoint_bytes": len(payload),
                "checkpoint_sha256": digest,
                "checkpoint_format_version": 1,
                "integrity_manifest": sidecar.name,
            }
        ),
        encoding="utf-8",
    )
    summary = {
        "run_dir": run.resolve().as_posix(),
        "step": 100_000,
        "checkpoint": {
            "name": checkpoint.name,
            "bytes": len(payload),
            "sha256": digest,
        },
        "integrity_manifest_name": sidecar.name,
    }
    return run, summary


def test_preparation_source_descriptors_bind_objective_reassessment() -> None:
    identities = {
        name: {
            "path": f"/tmp/{name}.json",
            "bytes": 1,
            "sha256": "a" * 64,
        }
        for name in preparation_module._SOURCE_NAMES
    }
    assert set(preparation_module._source_descriptors(identities)) == set(
        preparation_module._SOURCE_NAMES
    )

    del identities["objective_reassessment"]
    with pytest.raises(ValueError, match="missing"):
        preparation_module._source_descriptors(identities)


def _preparation(tmp_path: Path) -> tuple[dict[str, object], dict[str, object]]:
    cofitok_run, cofitok_summary = _source_run(tmp_path, "cofitok", payload=b"cofitok")
    dense_run, dense_summary = _source_run(tmp_path, "dense", payload=b"dense")
    del cofitok_run, dense_run
    preparation = {
        "evidence": {
            "training": {
                "source_checkpoints": {
                    "cofitok": cofitok_summary,
                    "dense_identity": dense_summary,
                }
            }
        }
    }
    return preparation, {"cofitok": cofitok_summary, "dense_identity": dense_summary}


def test_physical_source_binding_rehashes_payload_sidecar_and_latest(tmp_path: Path) -> None:
    preparation, summaries = _preparation(tmp_path)
    bindings = authorization._physical_source_bindings(preparation)
    assert bindings["cofitok"]["checkpoint"]["sha256"] == summaries["cofitok"]["checkpoint"]["sha256"]
    assert bindings["dense_identity"]["latest"]["bytes"] > 0

    latest = Path(bindings["dense_identity"]["latest"]["path"])
    latest.write_text(latest.read_text(encoding="utf-8").replace('"step": 100000', '"step": 99999'), encoding="utf-8")
    with pytest.raises(ValueError, match="latest pointer"):
        authorization._physical_source_bindings(preparation)


def test_preparation_sources_maps_report_names_to_method_ids(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cofitok_report = tmp_path / "cofitok_training_report.json"
    dense_report = tmp_path / "dense_training_report.json"
    cofitok_report.write_text(
        json.dumps({"config": {"name": "cofitok-source"}}), encoding="utf-8"
    )
    dense_report.write_text(
        json.dumps({"config": {"name": "dense-source"}}), encoding="utf-8"
    )
    preparation = {
        "sources": {
            "cofitok_training_report": _identity(cofitok_report),
            "dense_training_report": _identity(dense_report),
        }
    }
    monkeypatch.setattr(
        authorization,
        "validate_source_checkpoint_bindings",
        lambda *_args, **_kwargs: None,
    )

    reports = authorization._preparation_sources(preparation)

    assert set(reports) == {"cofitok", "dense_identity"}
    assert reports["cofitok"]["config"]["name"] == "cofitok-source"
    assert reports["dense_identity"]["config"]["name"] == "dense-source"


def test_config_binding_allows_only_name_and_horizon(tmp_path: Path) -> None:
    source = {
        "name": "source",
        "data": {"dataset": "imagenet_256", "batch_size": 64},
        "runtime": {"steps": 100_000},
    }
    target = deepcopy(source)
    target["name"] = "target"
    target["runtime"]["steps"] = 110_000
    paths = {}
    reports = {}
    for method in ("cofitok", "dense_identity"):
        config = tmp_path / f"{method}.json"
        config.write_text(json.dumps(target), encoding="utf-8")
        paths[method] = _identity(config)
        reports[method] = {"config": source}
    assert authorization._validate_config_bindings(paths, reports=reports) == paths

    target["data"]["batch_size"] = 32
    (tmp_path / "cofitok.json").write_text(json.dumps(target), encoding="utf-8")
    paths["cofitok"] = _identity(tmp_path / "cofitok.json")
    with pytest.raises(ValueError, match="outside name/runtime.steps"):
        authorization._validate_config_bindings(paths, reports=reports)


def test_config_binding_compares_resolved_defaults(tmp_path: Path) -> None:
    source = {
        "name": "source",
        "data": {"dataset": "imagenet_256", "batch_size": 64},
        "diffusion": {"beta_start": 0.0001},
        "runtime": {"steps": 100_000},
    }
    target = {
        "name": "target",
        "data": {"dataset": "imagenet_256", "batch_size": 64},
        "runtime": {"steps": 110_000},
    }
    paths = {}
    reports = {}
    for method in ("cofitok", "dense_identity"):
        config = tmp_path / f"{method}.json"
        config.write_text(json.dumps(target), encoding="utf-8")
        paths[method] = _identity(config)
        reports[method] = {"config": source}

    assert authorization._validate_config_bindings(paths, reports=reports) == paths


def test_result_layout_binds_method_specific_prefixes(tmp_path: Path) -> None:
    cofitok = layout(tmp_path, "cofitok")
    dense = layout(tmp_path, "dense_identity")
    assert cofitok["generated"].name == "prefix_8"
    assert dense["generated"].name == "prefix_1"
    assert cofitok["training_report"].name == "training_report.json"
    assert dense["checkpoint_report"].name == "checkpoint_evaluation_report.json"


def test_exposure_gate_binds_source_scheduler_horizon() -> None:
    preparation = {
        "candidate_arms": {
            "exposure_continuation": {
                "controlled_change": "training_exposure_only",
                "initialization": "exact_100k_checkpoint_resume_only",
                "model_layout_change_allowed": False,
                "output_root": "/root/autodl-tmp/CoFiTok/checkpoints/generation/exposure",
            },
            "capacity_qualification": {
                "controlled_change": "model_capacity_only",
                "initialization": "fresh_matched_initialization_required",
                "candidate_base_channels": 256,
                "output_root": "/root/autodl-tmp/CoFiTok/checkpoints/generation/capacity",
            },
        }
    }
    contract = gate._arm_contract(preparation, "exposure_continuation")
    assert contract["scheduler"] == gate.EXPOSURE_SCHEDULER_CONTRACT
    assert contract["scheduler"]["effective_horizon_steps"] == gate.SOURCE_STEP
    assert contract["scheduler"]["target_horizon_steps"] == gate.EXPOSURE_TARGET_STEP


def _contract() -> dict[str, object]:
    return {
        "arm_id": "exposure_continuation",
        "source_stage": "stability_quality_bridge",
        "controlled_change": "training_exposure_only",
        "methods": ["cofitok", "dense_identity"],
        "dataset": "imagenet_256",
        "effective_batch_size": 64,
        "source_step": 100_000,
        "source_images_seen_per_method": 6_400_000,
        "matched_pair_required": True,
        "objective_change_allowed": False,
        "conditioning_change_allowed": False,
        "formal_quality_claim_allowed": False,
        "evaluation": deepcopy(authorization.EVALUATION_CONTRACT),
        "output_root": "/root/autodl-tmp/CoFiTok/checkpoints/generation/exposure/test",
        "initialization": "exact_100k_checkpoint_resume_only",
        "target_step": 110_000,
        "additional_steps": 10_000,
        "resume_checkpoint_step": 100_000,
        "resume_checkpoint_required": True,
        "source_checkpoint_binding_required": True,
        "new_output_root_required": True,
        "automatic_300k_escalation_allowed": False,
        "scheduler": deepcopy(authorization.EXPOSURE_SCHEDULER_CONTRACT),
    }


def _fake_authorization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    source = {
        "run_dir": "/root/source/cofitok",
        "step": 100_000,
        "checkpoint": {"path": "/root/source/cofitok/checkpoint.pt", "bytes": 10, "sha256": "a" * 64},
        "integrity_manifest": {"path": "/root/source/cofitok/checkpoint.pt.integrity.json", "bytes": 11, "sha256": "b" * 64},
        "latest": {"path": "/root/source/cofitok/latest.json", "bytes": 12, "sha256": "c" * 64},
    }
    dense = deepcopy(source)
    dense["run_dir"] = "/root/source/dense"
    for item in (dense["checkpoint"], dense["integrity_manifest"], dense["latest"]):
        item["path"] = item["path"].replace("cofitok", "dense")
    sources = {"cofitok": source, "dense_identity": dense}
    gate = {
        "selected_arm": "exposure_continuation",
        "execution_ready": False,
        "terminal_status": "hold",
        "generation_advantage_proven": False,
        "qualification_contract": _contract(),
        "source_checkpoint": source,
    }
    preparation: dict[str, object] = {}
    execution = {"revision": "d" * 40, "tree": "e" * 40, "branch": "test", "tracked_dirty": False}
    configs = {
        "cofitok": {"path": "/tmp/cofitok.json", "bytes": 1, "sha256": "1" * 64},
        "dense_identity": {"path": "/tmp/dense.json", "bytes": 1, "sha256": "2" * 64},
    }
    prep_id = {"path": "/tmp/prep.json", "bytes": 1, "sha256": "3" * 64}
    gate_id = {"path": "/tmp/gate.json", "bytes": 1, "sha256": "4" * 64}
    standing_id = {"path": "/tmp/standing.json", "bytes": 1, "sha256": "5" * 64}
    live = {
        "runtime_environment_sha256": "6" * 64,
        "dataset_identity_sha256": "7" * 64,
        "gpu_inventory": [{"index": 0, "memory_used_mib": 0, "memory_total_mib": 100, "utilization_percent": 0}],
        "gpu_compute_processes": [],
        "conflicting_processes": [],
        "output_root": _contract()["output_root"],
        "output_root_absent": True,
        "execution_lock_free": True,
        "free_bytes": 200 * 1024**3,
    }
    source_checkout = deepcopy(authorization.SOURCE_CHECKOUT)
    execution_checkout = deepcopy(execution)
    stage = {
        "schema_version": authorization.STAGE_AUTHORIZATION_SCHEMA,
        "role": authorization.STAGE_AUTHORIZATION_ROLE,
        "status": "approved",
        "scope": authorization.STAGE_AUTHORIZATION_SCOPE,
        "decision": authorization.STAGE_AUTHORIZATION_DECISION,
        "selection": {
            "source_revision": source_checkout["revision"],
            "source_tree": source_checkout["tree"],
            "source_branch": source_checkout["branch"],
            "source_step": 100_000,
            "target_step": 110_000,
            "execution_revision": execution_checkout["revision"],
            "execution_tree": execution_checkout["tree"],
            "execution_branch": execution_checkout["branch"],
            "output_root": Path(_contract()["output_root"]).resolve().as_posix(),
        },
        "approval_record": {
            "approved_by": "test",
            "approved_at": "2026-09-01T00:00:00Z",
        },
        "source_checkout": source_checkout,
        "execution_checkout": execution_checkout,
        "authorization_boundary": deepcopy(authorization.STAGE_AUTHORIZATION_BOUNDARY),
    }
    stage_path = tmp_path / "stage_authorization.json"
    stage_path.write_text(json.dumps(stage), encoding="utf-8")
    stage_identity = _identity(stage_path)
    monkeypatch.setattr(authorization, "validate_execution_gate", lambda *args, **kwargs: gate)
    monkeypatch.setattr(authorization, "_validate_standing_authorization", lambda *_args: None)
    monkeypatch.setattr(authorization, "_preparation_sources", lambda *_args: {"cofitok": {"config": {}}, "dense_identity": {"config": {}}})
    monkeypatch.setattr(authorization, "_validate_config_bindings", lambda *_args, **_kwargs: configs)
    monkeypatch.setattr(authorization, "_validate_source_bindings", lambda *_args, **_kwargs: sources)
    auth = authorization.build_authorization(
        gate=gate,
        preparation=preparation,
        preparation_identity=prep_id,
        gate_identity=gate_id,
        standing_identity=standing_id,
        execution_checkout=execution,
        config_identities=configs,
        source_checkpoints=sources,
        live_prelaunch=live,
        stage_authorization=stage,
        stage_authorization_identity=stage_identity,
    )
    return auth, gate


def test_validation_rejects_evaluation_contract_tampering(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    auth, gate = _fake_authorization(monkeypatch, tmp_path)
    tampered = deepcopy(auth)
    tampered["evaluation"]["sample_steps"] = 50
    with pytest.raises(ValueError, match="evaluation contract"):
        authorization.validate_authorization_contract(
            tampered,
            gate=gate,
            preparation={},
            preparation_identity=auth["preparation"],
            gate_identity=auth["candidate_gate"],
            standing_identity=auth["standing_authorization"],
            execution_checkout=auth["execution_checkout"],
            config_identities=auth["configs"],
        )


def test_validation_rejects_tampered_user_stage_authorization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    auth, gate = _fake_authorization(monkeypatch, tmp_path)
    tampered = deepcopy(auth)
    tampered["user_stage_authorization"]["selection"]["target_step"] = 120_000
    with pytest.raises(ValueError, match="user stage authorization"):
        authorization.validate_authorization_contract(
            tampered,
            gate=gate,
            preparation={},
            preparation_identity=auth["preparation"],
            gate_identity=auth["candidate_gate"],
            standing_identity=auth["standing_authorization"],
            execution_checkout=auth["execution_checkout"],
            config_identities=auth["configs"],
        )
