from __future__ import annotations

import pytest

from cofitok.diffusion import select_sampling_timesteps
from cofitok.generation import (
    INFERENCE_API,
    SAMPLING_PROTOCOL_SCHEMA,
    sampling_protocol_contract,
)
from scripts.build_large_scale_generation_comparison import (
    SOURCE_REPORT_PROFILES,
    build_report,
    render_csv,
    render_markdown,
    source_report_identity,
    verify_comparison_source_reports,
)
from scripts.build_generation_gate_report import (
    _class_fidelity_evidence as gate_class_fidelity_evidence,
)


def _training(parameters: int, token_count: int) -> dict:
    return {
        "target_steps": 300_000,
        "parameter_count": parameters,
        "elapsed_seconds": 100_000.0,
        "peak_vram_bytes": 24 * 1024**3,
        "final_metrics": {"samples_seen": 19_200_000},
        "config": {
            "data": {"dataset": "imagenet_256", "batch_size": 16},
            "diffusion": {"num_train_timesteps": 1000},
            "runtime": {"device": "cuda"},
            "optimization": {"gradient_accumulation_steps": 4},
            "model": {"image_size": 256, "token_count": token_count},
        },
    }


def _sampling(token_count: int) -> dict:
    return {
        "protocol_schema": SAMPLING_PROTOCOL_SCHEMA,
        "inference_api": INFERENCE_API,
        "sampler": "ddim",
        "num_samples": 50_000,
        "start_index": 0,
        "batch_size": 64,
        "sample_steps": 250,
        "num_train_timesteps": 1000,
        "actual_timesteps": select_sampling_timesteps(1000, 250),
        "prefix_budgets": [token_count],
        "guidance_scale": 1.5,
        "guidance_rescale": 0.0,
        "cfg_batch_mode": "batched",
        "eta": 0.0,
        "clip_x0": True,
        "seed": 0,
        "precision": "bf16",
        "class_schedule": "balanced_modulo",
        "random_stream": {
            "prefix_budgets_share_stream": True,
            "batch_size_invariant": True,
            "resume_index_invariant": True,
        },
    }


def _generation(
    fid: float,
    checkpoint_sha: str,
    sample_sha: str,
    token_count: int,
) -> dict:
    return {
        "counts": {"real_image_count": 50_000, "generated_image_count": 50_000},
        "implementation": {"package": "torch_fidelity", "version": "0.4.0"},
        "runtime_environment_sha256": "f" * 64,
        "real_set": {
            "digest_schema": "cofitok_image_tree_sha256_v1",
            "sha256": "e" * 64,
            "image_count": 50_000,
        },
        "metrics": {
            "frechet_inception_distance": fid,
            "inception_score_mean": 30.0,
            "precision": 0.7,
            "recall": 0.5,
        },
        "sample_provenance": {
            "checkpoint_sha256": checkpoint_sha,
            "sample_set_sha256": sample_sha,
            "weights": "ema",
            "sampling": _sampling(token_count),
            "sampling_progress": {
                "status": "completed",
                "completed_samples": 50_000,
                "cumulative_elapsed_seconds": 10_000.0,
                "invocation": 1,
            },
        },
    }


def _official() -> dict:
    rows = []
    for alias, method, fid in (
        ("d_ar", "D-AR", 2.6281),
        ("mar", "MAR", 2.3385),
        ("retok", "ReTok", 2.2189),
    ):
        rows.append(
            {
                "alias": alias,
                "method": method,
                "dataset": "imagenet_256",
                "resolution": 256,
                "status": "completed_eval_only_50k",
                "sample_count": 50_000,
                "fid": fid,
                "inception_score": 200.0,
                "precision": 0.8,
                "recall": 0.6,
                "protocol": "official pretrained eval-only",
                "paper_table_role": "secondary related-method only",
                "metrics_txt": f"/{alias}.txt",
            }
        )
    return {"schema_version": 1, "rows": rows}


def _source_reports() -> dict:
    root = "/root/autodl-tmp/CoFiTok"
    return {
        "cofitok_training": {
            "path": f"{root}/checkpoints/generation/imagenet256_full_cofitok_k8_300k/training_report.json",
            "bytes": 100,
            "sha256": "1" * 64,
        },
        "dense_training": {
            "path": f"{root}/checkpoints/generation/imagenet256_full_dense_300k/training_report.json",
            "bytes": 100,
            "sha256": "2" * 64,
        },
        "cofitok_generation": {
            "path": f"{root}/checkpoints/generation/imagenet256_full_cofitok_k8_300k/samples_50k_ddim250_cfg15/metrics/generation_metrics_report.json",
            "bytes": 100,
            "sha256": "3" * 64,
        },
        "dense_generation": {
            "path": f"{root}/checkpoints/generation/imagenet256_full_dense_300k/samples_50k_ddim250_cfg15/metrics/generation_metrics_report.json",
            "bytes": 100,
            "sha256": "4" * 64,
        },
        "final_gate": {
            "path": f"{root}/CoFiTok-internal/artifacts/reports/generation/imagenet256_full_matched_300k/final_generation_gate.json",
            "bytes": 100,
            "sha256": "5" * 64,
        },
        "training_contention": {
            "path": f"{root}/checkpoints/generation/generation_full_matched_300k_monitor.json",
            "bytes": 100,
            "sha256": "6" * 64,
        },
    }


def _contention(*, direct: bool = True) -> dict:
    reason = (
        "exclusive_gpu_observation_coverage"
        if direct
        else "external_gpu_contention_observed"
    )
    return {
        "monitor": "generation_full_matched_300k",
        "status": "pass",
        "stage": "complete",
        "git": {
            "revision": "a" * 40,
            "branch": "scale/generative-system",
            "tracked_dirty": False,
        },
        "gpu_contention": {
            "schema_version": 1,
            "role": "generation_gpu_contention_evidence",
            "status": "pass_exclusive" if direct else "pass_observational_only",
            "binding": {
                "monitor_name": "generation_full_matched_300k",
                "training_revision": "a" * 40,
                "training_branch": "scale/generative-system",
            },
            "poll_seconds": 300.0,
            "observation_count": 10,
            "coverage": {
                "complete": True,
                "started_before_training": True,
                "saw_training_active": True,
                "completed_after_training": True,
                "continuous": True,
                "all_gpu_queries_complete": True,
                "maximum_gap_seconds": 300.0,
            },
            "unrelated_gpu_compute": {
                "observed": not direct,
                "identity_overflow": False,
                "observation_count": 1 if not direct else 0,
                "identities": ([{"identity_key": "99:100"}] if not direct else []),
            },
            "training_wall_clock": {
                "measurement": "raw_process_wall_clock",
                "direct_comparison_allowed": direct,
                "reason": reason,
            },
        },
    }


def _gate(status: str = "pass") -> dict:
    cofitok_contract = sampling_protocol_contract(
        _sampling(8), stage="full", expected_num_train_timesteps=1000
    )
    dense_contract = sampling_protocol_contract(
        _sampling(1), stage="full", expected_num_train_timesteps=1000
    )
    return {
        "stage": "full",
        "status": status,
        "decision": "large_scale_generation_ready" if status == "pass" else "hold",
        "summary": {"cofitok_fid": 12.0, "dense_fid": 11.8},
        "gates": [
            {
                "name": "matched_sampling_provenance",
                "passed": True,
                "evidence": {
                    "cofitok_checkpoint_sha256": "a" * 64,
                    "dense_checkpoint_sha256": "b" * 64,
                    "cofitok_sample_set_sha256": "c" * 64,
                    "dense_sample_set_sha256": "d" * 64,
                },
            },
            {
                "name": "matched_real_set_provenance",
                "passed": True,
                "evidence": {
                    method: {
                        "digest_schema": "cofitok_image_tree_sha256_v1",
                        "sha256": "e" * 64,
                        "image_count": 50_000,
                    }
                    for method in ("cofitok", "dense_identity")
                },
            },
            {
                "name": "formal_sampling_protocol",
                "passed": True,
                "evidence": {
                    "stage": "full",
                    "cofitok": cofitok_contract,
                    "dense_identity": dense_contract,
                },
            },
        ],
    }


def _report(
    official: dict | None = None,
    gate: dict | None = None,
    contention: dict | None = None,
    cofitok_training: dict | None = None,
    dense_training: dict | None = None,
) -> dict:
    return build_report(
        cofitok_training=cofitok_training or _training(62_950_800, 8),
        dense_training=dense_training or _training(62_824_707, 1),
        cofitok_generation=_generation(12.0, "a" * 64, "c" * 64, 8),
        dense_generation=_generation(11.8, "b" * 64, "d" * 64, 1),
        final_gate=gate or _gate(),
        official_related=official or _official(),
        training_contention=contention or _contention(),
        official_source_path="/reports/official_related_methods_table.json",
        official_source_sha256="e" * 64,
        source_reports=_source_reports(),
    )


def test_comparison_separates_matched_and_official_protocols() -> None:
    report = _report()

    assert report["status"] == "ready"
    assert report["schema_version"] == 8
    assert report["source_reports"] == _source_reports()
    assert len(report["matched_training_rows"]) == 2
    assert len(report["official_context_rows"]) == 3
    assert report["comparison_policy"]["cross_tier_numeric_ranking_allowed"] is False
    assert all(row["directly_comparable_to_cofitok"] for row in report["matched_training_rows"])
    assert not any(row["directly_comparable_to_cofitok"] for row in report["official_context_rows"])
    assert {row["alias"] for row in report["official_context_rows"]} == {
        "d_ar",
        "mar",
        "retok",
    }
    assert report["official_context_source"]["sha256"] == "e" * 64
    assert report["matched_training_rows"][0]["effective_batch_size"] == 64
    assert report["matched_training_rows"][0]["training_images_seen"] == 19_200_000
    assert report["matched_training_rows"][0][
        "training_wall_clock_directly_comparable"
    ] is True
    assert report["training_budget_policy"] == {
        "basis": "matched_steps_and_training_images",
        "matched_axis_values": {
            "dataset": "imagenet_256",
            "resolution": 256,
            "effective_batch_size": 64,
            "optimizer_steps": 300_000,
            "training_images_seen": 19_200_000,
        },
        "equal_wall_clock_budget": False,
        "equal_gpu_hours_budget": False,
        "equal_training_flops_budget": False,
        "compute_matched_claim_allowed": False,
        "direct_quality_comparison_allowed": True,
        "training_cost_fields_role": "measured_outcomes",
        "cost_efficiency_ranking_allowed": True,
        "cost_efficiency_ranking_reason": "exclusive_gpu_observation_coverage",
    }
    assert report["matched_training_rows"][0]["compute_matched_claim_allowed"] is False
    assert report["matched_training_rows"][0]["peak_vram_bytes"] == 24 * 1024**3
    assert report["matched_training_rows"][0]["sampling_images_per_second"] == 5.0
    assert report["matched_training_rows"][0]["sample_batch_size"] == 64
    assert report["matched_training_rows"][0]["clip_x0"] is True
    assert report["matched_training_rows"][0]["sampling_seed"] == 0
    assert report["matched_training_rows"][0]["real_set_sha256"] == "e" * 64
    assert "not a direct ranking" in render_markdown(report)
    assert "VRAM GiB" in render_markdown(report)
    assert "sample img/s" in render_markdown(report)
    assert "sample batch" in render_markdown(report)
    assert "train h (raw)" in render_markdown(report)
    assert "not an equal wall-clock, GPU-hours, or FLOPs budget" in render_markdown(
        report
    )
    assert "matched_training_direct" in render_csv(report)
    assert "matched_steps_and_training_images" in render_csv(report)
    assert "official_pretrained_contextual" in render_csv(report)


def test_comparison_preserves_hold_decision() -> None:
    assert _report(gate=_gate("fail"))["status"] == "hold"


def test_comparison_labels_contended_training_time_as_observational_only() -> None:
    report = _report(contention=_contention(direct=False))

    assert report["comparison_policy"][
        "training_wall_clock_direct_comparison_allowed"
    ] is False
    assert all(
        row["training_wall_clock_directly_comparable"] is False
        for row in report["matched_training_rows"]
    )
    assert report["training_budget_policy"]["cost_efficiency_ranking_allowed"] is False
    markdown = render_markdown(report)
    assert "raw observations only" in markdown
    assert "external_gpu_contention_observed" in markdown


def test_comparison_rejects_mismatched_training_step_budget() -> None:
    dense_training = _training(62_824_707, 1)
    dense_training["target_steps"] = 299_999
    dense_training["final_metrics"]["samples_seen"] = 299_999 * 64

    with pytest.raises(ValueError, match="optimizer_steps"):
        _report(dense_training=dense_training)


def test_comparison_rejects_mismatched_effective_batch_budget() -> None:
    dense_training = _training(62_824_707, 1)
    dense_training["config"]["data"]["batch_size"] = 8
    dense_training["final_metrics"]["samples_seen"] = 300_000 * 32

    with pytest.raises(ValueError, match="effective_batch_size"):
        _report(dense_training=dense_training)


def test_comparison_rejects_mismatched_real_set() -> None:
    dense = _generation(11.8, "b" * 64, "d" * 64, 1)
    dense["real_set"]["sha256"] = "8" * 64

    with pytest.raises(ValueError, match="different real sets"):
        build_report(
            cofitok_training=_training(62_950_800, 8),
            dense_training=_training(62_824_707, 1),
            cofitok_generation=_generation(12.0, "a" * 64, "c" * 64, 8),
            dense_generation=dense,
            final_gate=_gate(),
            official_related=_official(),
            training_contention=_contention(),
            official_source_path="/reports/official_related_methods_table.json",
            official_source_sha256="e" * 64,
            source_reports=_source_reports(),
        )


def test_comparison_rejects_unsafe_external_table_role() -> None:
    official = _official()
    official["rows"][0]["paper_table_role"] = "direct comparison"

    with pytest.raises(ValueError, match="unsafe table role"):
        _report(official=official)


def test_comparison_rejects_gate_from_another_sample_set() -> None:
    gate = _gate()
    gate["gates"][0]["evidence"]["cofitok_sample_set_sha256"] = "e" * 64

    with pytest.raises(ValueError, match="sample-set hash does not match"):
        _report(gate=gate)


def test_comparison_rejects_incomplete_sampling_progress() -> None:
    generation = _generation(12.0, "a" * 64, "c" * 64, 8)
    generation["sample_provenance"]["sampling_progress"]["status"] = "running"

    with pytest.raises(ValueError, match="sampling progress is incomplete"):
        build_report(
            cofitok_training=_training(62_950_800, 8),
            dense_training=_training(62_824_707, 1),
            cofitok_generation=generation,
            dense_generation=_generation(11.8, "b" * 64, "d" * 64, 1),
            final_gate=_gate(),
            official_related=_official(),
            training_contention=_contention(),
            official_source_path="/reports/official_related_methods_table.json",
            official_source_sha256="e" * 64,
            source_reports=_source_reports(),
        )


def test_comparison_rejects_external_method_identity_mismatch() -> None:
    official = _official()
    official["rows"][0]["method"] = "MAR"

    with pytest.raises(ValueError, match="method identity mismatch"):
        _report(official=official)


def test_comparison_rejects_external_metric_out_of_range() -> None:
    official = _official()
    official["rows"][0]["precision"] = 1.1

    with pytest.raises(ValueError, match="metrics are out of range"):
        _report(official=official)


def test_comparison_rejects_matched_weakened_sampling_protocol() -> None:
    cofitok = _generation(12.0, "a" * 64, "c" * 64, 8)
    dense = _generation(11.8, "b" * 64, "d" * 64, 1)
    cofitok["sample_provenance"]["sampling"]["clip_x0"] = False
    dense["sample_provenance"]["sampling"]["clip_x0"] = False

    with pytest.raises(ValueError, match="formal sampling protocol is invalid"):
        build_report(
            cofitok_training=_training(62_950_800, 8),
            dense_training=_training(62_824_707, 1),
            cofitok_generation=cofitok,
            dense_generation=dense,
            final_gate=_gate(),
            official_related=_official(),
            training_contention=_contention(),
            official_source_path="/reports/official_related_methods_table.json",
            official_source_sha256="e" * 64,
            source_reports=_source_reports(),
        )


def test_comparison_rejects_out_of_range_matched_metrics() -> None:
    cofitok = _generation(12.0, "a" * 64, "c" * 64, 8)
    cofitok["metrics"]["precision"] = 1.1

    with pytest.raises(ValueError, match="distribution metrics are out of range"):
        build_report(
            cofitok_training=_training(62_950_800, 8),
            dense_training=_training(62_824_707, 1),
            cofitok_generation=cofitok,
            dense_generation=_generation(11.8, "b" * 64, "d" * 64, 1),
            final_gate=_gate(),
            official_related=_official(),
            training_contention=_contention(),
            official_source_path="/reports/official_related_methods_table.json",
            official_source_sha256="e" * 64,
            source_reports=_source_reports(),
        )


def test_comparison_source_verification_detects_changed_file(tmp_path) -> None:
    paths = {
        "cofitok_training": tmp_path
        / "imagenet256_full_cofitok_k8_300k/training_report.json",
        "dense_training": tmp_path
        / "imagenet256_full_dense_300k/training_report.json",
        "cofitok_generation": tmp_path
        / "imagenet256_full_cofitok_k8_300k/samples_50k_ddim250_cfg15/metrics/generation_metrics_report.json",
        "dense_generation": tmp_path
        / "imagenet256_full_dense_300k/samples_50k_ddim250_cfg15/metrics/generation_metrics_report.json",
        "final_gate": tmp_path
        / "artifacts/reports/generation/imagenet256_full_matched_300k/final_generation_gate.json",
        "training_contention": tmp_path
        / "generation_full_matched_300k_monitor.json",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"status":"completed"}\n', encoding="utf-8")
    report = {
        "source_reports": {
            name: source_report_identity(path) for name, path in paths.items()
        }
    }

    assert verify_comparison_source_reports(report)["status"] == "verified"

    paths["cofitok_generation"].write_text(
        '{"status":"changed"}\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="changed after binding: cofitok_generation"):
        verify_comparison_source_reports(report)


def test_comparison_accepts_stability_full_source_profile(tmp_path) -> None:
    source_reports = {}
    for name, suffix in SOURCE_REPORT_PROFILES["stability_full"].items():
        path = tmp_path / suffix
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"status":"completed"}\n', encoding="utf-8")
        source_reports[name] = source_report_identity(path)

    report = _report()
    report["source_profile"] = "stability_full"
    report["source_reports"] = source_reports

    verified = verify_comparison_source_reports(report)

    assert verified["status"] == "verified"
    assert verified["source_profile"] == "stability_full"


def test_stability_comparison_requires_and_reports_class_fidelity(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.test_generation_class_fidelity import (
        EXPECTED_GIT,
        _passing_qualification,
    )

    class_fidelity = _passing_qualification(tmp_path, monkeypatch)
    class_fidelity["sampling_contract"].update(
        {
            "cofitok_checkpoint_sha256": "a" * 64,
            "dense_checkpoint_sha256": "b" * 64,
            "cofitok_sample_set_sha256": "c" * 64,
            "dense_sample_set_sha256": "d" * 64,
        }
    )
    contract = class_fidelity["sampling_contract"]
    gate = _gate()
    gate["provenance_contract"] = {
        "evaluation_revision": EXPECTED_GIT["revision"],
        "evaluation_branch": EXPECTED_GIT["branch"],
    }
    gate["gates"].append(
        {
            "name": "class_conditional_fidelity",
            "passed": True,
            "evidence": gate_class_fidelity_evidence(
                class_fidelity,
                stage="full",
                expected_evaluation_revision=EXPECTED_GIT["revision"],
                expected_evaluation_branch=EXPECTED_GIT["branch"],
                expected_sampling_protocol=contract["sampling"],
                expected_cofitok_checkpoint_sha256="a" * 64,
                expected_dense_checkpoint_sha256="b" * 64,
                expected_cofitok_sample_set_sha256="c" * 64,
                expected_dense_sample_set_sha256="d" * 64,
            ),
        }
    )
    source_reports = {
        name: {
            "path": (
                "/root/autodl-tmp/CoFiTok/checkpoints/generation/" + suffix
            ),
            "bytes": 100,
            "sha256": f"{index:x}" * 64,
        }
        for index, (name, suffix) in enumerate(
            SOURCE_REPORT_PROFILES["stability_full"].items(),
            start=1,
        )
    }
    report = build_report(
        cofitok_training=_training(62_950_800, 8),
        dense_training=_training(62_824_707, 1),
        cofitok_generation=_generation(12.0, "a" * 64, "c" * 64, 8),
        dense_generation=_generation(11.8, "b" * 64, "d" * 64, 1),
        final_gate=gate,
        official_related=_official(),
        training_contention=_contention(),
        class_fidelity_qualification=class_fidelity,
        official_source_path="/reports/official_related_methods_table.json",
        official_source_sha256="e" * 64,
        source_reports=source_reports,
        source_profile="stability_full",
    )

    assert report["status"] == "ready"
    assert report["class_fidelity"]["valid"] is True
    assert report["matched_training_rows"][0]["class_top1_accuracy"] == 0.21
    assert report["matched_training_rows"][1]["class_top5_accuracy"] == 0.43
    assert report["matched_summary"]["cofitok_minus_dense_class_top1"] == pytest.approx(
        -0.01
    )
    assert "class top-1" in render_markdown(report)
    assert "class_top1_accuracy" in render_csv(report)

    with pytest.raises(ValueError, match="requires class-fidelity qualification"):
        build_report(
            cofitok_training=_training(62_950_800, 8),
            dense_training=_training(62_824_707, 1),
            cofitok_generation=_generation(12.0, "a" * 64, "c" * 64, 8),
            dense_generation=_generation(11.8, "b" * 64, "d" * 64, 1),
            final_gate=gate,
            official_related=_official(),
            training_contention=_contention(),
            official_source_path="/reports/official_related_methods_table.json",
            official_source_sha256="e" * 64,
            source_reports=source_reports,
            source_profile="stability_full",
        )
