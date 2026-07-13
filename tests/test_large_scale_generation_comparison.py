from __future__ import annotations

import pytest

from scripts.build_large_scale_generation_comparison import (
    build_report,
    render_csv,
    render_markdown,
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
            "runtime": {"device": "cuda"},
            "optimization": {"gradient_accumulation_steps": 4},
            "model": {"image_size": 256, "token_count": token_count},
        },
    }


def _generation(fid: float, checkpoint_sha: str, sample_sha: str) -> dict:
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
            "sampling": {
                "sample_steps": 250,
                "guidance_scale": 1.5,
                "batch_size": 64,
            },
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


def _gate(status: str = "pass") -> dict:
    return {
        "stage": "full",
        "status": status,
        "decision": "large_scale_generation_ready" if status == "pass" else "hold",
        "summary": {"cofitok_fid": 12.0, "dense_fid": 11.8},
        "gates": [
            {
                "name": "matched_sampling_provenance",
                "evidence": {
                    "cofitok_checkpoint_sha256": "a" * 64,
                    "dense_checkpoint_sha256": "b" * 64,
                    "cofitok_sample_set_sha256": "c" * 64,
                    "dense_sample_set_sha256": "d" * 64,
                },
            },
            {
                "name": "matched_real_set_provenance",
                "evidence": {
                    method: {
                        "digest_schema": "cofitok_image_tree_sha256_v1",
                        "sha256": "e" * 64,
                        "image_count": 50_000,
                    }
                    for method in ("cofitok", "dense_identity")
                },
            },
        ],
    }


def _report(official: dict | None = None, gate: dict | None = None) -> dict:
    return build_report(
        cofitok_training=_training(62_950_800, 8),
        dense_training=_training(62_824_707, 1),
        cofitok_generation=_generation(12.0, "a" * 64, "c" * 64),
        dense_generation=_generation(11.8, "b" * 64, "d" * 64),
        final_gate=gate or _gate(),
        official_related=official or _official(),
        official_source_path="/reports/official_related_methods_table.json",
        official_source_sha256="e" * 64,
    )


def test_comparison_separates_matched_and_official_protocols() -> None:
    report = _report()

    assert report["status"] == "ready"
    assert report["schema_version"] == 3
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
    assert report["matched_training_rows"][0]["peak_vram_bytes"] == 24 * 1024**3
    assert report["matched_training_rows"][0]["sampling_images_per_second"] == 5.0
    assert report["matched_training_rows"][0]["sample_batch_size"] == 64
    assert report["matched_training_rows"][0]["real_set_sha256"] == "e" * 64
    assert "not a direct ranking" in render_markdown(report)
    assert "VRAM GiB" in render_markdown(report)
    assert "sample img/s" in render_markdown(report)
    assert "sample batch" in render_markdown(report)
    assert "matched_training_direct" in render_csv(report)
    assert "official_pretrained_contextual" in render_csv(report)


def test_comparison_preserves_hold_decision() -> None:
    assert _report(gate=_gate("fail"))["status"] == "hold"


def test_comparison_rejects_mismatched_real_set() -> None:
    dense = _generation(11.8, "b" * 64, "d" * 64)
    dense["real_set"]["sha256"] = "8" * 64

    with pytest.raises(ValueError, match="different real sets"):
        build_report(
            cofitok_training=_training(62_950_800, 8),
            dense_training=_training(62_824_707, 1),
            cofitok_generation=_generation(12.0, "a" * 64, "c" * 64),
            dense_generation=dense,
            final_gate=_gate(),
            official_related=_official(),
            official_source_path="/reports/official_related_methods_table.json",
            official_source_sha256="e" * 64,
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
    generation = _generation(12.0, "a" * 64, "c" * 64)
    generation["sample_provenance"]["sampling_progress"]["status"] = "running"

    with pytest.raises(ValueError, match="sampling progress is incomplete"):
        build_report(
            cofitok_training=_training(62_950_800, 8),
            dense_training=_training(62_824_707, 1),
            cofitok_generation=generation,
            dense_generation=_generation(11.8, "b" * 64, "d" * 64),
            final_gate=_gate(),
            official_related=_official(),
            official_source_path="/reports/official_related_methods_table.json",
            official_source_sha256="e" * 64,
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
