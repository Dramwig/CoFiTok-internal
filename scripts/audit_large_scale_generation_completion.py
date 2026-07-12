from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable

from cofitok.reporting import write_json_report

try:
    from scripts.validate_generation_training_pair import validate_training_pair
except ModuleNotFoundError:
    from validate_generation_training_pair import validate_training_pair


PINNED_10PCT_REVISION = "781a01444fddbf0d48a427ba58bdeed50167b5be"
MILESTONE_STEPS = (50_000, 100_000, 200_000, 300_000)


def _check(
    name: str,
    payloads: list[dict[str, Any] | None],
    validator: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    if any(payload is None for payload in payloads):
        return {"name": name, "status": "missing", "evidence": None}
    try:
        evidence = validator()
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        return {
            "name": name,
            "status": "fail",
            "error": str(error),
            "evidence": None,
        }
    return {"name": name, "status": "pass", "evidence": evidence}


def _gate_evidence(
    gate: dict[str, Any], *, stage: str, decision: str
) -> dict[str, Any]:
    if gate.get("stage") != stage:
        raise ValueError(f"expected {stage} gate stage")
    if gate.get("status") != "pass" or gate.get("decision") != decision:
        raise ValueError(f"{stage} gate did not authorize {decision}")
    failed = [row.get("name") for row in gate.get("gates", []) if row.get("passed") is not True]
    if failed:
        raise ValueError(f"{stage} gate contains failed checks: {', '.join(map(str, failed))}")
    return {"decision": decision, "gate_count": len(gate.get("gates", []))}


def _training_audit_evidence(audits: dict[str, dict[str, Any]]) -> dict[str, Any]:
    evidence = {}
    for method, report in audits.items():
        if report.get("status") != "complete":
            raise ValueError(f"{method} full training audit is not complete")
        if report.get("issues"):
            raise ValueError(f"{method} full training audit contains issues")
        validation = report.get("validation", {})
        if validation.get("logging_complete") is not True:
            raise ValueError(f"{method} full validation logging is incomplete")
        checkpoint = report.get("checkpoint", {})
        if checkpoint.get("missing_required_steps"):
            raise ValueError(f"{method} protected checkpoints are missing")
        evidence[method] = {
            "last_step": int(report["last_step"]),
            "validation_event_count": int(validation["event_count"]),
            "checkpoint_steps": list(checkpoint["steps"]),
        }
    if any(row["last_step"] != 300_000 for row in evidence.values()):
        raise ValueError("full training audit did not reach exactly 300000 steps")
    return evidence


def _runtime_selection_evidence(
    selection: dict[str, Any],
    training_reports: dict[str, dict[str, Any]],
    *,
    expected_revision: str,
) -> dict[str, Any]:
    if selection.get("status") != "selected":
        raise ValueError("full training runtime selection is incomplete")
    if selection.get("git_revision") != expected_revision:
        raise ValueError("runtime selection revision differs from full training")
    selected = selection.get("selected", {})
    micro_batch = int(selected.get("micro_batch_size", -1))
    accumulation = int(selected.get("gradient_accumulation_steps", -1))
    effective_batch = int(selected.get("effective_batch_size", -1))
    if micro_batch < 1 or accumulation < 1 or effective_batch != 64:
        raise ValueError("selected full runtime is invalid")
    if micro_batch * accumulation != effective_batch:
        raise ValueError("selected full runtime changes effective batch")
    for method, report in training_reports.items():
        config = report.get("config", {})
        if int(config.get("data", {}).get("batch_size", -1)) != micro_batch:
            raise ValueError(f"{method} training did not use selected microbatch")
        if (
            int(
                config.get("optimization", {}).get(
                    "gradient_accumulation_steps", -1
                )
            )
            != accumulation
        ):
            raise ValueError(f"{method} training did not use selected accumulation")
    return {
        "micro_batch_size": micro_batch,
        "gradient_accumulation_steps": accumulation,
        "effective_batch_size": effective_batch,
        "estimated_speedup_over_16x4": float(
            selected["estimated_speedup_over_16x4"]
        ),
    }


def _milestone_evidence(
    milestones: dict[int, dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    evidence = {}
    warnings = []
    for step in MILESTONE_STEPS:
        report = milestones[step]
        if report.get("status") != "completed":
            raise ValueError(f"milestone {step} is incomplete")
        if int(report.get("milestone_step", -1)) != step:
            raise ValueError(f"milestone {step} step mismatch")
        if int(report.get("expected_samples", -1)) != 2_048:
            raise ValueError(f"milestone {step} sample count mismatch")
        methods = report.get("methods", {})
        if set(methods) != {"cofitok", "dense_identity"}:
            raise ValueError(f"milestone {step} method pair mismatch")
        for method, row in methods.items():
            if int(row.get("checkpoint_step", -1)) != step:
                raise ValueError(f"milestone {step} {method} checkpoint mismatch")
            if int(row.get("sample_count", -1)) != 2_048:
                raise ValueError(f"milestone {step} {method} samples mismatch")
        alerts = list(report.get("quality_alerts", []))
        warnings.extend(f"milestone_{step}:{alert}" for alert in alerts)
        evidence[str(step)] = {
            "quality_alerts": alerts,
            "cofitok_fid": float(methods["cofitok"]["fid"]),
            "dense_fid": float(methods["dense_identity"]["fid"]),
        }
    return evidence, warnings


def _generation_evidence(
    reports: dict[str, dict[str, Any]],
    training_reports: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    evidence = {}
    for method, report in reports.items():
        if report.get("status") != "completed":
            raise ValueError(f"{method} formal generation metrics are incomplete")
        if int(report.get("counts", {}).get("generated_image_count", -1)) != 50_000:
            raise ValueError(f"{method} formal generated sample count is not 50000")
        provenance = report.get("sample_provenance", {})
        progress = provenance.get("sampling_progress", {})
        inference_api = provenance.get("sampling", {}).get("inference_api", {})
        if int(provenance.get("checkpoint_step", -1)) != 300_000:
            raise ValueError(f"{method} formal samples do not use the 300K checkpoint")
        if provenance.get("weights") != "ema":
            raise ValueError(f"{method} formal samples do not use EMA weights")
        if len(str(provenance.get("checkpoint_sha256", ""))) != 64:
            raise ValueError(f"{method} checkpoint SHA256 is malformed")
        if len(str(provenance.get("sample_set_sha256", ""))) != 64:
            raise ValueError(f"{method} sample-set SHA256 is malformed")
        if not str(provenance.get("checkpoint_integrity_manifest", "")).endswith(
            ".pt.integrity.json"
        ):
            raise ValueError(f"{method} checkpoint integrity manifest is missing")
        latest = training_reports[method].get("latest_checkpoint", {})
        if latest.get("checkpoint_sha256") != provenance["checkpoint_sha256"]:
            raise ValueError(f"{method} training and sampling checkpoint SHA256 differ")
        if Path(str(latest.get("integrity_manifest", ""))).name != Path(
            str(provenance["checkpoint_integrity_manifest"])
        ).name:
            raise ValueError(f"{method} training and sampling integrity manifests differ")
        if progress.get("status") != "completed":
            raise ValueError(f"{method} formal sampling progress is incomplete")
        if inference_api != {
            "name": "cofitok.generation.GenerationSession",
            "version": 1,
        }:
            raise ValueError(f"{method} formal sampling bypassed the stable inference API")
        if int(progress.get("completed_samples", -1)) != 50_000:
            raise ValueError(f"{method} formal sampling progress count mismatch")
        elapsed = float(progress.get("cumulative_elapsed_seconds", math.nan))
        if not math.isfinite(elapsed) or elapsed <= 0.0:
            raise ValueError(f"{method} formal sampling elapsed time is invalid")
        evidence[method] = {
            "checkpoint_sha256": provenance["checkpoint_sha256"],
            "sample_set_sha256": provenance["sample_set_sha256"],
            "sampling_elapsed_seconds": elapsed,
        }
    return evidence


def _sampling_runtime_selection_evidence(
    selection: dict[str, Any],
    generation_reports: dict[str, dict[str, Any]],
    *,
    expected_revision: str,
) -> dict[str, Any]:
    if selection.get("status") != "selected":
        raise ValueError("formal sampling runtime selection is incomplete")
    if selection.get("git_revision") != expected_revision:
        raise ValueError("sampling runtime selection revision differs from full training")
    policy = selection.get("policy", {})
    if policy.get("shared_candidate_required") is not True:
        raise ValueError("sampling runtime selection is not shared")
    if policy.get("batch_size_invariant_random_stream_required") is not True:
        raise ValueError("sampling selection omits batch-invariant random streams")
    batch_size = int(selection.get("selected", {}).get("batch_size", -1))
    if batch_size < 1:
        raise ValueError("selected formal sampling batch is invalid")
    identities = selection.get("checkpoints", {})
    for method in ("cofitok", "dense_identity"):
        provenance = generation_reports[method].get("sample_provenance", {})
        if identities.get(method, {}).get("sha256") != provenance.get(
            "checkpoint_sha256"
        ):
            raise ValueError(f"{method} sampling selection checkpoint SHA256 differs")
        if int(identities.get(method, {}).get("step", -1)) != 300_000:
            raise ValueError(f"{method} sampling selection checkpoint is not step 300K")
        if int(provenance.get("sampling", {}).get("batch_size", -1)) != batch_size:
            raise ValueError(f"{method} formal generation ignored selected sampling batch")
        random_stream = provenance.get("sampling", {}).get("random_stream", {})
        if random_stream.get("batch_size_invariant") is not True:
            raise ValueError(f"{method} formal sampling random stream is batch-dependent")
    return {
        "batch_size": batch_size,
        "estimated_speedup_over_baseline": float(
            selection["selected"]["estimated_speedup_over_baseline"]
        ),
        "checkpoint_sha256": {
            method: identities[method]["sha256"]
            for method in ("cofitok", "dense_identity")
        },
    }


def _final_gate_evidence(
    gate: dict[str, Any], generation_reports: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    evidence = _gate_evidence(
        gate,
        stage="full",
        decision="large_scale_generation_ready",
    )
    matches = [
        row.get("evidence", {})
        for row in gate.get("gates", [])
        if row.get("name") == "matched_sampling_provenance"
    ]
    if len(matches) != 1:
        raise ValueError("final gate lacks unique matched sampling provenance")
    gate_sampling = matches[0]
    for method, prefix in (("cofitok", "cofitok"), ("dense_identity", "dense")):
        provenance = generation_reports[method]["sample_provenance"]
        if gate_sampling.get(f"{prefix}_checkpoint_sha256") != provenance.get(
            "checkpoint_sha256"
        ):
            raise ValueError(f"final gate {method} checkpoint SHA256 differs")
        if gate_sampling.get(f"{prefix}_sample_set_sha256") != provenance.get(
            "sample_set_sha256"
        ):
            raise ValueError(f"final gate {method} sample-set SHA256 differs")
    evidence["sampling_provenance_bound"] = True
    return evidence


def _comparison_evidence(
    report: dict[str, Any], generation_reports: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    if report.get("status") != "ready":
        raise ValueError("large-scale comparison is not ready")
    rows = report.get("matched_training_rows", [])
    if len(rows) != 2 or any(int(row.get("sample_count", -1)) != 50_000 for row in rows):
        raise ValueError("large-scale comparison lacks the matched 50K pair")
    if len(report.get("official_context_rows", [])) != 3:
        raise ValueError("large-scale comparison lacks official contextual rows")
    indexed = {row.get("method"): row for row in rows}
    expected = {
        "CoFiTok K=8": generation_reports["cofitok"]["sample_provenance"],
        "Dense identity": generation_reports["dense_identity"]["sample_provenance"],
    }
    if set(indexed) != set(expected):
        raise ValueError("large-scale comparison method identities differ")
    for method, provenance in expected.items():
        row = indexed[method]
        if row.get("checkpoint_sha256") != provenance.get("checkpoint_sha256"):
            raise ValueError(f"comparison {method} checkpoint SHA256 differs")
        if row.get("sample_set_sha256") != provenance.get("sample_set_sha256"):
            raise ValueError(f"comparison {method} sample-set SHA256 differs")
    return {
        "matched_methods": [row["method"] for row in rows],
        "official_context_count": 3,
    }


def build_completion_audit(
    *,
    expected_10pct_revision: str,
    expected_full_revision: str,
    cofitok_10pct_training: dict[str, Any] | None,
    dense_10pct_training: dict[str, Any] | None,
    scaling_gate: dict[str, Any] | None,
    cofitok_full_training: dict[str, Any] | None,
    dense_full_training: dict[str, Any] | None,
    cofitok_training_audit: dict[str, Any] | None,
    dense_training_audit: dict[str, Any] | None,
    runtime_selection: dict[str, Any] | None,
    sampling_runtime_selection: dict[str, Any] | None,
    milestones: dict[int, dict[str, Any] | None],
    cofitok_generation: dict[str, Any] | None,
    dense_generation: dict[str, Any] | None,
    final_gate: dict[str, Any] | None,
    comparison: dict[str, Any] | None,
) -> dict[str, Any]:
    if len(expected_10pct_revision) != 40 or len(expected_full_revision) != 40:
        raise ValueError("completion audit requires full 40-character revisions")

    checks = []
    checks.append(
        _check(
            "ten_percent_matched_training",
            [cofitok_10pct_training, dense_10pct_training],
            lambda: validate_training_pair(
                cofitok_10pct_training,
                dense_10pct_training,
                expected_steps=50_000,
                expected_revision=expected_10pct_revision,
            ),
        )
    )
    checks.append(
        _check(
            "scaling_promotion_gate",
            [scaling_gate],
            lambda: _gate_evidence(
                scaling_gate,
                stage="scaling",
                decision="promote_to_full_imagenet256",
            ),
        )
    )
    checks.append(
        _check(
            "full_matched_training",
            [cofitok_full_training, dense_full_training],
            lambda: validate_training_pair(
                cofitok_full_training,
                dense_full_training,
                expected_steps=300_000,
                expected_revision=expected_full_revision,
                expected_dataset="imagenet_256",
            ),
        )
    )
    checks.append(
        _check(
            "full_runtime_selection",
            [runtime_selection, cofitok_full_training, dense_full_training],
            lambda: _runtime_selection_evidence(
                runtime_selection,
                {
                    "cofitok": cofitok_full_training,
                    "dense_identity": dense_full_training,
                },
                expected_revision=expected_full_revision,
            ),
        )
    )
    checks.append(
        _check(
            "full_training_audits",
            [cofitok_training_audit, dense_training_audit],
            lambda: _training_audit_evidence(
                {
                    "cofitok": cofitok_training_audit,
                    "dense_identity": dense_training_audit,
                }
            ),
        )
    )

    milestone_payloads = [milestones.get(step) for step in MILESTONE_STEPS]
    milestone_warnings: list[str] = []

    def validate_milestones() -> dict[str, Any]:
        evidence, warnings = _milestone_evidence(
            {step: milestones[step] for step in MILESTONE_STEPS}
        )
        milestone_warnings.extend(warnings)
        return evidence

    checks.append(
        _check("full_milestone_evaluations", milestone_payloads, validate_milestones)
    )
    checks.append(
        _check(
            "formal_50k_generation",
            [
                cofitok_generation,
                dense_generation,
                cofitok_full_training,
                dense_full_training,
            ],
            lambda: _generation_evidence(
                {"cofitok": cofitok_generation, "dense_identity": dense_generation},
                {
                    "cofitok": cofitok_full_training,
                    "dense_identity": dense_full_training,
                },
            ),
        )
    )
    checks.append(
        _check(
            "formal_sampling_runtime_selection",
            [sampling_runtime_selection, cofitok_generation, dense_generation],
            lambda: _sampling_runtime_selection_evidence(
                sampling_runtime_selection,
                {"cofitok": cofitok_generation, "dense_identity": dense_generation},
                expected_revision=expected_full_revision,
            ),
        )
    )
    checks.append(
        _check(
            "final_generation_gate",
            [final_gate],
            lambda: _final_gate_evidence(
                final_gate,
                {"cofitok": cofitok_generation, "dense_identity": dense_generation},
            ),
        )
    )
    checks.append(
        _check(
            "final_comparison_report",
            [comparison],
            lambda: _comparison_evidence(
                comparison,
                {"cofitok": cofitok_generation, "dense_identity": dense_generation},
            ),
        )
    )

    failed = [row["name"] for row in checks if row["status"] == "fail"]
    missing = [row["name"] for row in checks if row["status"] == "missing"]
    complete = not failed and not missing
    return {
        "schema_version": 1,
        "status": "complete" if complete else ("failed" if failed else "in_progress"),
        "complete": complete,
        "expected_revisions": {
            "ten_percent_training": expected_10pct_revision,
            "full_training": expected_full_revision,
        },
        "checks": checks,
        "failed_checks": failed,
        "missing_checks": missing,
        "warnings": milestone_warnings,
    }


def _read_optional(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit end-to-end completion of large-scale CoFiTok generation."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--output-root",
        default="/root/autodl-tmp/CoFiTok/checkpoints/generation",
    )
    parser.add_argument("--expected-10pct-revision", default=PINNED_10PCT_REVISION)
    parser.add_argument("--expected-full-revision", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    project = Path(args.project_root).resolve()
    output_root = Path(args.output_root).resolve()
    report_root = project / "artifacts/reports/generation"
    ten_root = report_root / "imagenet256_10pct_matched_50k_2026-07-12"
    full_root = report_root / "imagenet256_full_matched_300k"
    cofitok_10 = output_root / "imagenet256_10pct_cofitok_k8_50k_2026-07-12"
    dense_10 = output_root / "imagenet256_10pct_dense_50k_2026-07-12"
    cofitok_full = output_root / "imagenet256_full_cofitok_k8_300k"
    dense_full = output_root / "imagenet256_full_dense_300k"

    audit = build_completion_audit(
        expected_10pct_revision=args.expected_10pct_revision,
        expected_full_revision=args.expected_full_revision,
        cofitok_10pct_training=_read_optional(cofitok_10 / "training_report.json"),
        dense_10pct_training=_read_optional(dense_10 / "training_report.json"),
        scaling_gate=_read_optional(ten_root / "promotion_gate.json"),
        cofitok_full_training=_read_optional(cofitok_full / "training_report.json"),
        dense_full_training=_read_optional(dense_full / "training_report.json"),
        cofitok_training_audit=_read_optional(full_root / "cofitok_training_audit.json"),
        dense_training_audit=_read_optional(full_root / "dense_training_audit.json"),
        runtime_selection=_read_optional(full_root / "runtime_selection.json"),
        sampling_runtime_selection=_read_optional(
            full_root / "sampling_runtime_selection.json"
        ),
        milestones={
            step: _read_optional(full_root / "milestones" / f"step_{step:08d}.json")
            for step in MILESTONE_STEPS
        },
        cofitok_generation=_read_optional(
            cofitok_full
            / "samples_50k_ddim250_cfg15/metrics/generation_metrics_report.json"
        ),
        dense_generation=_read_optional(
            dense_full
            / "samples_50k_ddim250_cfg15/metrics/generation_metrics_report.json"
        ),
        final_gate=_read_optional(full_root / "final_generation_gate.json"),
        comparison=_read_optional(
            full_root / "comparison/large_scale_generation_comparison.json"
        ),
    )
    write_json_report(args.output, audit)
    print(json.dumps(audit, indent=2, sort_keys=True))
    if not audit["complete"] and not args.allow_incomplete:
        raise SystemExit("large-scale generation completion audit did not pass")


if __name__ == "__main__":
    main()
