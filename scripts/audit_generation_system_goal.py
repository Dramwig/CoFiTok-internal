"""Fail-closed, read-only audit for the large-model generation-system goal.

This auditor deliberately sits above the older scoped/MVP goal audit.  It
checks the evidence categories required for a usable generation system:
training scale, sample quality, a fair direct baseline comparison, stable
inference, and physically reproducible checkpoints.  It never launches a
stage, sends a process signal, or changes an upstream decision.  A hold in a
source report remains a hold in the aggregate result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Mapping

try:
    from cofitok.path_security import reject_symlink_chain
except ModuleNotFoundError:  # pragma: no cover - compatibility with pinned older checkouts
    def reject_symlink_chain(path: str | Path, *, name: str) -> Path:
        """Keep the auditor source-compatible with pre-hardening checkouts."""
        absolute = Path(os.path.abspath(Path(path).expanduser()))
        current = absolute
        while True:
            if current.is_symlink():
                raise ValueError(f"{name} path must not contain a symlink: {current}")
            if current.parent == current:
                break
            current = current.parent
        return absolute
from cofitok.reporting import file_sha256, write_json_report
from cofitok.generation.protocol import sampling_protocol_contract
from cofitok.training.checkpointing import checkpoint_integrity_path, verify_training_checkpoint


SCHEMA_VERSION = 1
DEFAULT_TARGET_STEPS = 300_000
DEFAULT_EFFECTIVE_BATCH = 64
DEFAULT_SAMPLE_COUNT = 50_000
DEFAULT_SAMPLE_STEPS = 250
DEFAULT_COFiTOK_RUN = "cofitok_rgbtail3_rollout_x0_u2_ema_teacher"
DEFAULT_DENSE_RUN = "dense_rollout_x0_u2_ema_teacher"
DEFAULT_QUALITY_RESULT = "reports/quality_bridge_result.json"
DEFAULT_COMPARISON = (
    "reports/quality_bridge_comparison_v4_authoritative/"
    "quality_bridge_comparison.json"
)
DEFAULT_TERMINAL_AUDIT = (
    "reports/terminal_completion_audit_v4_authoritative/"
    "terminal_completion_audit.json"
)


def _read_json(path: str | Path, *, name: str) -> dict[str, Any]:
    safe = reject_symlink_chain(path, name=name)
    if not safe.is_file():
        raise FileNotFoundError(f"{name} is missing: {safe}")
    try:
        payload = json.loads(safe.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} is not valid JSON: {safe}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a JSON object: {safe}")
    return payload


def _identity(path: str | Path, *, name: str) -> dict[str, Any]:
    safe = reject_symlink_chain(path, name=name)
    if not safe.is_file():
        raise FileNotFoundError(f"{name} is missing: {safe}")
    return {
        "path": safe.resolve().as_posix(),
        "bytes": safe.stat().st_size,
        "sha256": file_sha256(safe),
    }


def _source_identity(descriptor: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    if set(descriptor) != {"path", "bytes", "sha256"}:
        raise ValueError(f"{name} descriptor fields differ")
    path = reject_symlink_chain(str(descriptor.get("path", "")), name=name)
    actual = _identity(path, name=name)
    try:
        expected_bytes = int(descriptor["bytes"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} byte count is malformed") from error
    expected_sha = str(descriptor["sha256"])
    if expected_bytes != actual["bytes"] or expected_sha != actual["sha256"]:
        raise ValueError(f"{name} identity differs")
    return actual


def _source_identities(sources: Any, *, name: str) -> dict[str, dict[str, Any]]:
    """Rehash every declared source instead of silently dropping bad entries."""
    if not isinstance(sources, Mapping) or not sources:
        raise ValueError(f"{name} source_reports are missing")
    identities: dict[str, dict[str, Any]] = {}
    for key, descriptor in sources.items():
        if not isinstance(descriptor, Mapping):
            raise ValueError(f"{name} source {key} descriptor is malformed")
        identities[str(key)] = _source_identity(
            descriptor,
            name=f"{name} source {key}",
        )
    return identities


def _normalized_method_label(value: Any) -> str:
    """Normalize display labels without changing the source comparison row."""
    return "".join(char.lower() for char in str(value) if char.isalnum())


def _direct_method_role(row: Mapping[str, Any]) -> str | None:
    """Classify a matched direct row from structural fields and its label.

    The authoritative comparison uses display labels such as ``CoFiTok K=8``
    and ``Dense identity``.  The tier/comparability fields are the primary
    provenance, while the normalized label identifies which of the two
    expected matched methods the row represents.
    """
    if str(row.get("comparison_tier")) != "matched_training_direct":
        return None
    directly_comparable = row.get("directly_comparable_to_cofitok")
    if directly_comparable is not True:
        return None
    label = " ".join(
        _normalized_method_label(row.get(key, ""))
        for key in ("method", "alias", "name", "model")
    )
    if "cofitok" in label and "dense" not in label:
        return "cofitok"
    if "dense" in label and ("identity" in label or "dense" in label):
        return "dense_identity"
    return None


def _check(
    name: str,
    validator: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    try:
        evidence = validator()
    except (AttributeError, FileNotFoundError, KeyError, OSError, TypeError, ValueError) as error:
        return {"name": name, "status": "fail", "error": str(error)}
    if not isinstance(evidence, dict):
        return {"name": name, "status": "fail", "error": "validator returned a non-object"}
    status = evidence.pop("status", "pass")
    if status not in {"pass", "hold", "incomplete", "missing", "fail"}:
        return {"name": name, "status": "fail", "error": f"invalid validator status: {status}"}
    return {"name": name, "status": status, "evidence": evidence}


def aggregate_checks(checks: list[dict[str, Any]]) -> dict[str, Any]:
    failed = [row["name"] for row in checks if row.get("status") == "fail"]
    missing = [
        row["name"]
        for row in checks
        if row.get("status") in {"missing", "incomplete"}
    ]
    held = [row["name"] for row in checks if row.get("status") == "hold"]
    complete = not failed and not missing and not held
    status = "pass" if complete else ("failed" if failed else "hold")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "complete": complete,
        "failed_checks": failed,
        "incomplete_checks": missing,
        "held_checks": held,
        "checks": checks,
    }


def _metrics_rows(path: Path, *, effective_batch: int) -> dict[str, Any]:
    safe = reject_symlink_chain(path, name="training metrics")
    if not safe.is_file():
        raise FileNotFoundError(f"training metrics is missing: {safe}")
    previous_step = -1
    rows = 0
    last: dict[str, Any] | None = None
    with safe.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"training metrics line {line_number} is invalid") from error
            if not isinstance(row, dict):
                raise ValueError(f"training metrics line {line_number} is not an object")
            try:
                step = int(row["step"])
                samples_seen = int(row["samples_seen"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"training metrics line {line_number} lacks integer step accounting") from error
            if step <= previous_step:
                raise ValueError("training metrics steps are not strictly increasing")
            if samples_seen != step * effective_batch:
                raise ValueError(
                    f"training metrics samples_seen mismatch at step {step}: "
                    f"{samples_seen} != {step * effective_batch}"
                )
            for key, value in row.items():
                if isinstance(value, float) and not math.isfinite(value):
                    raise ValueError(f"training metrics value {key} is not finite at step {step}")
            previous_step = step
            rows += 1
            last = row
    if last is None:
        raise ValueError("training metrics is empty")
    return {
        "path": safe.resolve().as_posix(),
        "bytes": safe.stat().st_size,
        "sha256": file_sha256(safe),
        "row_count": rows,
        "last_step": previous_step,
        "last_samples_seen": int(last["samples_seen"]),
    }


def training_scale_evidence(
    system_root: str | Path,
    *,
    target_steps: int,
    effective_batch: int,
    cofitok_run: str = DEFAULT_COFiTOK_RUN,
    dense_run: str = DEFAULT_DENSE_RUN,
) -> dict[str, Any]:
    if target_steps < 1 or effective_batch < 1:
        raise ValueError("target_steps and effective_batch must be positive")
    root = reject_symlink_chain(system_root, name="generation system root")
    methods: dict[str, Any] = {}
    for method, run_name in (("cofitok", cofitok_run), ("dense_identity", dense_run)):
        run = reject_symlink_chain(root / run_name, name=f"{method} run")
        report_path = run / "training_report.json"
        latest_path = run / "latest.json"
        report = _read_json(report_path, name=f"{method} training report")
        latest = _read_json(latest_path, name=f"{method} latest pointer")
        metrics = _metrics_rows(run / "train_metrics.jsonl", effective_batch=effective_batch)
        completed = int(report.get("completed_steps", -1))
        latest_step = int(latest.get("step", -1))
        if report.get("training_complete") is not True or completed != target_steps:
            return {
                "status": "incomplete",
                "target_steps": target_steps,
                "method": method,
                "observed_completed_steps": completed,
                "latest_step": latest_step,
                "metrics": metrics,
            }
        if latest_step != target_steps or metrics["last_step"] != target_steps:
            raise ValueError(f"{method} latest/metrics do not reach target steps")
        if metrics["last_samples_seen"] != target_steps * effective_batch:
            raise ValueError(f"{method} final samples_seen is not step*effective_batch")
        git = report.get("git")
        if not isinstance(git, Mapping) or git.get("dirty") is not False:
            raise ValueError(f"{method} training Git provenance is not clean")
        methods[method] = {
            "training_report": _identity(report_path, name=f"{method} training report"),
            "latest": _identity(latest_path, name=f"{method} latest pointer"),
            "completed_steps": completed,
            "samples_seen": metrics["last_samples_seen"],
            "metric_rows": metrics["row_count"],
            "git": dict(git),
        }
    revisions = {row["git"].get("revision") for row in methods.values()}
    branches = {row["git"].get("branch") for row in methods.values()}
    if len(revisions) != 1 or len(branches) != 1:
        raise ValueError("matched methods do not share one training revision and branch")
    return {
        "status": "pass",
        "target_steps": target_steps,
        "effective_batch": effective_batch,
        "methods": methods,
        "shared_revision": next(iter(revisions)),
        "shared_branch": next(iter(branches)),
    }


def sample_quality_evidence(
    system_root: str | Path,
    *,
    quality_result_path: str | Path | None = None,
) -> dict[str, Any]:
    root = reject_symlink_chain(system_root, name="generation system root")
    path = (
        reject_symlink_chain(quality_result_path, name="quality bridge result")
        if quality_result_path is not None
        else root / DEFAULT_QUALITY_RESULT
    )
    result = _read_json(path, name="quality bridge result")
    screen = result.get("quality_screen")
    if not isinstance(screen, Mapping):
        raise ValueError("quality bridge result lacks quality_screen")
    failed_checks = screen.get("failed_checks", [])
    if not isinstance(failed_checks, list):
        raise ValueError("quality_screen.failed_checks is malformed")
    checks = screen.get("checks")
    if not isinstance(checks, list) or not checks:
        raise ValueError("quality_screen.checks are missing")
    if not isinstance(screen.get("thresholds"), Mapping):
        raise ValueError("quality_screen.thresholds are missing")
    if screen.get("non_authorizing") is not True:
        raise ValueError("quality_screen is not marked non-authorizing")
    observed_failed: list[str] = []
    seen_names: set[str] = set()

    def _check_finite(value: Any, *, label: str) -> None:
        if isinstance(value, bool):
            return
        if isinstance(value, (int, float)):
            if not math.isfinite(float(value)):
                raise ValueError(f"{label} is not finite")
            return
        if isinstance(value, Mapping):
            for key, nested in value.items():
                _check_finite(nested, label=f"{label}.{key}")
            return
        if isinstance(value, list):
            for index, nested in enumerate(value):
                _check_finite(nested, label=f"{label}[{index}]")

    for index, check in enumerate(checks):
        if not isinstance(check, Mapping):
            raise ValueError(f"quality_screen check {index} is malformed")
        name = check.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"quality_screen check {index} name is malformed")
        if name in seen_names:
            raise ValueError(f"quality_screen check {name} is duplicated")
        seen_names.add(name)
        if not isinstance(check.get("passed"), bool):
            raise ValueError(f"quality_screen check {name} passed flag is malformed")
        if "observed" not in check or "threshold" not in check:
            raise ValueError(f"quality_screen check {name} lacks observed/threshold")
        _check_finite(check["observed"], label=f"quality_screen check {name} observed")
        _check_finite(check["threshold"], label=f"quality_screen check {name} threshold")
        if check["passed"] is False:
            observed_failed.append(name)
    if failed_checks != observed_failed:
        raise ValueError("quality_screen.failed_checks do not match check outcomes")
    screen_status = screen.get("status")
    if screen_status not in {"pass", "hold"}:
        raise ValueError("quality_screen.status is malformed")
    expected_screen_status = "pass" if not observed_failed else "hold"
    if screen_status != expected_screen_status:
        raise ValueError("quality_screen.status does not match check outcomes")
    source_identities = _source_identities(
        result.get("source_reports"),
        name="quality result",
    )
    evidence = {
        "quality_result": _identity(path, name="quality bridge result"),
        "quality_screen_status": screen.get("status"),
        "quality_check_count": len(checks),
        "failed_checks": list(failed_checks),
        "source_reports_verified": len(source_identities),
        "source_reports": source_identities,
        "non_authorizing": result.get("authorization_boundary", {}).get(
            "quality_bridge_execution_allowed"
        ) is False,
    }
    if result.get("status") != "completed":
        evidence["reason"] = "quality bridge result is not completed"
        evidence["status"] = "hold"
    elif screen.get("status") != "pass" or failed_checks:
        evidence["reason"] = "one or more predeclared sample-quality checks remain held"
        evidence["status"] = "hold"
    else:
        evidence["status"] = "pass"
    return evidence


def baseline_fairness_evidence(
    system_root: str | Path,
    *,
    comparison_path: str | Path | None = None,
) -> dict[str, Any]:
    root = reject_symlink_chain(system_root, name="generation system root")
    path = (
        reject_symlink_chain(comparison_path, name="baseline comparison")
        if comparison_path is not None
        else root / DEFAULT_COMPARISON
    )
    comparison = _read_json(path, name="baseline comparison")
    rows = comparison.get("matched_training_rows")
    policy = comparison.get("comparison_policy")
    context_rows = comparison.get("official_context_rows")
    if not isinstance(rows, list) or len(rows) != 2:
        raise ValueError("baseline comparison must contain exactly two matched rows")
    if not isinstance(policy, Mapping):
        raise ValueError("baseline comparison policy is missing")
    direct_roles: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("matched direct rows must be objects")
        role = _direct_method_role(row)
        if role is None:
            raise ValueError("matched direct row lacks CoFiTok/dense identity provenance")
        if role in direct_roles:
            raise ValueError(f"duplicate matched direct method role: {role}")
        direct_roles[role] = row
    if set(direct_roles) != {"cofitok", "dense_identity"}:
        raise ValueError("matched direct rows are not exactly one CoFiTok and one dense identity row")
    if policy.get("primary_direct_tier") != "matched_training_direct":
        raise ValueError("comparison does not identify the matched direct tier")
    if policy.get("cross_tier_numeric_ranking_allowed") is not False:
        raise ValueError("cross-tier numeric ranking is not disabled")
    if not isinstance(context_rows, list):
        raise ValueError("official contextual rows are missing")
    for row in context_rows:
        if not isinstance(row, Mapping):
            raise ValueError("official contextual rows must be objects")
        if row.get("comparison_tier") != "official_pretrained_contextual":
            raise ValueError("official contextual row has the wrong comparison tier")
        if row.get("directly_comparable_to_cofitok") is not False:
            raise ValueError("official contextual row is marked directly comparable")
    source_identities = _source_identities(
        comparison.get("source_reports"),
        name="baseline comparison",
    )
    return {
        "status": "pass",
        "comparison": _identity(path, name="baseline comparison"),
        "direct_methods": sorted(str(row.get("method")) for row in rows),
        "direct_method_roles": {
            role: str(row.get("method")) for role, row in sorted(direct_roles.items())
        },
        "direct_row_count": len(rows),
        "official_context_row_count": len(context_rows),
        "cross_tier_numeric_ranking_allowed": False,
        "source_reports_verified": len(source_identities),
        "source_reports": source_identities,
        "scientific_status": comparison.get("terminal_status", comparison.get("status")),
    }


def stable_inference_evidence(
    system_root: str | Path,
    *,
    expected_sample_count: int,
    expected_sample_steps: int,
    cofitok_run: str = DEFAULT_COFiTOK_RUN,
    dense_run: str = DEFAULT_DENSE_RUN,
) -> dict[str, Any]:
    if expected_sample_count < 1 or expected_sample_steps < 1:
        raise ValueError("expected sample count and sample steps must be positive")
    root = reject_symlink_chain(system_root, name="generation system root")
    methods: dict[str, Any] = {}
    for method, run_name, prefix in (
        ("cofitok", cofitok_run, "8"),
        ("dense_identity", dense_run, "1"),
    ):
        sample_root = root / run_name / "terminal_100k" / "samples_10000_ddim100_cfg15"
        preflight = _read_json(
            root / run_name / "terminal_100k" / "sampling_preflight.json",
            name=f"{method} sampling preflight",
        )
        report = _read_json(sample_root / "sampling_report.json", name=f"{method} sampling report")
        progress = _read_json(sample_root / "sampling_progress.json", name=f"{method} sampling progress")
        metrics = _read_json(
            sample_root / "metrics" / "generation_metrics_report.json",
            name=f"{method} generation metrics",
        )
        sampling = report.get("sampling")
        if not isinstance(sampling, Mapping):
            raise ValueError(f"{method} sampling report lacks protocol")
        actual_count = int(sampling.get("num_samples", -1))
        actual_steps = int(sampling.get("sample_steps", -1))
        if report.get("status") != "completed" or progress.get("status") != "completed":
            raise ValueError(f"{method} sampling is not completed")
        if actual_count != expected_sample_count or actual_steps != expected_sample_steps:
            return {
                "status": "hold",
                "reason": f"{method} sampling protocol is {actual_count} samples/DDIM-{actual_steps}, "
                f"expected {expected_sample_count}/DDIM-{expected_sample_steps}",
                "method": method,
                "observed_sample_count": actual_count,
                "observed_sample_steps": actual_steps,
            }
        protocol = sampling_protocol_contract(
            dict(sampling),
            stage="scaling",
            expected_num_train_timesteps=1_000,
        )
        if not protocol["valid"]:
            raise ValueError(
                f"{method} sampling protocol is not the formal matched contract: "
                f"{', '.join(protocol['issues'])}"
            )
        if int(progress.get("completed_samples", -1)) != expected_sample_count:
            raise ValueError(f"{method} sampling progress count differs")
        if int(progress.get("total_samples", -1)) != expected_sample_count:
            raise ValueError(f"{method} sampling progress budget differs")
        progress_sets = progress.get("sample_sets")
        report_sets = report.get("sample_sets")
        if not isinstance(progress_sets, Mapping) or not isinstance(report_sets, Mapping):
            raise ValueError(f"{method} sampling progress/sample report sets are missing")
        progress_set = progress_sets.get(prefix)
        report_set = report_sets.get(prefix)
        if not isinstance(progress_set, Mapping) or not isinstance(report_set, Mapping):
            raise ValueError(f"{method} sampling progress/sample report prefix is missing")
        if int(progress_set.get("count", -1)) != actual_count:
            raise ValueError(f"{method} sampling progress prefix count differs")
        if int(report_set.get("count", -1)) != actual_count:
            raise ValueError(f"{method} sampling report prefix count differs")
        if progress_set.get("count") != report_set.get("count"):
            raise ValueError(f"{method} sampling progress count binding differs")
        if progress_set.get("sha256") != report_set.get("sha256"):
            raise ValueError(f"{method} sampling progress digest binding differs")
        sample_digest = report_set.get("sha256")
        if (
            not isinstance(sample_digest, str)
            or len(sample_digest) != 64
            or any(char not in "0123456789abcdefABCDEF" for char in sample_digest)
        ):
            raise ValueError(f"{method} sample-set digest is malformed")
        if sampling.get("prefix_budgets") != [int(prefix)]:
            raise ValueError(f"{method} sampling prefix budget is not the formal matched contract")
        random_stream = sampling.get("random_stream")
        if not isinstance(random_stream, Mapping) or not all(
            random_stream.get(key) is True
            for key in ("batch_size_invariant", "prefix_budgets_share_stream", "resume_index_invariant")
        ):
            raise ValueError(f"{method} sampling random-stream invariants are incomplete")
        if (
            random_stream.get("scope") != "per_global_sample_index"
            or random_stream.get("seed_formula") != "(seed + global_index) mod 2^63"
        ):
            raise ValueError(f"{method} sampling random-stream contract is incomplete")
        sample_digest_contract = sampling.get("sample_set_digest")
        if sample_digest_contract != {
            "algorithm": "sha256",
            "framing": "filename_utf8_nul_file_bytes_nul",
        }:
            raise ValueError(f"{method} sampling sample-set digest contract is incomplete")
        manifest_path = sample_root / "sampling_manifest.json"
        manifest = _read_json(manifest_path, name=f"{method} sampling manifest")
        manifest_identity = _identity(manifest_path, name=f"{method} sampling manifest")
        if report.get("sampling_manifest_sha256") != manifest_identity["sha256"]:
            raise ValueError(f"{method} sampling report manifest binding differs")
        if progress.get("sampling_manifest_sha256") != manifest_identity["sha256"]:
            raise ValueError(f"{method} sampling progress manifest binding differs")
        if manifest.get("sampling") != dict(sampling):
            raise ValueError(f"{method} sampling manifest protocol differs")
        for key in ("checkpoint_sha256", "runtime_environment_sha256", "checkpoint_step", "weights"):
            if manifest.get(key) != report.get(key):
                raise ValueError(f"{method} sampling manifest {key} binding differs")
        metrics_sample_provenance = metrics.get("sample_provenance")
        if not isinstance(metrics_sample_provenance, Mapping):
            raise ValueError(f"{method} generation metrics lacks sample_provenance")
        for payload, label in ((preflight, "preflight"), (report, "report")):
            if payload.get("checkpoint_sha256") != report.get("checkpoint_sha256"):
                raise ValueError(f"{method} {label} checkpoint binding differs")
            if payload.get("runtime_environment_sha256") != report.get("runtime_environment_sha256"):
                raise ValueError(f"{method} {label} runtime binding differs")
        if metrics_sample_provenance.get("checkpoint_sha256") != report.get("checkpoint_sha256"):
            raise ValueError(f"{method} metrics checkpoint binding differs")
        if metrics_sample_provenance.get("runtime_environment_sha256") != report.get(
            "runtime_environment_sha256"
        ):
            raise ValueError(f"{method} metrics sampling runtime binding differs")
        manifest_identity_in_metrics = metrics_sample_provenance.get("manifest_identity")
        if not isinstance(manifest_identity_in_metrics, Mapping):
            raise ValueError(f"{method} metrics sampling manifest identity is missing")
        if manifest_identity_in_metrics.get("sha256") != manifest_identity["sha256"]:
            raise ValueError(f"{method} metrics sampling manifest binding differs")
        metrics_protocol = metrics_sample_provenance.get("sampling_protocol_contract")
        if (
            not isinstance(metrics_protocol, Mapping)
            or metrics_protocol.get("valid") is not True
            or metrics_protocol.get("stage") != "scaling"
            or metrics_protocol.get("issues") != []
        ):
            raise ValueError(f"{method} metrics sampling protocol contract is not valid")
        sample_sets = report.get("sample_sets")
        if not isinstance(sample_sets, Mapping) or not isinstance(sample_sets.get(prefix), Mapping):
            raise ValueError(f"{method} sample-set digest is missing")
        methods[method] = {
            "sample_count": actual_count,
            "sample_steps": actual_steps,
            "checkpoint_sha256": report.get("checkpoint_sha256"),
            "runtime_environment_sha256": report.get("runtime_environment_sha256"),
            "evaluator_runtime_environment_sha256": metrics.get("runtime_environment_sha256"),
            "sample_set_sha256": sample_digest,
            "sampling_manifest_sha256": manifest_identity["sha256"],
            "sampling_protocol_issues": list(protocol["issues"]),
            "progress_status": progress.get("status"),
        }
    return {"status": "pass", "expected_sample_count": expected_sample_count, "expected_sample_steps": expected_sample_steps, "methods": methods}


def checkpoint_reproducibility_evidence(
    system_root: str | Path,
    *,
    target_steps: int,
    cofitok_run: str = DEFAULT_COFiTOK_RUN,
    dense_run: str = DEFAULT_DENSE_RUN,
) -> dict[str, Any]:
    root = reject_symlink_chain(system_root, name="generation system root")
    methods: dict[str, Any] = {}
    for method, run_name in (("cofitok", cofitok_run), ("dense_identity", dense_run)):
        run = reject_symlink_chain(root / run_name, name=f"{method} run")
        latest = _read_json(run / "latest.json", name=f"{method} latest pointer")
        training_report_path = run / "training_report.json"
        training_report = _read_json(
            training_report_path,
            name=f"{method} training report",
        )
        checkpoint_name = str(latest.get("checkpoint", ""))
        if not checkpoint_name or Path(checkpoint_name).name != checkpoint_name:
            raise ValueError(f"{method} latest checkpoint name is not local")
        checkpoint = reject_symlink_chain(run / checkpoint_name, name=f"{method} checkpoint")
        integrity = verify_training_checkpoint(checkpoint)
        if int(integrity.get("step", -1)) != int(latest.get("step", -2)):
            raise ValueError(f"{method} latest step differs from physical checkpoint")
        sidecar = checkpoint_integrity_path(checkpoint)
        expected_binding = {
            "checkpoint": checkpoint.name,
            "checkpoint_bytes": integrity["checkpoint_bytes"],
            "checkpoint_sha256": integrity["checkpoint_sha256"],
            "checkpoint_format_version": integrity["checkpoint_format_version"],
            "step": integrity["step"],
            "integrity_manifest": sidecar.name,
        }
        for binding_name, binding in (
            ("latest.json", latest),
            ("training_report.latest_checkpoint", training_report.get("latest_checkpoint")),
        ):
            if not isinstance(binding, Mapping):
                raise ValueError(f"{method} {binding_name} binding is missing")
            for key, expected in expected_binding.items():
                if binding.get(key) != expected:
                    raise ValueError(
                        f"{method} {binding_name} {key} does not match checkpoint integrity"
                    )
        if int(integrity.get("step", -1)) != target_steps:
            return {
                "status": "incomplete",
                "method": method,
                "target_steps": target_steps,
                "observed_step": int(integrity.get("step", -1)),
                "checkpoint": _identity(checkpoint, name=f"{method} checkpoint"),
            }
        if not sidecar.is_file():
            raise FileNotFoundError(f"{method} checkpoint sidecar is missing")
        methods[method] = {
            "checkpoint": _identity(checkpoint, name=f"{method} checkpoint"),
            "integrity_manifest": _identity(sidecar, name=f"{method} checkpoint sidecar"),
            "training_report": _identity(training_report_path, name=f"{method} training report"),
            "step": int(integrity["step"]),
            "physical_sha256_verified": True,
            "latest_binding": {
                key: latest.get(key)
                for key in expected_binding
            },
        }
    return {"status": "pass", "target_steps": target_steps, "methods": methods}


def authorization_boundary_evidence(
    system_root: str | Path,
    *,
    quality_result_path: str | Path | None = None,
    terminal_audit_path: str | Path | None = None,
) -> dict[str, Any]:
    root = reject_symlink_chain(system_root, name="generation system root")
    quality_path = (
        reject_symlink_chain(quality_result_path, name="quality bridge result")
        if quality_result_path is not None
        else root / DEFAULT_QUALITY_RESULT
    )
    result = _read_json(quality_path, name="quality bridge result")
    terminal = None
    if terminal_audit_path is not None or (root / DEFAULT_TERMINAL_AUDIT).is_file():
        terminal = _read_json(
            terminal_audit_path or root / DEFAULT_TERMINAL_AUDIT,
            name="terminal completion audit",
        )
    result_boundary = result.get("authorization_boundary")
    if not isinstance(result_boundary, Mapping):
        raise ValueError("quality bridge result authorization boundary is missing")
    required_quality_false = (
        "full_training_launch_allowed",
        "full_300k_launch_allowed",
        "quality_bridge_execution_allowed",
        "release_authorization_allowed",
    )
    boundaries: list[tuple[str, Mapping[str, Any]]] = [("quality bridge result", result_boundary)]
    if terminal is not None:
        terminal_boundary = terminal.get("authorization_boundary")
        if not isinstance(terminal_boundary, Mapping):
            raise ValueError("terminal completion audit authorization boundary is missing")
        boundaries.append(("terminal completion audit", terminal_boundary))
    required_false = (
        "full_training_launch_allowed",
        "full_300k_launch_allowed",
        "gpu_execution_allowed",
        "training_launch_allowed",
        "sampling_launch_allowed",
        "promotion_allowed",
        "promotion_authorization_allowed",
        "release_allowed",
        "release_authorization_allowed",
        "process_signals_allowed",
    )
    # Some downstream waiters duplicate launch permissions under a top-level
    # ``scope`` object.  It may also be a descriptive scope with no permission
    # fields, so only treat it as an authorization surface when it exposes at
    # least one known permission key.  Once exposed, the complete fail-closed
    # set is required and every value must be explicitly ``False``.
    for source_name, source in (("quality bridge result", result), ("terminal completion audit", terminal)):
        if source is None:
            continue
        scope = source.get("scope")
        if isinstance(scope, Mapping) and any(key in scope for key in required_false):
            boundaries.append((f"{source_name} scope", scope))
    for source_name, boundary in boundaries:
        required = required_quality_false if source_name == "quality bridge result" else required_false
        for key in required:
            if key not in boundary:
                raise ValueError(f"{source_name} authorization boundary {key} is missing")
            if boundary.get(key) is not False:
                raise ValueError(f"authorization boundary {key} is not false")
        for key in required_false:
            if key in boundary and boundary.get(key) is not False:
                raise ValueError(f"authorization boundary {key} is not false")
    proven = [source.get("generation_advantage_proven") for source in (result, terminal) if isinstance(source, Mapping)]
    if any(value is True for value in proven):
        raise ValueError("an upstream report incorrectly claims generation advantage")
    return {
        "status": "pass",
        "checked_reports": len(boundaries),
        "generation_advantage_proven": False,
        "gpu_and_launch_permissions_false": True,
        "terminal_status": terminal.get("terminal_status") if isinstance(terminal, Mapping) else None,
    }


def build_generation_system_goal_audit(
    *,
    system_root: str | Path,
    target_steps: int = DEFAULT_TARGET_STEPS,
    effective_batch: int = DEFAULT_EFFECTIVE_BATCH,
    expected_sample_count: int = DEFAULT_SAMPLE_COUNT,
    expected_sample_steps: int = DEFAULT_SAMPLE_STEPS,
    cofitok_run: str = DEFAULT_COFiTOK_RUN,
    dense_run: str = DEFAULT_DENSE_RUN,
    quality_result_path: str | Path | None = None,
    comparison_path: str | Path | None = None,
    terminal_audit_path: str | Path | None = None,
) -> dict[str, Any]:
    checks = [
        _check(
            "training_scale",
            lambda: training_scale_evidence(
                system_root,
                target_steps=target_steps,
                effective_batch=effective_batch,
                cofitok_run=cofitok_run,
                dense_run=dense_run,
            ),
        ),
        _check(
            "sample_quality",
            lambda: sample_quality_evidence(
                system_root,
                quality_result_path=quality_result_path,
            ),
        ),
        _check(
            "strong_baseline_fairness",
            lambda: baseline_fairness_evidence(
                system_root,
                comparison_path=comparison_path,
            ),
        ),
        _check(
            "stable_inference",
            lambda: stable_inference_evidence(
                system_root,
                expected_sample_count=expected_sample_count,
                expected_sample_steps=expected_sample_steps,
                cofitok_run=cofitok_run,
                dense_run=dense_run,
            ),
        ),
        _check(
            "checkpoint_reproducibility",
            lambda: checkpoint_reproducibility_evidence(
                system_root,
                target_steps=target_steps,
                cofitok_run=cofitok_run,
                dense_run=dense_run,
            ),
        ),
        _check(
            "authorization_boundary",
            lambda: authorization_boundary_evidence(
                system_root,
                quality_result_path=quality_result_path,
                terminal_audit_path=terminal_audit_path,
            ),
        ),
    ]
    audit = aggregate_checks(checks)
    audit.update(
        {
            "role": "cofitok_generation_system_goal_audit",
            "target": {
                "training_steps_per_method": target_steps,
                "effective_batch": effective_batch,
                "sample_count_per_method": expected_sample_count,
                "sample_steps": expected_sample_steps,
            },
            "generation_advantage_proven": False,
            "read_only": True,
            "system_root": reject_symlink_chain(system_root, name="generation system root").resolve().as_posix(),
        }
    )
    return audit


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a fail-closed, read-only audit of the CoFiTok large-model "
            "generation-system objective."
        )
    )
    parser.add_argument("--system-root", required=True)
    parser.add_argument("--target-steps", type=int, default=DEFAULT_TARGET_STEPS)
    parser.add_argument("--effective-batch", type=int, default=DEFAULT_EFFECTIVE_BATCH)
    parser.add_argument("--expected-sample-count", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--expected-sample-steps", type=int, default=DEFAULT_SAMPLE_STEPS)
    parser.add_argument("--cofitok-run", default=DEFAULT_COFiTOK_RUN)
    parser.add_argument("--dense-run", default=DEFAULT_DENSE_RUN)
    parser.add_argument("--quality-result")
    parser.add_argument("--comparison")
    parser.add_argument("--terminal-audit")
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    audit = build_generation_system_goal_audit(
        system_root=args.system_root,
        target_steps=args.target_steps,
        effective_batch=args.effective_batch,
        expected_sample_count=args.expected_sample_count,
        expected_sample_steps=args.expected_sample_steps,
        cofitok_run=args.cofitok_run,
        dense_run=args.dense_run,
        quality_result_path=args.quality_result,
        comparison_path=args.comparison,
        terminal_audit_path=args.terminal_audit,
    )
    write_json_report(args.output, audit)
    print(json.dumps(audit, indent=2, sort_keys=True))
    if not audit["complete"] and not args.allow_incomplete:
        raise SystemExit("generation system goal audit did not pass")


if __name__ == "__main__":
    main()
