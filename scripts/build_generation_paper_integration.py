from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any, Callable

from cofitok.generation import verify_generation_release_receipt
from cofitok.generation_gate import validate_generation_gate_authorization
from cofitok.generation_gate_sources import verify_generation_gate_source_reports
from cofitok.reporting import file_sha256, write_json_report, write_text_report

try:
    from scripts.build_large_scale_generation_comparison import (
        COMPARISON_REPORT_SCHEMA_VERSION,
        build_report as build_comparison_report,
        verify_comparison_source_reports,
    )
except ModuleNotFoundError:  # Direct execution with scripts/ on sys.path.
    from build_large_scale_generation_comparison import (  # type: ignore[no-redef]
        COMPARISON_REPORT_SCHEMA_VERSION,
        build_report as build_comparison_report,
        verify_comparison_source_reports,
    )


PAPER_INTEGRATION_SCHEMA_VERSION = 1
PAPER_INTEGRATION_ROLE = "terminal_generation_paper_integration_bundle"
PAPER_INTEGRATION_STATUS = "integration_ready"
MATCHED_METHODS = ("CoFiTok K=8", "Dense identity")
OFFICIAL_METHODS = {"d_ar": "D-AR", "mar": "MAR", "retok": "ReTok"}
OUTPUT_FILENAMES = {
    "matched_table": "generation_matched_table.tex",
    "official_context_table": "generation_official_context_table.tex",
    "claims": "generation_claims.tex",
}
_SHA1 = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_PROFILE_CONTRACTS = {
    "large_scale_generation_v1": {
        "audit_status": "complete",
        "comparison_profile": "full",
        "final_gate_check": "final_generation_gate",
        "comparison_check": "final_comparison_report",
        "release_check": "deployable_ema_inference_artifacts",
    },
    "stability_generation_system_v1": {
        "audit_status": "pass",
        "comparison_profile": "stability_full",
        "final_gate_check": "stability_final_gate",
        "comparison_check": "stability_strong_baseline_comparison",
        "release_check": "stability_release_authorized_inference",
    },
}


def _read_object(path: str | Path, *, name: str) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f"{name} is missing or is a symlink: {source}")
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return payload


def file_identity(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f"paper-integration source is missing or a symlink: {source}")
    resolved = source.resolve(strict=True)
    return {
        "path": resolved.as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": file_sha256(resolved),
    }


def checkout_identity(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve(strict=True)

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    identity = {
        "revision": git("rev-parse", "HEAD^{commit}"),
        "tree": git("rev-parse", "HEAD^{tree}"),
        "branch": git("branch", "--show-current"),
        "tracked_dirty": bool(
            git("status", "--porcelain=v1", "--untracked-files=no")
        ),
    }
    return _validate_git_identity(identity)


def _validate_git_identity(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "revision",
        "tree",
        "branch",
        "tracked_dirty",
    }:
        raise ValueError("paper-integration Git identity is malformed")
    if (
        _SHA1.fullmatch(str(value["revision"])) is None
        or _SHA1.fullmatch(str(value["tree"])) is None
        or not str(value["branch"])
        or value["tracked_dirty"] is not False
    ):
        raise ValueError("paper integration requires a clean named Git checkout")
    return dict(value)


def _profile(audit: dict[str, Any]) -> tuple[str, dict[str, str]]:
    profile = str(audit.get("profile", "large_scale_generation_v1"))
    contract = _PROFILE_CONTRACTS.get(profile)
    if contract is None:
        raise ValueError(f"unsupported terminal completion profile: {profile}")
    if (
        audit.get("schema_version") != 1
        or audit.get("status") != contract["audit_status"]
        or audit.get("complete") is not True
        or audit.get("failed_checks") != []
        or audit.get("missing_checks") != []
    ):
        raise ValueError("terminal generation completion audit did not pass")
    return profile, contract


def _passing_check(audit: dict[str, Any], name: str) -> dict[str, Any]:
    rows = [
        row
        for row in audit.get("checks", [])
        if isinstance(row, dict) and row.get("name") == name
    ]
    if len(rows) != 1 or rows[0].get("status") != "pass":
        raise ValueError(f"terminal completion audit lacks passing check: {name}")
    evidence = rows[0].get("evidence")
    if not isinstance(evidence, dict):
        raise ValueError(f"terminal completion audit check has no evidence: {name}")
    return evidence


def _load_bound_comparison_sources(
    comparison: dict[str, Any],
    *,
    provided_final_gate: dict[str, Any],
    provided_final_gate_identity: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    verified = verify_comparison_source_reports(comparison)
    source_reports = verified["source_reports"]
    loaded = {
        name: _read_object(identity["path"], name=f"comparison source {name}")
        for name, identity in source_reports.items()
    }
    bound_gate = source_reports.get("final_gate")
    if bound_gate != provided_final_gate_identity or loaded.get("final_gate") != provided_final_gate:
        raise ValueError("provided final gate differs from the comparison-bound final gate")

    official = comparison.get("official_context_source")
    if not isinstance(official, dict):
        raise ValueError("comparison lacks official contextual source binding")
    official_identity = file_identity(str(official.get("path", "")))
    if (
        official_identity["sha256"] != official.get("sha256")
        or official.get("schema_version") != 1
    ):
        raise ValueError("official contextual source changed after comparison binding")
    official_payload = _read_object(
        official_identity["path"], name="official contextual comparison source"
    )

    rebuilt = build_comparison_report(
        cofitok_training=loaded["cofitok_training"],
        dense_training=loaded["dense_training"],
        cofitok_generation=loaded["cofitok_generation"],
        dense_generation=loaded["dense_generation"],
        final_gate=loaded["final_gate"],
        official_related=official_payload,
        training_contention=loaded["training_contention"],
        class_fidelity_qualification=loaded.get("class_fidelity_qualification"),
        official_source_path=str(official["path"]),
        official_source_sha256=str(official["sha256"]),
        source_reports=comparison["source_reports"],
        source_profile=str(comparison.get("source_profile", "full")),
    )
    if rebuilt != comparison:
        raise ValueError("large-scale comparison differs from its physical sources")
    return verified, official_identity, loaded


def _validate_report_shape(comparison: dict[str, Any]) -> None:
    if (
        comparison.get("schema_version") != COMPARISON_REPORT_SCHEMA_VERSION
        or comparison.get("status") != "ready"
        or comparison.get("final_gate")
        != {"status": "pass", "decision": "large_scale_generation_ready"}
    ):
        raise ValueError("large-scale comparison is not terminally ready")
    policy = comparison.get("comparison_policy", {})
    if (
        policy.get("primary_direct_tier") != "matched_training_direct"
        or policy.get("external_context_tier") != "official_pretrained_contextual"
        or policy.get("cross_tier_numeric_ranking_allowed") is not False
    ):
        raise ValueError("large-scale comparison tier policy is unsafe")
    matched = comparison.get("matched_training_rows")
    if (
        not isinstance(matched, list)
        or [row.get("method") for row in matched] != list(MATCHED_METHODS)
    ):
        raise ValueError("matched generation rows have unexpected identities or order")
    for row in matched:
        if (
            row.get("comparison_tier") != "matched_training_direct"
            or row.get("directly_comparable_to_cofitok") is not True
            or row.get("dataset") != "imagenet_256"
            or int(row.get("resolution", -1)) != 256
            or int(row.get("training_steps", -1)) != 300_000
            or int(row.get("sample_count", -1)) != 50_000
            or row.get("weights") != "ema"
            or row.get("sampler") != "ddim"
            or int(row.get("sample_steps", -1)) != 250
        ):
            raise ValueError(f"matched paper row has unsafe protocol metadata: {row.get('method')}")
        for field in ("fid", "inception_score", "precision", "recall"):
            value = float(row.get(field, math.nan))
            if not math.isfinite(value):
                raise ValueError(f"matched paper metric is non-finite: {row.get('method')}/{field}")
    official = comparison.get("official_context_rows")
    if not isinstance(official, list):
        raise ValueError("official contextual rows are missing")
    indexed = {str(row.get("alias")): row for row in official}
    if set(indexed) != set(OFFICIAL_METHODS):
        raise ValueError("official contextual paper rows have unexpected aliases")
    for alias, method in OFFICIAL_METHODS.items():
        row = indexed[alias]
        if (
            row.get("method") != method
            or row.get("comparison_tier") != "official_pretrained_contextual"
            or row.get("directly_comparable_to_cofitok") is not False
            or row.get("paper_table_role") != "secondary related-method only"
            or row.get("source_status") != "completed_eval_only_50k"
            or int(row.get("sample_count", -1)) != 50_000
        ):
            raise ValueError(f"official contextual paper row is unsafe: {alias}")


def _validate_audit_bindings(
    audit: dict[str, Any],
    comparison: dict[str, Any],
    gate_identity: dict[str, Any],
    gate_source_verification: dict[str, Any],
) -> tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]]:
    profile, contract = _profile(audit)
    gate_evidence = _passing_check(audit, contract["final_gate_check"])
    comparison_evidence = _passing_check(audit, contract["comparison_check"])
    release_evidence = _passing_check(audit, contract["release_check"])
    if comparison.get("source_profile") != contract["comparison_profile"]:
        raise ValueError("comparison profile differs from terminal completion profile")
    expected_source_shas = {
        name: identity["sha256"]
        for name, identity in comparison["source_reports"].items()
    }
    if comparison_evidence.get("source_report_sha256") != expected_source_shas:
        raise ValueError("terminal audit comparison evidence differs from comparison sources")
    if comparison_evidence.get("matched_methods") != list(MATCHED_METHODS):
        raise ValueError("terminal audit comparison method identities differ")
    if comparison_evidence.get("cross_tier_numeric_ranking_allowed") is not False:
        raise ValueError("terminal audit permits unsafe cross-tier ranking")
    if profile == "stability_generation_system_v1":
        if gate_evidence.get("gate_sha256") != gate_identity["sha256"]:
            raise ValueError("terminal audit final-gate SHA256 differs")
        expected_gate_source_shas = {
            name: identity["sha256"]
            for name, identity in gate_source_verification["source_reports"].items()
        }
        if gate_evidence.get("source_report_sha256") != expected_gate_source_shas:
            raise ValueError("terminal audit final-gate sources differ")
    elif gate_evidence.get("source_reports") != gate_source_verification["source_reports"]:
        raise ValueError("terminal audit final-gate sources differ")
    return profile, gate_evidence, comparison_evidence, release_evidence


def _verify_release(
    receipt_path: str | Path,
    receipt: dict[str, Any],
    audit_identity: dict[str, Any],
    profile: str,
    *,
    verifier: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if (
        receipt.get("schema_version") != 1
        or receipt.get("receipt_type") != "cofitok_generation_release_receipt"
        or receipt.get("status") != "completed"
        or receipt.get("completion_profile") != profile
        or receipt.get("completion_audit") != audit_identity
    ):
        raise ValueError("generation release receipt differs from terminal audit")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {"cofitok", "dense_identity"}:
        raise ValueError("generation release receipt artifact set is incomplete")
    verify = verifier or verify_generation_release_receipt
    verified = {}
    for method, row in artifacts.items():
        if not isinstance(row, dict):
            raise ValueError(f"generation release artifact row is malformed: {method}")
        validation = verify(receipt_path, str(row.get("path", "")))
        if (
            validation.get("method") != method
            or validation.get("completion_profile") != profile
            or validation.get("completion_audit") != audit_identity
        ):
            raise ValueError(f"generation release artifact verification differs: {method}")
        verified[method] = validation
    return verified


def _tex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in text)


def _number(value: Any, digits: int = 3) -> str:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("paper metric is non-finite")
    return f"{numeric:.{digits}f}"


def _integer(value: Any) -> str:
    return f"{int(value):,}".replace(",", "{,}")


def render_matched_table(comparison: dict[str, Any]) -> str:
    rows = comparison["matched_training_rows"]
    lines = [
        "% Auto-generated from terminally audited CoFiTok generation evidence.",
        "% Do not edit numeric values by hand.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Method & Params & Steps & Train images & Samples & FID$\downarrow$ & IS$\uparrow$ & Prec.$\uparrow$ & Rec.$\uparrow$ \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            "{} & {} & {} & {} & {} & {} & {} & {} & {} \\\\".format(
                _tex_escape(row["method"]),
                _integer(row["parameter_count"]),
                _integer(row["training_steps"]),
                _integer(row["training_images_seen"]),
                _integer(row["sample_count"]),
                _number(row["fid"]),
                _number(row["inception_score"]),
                _number(row["precision"]),
                _number(row["recall"]),
            )
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{ImageNet-256 generation under matched dataset, shared "
                r"backbone contract, optimizer schedule, effective batch, optimizer "
                r"steps, training images, and a 50K EMA DDIM-250 evaluation. Training "
                r"wall-clock, GPU-hours, and FLOPs are measured outcomes rather than "
                r"equalized budgets.}"
            ),
            r"\label{tab:cofitok-full-generation-matched}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def render_official_context_table(comparison: dict[str, Any]) -> str:
    indexed = {row["alias"]: row for row in comparison["official_context_rows"]}
    lines = [
        "% Auto-generated contextual rows; intentionally separate from the matched table.",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Method & FID$\downarrow$ & IS$\uparrow$ & Prec.$\uparrow$ & Rec.$\uparrow$ \\",
        r"\midrule",
    ]
    for alias in ("d_ar", "mar", "retok"):
        row = indexed[alias]
        lines.append(
            "{} & {} & {} & {} & {} \\\\".format(
                _tex_escape(row["method"]),
                _number(row["fid"]),
                _number(row["inception_score"]),
                _number(row["precision"]),
                _number(row["recall"]),
            )
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{Official pretrained ImageNet-256 context. These eval-only "
                r"rows use different checkpoints, training budgets, and the ADM "
                r"TensorFlow evaluator, so they are not numerically ranked against "
                r"Table~\ref{tab:cofitok-full-generation-matched}.}"
            ),
            r"\label{tab:cofitok-official-generation-context}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def render_claims(comparison: dict[str, Any]) -> str:
    cofitok, dense = comparison["matched_training_rows"]
    relative = (float(cofitok["fid"]) / float(dense["fid"]) - 1.0) * 100.0
    if relative <= 0.0:
        fid_clause = (
            f"reducing FID by {_number(abs(relative), 2)}\\% relative to the "
            "matched dense predictor"
        )
    else:
        fid_clause = (
            f"a {_number(relative, 2)}\\% FID increase relative to the matched "
            "dense predictor while remaining inside the predeclared final gate"
        )
    class_clause = ""
    if cofitok.get("class_top1_accuracy") is not None:
        class_clause = (
            " Class fidelity is reported on the same generated sets: CoFiTok/dense "
            f"top-1 are {_number(cofitok['class_top1_accuracy'])}/"
            f"{_number(dense['class_top1_accuracy'])}, and top-5 are "
            f"{_number(cofitok['class_top5_accuracy'])}/"
            f"{_number(dense['class_top5_accuracy'])}."
        )
    return (
        "% Auto-generated claim text from terminally audited evidence.\n"
        "\\paragraph{Large-scale generation.} Under matched ImageNet-256 training "
        f"({_integer(cofitok['training_steps'])} optimizer steps and "
        f"{_integer(cofitok['training_images_seen'])} training images per method) "
        "and the same 50K-sample EMA DDIM-250 protocol, CoFiTok K=8 obtains "
        f"FID/IS/precision/recall of {_number(cofitok['fid'])}/"
        f"{_number(cofitok['inception_score'])}/{_number(cofitok['precision'])}/"
        f"{_number(cofitok['recall'])}, versus {_number(dense['fid'])}/"
        f"{_number(dense['inception_score'])}/{_number(dense['precision'])}/"
        f"{_number(dense['recall'])} for dense identity, {fid_clause}."
        f"{class_clause} Official D-AR, MAR, and ReTok values are shown only as "
        "pretrained contextual rows because their training and evaluator protocols "
        "differ. These results support the scoped claim that pixel-space diffusion "
        "noise prediction can be factorized into ordered compressed denoising "
        "components expanded by a restricted, condition-free synthesis operator, "
        "with controllable prefix denoising; they do not establish broad generation "
        "SOTA.\n"
    )


def _text_descriptor(path: Path, content: str) -> dict[str, Any]:
    payload = content.encode("utf-8")
    return {
        "path": path.resolve().as_posix(),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def build_paper_integration_manifest(
    *,
    completion_audit_path: str | Path,
    comparison_path: str | Path,
    final_gate_path: str | Path,
    release_receipt_path: str | Path,
    output_dir: str | Path,
    implementation_git: dict[str, Any],
    release_verifier: Callable[..., dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    implementation = _validate_git_identity(implementation_git)
    audit = _read_object(completion_audit_path, name="terminal completion audit")
    comparison = _read_object(comparison_path, name="large-scale comparison")
    final_gate = _read_object(final_gate_path, name="final generation gate")
    release = _read_object(release_receipt_path, name="generation release receipt")
    identities = {
        "completion_audit": file_identity(completion_audit_path),
        "comparison": file_identity(comparison_path),
        "final_gate": file_identity(final_gate_path),
        "release_receipt": file_identity(release_receipt_path),
    }
    _validate_report_shape(comparison)
    gate_authorization = validate_generation_gate_authorization(
        final_gate, expected_stage="full"
    )
    if gate_authorization.get("decision") != "large_scale_generation_ready":
        raise ValueError("final generation gate did not authorize release readiness")
    gate_source_verification = verify_generation_gate_source_reports(final_gate)
    comparison_source_verification, official_identity, _ = (
        _load_bound_comparison_sources(
            comparison,
            provided_final_gate=final_gate,
            provided_final_gate_identity=identities["final_gate"],
        )
    )
    profile, gate_evidence, comparison_evidence, release_evidence = (
        _validate_audit_bindings(
            audit,
            comparison,
            identities["final_gate"],
            gate_source_verification,
        )
    )
    release_verifications = _verify_release(
        release_receipt_path,
        release,
        identities["completion_audit"],
        profile,
        verifier=release_verifier,
    )

    output_root = Path(output_dir)
    if output_root.exists() and output_root.is_symlink():
        raise ValueError("paper-integration output directory may not be a symlink")
    output_root = output_root.resolve()
    contents = {
        "matched_table": render_matched_table(comparison),
        "official_context_table": render_official_context_table(comparison),
        "claims": render_claims(comparison),
    }
    outputs = {
        name: _text_descriptor(output_root / OUTPUT_FILENAMES[name], content)
        for name, content in contents.items()
    }
    manifest = {
        "schema_version": PAPER_INTEGRATION_SCHEMA_VERSION,
        "role": PAPER_INTEGRATION_ROLE,
        "status": PAPER_INTEGRATION_STATUS,
        "completion_profile": profile,
        "implementation_git": implementation,
        "completion_expectations": release.get("completion_expectations"),
        "sources": {
            **identities,
            "official_context": official_identity,
            "comparison_source_reports": comparison_source_verification[
                "source_reports"
            ],
            "final_gate_source_reports": gate_source_verification[
                "source_reports"
            ],
            "final_gate_diagnostic_reports": gate_source_verification.get(
                "diagnostic_reports"
            ),
        },
        "terminal_audit_evidence": {
            "final_gate": gate_evidence,
            "comparison": comparison_evidence,
            "release": release_evidence,
            "warnings": list(audit.get("warnings", [])),
        },
        "release_verifications": release_verifications,
        "paper_policy": {
            "venue": "venue_neutral",
            "matched_direct_tier": "matched_training_direct",
            "official_context_tier": "official_pretrained_contextual",
            "cross_tier_numeric_ranking_allowed": False,
            "broad_generation_sota_claim_allowed": False,
            "claim_scope": (
                "ordered restricted dense-noise factorization with controllable "
                "prefix denoising"
            ),
            "paper_consumer_files_mutated_by_builder": False,
            "generated_snippets_may_be_written_under_output_root": True,
            "post_application_lock_required": True,
        },
        "authorization_boundary": {
            "gpu_execution_authorized": False,
            "training_authorized": False,
            "evaluation_authorized": False,
            "paper_application_performed": False,
        },
        "results": {
            "matched_training_rows": comparison["matched_training_rows"],
            "official_context_rows": comparison["official_context_rows"],
            "matched_summary": comparison["matched_summary"],
        },
        "output_root": output_root.as_posix(),
        "outputs": outputs,
    }
    return manifest, contents


def _write_exact_text(path: Path, content: str) -> None:
    expected = content.encode("utf-8")
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != expected:
            raise ValueError(f"existing paper-integration output differs: {path}")
        return
    write_text_report(path, content)
    if path.read_bytes() != expected:
        raise ValueError(f"written paper-integration output differs: {path}")


def _write_exact_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        existing = _read_object(path, name="paper-integration manifest")
        if existing != payload:
            raise ValueError("existing paper-integration manifest differs")
        return
    write_json_report(path, payload)
    written = _read_object(path, name="paper-integration manifest")
    if written != payload:
        raise ValueError("written paper-integration manifest differs")


def write_paper_integration_bundle(
    *,
    completion_audit_path: str | Path,
    comparison_path: str | Path,
    final_gate_path: str | Path,
    release_receipt_path: str | Path,
    output_dir: str | Path,
    implementation_git: dict[str, Any],
    release_verifier: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    manifest, contents = build_paper_integration_manifest(
        completion_audit_path=completion_audit_path,
        comparison_path=comparison_path,
        final_gate_path=final_gate_path,
        release_receipt_path=release_receipt_path,
        output_dir=output_dir,
        implementation_git=implementation_git,
        release_verifier=release_verifier,
    )
    root = Path(manifest["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    for name, content in contents.items():
        _write_exact_text(root / OUTPUT_FILENAMES[name], content)
    for name, descriptor in manifest["outputs"].items():
        if file_identity(descriptor["path"]) != descriptor:
            raise ValueError(f"paper-integration output identity differs: {name}")
    _write_exact_json(root / "generation_paper_integration_manifest.json", manifest)
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build immutable, venue-neutral LaTeX snippets only after the terminal "
            "generation audit and release receipt physically validate."
        )
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--completion-audit", required=True)
    parser.add_argument("--comparison", required=True)
    parser.add_argument("--final-gate", required=True)
    parser.add_argument("--release-receipt", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = write_paper_integration_bundle(
        completion_audit_path=args.completion_audit,
        comparison_path=args.comparison,
        final_gate_path=args.final_gate,
        release_receipt_path=args.release_receipt,
        output_dir=args.output_dir,
        implementation_git=checkout_identity(args.project_root),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
