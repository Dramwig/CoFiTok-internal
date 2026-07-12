from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.validate_dataset_conditions import check_downsampled_strict_record, validate_dataset_conditions
from scripts.validate_idea_requirements import validate_requirements
from scripts.validate_mvp_evidence import validate_evidence
from scripts.validate_publication_readiness import validate_readiness
from scripts.validate_summary_consistency import validate_summary
from scripts.validate_synthesis_contract import validate_configs


@dataclass(frozen=True)
class GoalGate:
    name: str
    status: str
    evidence: dict[str, Any]
    missing: list[str]


@dataclass(frozen=True)
class StrictGap:
    name: str
    status: str
    evidence: dict[str, Any]
    reason: str
    next_action: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a strict CoFiTok goal-completion audit. This aggregates the "
            "existing MVP/publication gates and separately records gaps that "
            "still prevent a full unqualified completion claim."
        )
    )
    parser.add_argument("--internal-root", default=".")
    parser.add_argument("--project-root", default="..")
    parser.add_argument("--summary-dir", default="artifacts/reports/summary_2026-07-08")
    parser.add_argument("--summary", default="artifacts/reports/summary_2026-07-08/experiment_summary.json")
    parser.add_argument("--queue-manifest", default="artifacts/runbooks/next_validation_queue_2026-07-08.sh.manifest.json")
    parser.add_argument("--conditions-dir", default="docs/experiment_conditions")
    parser.add_argument("--config-glob", default="configs/*.json")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    parser.add_argument("--require-scoped-ready", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _rows(summary: dict[str, Any], key: str) -> list[dict[str, Any]]:
    rows = summary.get(key, [])
    if not isinstance(rows, list):
        raise AssertionError(f"summary[{key!r}] is not a list")
    return rows


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _max_number(rows: Iterable[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if _number(row.get(key))]
    return max(values) if values else 0.0


def _quality_row_ok(row: dict[str, Any], min_images: int) -> bool:
    return (
        _number(row.get("final_mse"))
        and _number(row.get("mse_auc"))
        and _number(row.get("lowres_frechet_proxy_auc"))
        and _number(row.get("image_count"))
        and int(row["image_count"]) >= min_images
    )


def full_validation_quality_evidence(summary: dict[str, Any]) -> dict[str, Any]:
    quality = _rows(summary, "quality")
    required = {
        ("tiny_imagenet_200", "epsilon_only"): 10000,
        ("tiny_imagenet_200", "light_denoise_path"): 10000,
        ("imagenet_1k_64x64_hf", "epsilon_only"): 50000,
        ("imagenet_1k_64x64_hf", "light_denoise_path"): 50000,
    }
    evidence: dict[str, Any] = {}
    missing = []
    for (dataset, variant), min_images in required.items():
        candidates = [
            row
            for row in quality
            if row.get("dataset") == dataset
            and row.get("variant") == variant
            and row.get("token_count") == 8
            and row.get("steps") == 20000
            and _quality_row_ok(row, min_images)
        ]
        key = f"{dataset}/{variant}"
        if candidates:
            best = max(candidates, key=lambda row: int(row.get("image_count") or 0))
            evidence[key] = {
                "report_dir": best.get("report_dir"),
                "image_count": best.get("image_count"),
                "final_mse": best.get("final_mse"),
                "mse_auc": best.get("mse_auc"),
                "lowres_frechet_proxy_auc": best.get("lowres_frechet_proxy_auc"),
            }
        else:
            evidence[key] = {"required_min_images": min_images}
            missing.append(key)
    evidence["missing"] = missing
    evidence["status"] = "ok" if not missing else "missing"
    return evidence


def _multiscale_20k_row_ok(row: dict[str, Any]) -> bool:
    is_multiscale = (
        row.get("predictor_type") == "multiscale_unet"
        or "multiscale" in str(row.get("config_name", "")).lower()
        or "multiscale" in str(row.get("report_dir", "")).lower()
    )
    zero_ratio = row.get("zero_token_ratio")
    zero_ok = _number(zero_ratio) and abs(float(zero_ratio)) <= 1e-8
    return (
        row.get("variant") == "light_denoise_path"
        and row.get("token_count") == 8
        and row.get("steps") == 20000
        and is_multiscale
        and row.get("synthesis_mode") == "restricted"
        and _number(row.get("final_clean_mse"))
        and _number(row.get("path_auc"))
        and _number(row.get("effective_tokens"))
        and _number(row.get("shuffled_final_ratio"))
        and zero_ok
    )


def multiscale_20k_evidence(summary: dict[str, Any]) -> dict[str, Any]:
    train = _rows(summary, "train")
    required_datasets = ["tiny_imagenet_200", "imagenet_1k_64x64_hf"]
    evidence: dict[str, Any] = {}
    missing = []
    for dataset in required_datasets:
        candidates = [row for row in train if row.get("dataset") == dataset and _multiscale_20k_row_ok(row)]
        if candidates:
            best = min(candidates, key=lambda row: float(row["final_clean_mse"]))
            evidence[dataset] = {
                "report_dir": best.get("report_dir"),
                "config_name": best.get("config_name"),
                "final_clean_mse": best.get("final_clean_mse"),
                "path_auc": best.get("path_auc"),
                "effective_tokens": best.get("effective_tokens"),
                "zero_token_ratio": best.get("zero_token_ratio"),
                "shuffled_final_ratio": best.get("shuffled_final_ratio"),
            }
        else:
            evidence[dataset] = {"required": "restricted light_denoise_path K8 multiscale_unet 20k train row"}
            missing.append(dataset)
    evidence["missing"] = missing
    evidence["status"] = "ok" if not missing else "missing"
    return evidence


def formal_generated_quality_evidence(summary: dict[str, Any]) -> dict[str, Any]:
    generated = _rows(summary, "generated_quality")
    required = {
        ("imagenet_1k_64x64_hf", "epsilon_only"): 50000,
        ("imagenet_1k_64x64_hf", "light_denoise_path"): 50000,
    }
    evidence: dict[str, Any] = {}
    missing = []
    for (dataset, variant), min_count in required.items():
        candidates = [
            row
            for row in generated
            if row.get("dataset") == dataset
            and row.get("variant") == variant
            and row.get("token_count") == 8
            and row.get("steps") == 20000
            and _number(row.get("sample_image_count"))
            and _number(row.get("real_image_count"))
            and int(row["sample_image_count"]) >= min_count
            and int(row["real_image_count"]) >= min_count
            and _number(row.get("inception_frechet"))
        ]
        key = f"{dataset}/{variant}"
        if candidates:
            best = max(
                candidates,
                key=lambda row: (
                    int(row.get("sample_image_count") or 0),
                    int(row.get("real_image_count") or 0),
                    int(
                        row.get("predictor_type") == "multiscale_unet"
                        or "multiscale" in str(row.get("report_dir", "")).lower()
                        or "multiscale" in str(row.get("config_name", "")).lower()
                    ),
                ),
            )
            evidence[key] = {
                "report_dir": best.get("report_dir"),
                "sample_image_count": best.get("sample_image_count"),
                "real_image_count": best.get("real_image_count"),
                "predictor_type": best.get("predictor_type"),
                "inception_frechet": best.get("inception_frechet"),
                "lowres_frechet_proxy": best.get("lowres_frechet_proxy"),
            }
        else:
            evidence[key] = {"required_sample_and_real_images": min_count}
            missing.append(key)
    evidence["missing"] = missing
    evidence["status"] = "ok" if not missing else "missing"
    return evidence


def official_fid_evidence(summary: dict[str, Any]) -> dict[str, Any]:
    official_rows = _rows(summary, "official_fid")
    required = {
        ("tiny_imagenet_200", "epsilon_only"): 10000,
        ("tiny_imagenet_200", "light_denoise_path"): 10000,
        ("imagenet_1k_64x64_hf", "epsilon_only"): 50000,
        ("imagenet_1k_64x64_hf", "light_denoise_path"): 50000,
    }
    evidence: dict[str, Any] = {}
    missing = []
    for (dataset, variant), min_count in required.items():
        candidates = [
            row
            for row in official_rows
            if row.get("dataset") == dataset
            and row.get("variant") == variant
            and row.get("token_count") == 8
            and row.get("steps") == 20000
            and row.get("status") == "ok"
            and row.get("implementation_package") == "pytorch-fid"
            and (
                row.get("predictor_type") == "multiscale_unet"
                or "multiscale" in str(row.get("report_dir", "")).lower()
                or "multiscale" in str(row.get("config_name", "")).lower()
            )
            and _number(row.get("official_fid"))
            and _number(row.get("generated_image_count"))
            and _number(row.get("real_image_count"))
            and int(row["generated_image_count"]) >= min_count
            and int(row["real_image_count"]) >= min_count
        ]
        key = f"{dataset}/{variant}"
        if candidates:
            best = max(
                candidates,
                key=lambda row: (
                    int(row.get("generated_image_count") or 0),
                    int(row.get("real_image_count") or 0),
                ),
            )
            evidence[key] = {
                "report_dir": best.get("report_dir"),
                "generated_image_count": best.get("generated_image_count"),
                "real_image_count": best.get("real_image_count"),
                "predictor_type": best.get("predictor_type"),
                "official_fid": best.get("official_fid"),
                "implementation_package": best.get("implementation_package"),
                "sample_steps": best.get("sample_steps"),
                "prefix_budget": best.get("prefix_budget"),
            }
        else:
            evidence[key] = {"required_generated_and_real_images": min_count}
            missing.append(key)
    evidence["missing"] = missing
    evidence["status"] = "ok" if not missing else "missing"
    evidence["row_count"] = len(official_rows)
    return evidence


def _gate(name: str, status: str, evidence: dict[str, Any] | None = None, missing: list[str] | None = None) -> GoalGate:
    return GoalGate(name=name, status=status, evidence=evidence or {}, missing=missing or [])


def _status_is_ok(status: str, accepted: set[str]) -> bool:
    return status in accepted


def check_paper_artifacts(project_root: Path) -> GoalGate:
    required = [
        "paper/README.md",
        "paper/references.bib",
        "paper/citation_audit.md",
        "paper/full_pdf_claim_audit_sources.json",
        "paper/latex/README.md",
        "paper/latex/Makefile",
        "paper/latex/main.tex",
        "paper/latex/main.pdf",
    ]
    missing = []
    evidence: dict[str, Any] = {}
    for relative in required:
        path = project_root / relative
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        evidence[relative] = {"exists": exists, "bytes": size}
        if not exists:
            missing.append(relative)
        elif relative.endswith(".pdf") and size < 1_000_000:
            missing.append(f"{relative}: unexpectedly small PDF")

    paper_sources = [
        "paper/latex/main.tex",
        "paper/venues/aaai27/main.tex",
        "paper/venues/aaai27/supplementary_aaai2027.tex",
    ]
    paper_text = "\n".join(
        (project_root / relative).read_text(encoding="utf-8")
        for relative in paper_sources
        if (project_root / relative).exists()
    )
    forbidden_claims = [
        "we are the first to propose coarse-to-fine visual tokens",
        "cofitok is the first to propose coarse-to-fine visual tokens",
        "cofitok solves image tokenization broadly",
        "we solve image tokenization broadly",
        "cofitok achieves unconditional generation quality wins",
        "cofitok demonstrates unconditional generation quality wins",
        "cofitok proves unconditional generation quality wins",
        "cofitok beats epsilon-only on generated-sample quality",
    ]
    forbidden_present = [
        claim for claim in forbidden_claims if claim in paper_text.lower()
    ]
    evidence["scanned_paper_sources"] = paper_sources
    evidence["forbidden_claims_present"] = forbidden_present
    evidence["venue_template"] = venue_template_artifacts(project_root)
    if forbidden_present:
        missing.extend(f"forbidden claim wording: {claim}" for claim in forbidden_present)

    return _gate("paper_artifacts", "ok" if not missing else "missing", evidence, missing)


def official_fid_protocol_artifacts(internal_root: Path | None = None) -> dict[str, Any]:
    required = [
        "scripts/export_official_fid_dirs.py",
        "scripts/evaluate_official_fid_dirs.py",
        "artifacts/runbooks/official_fid_protocol_2026-07-08.sh",
        "docs/records/2026-07-08_official_fid_protocol.md",
    ]
    if internal_root is None:
        return {"checked": False, "status": "unknown", "required": required}

    evidence: dict[str, Any] = {"checked": True, "files": {}}
    missing = []
    for relative in required:
        path = internal_root / relative
        exists = path.is_file()
        evidence["files"][relative] = {"exists": exists, "bytes": path.stat().st_size if exists else 0}
        if not exists:
            missing.append(relative)
    evidence["missing"] = missing
    evidence["status"] = "ready" if not missing else "missing"
    return evidence


def venue_template_artifacts(project_root: Path | None = None) -> dict[str, Any]:
    required_files = [
        "paper/venues/aaai27/README.md",
        "paper/venues/aaai27/Makefile",
        "paper/venues/aaai27/aaai2027.sty",
        "paper/venues/aaai27/aaai2027.bst",
        "paper/venues/aaai27/ReproducibilityChecklist.tex",
        "paper/venues/aaai27/main.tex",
        "paper/venues/aaai27/main_aaai2027.tex",
        "paper/venues/aaai27/supplementary_aaai2027.tex",
        "paper/venues/aaai27/main_aaai2027.pdf",
        "paper/venues/aaai27/supplementary_aaai2027.pdf",
    ]
    required_snippets = [
        r"\documentclass[letterpaper]{article}",
        r"\usepackage[submission]{aaai2027}",
        r"\author{Anonymous Submission}",
        r"\affiliations{}",
        r"\bibliography{../../references}",
    ]
    forbidden_snippets = [
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{hyperref}",
        r"\bibliographystyle{plainnat}",
    ]
    if project_root is None:
        return {"checked": False, "status": "unknown", "required": required_files}

    evidence: dict[str, Any] = {"checked": True, "files": {}, "template": "aaai27"}
    missing = []
    for relative in required_files:
        path = project_root / relative
        exists = path.is_file()
        size = path.stat().st_size if exists else 0
        evidence["files"][relative] = {"exists": exists, "bytes": size}
        if not exists:
            missing.append(relative)
        elif relative.endswith(".pdf") and size < 100_000:
            missing.append(f"{relative}: unexpectedly small PDF")

    main_path = project_root / "paper/venues/aaai27/main.tex"
    main_text = main_path.read_text(encoding="utf-8") if main_path.exists() else ""
    missing_snippets = [snippet for snippet in required_snippets if snippet not in main_text]
    forbidden_present = [snippet for snippet in forbidden_snippets if snippet in main_text]
    evidence["required_snippet_count"] = len(required_snippets)
    evidence["missing_snippets"] = missing_snippets
    evidence["forbidden_snippets_present"] = forbidden_present
    missing.extend(f"missing AAAI snippet: {snippet}" for snippet in missing_snippets)
    missing.extend(f"forbidden AAAI snippet: {snippet}" for snippet in forbidden_present)
    evidence["missing"] = missing
    evidence["status"] = "ok" if not missing else "missing"
    return evidence


def downsampled_imagenet64_artifacts(internal_root: Path | None = None) -> dict[str, Any]:
    if internal_root is None:
        return {
            "checked": False,
            "status": "unknown",
            "missing": ["internal_root was not provided"],
        }
    try:
        result = check_downsampled_strict_record(internal_root / "docs/experiment_conditions")
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError) as exc:
        return {
            "checked": True,
            "status": "missing",
            "missing": [f"{type(exc).__name__}: {exc}"],
        }
    evidence = asdict(result)
    evidence["checked"] = True
    return evidence


def strict_gaps(
    summary: dict[str, Any],
    internal_root: Path | None = None,
    project_root: Path | None = None,
) -> list[StrictGap]:
    generated = _rows(summary, "generated_quality")

    max_generated = int(_max_number(generated, "sample_image_count"))
    max_real = int(_max_number(generated, "real_image_count"))
    generated_protocol = (
        f"{max_generated}/{max_real}" if max_generated > 0 and max_real > 0 else "no generated-sample"
    )
    formal_generated = formal_generated_quality_evidence(summary)
    official_fid = official_fid_evidence(summary)
    full_validation = full_validation_quality_evidence(summary)
    multiscale_20k = multiscale_20k_evidence(summary)
    official_protocol = official_fid_protocol_artifacts(internal_root)
    venue_template = venue_template_artifacts(project_root)
    downsampled_source = downsampled_imagenet64_artifacts(internal_root)

    if formal_generated["status"] == "ok":
        generation_reason = (
            "Generated-sample evidence includes HF ImageNet-64 50000/50000 streamed/DDIM "
            "Inception-Frechet rows for epsilon-only and K8 light, including matched "
            "multiscale 20k backbone rows when available, but this still does not "
            "establish a robust unconditional generation-quality win across datasets or "
            "an external official FID benchmark claim."
        )
        generation_next_action = (
            "Use an agreed official FID implementation and/or stronger trained models only if "
            "unconditional generation quality becomes a central paper claim."
        )
    else:
        generation_reason = (
            f"Generated-sample evidence now reaches {generated_protocol} streamed/DDIM "
            "Inception-Frechet evidence, but this is not a 50k official FID protocol "
            "or a robust unconditional generation-quality win across datasets."
        )
        generation_next_action = (
            "Only if generation quality becomes central, run larger DDIM sampling and formal FID/LPIPS protocol."
        )
    if official_fid["status"] == "ok":
        generation_reason = (
            "Official pytorch-fid reports exist for matched Tiny ImageNet-200 10000/10000 "
            "and HF ImageNet-family 50000/50000 multiscale 20k epsilon-only/K8 light rows. "
            "They support reporting generation quality as a measured result, not as a broad "
            "unconditional generation-quality win."
        )
        generation_next_action = (
            "Do not claim broad generation-quality superiority; only run stronger/longer models if "
            "the paper later makes generation quality central."
        )
    elif official_protocol["status"] == "ready":
        generation_reason += (
            " An external pytorch-fid export/evaluation protocol exists, but full-scale "
            "official FID reports have not been generated."
        )

    gaps = []
    if official_fid["status"] != "ok":
        gaps.append(
            StrictGap(
                name="official_generation_quality_protocol",
                status="open",
                evidence={
                    "max_generated_samples": max_generated,
                    "max_real_images": max_real,
                    "hf_50k_streamed_protocol": formal_generated,
                    "official_fid_protocol_artifacts": official_protocol,
                    "official_fid_reports": official_fid,
                },
                reason=generation_reason,
                next_action=generation_next_action,
            )
        )
    if downsampled_source["status"] != "ok":
        gaps.append(
            StrictGap(
                name="exact_downsampled_imagenet64_source",
                status="open",
                evidence={
                    "fallback": "imagenet_1k_64x64_hf",
                    "downsampled_source": downsampled_source,
                },
                reason="The exact downsampled_imagenet/64x64 source is not verified in the local experiment-condition records.",
                next_action="Add the strict source record and manifest for the Academic Torrents/ImageNet-64 payload, or keep fallback naming.",
            )
        )
    if venue_template["status"] != "ok":
        gaps.append(
            StrictGap(
                name="venue_template_and_submission_format",
                status="open",
                evidence={
                    "venue_neutral_latex": True,
                    "full_pdf_claim_audit": True,
                    "venue_template": venue_template,
                },
                reason=(
                    "A venue-neutral LaTeX draft and full-PDF related-work claim audit exist, "
                    "but no target venue template/proceedings-metadata pass/camera-ready layout is finished."
                ),
                next_action="Choose target venue/template and port the current LaTeX draft into that format.",
            ),
        )
    if full_validation["status"] != "ok":
        insert_at = 1 if official_fid["status"] != "ok" else 0
        gaps.insert(
            insert_at,
            StrictGap(
                name="full_validation_reconstruction_sweep",
                status="open",
                evidence=full_validation,
                reason="Full-validation reconstruction/prefix quality rows are missing for one or more main Tiny/HF variants.",
                next_action="Run full validation reconstruction/prefix sweeps for Tiny/HF epsilon-only and K8 light main rows.",
            ),
        )
    if multiscale_20k["status"] != "ok":
        insert_at = (1 if official_fid["status"] != "ok" else 0) + (
            1 if full_validation["status"] != "ok" else 0
        )
        gaps.insert(
            insert_at,
            StrictGap(
                name="longer_stronger_backbone_validation",
                status="open",
                evidence=multiscale_20k,
                reason="The stronger multiscale U-Net path is missing matched 20k train-scale validation on Tiny/HF.",
                next_action="Run restricted K8 light_denoise_path multiscale_unet 20k rows on Tiny ImageNet-200 and HF ImageNet-1K 64x64.",
            ),
        )
    return gaps


def validate_goal_completion(
    internal_root: Path,
    project_root: Path,
    summary_dir: Path,
    summary_path: Path,
    queue_manifest_path: Path,
    conditions_dir: Path,
    config_glob: str,
) -> dict[str, Any]:
    internal_root = internal_root.resolve()
    project_root = project_root.resolve()
    summary_dir = (internal_root / summary_dir).resolve() if not summary_dir.is_absolute() else summary_dir.resolve()
    summary_path = (internal_root / summary_path).resolve() if not summary_path.is_absolute() else summary_path.resolve()
    queue_manifest_path = (
        (internal_root / queue_manifest_path).resolve()
        if not queue_manifest_path.is_absolute()
        else queue_manifest_path.resolve()
    )
    conditions_dir = (internal_root / conditions_dir).resolve() if not conditions_dir.is_absolute() else conditions_dir.resolve()
    summary = _read_json(summary_path)

    report_doc = internal_root / "docs/reports/cofitok_mvp_report_2026-07-08.md"
    completion_doc = internal_root / "docs/records/2026-07-08_completion_audit.md"

    summary_result = validate_summary(summary_dir, docs=[report_doc, completion_doc])
    mvp_result = validate_evidence(summary_path)
    idea_result = validate_requirements(internal_root, summary_path, queue_manifest_path)
    publication_result = validate_readiness(summary_path)
    dataset_result = validate_dataset_conditions(conditions_dir)
    synthesis_paths = sorted(internal_root.glob(config_glob))
    synthesis_result = validate_configs(synthesis_paths, static_only=True)
    paper_gate = check_paper_artifacts(project_root)

    gates = [
        _gate("summary_consistency", summary_result["status"], summary_result),
        _gate("mvp_evidence", mvp_result["status"], mvp_result),
        _gate("idea_requirements", idea_result["status"], idea_result),
        _gate("publication_readiness", publication_result["status"], publication_result),
        _gate("dataset_conditions", dataset_result["status"], dataset_result),
        _gate("synthesis_contract_static", synthesis_result["status"], synthesis_result),
        paper_gate,
    ]
    failed = [gate for gate in gates if gate.status not in {"ok", "ready"}]
    gaps = strict_gaps(summary, internal_root=internal_root, project_root=project_root)

    if failed:
        status = "missing"
    elif gaps:
        status = "scoped_ready_with_open_gaps"
    else:
        status = "complete"

    return {
        "objective": "Implement CoFiTok denoising-token diffusion and validate effectiveness on multiple common datasets.",
        "status": status,
        "scoped_claim_ready": not failed,
        "strict_completion_ready": status == "complete",
        "gate_count": len(gates),
        "failed_gate_count": len(failed),
        "open_gap_count": len(gaps),
        "gates": [asdict(gate) for gate in gates],
        "strict_gaps": [asdict(gap) for gap in gaps],
        "summary_counts": summary.get("counts", {}),
        "internal_root": internal_root.as_posix(),
        "project_root": project_root.as_posix(),
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# CoFiTok Goal Completion Audit",
        "",
        f"Status: `{result['status']}`",
        "",
        f"Scoped claim ready: `{result['scoped_claim_ready']}`",
        f"Strict completion ready: `{result['strict_completion_ready']}`",
        "",
        "This audit keeps the original goal intact. It does not mark the project complete while strict gaps remain.",
        "",
        "## Gates",
        "",
        "| gate | status | evidence |",
        "|---|---|---|",
    ]
    for gate in result["gates"]:
        evidence = gate.get("evidence", {})
        bits = []
        for key in ["status", "check_count", "ok_count", "ready_count", "missing_count", "checked_count", "restricted_ok_count", "ablation_count"]:
            if key in evidence:
                bits.append(f"{key}={evidence[key]}")
        if gate.get("missing"):
            bits.append("missing=" + "; ".join(gate["missing"][:3]))
        lines.append(f"| `{gate['name']}` | `{gate['status']}` | {'<br>'.join(bits) or 'see JSON'} |")

    lines.extend(
        [
            "",
            "## Strict Open Gaps",
            "",
            "| gap | status | reason | next action |",
            "|---|---|---|---|",
        ]
    )
    for gap in result["strict_gaps"]:
        lines.append(f"| `{gap['name']}` | `{gap['status']}` | {gap['reason']} | {gap['next_action']} |")

    lines.extend(
        [
            "",
            "## Summary Counts",
            "",
            "```text",
        ]
    )
    for key, value in sorted(result["summary_counts"].items()):
        lines.append(f"{key}: {value}")
    lines.extend(["```", ""])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    result = validate_goal_completion(
        internal_root=Path(args.internal_root),
        project_root=Path(args.project_root),
        summary_dir=Path(args.summary_dir),
        summary_path=Path(args.summary),
        queue_manifest_path=Path(args.queue_manifest),
        conditions_dir=Path(args.conditions_dir),
        config_glob=args.config_glob,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_md:
        path = Path(args.output_md)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_markdown(result), encoding="utf-8")
    if args.require_scoped_ready and not result["scoped_claim_ready"]:
        raise SystemExit(1)
    if args.require_complete and not result["strict_completion_ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
