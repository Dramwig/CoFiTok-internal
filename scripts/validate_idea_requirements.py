from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


CORE_DATASETS = {"cifar10", "tiny_imagenet_200", "imagenet_1k_64x64_hf"}
MAIN_DATASETS = {"tiny_imagenet_200", "imagenet_1k_64x64_hf"}
ORDERS = {"ordered", "random", "reverse"}


@dataclass(frozen=True)
class RequirementResult:
    name: str
    group: str
    status: str
    evidence: dict[str, Any]
    missing: list[str]
    next_action: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Map the CoFiTok idea requirements to current code, evidence, and queued validation."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--summary", default="artifacts/reports/summary_2026-07-08/experiment_summary.json")
    parser.add_argument("--queue-manifest", default="artifacts/runbooks/next_validation_queue_2026-07-08.sh.manifest.json")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    parser.add_argument("--require-mvp-ok", action="store_true")
    parser.add_argument("--require-publication-ready", action="store_true")
    return parser.parse_args()


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _rows(summary: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = summary.get(key, [])
    if not isinstance(value, list):
        raise AssertionError(f"summary[{key!r}] is not a list")
    return value


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _find(rows: Iterable[dict[str, Any]], **criteria: Any) -> list[dict[str, Any]]:
    matched = []
    for row in rows:
        if all(row.get(key) == value for key, value in criteria.items()):
            matched.append(row)
    return matched


def _ok(
    name: str,
    group: str,
    evidence: dict[str, Any],
    next_action: str = "Keep as invariant.",
) -> RequirementResult:
    return RequirementResult(name, group, "ok", evidence, [], next_action)


def _missing(
    name: str,
    group: str,
    evidence: dict[str, Any],
    missing: list[str],
    next_action: str,
    status: str = "missing",
) -> RequirementResult:
    return RequirementResult(name, group, status, evidence, missing, next_action)


def _snippet_check(text: str, snippets: list[str]) -> tuple[list[str], dict[str, int]]:
    missing = [snippet for snippet in snippets if snippet not in text]
    return missing, {"required_snippet_count": len(snippets), "present_count": len(snippets) - len(missing)}


def check_factorization_code(root: Path) -> RequirementResult:
    path = root / "src/cofitok/models/cofitok.py"
    text = _read_text(path)
    snippets = [
        "tokens: list[torch.Tensor]",
        "components: list[torch.Tensor]",
        "prefix_epsilons: list[torch.Tensor]",
        "components = self.synthesis(tokens)",
        "running = running + component",
        "prefix_epsilons.append(running)",
        "epsilon=prefix_epsilons[-1]",
    ]
    missing, counts = _snippet_check(text, snippets)
    evidence = {"path": path.as_posix(), **counts}
    if missing:
        return _missing(
            "dense_noise_factorization_code",
            "method",
            evidence,
            missing,
            "Restore ordered token -> component -> prefix epsilon accumulation in CoFiTokTiny.",
        )
    return _ok("dense_noise_factorization_code", "method", evidence)


def check_restricted_synthesis_code(root: Path) -> RequirementResult:
    path = root / "src/cofitok/models/synthesis.py"
    text = _read_text(path)
    snippets = [
        "Maps one denoising token to one dense noise component",
        "def forward(self, token: torch.Tensor)",
        "bias=False",
        "self.local(self.proj(token))",
        "zero_components_like",
        "deep synthesis",
    ]
    forbidden = ["self.bias = nn.Parameter", "nn.MultiheadAttention", "Transformer", "learned_constant"]
    missing, counts = _snippet_check(text, snippets)
    forbidden_present = [snippet for snippet in forbidden if snippet in text]
    evidence = {"path": path.as_posix(), **counts, "forbidden_present": forbidden_present}
    if missing or forbidden_present:
        return _missing(
            "restricted_synthesis_operator_code",
            "method",
            evidence,
            [*missing, *[f"forbidden: {item}" for item in forbidden_present]],
            "Keep default S_k token-only, local, bias-free, and reserve deep synthesis for explicit ablations.",
        )
    return _ok("restricted_synthesis_operator_code", "method", evidence)


def check_losses_and_diagnostics_code(root: Path) -> RequirementResult:
    losses_path = root / "src/cofitok/training/losses.py"
    diag_path = root / "src/cofitok/diagnostics.py"
    losses = _read_text(losses_path)
    diagnostics = _read_text(diag_path)
    snippets = [
        "monotonic_loss",
        "zero_token_loss",
        "component_decorrelation",
        "denoise_path_prefix",
        "denoise_path_component",
        "random_tokens",
        "shuffle_tokens_across_batch",
        "zero_components",
        "shuffled_component_relative_mse",
    ]
    combined = losses + "\n" + diagnostics
    missing, counts = _snippet_check(combined, snippets)
    evidence = {"losses": losses_path.as_posix(), "diagnostics": diag_path.as_posix(), **counts}
    if missing:
        return _missing(
            "losses_and_diagnostics_code",
            "method",
            evidence,
            missing,
            "Restore prefix/monotonic/zero-token losses and zero/random/shuffle diagnostics.",
        )
    return _ok("losses_and_diagnostics_code", "method", evidence)


def check_dataset_evidence(summary: dict[str, Any]) -> RequirementResult:
    train_datasets = {str(row.get("dataset")) for row in _rows(summary, "train")}
    missing = sorted(CORE_DATASETS - train_datasets)
    evidence = {"train_datasets": sorted(train_datasets), "required_datasets": sorted(CORE_DATASETS)}
    if missing:
        return _missing(
            "multi_dataset_coverage",
            "evidence",
            evidence,
            missing,
            "Run at least smoke/training evidence for the missing common datasets.",
        )
    return _ok("multi_dataset_coverage", "evidence", evidence)


def check_prefix_evidence(summary: dict[str, Any]) -> RequirementResult:
    train = _rows(summary, "train")
    evidence: dict[str, Any] = {}
    missing = []
    for dataset in sorted(MAIN_DATASETS):
        rows = _find(train, dataset=dataset, variant="light_denoise_path", token_count=8, steps=20000)
        valid = [
            row
            for row in rows
            if _number(row.get("path_auc"))
            and _number(row.get("final_clean_mse"))
            and _number(row.get("effective_tokens"))
            and _number(row.get("zero_token_ratio"))
            and float(row.get("zero_token_ratio")) == 0.0
        ]
        evidence[dataset] = {"matching_rows": len(valid)}
        if not valid:
            missing.append(f"{dataset}: missing K8 20k light denoise-path prefix evidence")
    if missing:
        return _missing(
            "prefix_denoising_evidence",
            "evidence",
            evidence,
            missing,
            "Run K8 20k light denoise-path train/eval reports with prefix path metrics.",
        )
    return _ok("prefix_denoising_evidence", "evidence", evidence)


def check_non_degenerate_diagnostics(summary: dict[str, Any]) -> RequirementResult:
    train = _rows(summary, "train")
    evidence: dict[str, Any] = {}
    missing = []
    for dataset in sorted(CORE_DATASETS):
        rows = [
            row
            for row in train
            if row.get("dataset") == dataset
            and row.get("synthesis_mode") == "restricted"
            and _number(row.get("zero_token_ratio"))
            and _number(row.get("random_token_ratio"))
            and _number(row.get("shuffled_final_ratio"))
        ]
        zero_ok = any(float(row["zero_token_ratio"]) == 0.0 for row in rows)
        shuffle_ok = any(float(row["shuffled_final_ratio"]) > 1.0 for row in rows)
        evidence[dataset] = {"rows": len(rows), "zero_ok": zero_ok, "shuffle_ok": shuffle_ok}
        if not zero_ok:
            missing.append(f"{dataset}: missing restricted zero-token ratio == 0")
        if not shuffle_ok:
            missing.append(f"{dataset}: missing shuffle mismatch diagnostic")
    if missing:
        return _missing(
            "zero_random_shuffle_diagnostics",
            "evidence",
            evidence,
            missing,
            "Regenerate restricted diagnostics for zero/random/shuffle token tests.",
        )
    return _ok("zero_random_shuffle_diagnostics", "evidence", evidence)


def check_order_and_scaling(summary: dict[str, Any]) -> RequirementResult:
    order_rows = _rows(summary, "order_eval")
    evidence: dict[str, Any] = {}
    missing = []
    for token_count in [4, 8, 16]:
        for seed in [103, 139]:
            rows_by_order = {
                order: _find(
                    order_rows,
                    dataset="imagenet_1k_64x64_hf",
                    variant="light_denoise_path",
                    token_count=token_count,
                    seed=seed,
                    component_order=order,
                )
                for order in ORDERS
            }
            key = f"imagenet_1k_64x64_hf/K{token_count}/seed{seed}"
            if not all(rows_by_order.values()):
                missing.append(f"{key}: missing ordered/random/reverse rows")
                evidence[key] = {"orders": sorted(order for order, rows in rows_by_order.items() if rows)}
                continue
            auc = {order: float(rows_by_order[order][0]["path_auc"]) for order in ORDERS}
            order_ok = auc["random"] > auc["ordered"] and auc["reverse"] > auc["ordered"]
            evidence[key] = {"path_auc": auc, "order_ok": order_ok}
            if not order_ok:
                missing.append(f"{key}: random/reverse do not worsen path AUC")
    if missing:
        return _missing(
            "order_ablation_and_token_scaling",
            "evidence",
            evidence,
            missing,
            "Complete ordered/random/reverse diagnostics for K=4,8,16 seed repeats.",
        )
    return _ok("order_ablation_and_token_scaling", "evidence", evidence)


def check_deep_and_simultaneous(summary: dict[str, Any]) -> RequirementResult:
    train = _rows(summary, "train")
    order_rows = _rows(summary, "order_eval")
    evidence: dict[str, Any] = {}
    missing = []
    for seed in [103, 139]:
        deep_rows = _find(
            train,
            dataset="imagenet_1k_64x64_hf",
            variant="deep_synthesis_ablation",
            token_count=8,
            seed=seed,
        )
        deep_ok = bool(deep_rows) and float(deep_rows[0].get("zero_token_ratio", 0.0)) > 0.01
        sim_orders = {
            order: _find(
                order_rows,
                dataset="imagenet_1k_64x64_hf",
                variant="simultaneous_predictor",
                token_count=8,
                seed=seed,
                component_order=order,
            )
            for order in ORDERS
        }
        sim_ok = False
        if all(sim_orders.values()):
            ordered = float(sim_orders["ordered"][0]["path_auc"])
            sim_ok = all(float(sim_orders[order][0]["path_auc"]) > ordered for order in ["random", "reverse"])
        evidence[f"seed{seed}"] = {"deep_ok": deep_ok, "simultaneous_order_ok": sim_ok}
        if not deep_ok:
            missing.append(f"seed{seed}: missing deep S_k degeneration signal")
        if not sim_ok:
            missing.append(f"seed{seed}: missing simultaneous predictor order ablation")
    if missing:
        return _missing(
            "degeneration_and_predictor_ablations",
            "evidence",
            evidence,
            missing,
            "Complete deep-S_k and simultaneous predictor seed-repeat ablations.",
        )
    return _ok("degeneration_and_predictor_ablations", "evidence", evidence)


def check_quality_and_sampling(summary: dict[str, Any]) -> RequirementResult:
    quality = _rows(summary, "quality")
    sampling = _rows(summary, "sampling")
    generated = _rows(summary, "generated_quality")
    evidence = {
        "quality_rows": len(quality),
        "sampling_rows": len(sampling),
        "generated_quality_rows": len(generated),
    }
    missing = []
    quality_ok = any(
        row.get("dataset") == "imagenet_1k_64x64_hf"
        and row.get("variant") == "light_denoise_path"
        and row.get("token_count") == 8
        and row.get("lpips_available") is True
        and row.get("inception_available") is True
        for row in quality
    )
    generated_ok = all(
        _find(generated, dataset=dataset, variant=variant, token_count=8, steps=20000)
        for dataset in MAIN_DATASETS
        for variant in ["epsilon_only", "light_denoise_path"]
    )
    sampling_ok = CORE_DATASETS.issubset({str(row.get("dataset")) for row in sampling})
    evidence.update({"quality_ok": quality_ok, "sampling_ok": sampling_ok, "generated_ok": generated_ok})
    if not quality_ok:
        missing.append("missing ImageNet-64 HF K8 quality row with LPIPS and Inception")
    if not sampling_ok:
        missing.append("missing sampling smoke rows for all core datasets")
    if not generated_ok:
        missing.append("missing generated-quality smoke rows for main datasets and variants")
    if missing:
        return _missing(
            "quality_and_sampling_evidence",
            "evidence",
            evidence,
            missing,
            "Run quality, sampling, and generated-sample evaluators for missing rows.",
        )
    return _ok("quality_and_sampling_evidence", "evidence", evidence)


def check_queued_publication_batch(queue_manifest: dict[str, Any]) -> RequirementResult:
    expected = {
        "run_count": 10,
        "quality_count": 10,
        "order_eval_count": 24,
        "sampling_count": 6,
        "generated_quality_count": 6,
        "sample_count": 2048,
        "sample_steps": 50,
        "real_count": 8192,
    }
    missing = [f"{key}: expected {value}, got {queue_manifest.get(key)}" for key, value in expected.items() if queue_manifest.get(key) != value]
    evidence = {key: queue_manifest.get(key) for key in expected}
    if missing:
        return _missing(
            "queued_publication_scale_batch",
            "publication",
            evidence,
            missing,
            "Regenerate next_validation_queue_2026-07-08.sh with the publication-readiness batch.",
        )
    return _ok(
        "queued_publication_scale_batch",
        "publication",
        evidence,
        "Run the queued batch on pro6000, then regenerate summaries and rerun publication readiness.",
    )


def check_publication_readiness(summary: dict[str, Any]) -> RequirementResult:
    # Keep this lightweight here; validate_publication_readiness.py remains the detailed gate.
    train = _rows(summary, "train")
    quality = _rows(summary, "quality")
    generated = _rows(summary, "generated_quality")
    evidence = {
        "main_20k_seed_counts": {},
        "max_quality_images": max([int(row.get("image_count", 0)) for row in quality if _number(row.get("image_count"))] or [0]),
        "max_generated_samples": max([int(row.get("sample_image_count", 0)) for row in generated if _number(row.get("sample_image_count"))] or [0]),
        "max_real_images": max([int(row.get("real_image_count", 0)) for row in generated if _number(row.get("real_image_count"))] or [0]),
    }
    missing = []
    for dataset in sorted(MAIN_DATASETS):
        for variant in ["epsilon_only", "light_denoise_path"]:
            seeds = {
                row.get("seed")
                for row in train
                if row.get("dataset") == dataset
                and row.get("variant") == variant
                and row.get("token_count") == 8
                and row.get("steps") == 20000
            }
            key = f"{dataset}/{variant}"
            evidence["main_20k_seed_counts"][key] = len(seeds)
            if len(seeds) < 2:
                missing.append(f"{key}: need two 20k seeds")
    if evidence["max_quality_images"] < 1024:
        missing.append("need >=1024-image quality slices")
    if evidence["max_generated_samples"] < 2048 or evidence["max_real_images"] < 8192:
        missing.append("need >=2048 generated samples vs >=8192 real images")
    if missing:
        return _missing(
            "publication_readiness",
            "publication",
            evidence,
            missing,
            "Execute the queued remote validation batch; current MVP evidence is not yet publication-scale.",
            status="pending",
        )
    return _ok("publication_readiness", "publication", evidence)


def validate_requirements(project_root: Path, summary_path: Path, queue_manifest_path: Path) -> dict[str, Any]:
    project_root = project_root.resolve()
    summary = _read_json(summary_path)
    queue_manifest = _read_json(queue_manifest_path)
    checks = [
        check_factorization_code(project_root),
        check_restricted_synthesis_code(project_root),
        check_losses_and_diagnostics_code(project_root),
        check_dataset_evidence(summary),
        check_prefix_evidence(summary),
        check_non_degenerate_diagnostics(summary),
        check_order_and_scaling(summary),
        check_deep_and_simultaneous(summary),
        check_quality_and_sampling(summary),
        check_queued_publication_batch(queue_manifest),
        check_publication_readiness(summary),
    ]
    missing = [check for check in checks if check.status == "missing"]
    pending = [check for check in checks if check.status == "pending"]
    return {
        "project_root": project_root.as_posix(),
        "summary": summary_path.as_posix(),
        "queue_manifest": queue_manifest_path.as_posix(),
        "status": "missing" if missing else ("mvp_ok_publication_pending" if pending else "ready"),
        "check_count": len(checks),
        "ok_count": sum(1 for check in checks if check.status == "ok"),
        "pending_count": len(pending),
        "missing_count": len(missing),
        "checks": [asdict(check) for check in checks],
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# CoFiTok Idea Requirements Matrix",
        "",
        f"Status: `{result['status']}`",
        "",
        "| requirement | group | status | evidence | next action |",
        "|---|---|---|---|---|",
    ]
    for check in result["checks"]:
        evidence_bits = []
        for key, value in check["evidence"].items():
            if isinstance(value, (str, int, float, bool)):
                evidence_bits.append(f"{key}={value}")
        if check["missing"]:
            evidence_bits.append("missing: " + "; ".join(check["missing"][:3]))
        evidence = "<br>".join(evidence_bits[:5]) or "see JSON"
        lines.append(
            f"| `{check['name']}` | {check['group']} | `{check['status']}` | {evidence} | {check['next_action']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    result = validate_requirements(Path(args.project_root), Path(args.summary), Path(args.queue_manifest))
    print(json.dumps(result, indent=2, sort_keys=True))

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_md:
        Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_md).write_text(render_markdown(result), encoding="utf-8")

    if args.require_mvp_ok and result["missing_count"]:
        raise SystemExit(1)
    if args.require_publication_ready and result["status"] != "ready":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
