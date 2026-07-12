from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paper-ready LaTeX tables from the evidence report.")
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _num(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: Any, digits: int = 2) -> str:
    number = _num(value)
    if number is None:
        return "--" if value in {None, ""} else _tex(str(value))
    return f"{number:.{digits}f}"


def _rank(value: Any) -> str:
    return "--" if value is None else str(value)


def _tex(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in value)


def _dataset_name(name: str) -> str:
    mapping = {
        "cifar10": "CIFAR-10",
        "tiny_imagenet_200": "Tiny-IN",
        "imagenet_1k_64x64_hf": "ImageNet-64 HF",
        "downsampled_imagenet_64": "ImageNet-64 strict",
        "ffhq_64": "FFHQ-64",
        "afhqv2_64": "AFHQv2-64",
        "imagenet_256_10pct": "ImageNet-256 10\\%",
        "imagenet_256": "ImageNet-256",
    }
    return mapping.get(name, _tex(name))


def _protocol_name(status: Any) -> str:
    mapping = {
        "completed_eval_only_50k": "official 50K eval-only",
    }
    value = str(status or "")
    return mapping.get(value, value)


def _table_env(label: str, caption: str, tabular: str, note: str = "") -> str:
    note_tex = f"\n  \\vspace{{0.3em}}\n  \\caption*{{\\footnotesize {note}}}" if note else ""
    return "\n".join(
        [
            r"\begin{table*}[t]",
            r"  \centering",
            r"  \scriptsize",
            f"  \\caption{{{caption}}}",
            f"  \\label{{{label}}}",
            r"  \resizebox{\textwidth}{!}{%",
            tabular,
            r"  }",
            note_tex,
            r"\end{table*}",
        ]
    )


def _p0_generation_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        r"  \begin{tabular}{lrrrrlrrrrlr}",
        r"    \toprule",
        r"    Dataset & C low & D low & DDPM low & EDM low & low winner & C Inc & D Inc & DDPM Inc & EDM Inc & Inc winner & C rank \\",
        r"    \midrule",
    ]
    for row in rows:
        lines.append(
            "    "
            + " & ".join(
                [
                    _dataset_name(str(row["dataset"])),
                    _fmt(row.get("cofitok_lowres")),
                    _fmt(row.get("dense_lowres")),
                    _fmt(row.get("improved_ddpm_lowres")),
                    _fmt(row.get("edm_lowres")),
                    _tex(str(row.get("lowres_winner", ""))),
                    _fmt(row.get("cofitok_inception")),
                    _fmt(row.get("dense_inception")),
                    _fmt(row.get("improved_ddpm_inception")),
                    _fmt(row.get("edm_inception")),
                    _tex(str(row.get("inception_winner", ""))),
                    _rank(row.get("cofitok_inception_rank")),
                ]
            )
            + r" \\"
        )
    lines.extend([r"    \bottomrule", r"  \end{tabular}%"])
    return _table_env(
        "tab:p0-generation-current",
        "Eight-dataset P0 generated-sample comparison under the locked short-budget protocols. Lower Frechet-style metrics are better.",
        "\n".join(lines),
        "C = CoFiTok; D = parameter-matched direct dense epsilon. Rows use the locked per-dataset short-budget protocol; they are not formal SOTA FID claims.",
    )


def _diagnostic_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        r"  \begin{tabular}{lrrrrrrrrr}",
        r"    \toprule",
        r"    Dataset & C PSNR & D PSNR & mask PSNR & C path & E path & C effK & E effK & C zero & deep-S zero \\",
        r"    \midrule",
    ]
    for row in rows:
        lines.append(
            "    "
            + " & ".join(
                [
                    _dataset_name(str(row["dataset"])),
                    _fmt(row.get("cofitok_psnr"), 3),
                    _fmt(row.get("dense_psnr"), 3),
                    _fmt(row.get("channel_mask_psnr"), 3),
                    _fmt(row.get("cofitok_path_auc"), 4),
                    _fmt(row.get("endpoint_only_path_auc"), 4),
                    _fmt(row.get("cofitok_effective_tokens"), 2),
                    _fmt(row.get("endpoint_only_effective_tokens"), 2),
                    _fmt(row.get("cofitok_zero_ratio"), 4),
                    _fmt(row.get("deep_synthesis_zero_ratio"), 4),
                ]
            )
            + r" \\"
        )
    lines.extend([r"    \bottomrule", r"  \end{tabular}%"])
    return _table_env(
        "tab:cofitok-diagnostics-current",
        "Diagnostics for the supported CoFiTok claim.",
        "\n".join(lines),
        "Lower path AUC indicates better prefix controllability. Zero-token ratio diagnoses whether the synthesis operator leaks structure at zero input.",
    )


def _imagenet256_confirmatory_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        r"  \begin{tabular}{rrrrrrrrr}",
        r"    \toprule",
        r"    Seed & E path & C ordered & non-ID mean & reverse & rank/24 & $\Delta$ CI low & D endpoint & C endpoint \\",
        r"    \midrule",
    ]
    for row in rows:
        lines.append(
            "    "
            + " & ".join(
                [
                    str(row["seed"]),
                    _fmt(row.get("endpoint_only_path_auc"), 4),
                    _fmt(row.get("cofitok_ordered_path_auc"), 4),
                    _fmt(row.get("exhaustive_nonidentity_mean_path_auc"), 4),
                    _fmt(row.get("cofitok_reverse_path_auc"), 4),
                    str(row.get("ordered_rank_of_24", "--")),
                    _fmt(row.get("nonidentity_delta_ci_low"), 4),
                    _fmt(row.get("dense_monolithic_endpoint_mse"), 4),
                    _fmt(row.get("cofitok_endpoint_mse"), 4),
                ]
            )
            + r" \\"
        )
    lines.extend([r"    \bottomrule", r"  \end{tabular}%"])
    return _table_env(
        "tab:imagenet256-confirmatory-order",
        "ImageNet-256 K4 20k confirmatory order and endpoint results over 1,024 validation images at $t=500$.",
        "\n".join(lines),
        (
            "E = endpoint-only factorized; C = CoFiTok; D = parameter-matched direct dense. "
            "non-ID averages the other 23 K4 permutations; $\\Delta$ CI low is the 95\\% paired-bootstrap "
            "lower bound for non-ID minus ordered path AUC."
        ),
    )


def _p1_table(rows: list[dict[str, Any]]) -> str:
    by_dataset: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_dataset.setdefault(str(row["dataset"]), {})[str(row["baseline"])] = row
    image_counts = sorted({_num(row.get("image_count")) for row in rows if _num(row.get("image_count")) is not None})
    eval_sizes = sorted({_num(row.get("eval_image_size")) for row in rows if _num(row.get("eval_image_size")) is not None})
    if len(image_counts) == 1 and len(eval_sizes) == 1:
        note = (
            f"Official pretrained reconstruction at {int(eval_sizes[0])}x{int(eval_sizes[0])} "
            f"on {int(image_counts[0])} images per dataset. These rows are not retrained generation baselines."
        )
    else:
        note = "Official pretrained reconstruction. These rows are not retrained generation baselines."
    lines = [
        r"  \begin{tabular}{lrrrrrr}",
        r"    \toprule",
        r"    Dataset & src & eval & FlexTok PSNR & FlexTok low & TiTok PSNR & TiTok low \\",
        r"    \midrule",
    ]
    for dataset, methods in sorted(by_dataset.items()):
        flex = methods.get("ml_flextok", {})
        titok = methods.get("titok_1d_tokenizer", {})
        source_resolution = flex.get("source_resolution") or titok.get("source_resolution")
        eval_size = flex.get("eval_image_size") or titok.get("eval_image_size")
        lines.append(
            "    "
            + " & ".join(
                [
                    _dataset_name(dataset),
                    _fmt(source_resolution, 0),
                    _fmt(eval_size, 0),
                    _fmt(flex.get("reconstruction_psnr_db")),
                    _fmt(flex.get("lowres_frechet_proxy"), 3),
                    _fmt(titok.get("reconstruction_psnr_db")),
                    _fmt(titok.get("lowres_frechet_proxy"), 3),
                ]
            )
            + r" \\"
        )
    lines.extend([r"    \bottomrule", r"  \end{tabular}%"])
    return _table_env(
        "tab:p1-tokenizer-recon-current",
        "Official-pretrained tokenizer reconstruction context (eval-only).",
        "\n".join(lines),
        note,
    )


def _official_related_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        r"  \begin{tabular}{lllrlrrrr}",
        r"    \toprule",
        r"    Method & Dataset & Protocol & Samples & FID & sFID & IS & Prec. & Rec. \\",
        r"    \midrule",
    ]
    for row in sorted(rows, key=lambda item: (str(item.get("method", "")), str(item.get("dataset", "")))):
        lines.append(
            "    "
            + " & ".join(
                [
                    _tex(str(row.get("method", ""))),
                    _dataset_name(str(row.get("dataset", ""))),
                    _tex(_protocol_name(row.get("status"))),
                    _fmt(row.get("sample_count"), 0),
                    _fmt(row.get("fid"), 3),
                    _fmt(row.get("sfid"), 3),
                    _fmt(row.get("inception_score"), 2),
                    _fmt(row.get("precision"), 3),
                    _fmt(row.get("recall"), 3),
                ]
            )
            + r" \\"
        )
    lines.extend([r"    \bottomrule", r"  \end{tabular}%"])
    return _table_env(
        "tab:official-related-eval-only",
        "Official ImageNet-256 related-method evaluations (50K samples).",
        "\n".join(lines),
        "These rows use official pretrained protocols where available and are secondary comparisons; they are not part of the matched-dataset/optimizer-step P0 matrix.",
    )


def _claim_table(evidence: dict[str, Any]) -> str:
    generation = evidence["generation_summary"]
    diagnostics = evidence["diagnostic_summary"]
    repeat = evidence.get("long_budget_repeat", {}).get("summary", {})
    confirmatory = evidence.get("imagenet256_confirmatory_summary", {})
    related = evidence.get("official_related_summary", {})
    core_ready = bool(
        evidence.get("claim_stance", {}).get("core_empirical_gates_ready")
    )
    stance = (
        "The core empirical gates support the scoped claim of ordered restricted dense-noise factorization "
        "with prefix-controllable denoising, not broad unconditional generation superiority."
        if core_ready
        else "One or more core empirical gates remain pending or failed; the scoped submission claim is not yet cleared."
    )
    lines = [
        r"  \begin{tabular}{p{0.24\textwidth}p{0.68\textwidth}}",
        r"    \toprule",
        r"    Claim axis & Evidence summary \\",
        r"    \midrule",
        (
            "    Generation quality & "
            f"CoFiTok is best on {generation['cofitok_best_lowres_count']}/{generation['dataset_count']} lowres rows "
            f"and {generation['cofitok_best_inception_count']}/{generation['dataset_count']} Inception rows; "
            f"average ranks are {generation['cofitok_average_lowres_rank']:.2f} and "
            f"{generation['cofitok_average_inception_rank']:.2f}. Do not claim generation SOTA. \\\\"
        ),
        (
            "    Prefix control & "
            f"CoFiTok path AUC is lower than the endpoint-only factorized control on "
            f"{diagnostics['cofitok_path_auc_better_than_endpoint_only']}/{diagnostics['dataset_count']} datasets. \\\\"
        ),
        (
            "    Repeated 20k evidence & "
            f"CoFiTok lowers path AUC in {repeat.get('path_auc_better_pairs', 0)}/"
            f"{repeat.get('pair_count', 0)} paired dataset-seed runs (mean relative reduction "
            f"{100 * repeat.get('mean_path_auc_relative_reduction', 0.0):.2f}\\%), while endpoint $x_0$ MSE at $t=500$ "
            f"changes by {100 * repeat.get('mean_final_mse_relative_change', 0.0):+.2f}\\% on average. \\\\"
        ),
        (
            "    ImageNet-256 scaling & "
            f"{_tex(str(confirmatory.get('text', 'Confirmatory report pending.')))} \\\\"
        ),
        (
            "    Restricted synthesis & "
            f"CoFiTok zero-token ratio is zero in current rows; deep-$S_k$ has nonzero zero-token ratio on "
            f"{diagnostics['deep_synthesis_nonzero_zero_ratio_count']}/{diagnostics['dataset_count']} datasets. \\\\"
        ),
        (
            "    Related methods & "
            f"{_tex(str(related.get('text', 'No official related-method summary.')))} \\\\"
        ),
        (
            "    Main paper stance & "
            f"{stance} \\\\"
        ),
        r"    \bottomrule",
        r"  \end{tabular}%",
    ]
    return _table_env(
        "tab:claim-support-current",
        "Claim-evidence boundary.",
        "\n".join(lines),
    )


def _preview_markdown(evidence: dict[str, Any]) -> str:
    generation = evidence["generation_summary"]
    diagnostics = evidence["diagnostic_summary"]
    official_rows = evidence.get("official_related_rows", [])
    confirmatory_rows = evidence.get("imagenet256_confirmatory", {}).get("rows", [])
    repeat = evidence.get("long_budget_repeat", {}).get("summary", {})
    return "\n".join(
        [
            "# Paper LaTeX Tables",
            "",
            "Generated from `paper_evidence_report.json`.",
            "",
            "## Included Tables",
            "",
            "- `tab:p0-generation-current`: P0 generation comparison.",
            "- `tab:cofitok-diagnostics-current`: prefix/factorization diagnostics.",
            "- `tab:imagenet256-confirmatory-order`: exhaustive ImageNet-256 order evidence.",
            "- `tab:p1-tokenizer-recon-current`: eval-only FlexTok/TiTok reconstruction.",
            "- `tab:official-related-eval-only`: secondary official related-method eval-only rows.",
            "- `tab:claim-support-current`: concise claim stance.",
            "",
            "## Key Numbers",
            "",
            f"- CoFiTok best lowres rows: {generation['cofitok_best_lowres_count']}/{generation['dataset_count']}.",
            f"- CoFiTok best Inception rows: {generation['cofitok_best_inception_count']}/{generation['dataset_count']}.",
            f"- CoFiTok path AUC better than endpoint-only factorized: {diagnostics['cofitok_path_auc_better_than_endpoint_only']}/{diagnostics['dataset_count']}.",
            f"- Matched 20k paired path-AUC wins: {repeat.get('path_auc_better_pairs', 0)}/{repeat.get('pair_count', 0)}.",
            f"- Mean PSNR gain over channel-mask: {diagnostics['mean_cofitok_minus_channel_mask_psnr']:.3f} dB.",
            f"- Official related-method rows: {len(official_rows)}.",
            "",
            "Use these tables with the scoped claim from `paper_claim_pack.md`.",
        ]
    ) + "\n"


def main() -> None:
    args = parse_args()
    evidence = _read_json(args.evidence)
    generation_rows = evidence["generation_rows"]
    diagnostic_rows = evidence["diagnostic_rows"]
    p1_rows = evidence["p1_rows"]
    official_rows = evidence.get("official_related_rows", [])
    confirmatory_rows = evidence.get("imagenet256_confirmatory", {}).get("rows", [])
    tables = [
        "% Auto-generated from paper_evidence_report.json. Do not hand-edit metrics here.",
        "% Required packages already present in the current AAAI draft: graphicx, booktabs, caption.",
        _p0_generation_table(generation_rows),
        _diagnostic_table(diagnostic_rows),
        _imagenet256_confirmatory_table(confirmatory_rows),
        _p1_table(p1_rows),
        _official_related_table(official_rows),
        _claim_table(evidence),
    ]
    main_tables = [
        "% Auto-generated from paper_evidence_report.json. Do not hand-edit metrics here.",
        "% Main-paper subset: CoFiTok diagnostics and ImageNet-256 confirmatory order evidence.",
        _diagnostic_table(diagnostic_rows),
        _imagenet256_confirmatory_table(confirmatory_rows),
    ]
    appendix_tables = [
        "% Auto-generated from paper_evidence_report.json. Do not hand-edit metrics here.",
        "% Appendix/supplement subset: generation context, tokenizer reconstruction, official related rows, and claim summary.",
        _p0_generation_table(generation_rows),
        _p1_table(p1_rows),
        _official_related_table(official_rows),
        _claim_table(evidence),
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "paper_tables.tex").write_text("\n\n".join(tables) + "\n", encoding="utf-8")
    (args.output_dir / "paper_tables_main.tex").write_text("\n\n".join(main_tables) + "\n", encoding="utf-8")
    (args.output_dir / "paper_tables_appendix.tex").write_text("\n\n".join(appendix_tables) + "\n", encoding="utf-8")
    (args.output_dir / "paper_tables_preview.md").write_text(_preview_markdown(evidence), encoding="utf-8")
    (args.output_dir / "paper_tables_compile_check.tex").write_text(
        "\n".join(
            [
                r"\documentclass{article}",
                r"\usepackage[margin=0.5in]{geometry}",
                r"\usepackage{graphicx}",
                r"\usepackage{booktabs}",
                r"\usepackage{caption}",
                r"\begin{document}",
                r"\input{paper_tables.tex}",
                r"\end{document}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    manifest = {
        "evidence": args.evidence.as_posix(),
        "evidence_sha256": _sha256(args.evidence),
        "output": {
            "compile_check_tex": (args.output_dir / "paper_tables_compile_check.tex").as_posix(),
            "paper_tables_appendix_tex": (args.output_dir / "paper_tables_appendix.tex").as_posix(),
            "paper_tables_main_tex": (args.output_dir / "paper_tables_main.tex").as_posix(),
            "paper_tables_tex": (args.output_dir / "paper_tables.tex").as_posix(),
            "paper_tables_preview": (args.output_dir / "paper_tables_preview.md").as_posix(),
        },
        "row_counts": {
            "generation_rows": len(generation_rows),
            "diagnostic_rows": len(diagnostic_rows),
            "official_related_rows": len(official_rows),
            "imagenet256_confirmatory_rows": len(confirmatory_rows),
            "p1_rows": len(p1_rows),
        },
    }
    (args.output_dir / "paper_tables_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output_dir / 'paper_tables.tex'}")


if __name__ == "__main__":
    main()

