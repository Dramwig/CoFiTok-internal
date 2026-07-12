from pathlib import Path

from scripts.validate_goal_completion import (
    check_paper_artifacts,
    downsampled_imagenet64_artifacts,
    formal_generated_quality_evidence,
    official_fid_evidence,
    official_fid_protocol_artifacts,
    render_markdown,
    strict_gaps,
    venue_template_artifacts,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_minimal_paper(project_root: Path, draft_text: str = "") -> None:
    required_text_files = [
        "paper/README.md",
        "paper/references.bib",
        "paper/citation_audit.md",
        "paper/full_pdf_claim_audit_sources.json",
        "paper/latex/README.md",
        "paper/latex/Makefile",
    ]
    for relative in required_text_files:
        _write(project_root / relative, f"payload for {relative}\n")
    _write(
        project_root / "paper/latex/main.tex",
        draft_text or "CoFiTok scoped draft.\n",
    )
    pdf = project_root / "paper/latex/main.pdf"
    pdf.parent.mkdir(parents=True, exist_ok=True)
    pdf.write_bytes(b"%PDF-1.5\n" + (b"0" * 1_000_000))


def _write_minimal_aaai27(project_root: Path) -> None:
    venue_root = project_root / "paper/venues/aaai27"
    _write(venue_root / "README.md", "AAAI-27 submission build.\n")
    _write(venue_root / "Makefile", "all:\n\tlatexmk -pdf main.tex\n")
    _write(venue_root / "aaai2027.sty", "style payload\n")
    _write(venue_root / "aaai2027.bst", "bst payload\n")
    _write(venue_root / "ReproducibilityChecklist.tex", "checklist payload\n")
    _write(
        venue_root / "main.tex",
        "\n".join(
            [
                r"\documentclass[letterpaper]{article}",
                r"\usepackage[submission]{aaai2027}",
                r"\author{Anonymous Submission}",
                r"\affiliations{}",
                r"\begin{document}",
                r"\maketitle",
                r"\bibliography{../../references}",
                r"\end{document}",
            ]
        ),
    )
    _write(venue_root / "main_aaai2027.tex", "\\input{main.tex}\n")
    _write(
        venue_root / "supplementary_aaai2027.tex",
        "\\documentclass[letterpaper]{article}\n\\begin{document}Supplement\\end{document}\n",
    )
    for filename, payload_bytes in (
        ("main_aaai2027.pdf", 1_000_000),
        ("supplementary_aaai2027.pdf", 200_000),
    ):
        (venue_root / filename).write_bytes(b"%PDF-1.5\n" + (b"0" * payload_bytes))


def _write_minimal_downsampled_source(internal_root: Path) -> None:
    conditions = internal_root / "docs/experiment_conditions"
    _write(
        conditions / "downsampled_imagenet_64_2026-07-09.md",
        """
# downsampled_imagenet_64 Dataset Record
Completed and verified from the strict Academic Torrents source.
https://academictorrents.com/details/96816a530ee002254d29bf7a61c0c158d3dedc3b
96816a530ee002254d29bf7a61c0c158d3dedc3b
aria2c magnet+DHT on pro6000
/root/autodl-tmp/CoFiTok/datasets/downsampled_imagenet_64/raw/small/train_64x64.tar
/root/autodl-tmp/CoFiTok/datasets/downsampled_imagenet_64/raw/small/valid_64x64.tar
e2a2c1947a748d0d256e98e6d82800d855441dc9e4de2e71a773f1324c4524d7
eeac03ec4f585baf871d2b4cb9c974bfe345e74878a74e5820fe6eb1041629cd
1,281,149
49,999
No crop, resize, normalization
""",
    )
    _write(
        conditions / "downsampled_imagenet_64_manifest_summary_2026-07-09.json",
        """
{
  "alias": "downsampled_imagenet_64",
  "status": "completed",
  "source": {"info_hash": "96816a530ee002254d29bf7a61c0c158d3dedc3b"},
  "raw_files": [
    {
      "name": "96816a530ee002254d29bf7a61c0c158d3dedc3b.torrent",
      "bytes": 240592,
      "sha256": "1104077da8c88db149d74ffb6d1f9fe17b32abf535880966ee1684559820cdfa"
    },
    {
      "name": "train_64x64.tar",
      "bytes": 12112588800,
      "sha256": "e2a2c1947a748d0d256e98e6d82800d855441dc9e4de2e71a773f1324c4524d7"
    },
    {
      "name": "valid_64x64.tar",
      "bytes": 477255680,
      "sha256": "eeac03ec4f585baf871d2b4cb9c974bfe345e74878a74e5820fe6eb1041629cd"
    }
  ],
  "splits": {"train": {"png_files": 1281149}, "valid": {"png_files": 49999}},
  "sample_png_verification": {
    "train": [{"name": "0000001.png", "size": [64, 64]}],
    "valid": [{"name": "00001.png", "size": [64, 64]}]
  }
}
""",
    )


def _fullval_quality_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset, variant, image_count in [
        ("tiny_imagenet_200", "epsilon_only", 10000),
        ("tiny_imagenet_200", "light_denoise_path", 10000),
        ("imagenet_1k_64x64_hf", "epsilon_only", 50000),
        ("imagenet_1k_64x64_hf", "light_denoise_path", 50000),
    ]:
        rows.append(
            {
                "dataset": dataset,
                "variant": variant,
                "token_count": 8,
                "steps": 20000,
                "image_count": image_count,
                "final_mse": 0.1,
                "mse_auc": 6.0,
                "lowres_frechet_proxy_auc": 5.0,
                "report_dir": f"quality_{dataset}_{variant}_fullval",
            }
        )
    return rows


def _multiscale_20k_train_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset in ["tiny_imagenet_200", "imagenet_1k_64x64_hf"]:
        rows.append(
            {
                "dataset": dataset,
                "variant": "light_denoise_path",
                "token_count": 8,
                "steps": 20000,
                "predictor_type": "multiscale_unet",
                "synthesis_mode": "restricted",
                "final_clean_mse": 0.1,
                "path_auc": 0.03,
                "effective_tokens": 6.7,
                "zero_token_ratio": 0.0,
                "shuffled_final_ratio": 100.0,
                "config_name": f"train_{dataset}_k8_denoisepath_p150_light_multiscale_20k_cuda",
                "report_dir": f"train_{dataset}_k8_denoisepath_p150_light_multiscale_20k_2026-07-08",
            }
        )
    return rows


def _formal_generated_quality_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for variant, inception in [("epsilon_only", 240.0), ("light_denoise_path", 269.0)]:
        rows.append(
            {
                "dataset": "imagenet_1k_64x64_hf",
                "variant": variant,
                "token_count": 8,
                "steps": 20000,
                "sample_image_count": 50000,
                "real_image_count": 50000,
                "inception_frechet": inception,
                "lowres_frechet_proxy": 6.5,
                "predictor_type": "tiny_conv",
                "report_dir": f"generated_quality_stream_hf_{variant}_50000",
            }
        )
    return rows


def _formal_generated_quality_rows_with_multiscale_controls() -> list[dict[str, object]]:
    rows = _formal_generated_quality_rows()
    for variant, inception in [("epsilon_only", 134.2), ("light_denoise_path", 136.8)]:
        rows.append(
            {
                "dataset": "imagenet_1k_64x64_hf",
                "variant": variant,
                "token_count": 8,
                "steps": 20000,
                "sample_image_count": 50000,
                "real_image_count": 50000,
                "inception_frechet": inception,
                "lowres_frechet_proxy": 5.6,
                "predictor_type": "multiscale_unet",
                "report_dir": f"generated_quality_stream_hf_{variant}_multiscale_50000",
            }
        )
    return rows


def _official_fid_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset, generated_count in [("tiny_imagenet_200", 10000), ("imagenet_1k_64x64_hf", 50000)]:
        for variant, fid in [("epsilon_only", 140.0), ("light_denoise_path", 134.0)]:
            rows.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "token_count": 8,
                    "steps": 20000,
                    "predictor_type": "multiscale_unet",
                    "generated_image_count": generated_count,
                    "real_image_count": generated_count,
                    "official_fid": fid,
                    "implementation_package": "pytorch-fid",
                    "status": "ok",
                    "sample_steps": 50,
                    "prefix_budget": 8,
                    "report_dir": f"official_fid_export_{dataset}_{variant}_multiscale_20k",
                }
            )
    return rows


def test_check_paper_artifacts_accepts_negated_generation_quality_claim(tmp_path) -> None:
    project_root = tmp_path / "project"
    _write_minimal_paper(
        project_root,
        "CoFiTok should not claim unconditional generation quality wins at this scale.\n",
    )

    gate = check_paper_artifacts(project_root)

    assert gate.status == "ok"
    assert gate.missing == []
    assert gate.evidence["forbidden_claims_present"] == []
    assert gate.evidence["venue_template"]["status"] == "missing"


def test_check_paper_artifacts_rejects_positive_overclaim(tmp_path) -> None:
    project_root = tmp_path / "project"
    _write_minimal_paper(
        project_root,
        "CoFiTok achieves unconditional generation quality wins at this scale.\n",
    )

    gate = check_paper_artifacts(project_root)

    assert gate.status == "missing"
    assert "cofitok achieves unconditional generation quality wins" in gate.evidence["forbidden_claims_present"]


def test_venue_template_artifacts_accepts_aaai27_submission_build(tmp_path) -> None:
    project_root = tmp_path / "project"
    _write_minimal_aaai27(project_root)

    evidence = venue_template_artifacts(project_root)

    assert evidence["status"] == "ok"
    assert evidence["missing"] == []
    assert evidence["template"] == "aaai27"


def test_strict_gaps_keep_completion_open_for_scoped_mvp_summary() -> None:
    summary = {
        "generated_quality": [{"sample_image_count": 2048, "real_image_count": 8192}],
        "quality": [{"image_count": 1024}],
        "train": [
            {
                "variant": "light_denoise_path",
                "token_count": 8,
                "steps": 10000,
                "predictor_type": "multiscale_unet",
            }
        ],
    }

    gaps = strict_gaps(summary)

    assert [gap.name for gap in gaps] == [
        "official_generation_quality_protocol",
        "full_validation_reconstruction_sweep",
        "longer_stronger_backbone_validation",
        "exact_downsampled_imagenet64_source",
        "venue_template_and_submission_format",
    ]
    assert all(gap.status == "open" for gap in gaps)
    assert gaps[0].evidence["max_generated_samples"] == 2048
    assert gaps[0].evidence["max_real_images"] == 8192
    assert gaps[0].evidence["hf_50k_streamed_protocol"]["status"] == "missing"


def test_strict_gaps_accept_full_validation_quality_evidence() -> None:
    summary = {
        "generated_quality": [{"sample_image_count": 2048, "real_image_count": 8192}],
        "quality": _fullval_quality_rows(),
        "train": [
            {
                "variant": "light_denoise_path",
                "token_count": 8,
                "steps": 10000,
                "predictor_type": "multiscale_unet",
            }
        ],
    }

    gaps = strict_gaps(summary)

    assert [gap.name for gap in gaps] == [
        "official_generation_quality_protocol",
        "longer_stronger_backbone_validation",
        "exact_downsampled_imagenet64_source",
        "venue_template_and_submission_format",
    ]


def test_strict_gaps_accept_full_validation_and_multiscale_20k_evidence() -> None:
    summary = {
        "generated_quality": [{"sample_image_count": 2048, "real_image_count": 8192}],
        "quality": _fullval_quality_rows(),
        "train": _multiscale_20k_train_rows(),
    }

    gaps = strict_gaps(summary)

    assert [gap.name for gap in gaps] == [
        "official_generation_quality_protocol",
        "exact_downsampled_imagenet64_source",
        "venue_template_and_submission_format",
    ]


def test_official_generation_gap_uses_largest_generated_protocol() -> None:
    summary = {
        "generated_quality": [
            {"sample_image_count": 2048, "real_image_count": 8192},
            {"sample_image_count": 8192, "real_image_count": 8192},
        ],
        "quality": _fullval_quality_rows(),
        "train": _multiscale_20k_train_rows(),
    }

    gap = strict_gaps(summary)[0]

    assert gap.name == "official_generation_quality_protocol"
    assert gap.evidence["max_generated_samples"] == 8192
    assert gap.evidence["max_real_images"] == 8192
    assert gap.evidence["hf_50k_streamed_protocol"]["status"] == "missing"
    assert "8192/8192" in gap.reason
    assert "50k official FID" in gap.reason
    assert "robust unconditional generation-quality win" in gap.reason


def test_formal_generated_quality_evidence_accepts_hf_50k_rows() -> None:
    evidence = formal_generated_quality_evidence({"generated_quality": _formal_generated_quality_rows()})

    assert evidence["status"] == "ok"
    assert evidence["missing"] == []
    assert evidence["imagenet_1k_64x64_hf/epsilon_only"]["sample_image_count"] == 50000
    assert evidence["imagenet_1k_64x64_hf/light_denoise_path"]["real_image_count"] == 50000


def test_formal_generated_quality_evidence_prefers_multiscale_when_counts_match() -> None:
    evidence = formal_generated_quality_evidence(
        {"generated_quality": _formal_generated_quality_rows_with_multiscale_controls()}
    )

    assert evidence["status"] == "ok"
    assert evidence["imagenet_1k_64x64_hf/epsilon_only"]["predictor_type"] == "multiscale_unet"
    assert evidence["imagenet_1k_64x64_hf/light_denoise_path"]["predictor_type"] == "multiscale_unet"


def test_generation_gap_remains_open_after_hf_50k_if_quality_win_is_not_claimed() -> None:
    summary = {
        "generated_quality": _formal_generated_quality_rows_with_multiscale_controls(),
        "quality": _fullval_quality_rows(),
        "train": _multiscale_20k_train_rows(),
    }

    gap = strict_gaps(summary)[0]

    assert gap.name == "official_generation_quality_protocol"
    assert gap.status == "open"
    assert gap.evidence["max_generated_samples"] == 50000
    assert gap.evidence["max_real_images"] == 50000
    assert gap.evidence["hf_50k_streamed_protocol"]["status"] == "ok"
    assert (
        gap.evidence["hf_50k_streamed_protocol"]["imagenet_1k_64x64_hf/epsilon_only"]["predictor_type"]
        == "multiscale_unet"
    )
    assert "50000/50000" in gap.reason
    assert "matched multiscale 20k backbone" in gap.reason
    assert "robust unconditional generation-quality win" in gap.reason


def test_official_fid_evidence_accepts_four_matched_reports() -> None:
    evidence = official_fid_evidence({"official_fid": _official_fid_rows()})

    assert evidence["status"] == "ok"
    assert evidence["missing"] == []
    assert evidence["tiny_imagenet_200/epsilon_only"]["generated_image_count"] == 10000
    assert evidence["imagenet_1k_64x64_hf/light_denoise_path"]["official_fid"] == 134.0


def test_official_fid_reports_close_generation_gap() -> None:
    summary = {
        "generated_quality": _formal_generated_quality_rows_with_multiscale_controls(),
        "official_fid": _official_fid_rows(),
        "quality": _fullval_quality_rows(),
        "train": _multiscale_20k_train_rows(),
    }

    gaps = strict_gaps(summary)

    assert [gap.name for gap in gaps] == [
        "exact_downsampled_imagenet64_source",
        "venue_template_and_submission_format",
    ]


def test_strict_gaps_close_venue_template_when_aaai27_build_exists(tmp_path) -> None:
    project_root = tmp_path / "project"
    _write_minimal_aaai27(project_root)
    summary = {
        "generated_quality": _formal_generated_quality_rows_with_multiscale_controls(),
        "official_fid": _official_fid_rows(),
        "quality": _fullval_quality_rows(),
        "train": _multiscale_20k_train_rows(),
    }

    gaps = strict_gaps(summary, project_root=project_root)

    assert [gap.name for gap in gaps] == ["exact_downsampled_imagenet64_source"]


def test_downsampled_imagenet64_artifacts_detect_ready_source(tmp_path: Path) -> None:
    _write_minimal_downsampled_source(tmp_path)

    evidence = downsampled_imagenet64_artifacts(tmp_path)

    assert evidence["status"] == "ok"
    assert evidence["missing"] == []
    assert evidence["evidence"]["source_info_hash"] == "96816a530ee002254d29bf7a61c0c158d3dedc3b"


def test_strict_gaps_close_exact_downsampled_when_source_record_exists(tmp_path) -> None:
    project_root = tmp_path / "project"
    internal_root = tmp_path / "internal"
    _write_minimal_aaai27(project_root)
    _write_minimal_downsampled_source(internal_root)
    summary = {
        "generated_quality": _formal_generated_quality_rows_with_multiscale_controls(),
        "official_fid": _official_fid_rows(),
        "quality": _fullval_quality_rows(),
        "train": _multiscale_20k_train_rows(),
    }

    gaps = strict_gaps(summary, internal_root=internal_root, project_root=project_root)

    assert gaps == []


def test_official_fid_protocol_artifacts_detect_ready_files(tmp_path) -> None:
    for relative in [
        "scripts/export_official_fid_dirs.py",
        "scripts/evaluate_official_fid_dirs.py",
        "artifacts/runbooks/official_fid_protocol_2026-07-08.sh",
        "docs/records/2026-07-08_official_fid_protocol.md",
    ]:
        _write(tmp_path / relative, "payload\n")

    evidence = official_fid_protocol_artifacts(tmp_path)

    assert evidence["status"] == "ready"
    assert evidence["missing"] == []


def test_strict_gap_mentions_ready_official_fid_protocol(tmp_path) -> None:
    for relative in [
        "scripts/export_official_fid_dirs.py",
        "scripts/evaluate_official_fid_dirs.py",
        "artifacts/runbooks/official_fid_protocol_2026-07-08.sh",
        "docs/records/2026-07-08_official_fid_protocol.md",
    ]:
        _write(tmp_path / relative, "payload\n")
    summary = {
        "generated_quality": _formal_generated_quality_rows_with_multiscale_controls(),
        "quality": _fullval_quality_rows(),
        "train": _multiscale_20k_train_rows(),
    }

    gap = strict_gaps(summary, internal_root=tmp_path)[0]

    assert gap.evidence["official_fid_protocol_artifacts"]["status"] == "ready"
    assert "pytorch-fid export/evaluation protocol exists" in gap.reason


def test_render_markdown_reports_scoped_ready_with_open_gaps() -> None:
    result = {
        "status": "scoped_ready_with_open_gaps",
        "scoped_claim_ready": True,
        "strict_completion_ready": False,
        "gates": [{"name": "paper_artifacts", "status": "ok", "evidence": {"status": "ok"}, "missing": []}],
        "strict_gaps": [
            {
                "name": "official_generation_quality_protocol",
                "status": "open",
                "reason": "Generated-sample evidence is smoke-scale.",
                "next_action": "Run formal FID only if needed.",
            }
        ],
        "summary_counts": {"train": 99},
    }

    markdown = render_markdown(result)

    assert "Status: `scoped_ready_with_open_gaps`" in markdown
    assert "Strict completion ready: `False`" in markdown
    assert "`official_generation_quality_protocol`" in markdown
