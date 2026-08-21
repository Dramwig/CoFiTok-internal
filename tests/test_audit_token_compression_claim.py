import hashlib
import json
from pathlib import Path

import pytest

from scripts import audit_token_compression_claim as audit


def _report(path: Path, model: dict[str, object]) -> Path:
    path.write_text(json.dumps({"config": {"model": model}}), encoding="utf-8")
    return path


def test_full_resolution_synthesis_mask_is_not_token_compression(tmp_path: Path) -> None:
    report = _report(
        tmp_path / "legacy.json",
        {
            "image_size": 64,
            "image_channels": 3,
            "token_count": 8,
            "token_channels": 16,
            "predictor_type": "tiny_conv",
            "predictor_use_feedback": True,
            "synthesis_mode": "restricted",
            "synthesis_active_token_channels": [4, 4, 8, 8, 12, 12, 16, 16],
            "synthesis_token_strides": [],
        },
    )

    result = audit.audit_reports([report])

    row = result["reports"][0]["audit"]
    assert row["emitted_token_layout"] == "full_resolution_token_fields"
    assert row["layout"]["dense_scalar_count"] == 3 * 64 * 64
    assert row["layout"]["scalar_counts"] == [16 * 64 * 64] * 8
    assert not row["claims"]["per_token_compression_supported"]
    assert not row["claims"]["aggregate_sequence_compression_supported"]
    assert "masks S_k inputs" in " ".join(row["notes"])
    assert result["decision"]["allowed_paper_wording"] == (
        "ordered restricted denoising components"
    )


def test_true_compressed_layout_separates_per_token_and_aggregate_claims(
    tmp_path: Path,
) -> None:
    report = _report(
        tmp_path / "rgbtail3.json",
        {
            "image_size": 256,
            "image_channels": 3,
            "token_count": 8,
            "token_channels": 8,
            "token_channel_schedule": [4, 4, 8, 8, 8, 1, 1, 1],
            "token_spatial_strides": [16, 16, 8, 8, 4, 1, 1, 1],
            "predictor_type": "scalable_unet",
            "predictor_use_feedback": True,
            "synthesis_mode": "fixed_basis",
        },
    )

    result = audit.audit_reports([report])

    row = result["reports"][0]["audit"]
    assert row["claims"]["per_token_compression_supported"]
    assert not row["claims"]["aggregate_sequence_compression_supported"]
    assert row["layout"]["total_scalar_count"] == 247_808
    assert row["layout"]["dense_scalar_count"] == 196_608
    assert math_isclose(row["layout"]["aggregate_token_to_dense_ratio"], 1.2604166666666667)
    assert result["decision"]["allowed_paper_wording"] == (
        "individually compressed denoising tokens"
    )
    assert not result["decision"][
        "aggregate_sequence_compression_supported_by_all_reports"
    ]


def test_mixed_evidence_set_fails_compressed_token_claim(tmp_path: Path) -> None:
    legacy = _report(
        tmp_path / "legacy.json",
        {
            "image_size": 32,
            "image_channels": 3,
            "token_count": 4,
            "token_channels": 8,
        },
    )
    compressed = _report(
        tmp_path / "compressed.json",
        {
            "image_size": 32,
            "image_channels": 3,
            "token_count": 4,
            "token_channels": 4,
            "token_channel_schedule": [1, 1, 2, 3],
            "token_spatial_strides": [8, 4, 2, 1],
        },
    )

    result = audit.audit_reports([legacy, compressed])

    assert result["decision"]["report_count"] == 2
    assert not result["decision"]["per_token_compression_supported_by_all_reports"]
    assert result["decision"]["compressed_token_wording_requires_separate_evidence"]


def test_source_identity_and_ambiguous_model_config_fail_closed(tmp_path: Path) -> None:
    report = _report(
        tmp_path / "one.json",
        {
            "image_size": 32,
            "image_channels": 3,
            "token_count": 4,
            "token_channels": 8,
        },
    )
    result = audit.audit_reports([report])
    raw = report.read_bytes()
    assert result["reports"][0]["source"]["sha256"] == hashlib.sha256(raw).hexdigest()

    ambiguous = tmp_path / "ambiguous.json"
    ambiguous.write_text(
        json.dumps(
            {
                "first": {"model": json.loads(report.read_text())["config"]["model"]},
                "second": {
                    "model": {
                        "image_size": 64,
                        "image_channels": 3,
                        "token_count": 8,
                        "token_channels": 16,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exactly one resolved model config"):
        audit.audit_reports([ambiguous])


def test_partial_variable_shape_config_is_rejected(tmp_path: Path) -> None:
    report = _report(
        tmp_path / "partial.json",
        {
            "image_size": 32,
            "image_channels": 3,
            "token_count": 4,
            "token_channels": 8,
            "token_channel_schedule": [1, 2, 4, 8],
        },
    )
    with pytest.raises(ValueError, match="must either both be populated"):
        audit.audit_reports([report])


def test_compact_payload_keeps_claim_and_source_bindings(tmp_path: Path) -> None:
    report = _report(
        tmp_path / "compressed.json",
        {
            "image_size": 32,
            "image_channels": 3,
            "token_count": 4,
            "token_channels": 3,
            "token_channel_schedule": [1, 1, 2, 3],
            "token_spatial_strides": [8, 4, 2, 1],
        },
    )
    full = audit.audit_reports([report])
    compact = audit.compact_payload(full)

    assert compact["decision"] == full["decision"]
    assert compact["reports"][0]["source"] == full["reports"][0]["source"]
    assert compact["reports"][0]["claims"] == full["reports"][0]["audit"]["claims"]
    assert "issues" not in compact["reports"][0]["layout"]


def math_isclose(left: float, right: float) -> bool:
    return abs(left - right) <= 1e-12
