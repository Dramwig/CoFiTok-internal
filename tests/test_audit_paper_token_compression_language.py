import json
from pathlib import Path

from scripts import audit_paper_token_compression_language as language


def _layout(*, per_token: bool, aggregate: bool) -> dict[str, object]:
    return {
        "decision": {
            "per_token_compression_supported_by_all_reports": per_token,
            "aggregate_sequence_compression_supported_by_all_reports": aggregate,
        }
    }


def test_locked_layout_rejects_compressed_token_wording(tmp_path: Path) -> None:
    paper = tmp_path / "main.tex"
    paper.write_text(
        "We factorize noise into a sequence of compressed denoising tokens.\n",
        encoding="utf-8",
    )

    result = language.audit_language(
        _layout(per_token=False, aggregate=False), [paper]
    )

    assert not result["decision"]["compatible"]
    violation = result["sources"][0]["unsupported_per_token_compression_wording"]
    assert violation[0]["line"] == 1
    assert violation[0]["match"] == "compressed denoising tokens"


def test_locked_layout_accepts_scoped_factorization_wording(tmp_path: Path) -> None:
    paper = tmp_path / "main.tex"
    paper.write_text(
        "We factorize noise into ordered restricted denoising components.\n",
        encoding="utf-8",
    )

    result = language.audit_language(
        _layout(per_token=False, aggregate=False), [paper]
    )

    assert result["decision"]["compatible"]


def test_locked_layout_rejects_split_compression_wording_in_figure_source(
    tmp_path: Path,
) -> None:
    generator = tmp_path / "make_method_figure.py"
    generator.write_text(
        'token_lines = ("compressed", "denoising token")\n',
        encoding="utf-8",
    )

    result = language.audit_language(
        _layout(per_token=False, aggregate=False), [generator]
    )

    assert not result["decision"]["compatible"]
    violation = result["sources"][0]["unsupported_figure_compression_wording"]
    assert violation[0]["match"] == "compressed"


def test_locked_layout_accepts_restricted_wording_in_svg(tmp_path: Path) -> None:
    svg = tmp_path / "method_overview.svg"
    svg.write_text(
        "<text>ordered restricted denoising components</text>\n",
        encoding="utf-8",
    )

    result = language.audit_language(
        _layout(per_token=False, aggregate=False), [svg]
    )

    assert result["decision"]["compatible"]


def test_per_token_evidence_still_rejects_aggregate_sequence_claim(tmp_path: Path) -> None:
    paper = tmp_path / "main.tex"
    paper.write_text(
        "Each token is individually compressed, yielding a compressed token sequence.\n",
        encoding="utf-8",
    )

    result = language.audit_language(
        _layout(per_token=True, aggregate=False), [paper]
    )

    assert not result["decision"]["compatible"]
    assert not result["sources"][0]["unsupported_per_token_compression_wording"]
    assert result["sources"][0][
        "unsupported_aggregate_sequence_compression_wording"
    ]


def test_fully_supported_layout_accepts_compression_language(tmp_path: Path) -> None:
    paper = tmp_path / "main.tex"
    paper.write_text(
        "Each token is individually compressed and the sequence is aggregate-compressed.\n",
        encoding="utf-8",
    )

    result = language.audit_language(
        _layout(per_token=True, aggregate=True), [paper]
    )

    assert result["decision"]["compatible"]
