from __future__ import annotations

import re
import subprocess
from pathlib import Path


INTERNAL_ROOT = Path(__file__).resolve().parents[1]


def _project_root() -> Path:
    direct_parent = INTERNAL_ROOT.parent
    if (direct_parent / "paper").is_dir():
        return direct_parent
    result = subprocess.run(
        [
            "git",
            "rev-parse",
            "--path-format=absolute",
            "--git-common-dir",
        ],
        cwd=INTERNAL_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    common_dir = Path(result.stdout.strip()).resolve()
    project_root = common_dir.parent.parent
    if not (project_root / "paper").is_dir():
        raise FileNotFoundError(
            f"CoFiTok paper root is unavailable from {INTERNAL_ROOT}"
        )
    return project_root


ROOT = _project_root()
MAIN = ROOT / "paper" / "venues" / "aaai27" / "main.tex"
VENUE_NEUTRAL_MAIN = ROOT / "paper" / "latex" / "main.tex"


def test_experiments_has_exactly_setup_results_and_ablations() -> None:
    source = MAIN.read_text(encoding="utf-8")
    experiments = source.split(r"\section{Experiments}", 1)[1].split(
        r"\section{Discussion}", 1
    )[0]
    subsections = re.findall(r"^\s*\\subsection\{([^}]+)\}", experiments, flags=re.MULTILINE)
    assert subsections == ["Setup", "Results", "Ablations"]


def test_old_experiment_subsection_titles_are_removed() -> None:
    source = MAIN.read_text(encoding="utf-8")
    forbidden = {
        "Experimental Setup",
        "Current All-Dataset Evidence Audit",
        "Endpoint Quality and Prefixes",
        "Order Sensitivity",
        "Restricted Synthesis and Degeneration",
        "Component Separation Pressure",
        "Fixed-Timestep and Generated-Sample Quality",
    }
    assert all(rf"\subsection{{{title}}}" not in source for title in forbidden)


def test_figure2_is_latex_layout_of_four_standalone_files() -> None:
    source = MAIN.read_text(encoding="utf-8")
    for filename in (
        "figure2a_tiny_prefix.png",
        "figure2b_imagenet64_prefix.png",
        "figure2c_restricted_synthesis.png",
        "figure2d_deep_synthesis.png",
    ):
        assert source.count(filename) == 1
    assert "cofitok_final_visual_claim_panel.png" not in source
    marker = source.index("figure2a_tiny_prefix.png")
    start = source.rfind(r"\begin{figure*}[t]", 0, marker)
    end = source.index(r"\end{figure*}", marker)
    figure = source[start:end]
    assert figure.count(r"\begin{minipage}") == 4


def test_venue_neutral_figure2_uses_the_same_four_standalone_files() -> None:
    source = VENUE_NEUTRAL_MAIN.read_text(encoding="utf-8")
    for filename in (
        "figure2a_tiny_prefix.png",
        "figure2b_imagenet64_prefix.png",
        "figure2c_restricted_synthesis.png",
        "figure2d_deep_synthesis.png",
    ):
        assert source.count(filename) == 1
    assert "prefix_comparison_20k.png" not in source
    assert "sk_diagnostic_panel.png" not in source
    marker = source.index("figure2a_tiny_prefix.png")
    start = source.rfind(r"\begin{figure}[t]", 0, marker)
    end = source.index(r"\end{figure}", marker)
    figure = source[start:end]
    assert figure.count(r"\begin{minipage}") == 4
    assert r"\label{fig:final-visual}" in figure
