import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "build_paper_latex_tables.py"
SPEC = importlib.util.spec_from_file_location("build_paper_latex_tables", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_confirmatory_table_exposes_exhaustive_order_statistics() -> None:
    latex = MODULE._imagenet256_confirmatory_table(
        [
            {
                "seed": 103,
                "endpoint_only_path_auc": 1.0,
                "cofitok_ordered_path_auc": 0.2,
                "exhaustive_nonidentity_mean_path_auc": 0.6,
                "cofitok_reverse_path_auc": 0.8,
                "ordered_rank_of_24": 1,
                "nonidentity_delta_ci_low": 0.3,
                "dense_monolithic_endpoint_mse": 0.1,
                "cofitok_endpoint_mse": 0.103,
            }
        ]
    )

    assert "non-ID mean" in latex
    assert "rank/24" in latex
    assert "paired-bootstrap" in latex
    assert "0.3000" in latex
