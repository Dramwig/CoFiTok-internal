import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "build_paper_evidence_report.py"


def load_module():
    spec = importlib.util.spec_from_file_location("build_paper_evidence_report", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_related_method_summary_tracks_completed_and_running() -> None:
    module = load_module()
    summary = module.summarize_official_related(
        [
            {"method": "D-AR", "status": "completed_eval_only_50k"},
            {"method": "MAR", "status": "sampling_or_eval_pending"},
            {"method": "ReTok", "status": "completed_eval_only_50k"},
        ]
    )

    assert summary["completed_50k"] == ["D-AR", "ReTok"]
    assert summary["pending"] == ["MAR"]

    corrected = module.summarize_official_related(
        [
            {
                "method": "MAR",
                "status": "community_non_ema_audit_completed_official_ema_pending",
            }
        ]
    )
    assert corrected["pending"] == ["MAR"]
    assert summary["smoke_only"] == []
    assert "outside P0 matched-dataset/step training" in summary["text"]


def test_confirmatory_summary_preserves_preregistered_decision() -> None:
    module = load_module()
    summary = module.summarize_imagenet256_confirmatory(
        {
            "decision": {
                "overall_pass": True,
                "mean_endpoint_relative_change": 0.032,
                "gates": {"a": True, "b": True, "c": True, "d": True},
            }
        }
    )

    assert summary["available"]
    assert summary["overall_pass"]
    assert summary["gate_pass_count"] == 4
    assert "+3.20%" in summary["text"]


def test_claim_pack_readiness_requires_confirmatory_and_three_official_rows() -> None:
    module = load_module()
    diagnostics = {
        "dataset_count": 8,
        "cofitok_path_auc_better_than_endpoint_only": 8,
        "cofitok_zero_ratio_zero_count": 8,
    }
    long_budget = {
        "summary": {
            "pair_count": 4,
            "path_auc_better_pairs": 4,
            "mean_final_mse_relative_change": 0.03,
        }
    }
    related = {"completed_50k": ["D-AR", "MAR", "ReTok"]}

    assert module.scoped_claim_pack_ready(
        diagnostics, long_budget, {"overall_pass": True}, related
    )
    assert not module.scoped_claim_pack_ready(
        diagnostics, long_budget, {"overall_pass": False}, related
    )
