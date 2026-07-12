import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "build_top_tier_claim_audit.py"
SPEC = importlib.util.spec_from_file_location("build_top_tier_claim_audit", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_scoped_claim_requires_all_protocol_and_scaling_gates() -> None:
    evidence = {
        "diagnostic_summary": {
            "dataset_count": 8,
            "cofitok_path_auc_better_than_endpoint_only": 8,
        },
        "diagnostic_rows": [{"cofitok_zero_ratio": 0.0} for _ in range(8)],
        "long_budget_repeat": {
            "summary": {
                "pair_count": 4,
                "path_auc_better_pairs": 4,
                "mean_final_mse_relative_change": 0.03,
            }
        },
        "imagenet256_confirmatory_summary": {"overall_pass": True},
        "official_related_summary": {"completed_50k": ["D-AR", "MAR", "ReTok"]},
        "generation_summary": {
            "dataset_count": 8,
            "cofitok_best_lowres_count": 0,
            "cofitok_best_inception_count": 1,
        },
    }
    coverage = {"scientific_status": "completed_defined_protocols"}

    result = MODULE.audit(evidence, coverage)

    assert result["decision"]["scoped_top_tier_evidence_ready"]
    assert not result["decision"]["broad_generation_superiority_supported"]
    assert not result["decision"]["top_tier_acceptance_guaranteed"]


def test_scoped_claim_fails_when_imagenet_confirmatory_fails() -> None:
    evidence = {
        "diagnostic_summary": {
            "dataset_count": 8,
            "cofitok_path_auc_better_than_endpoint_only": 8,
        },
        "diagnostic_rows": [{"cofitok_zero_ratio": 0.0} for _ in range(8)],
        "long_budget_repeat": {
            "summary": {
                "pair_count": 4,
                "path_auc_better_pairs": 4,
                "mean_final_mse_relative_change": 0.03,
            }
        },
        "imagenet256_confirmatory_summary": {"overall_pass": False},
        "official_related_summary": {"completed_50k": ["D-AR", "MAR", "ReTok"]},
        "generation_summary": {"dataset_count": 8},
    }
    coverage = {"scientific_status": "completed_defined_protocols"}

    result = MODULE.audit(evidence, coverage)

    assert not result["decision"]["scoped_top_tier_evidence_ready"]
    assert not result["checks"]["imagenet256_confirmatory_all_gates"]


def test_missing_zero_token_evidence_cannot_pass() -> None:
    evidence = {
        "diagnostic_summary": {
            "dataset_count": 8,
            "cofitok_path_auc_better_than_endpoint_only": 8,
        },
        "diagnostic_rows": [{"cofitok_zero_ratio": 0.0} for _ in range(7)]
        + [{}],
        "long_budget_repeat": {
            "summary": {
                "pair_count": 4,
                "path_auc_better_pairs": 4,
                "mean_final_mse_relative_change": 0.03,
            }
        },
        "imagenet256_confirmatory_summary": {"overall_pass": True},
        "official_related_summary": {
            "completed_50k": ["D-AR", "MAR", "ReTok"]
        },
        "generation_summary": {"dataset_count": 8},
    }
    coverage = {"scientific_status": "completed_defined_protocols"}

    result = MODULE.audit(evidence, coverage)

    assert not result["checks"]["restricted_zero_token_contract"]
    assert not result["decision"]["scoped_top_tier_evidence_ready"]
