import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "build_protocol_coverage_report.py"


def load_module():
    spec = importlib.util.spec_from_file_location("build_protocol_coverage_report", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_separates_defined_protocols_from_blocked_product(monkeypatch) -> None:
    module = load_module()
    monkeypatch.setattr(module, "PRIMARY_FAIR_METHODS", {"ours"})
    monkeypatch.setattr(module, "TOKENIZER_EVAL_METHODS", {"tokenizer"})
    monkeypatch.setattr(module, "GUARDED_MATRIX_METHODS", {"d_ar"})
    matrix = {
        "rows": [
            {"dataset": "a", "method": "ours", "status": "completed"},
            {"dataset": "a", "method": "tokenizer", "status": "completed_eval_only"},
            {"dataset": "a", "method": "d_ar", "status": "protocol_blocked"},
        ]
    }
    official = {
        "rows": [
            {"method": "D-AR", "status": "completed_eval_only_50k"},
            {"method": "MAR", "status": "sampling_or_eval_pending"},
            {"method": "ReTok", "status": "completed_eval_only_50k"},
        ]
    }

    payload = module.build(matrix, official)

    assert payload["scientific_status"] == "official_eval_running"

    official["rows"][1]["status"] = "official_ema_sampling_or_eval_pending"
    payload = module.build(matrix, official)
    assert payload["scientific_status"] == "official_eval_running"
    assert payload["views"]["primary_matched_dataset_step_training"]["complete"] is True
    assert payload["views"]["tokenizer_reconstruction_eval_only"]["complete"] is True
    assert payload["views"]["official_imagenet256_eval_only"]["complete"] is False
    assert payload["literal_cartesian_product_complete"] is False
