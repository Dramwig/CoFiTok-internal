import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "evaluate_quality.py"
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("evaluate_quality", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_prefix_budgets_keep_endpoint_and_sort() -> None:
    assert MODULE._prefix_budgets("4,1,2,2", token_count=4) == [1, 2, 4]
    assert MODULE._prefix_budgets("1,2", token_count=4) == [1, 2, 4]


def test_evaluation_progress_power_cli_is_optional(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_quality.py",
            "--config",
            "config.json",
            "--checkpoint",
            "checkpoint.pt",
            "--output-dir",
            "out",
            "--evaluation-progress-power",
            "1.5",
        ],
    )
    assert MODULE.parse_args().evaluation_progress_power == 1.5
