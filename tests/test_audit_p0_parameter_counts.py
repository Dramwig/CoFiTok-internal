import importlib.util
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "baselines" / "audit_p0_parameter_counts.py"
SPEC = importlib.util.spec_from_file_location("audit_p0_parameter_counts", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_audit_improved_diffusion_state_dict(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ema.pt"
    torch.save(
        {
            "weight": torch.zeros(3, 4),
            "bias": torch.zeros(3),
        },
        checkpoint,
    )
    report = tmp_path / "baseline_train_report.json"
    report.write_text(
        json.dumps(
            {
                "baseline": "improved_diffusion",
                "dataset": "cifar10",
                "run_name": "run",
                "repo_commit": "abc",
                "checkpoints": {"sample_checkpoint": str(checkpoint)},
            }
        ),
        encoding="utf-8",
    )

    row = MODULE.audit_report(report)

    assert row["parameter_count"] == 15
    assert row["count_method"] == "ema_state_dict_tensor_elements"


def test_contract_requires_unique_full_pinned_cartesian_product(monkeypatch) -> None:
    monkeypatch.setattr(MODULE, "SUPPORTED", {"a", "b"})
    monkeypatch.setattr(MODULE, "DATASETS", {"x", "y"})
    monkeypatch.setattr(MODULE, "PINNED_COMMITS", {"a": "ca", "b": "cb"})
    rows = [
        {
            "baseline": baseline,
            "dataset": dataset,
            "repo_commit": "ca" if baseline == "a" else "cb",
            "parameter_count": 10,
        }
        for baseline in ("a", "b")
        for dataset in ("x", "y")
    ]

    assert MODULE.contract_violations(rows) == []
    rows[-1]["repo_commit"] = "wrong"
    assert any("commit mismatch" in value for value in MODULE.contract_violations(rows))
