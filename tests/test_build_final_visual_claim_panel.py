import importlib.util
import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "build_final_visual_claim_panel.py"
SPEC = importlib.util.spec_from_file_location("build_final_visual_claim_panel", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_panel_uses_endpoint_only_diagnostic_key(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.json"
    evidence.write_text(
        json.dumps(
            {
                "generation_summary": {
                    "cofitok_best_lowres_count": 1,
                    "dataset_count": 8,
                },
                "diagnostic_summary": {
                    "cofitok_path_auc_better_than_endpoint_only": 8,
                    "dataset_count": 8,
                    "deep_synthesis_nonzero_zero_ratio_count": 8,
                },
            }
        ),
        encoding="utf-8",
    )
    visuals = tmp_path / "visuals"
    visuals.mkdir()
    for name in ("prefix_comparison_20k.png", "sk_diagnostic_panel.png"):
        Image.new("RGB", (64, 64), "white").save(visuals / name)

    manifest = MODULE.build_panel(evidence, visuals, tmp_path / "output")

    assert Path(manifest["path"]).is_file()
    assert manifest["metrics"]["cofitok_path_auc_better_than_endpoint_only"] == 8
    assert manifest["metrics"]["core_empirical_gates_ready"] is False
    assert "remains pending" in manifest["notes"][1]
