from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "build_figure2_subfigures.py"


def load_module():
    spec = importlib.util.spec_from_file_location("build_figure2_subfigures", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_builds_four_standalone_subfigures_without_baked_labels(tmp_path: Path) -> None:
    module = load_module()
    reports = tmp_path / "reports"
    for index, spec in enumerate(module.SUBFIGURE_SPECS):
        for side, relative in enumerate((spec.left_source, spec.right_source)):
            path = reports / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (40 + side, 30 + index), (30 * index, 40 * side, 100)).save(path)

    output = tmp_path / "figure2"
    manifest = module.build_subfigures(reports, output, gap=7)

    assert manifest["subfigure_count"] == 4
    assert manifest["latex_layout"].startswith("2x2")
    assert len({row["path"] for row in manifest["subfigures"]}) == 4
    for row in manifest["subfigures"]:
        path = Path(row["path"])
        assert path.is_file()
        assert row["sha256"]
        with Image.open(path) as image:
            assert image.width == 40 + 7 + 41
            assert image.height >= 30

    persisted = json.loads((output / "figure2_subfigures_manifest.json").read_text(encoding="utf-8"))
    assert persisted["subfigure_count"] == 4
    assert "No title" in persisted["notes"][1]


def test_pair_images_rejects_negative_gap() -> None:
    module = load_module()
    image = Image.new("RGB", (4, 4), "white")
    try:
        module._pair_images(image, image, -1)
    except ValueError as error:
        assert "gap" in str(error)
    else:
        raise AssertionError("negative gap must fail")
