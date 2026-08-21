import json
from pathlib import Path

from PIL import Image

from scripts.make_method_figure import (
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    METHOD_SUBTITLE,
    build_method_scene,
    make_method_figure,
)


def test_method_scene_encodes_core_constraints() -> None:
    boxes, arrows = build_method_scene()
    joined = " ".join([box.title for box in boxes] + [line for box in boxes for line in box.lines])

    assert "Default S_k restrictions" in joined
    assert "no x_t, t, label, prompt, z_<k, skips, or constants" in joined
    assert "S_k(0)=0" in joined
    assert any(arrow.dashed and arrow.label == "blocked into S_k" for arrow in arrows)


def test_make_method_figure_writes_png_svg_and_manifest(tmp_path) -> None:
    manifest = make_method_figure(tmp_path / "method")

    assert manifest["artifact_count"] == 2
    assert manifest["subtitle"] == METHOD_SUBTITLE
    assert "S_k receives only the current token z_k." in manifest["constraints"]

    png_path = Path(manifest["figures"][0]["path"])
    svg_path = Path(manifest["figures"][1]["path"])
    assert png_path.exists()
    assert svg_path.exists()

    with Image.open(png_path) as image:
        assert image.size == (CANVAS_WIDTH, CANVAS_HEIGHT)

    svg_text = svg_path.read_text(encoding="utf-8")
    assert "CoFiTok method overview" in svg_text
    assert METHOD_SUBTITLE in svg_text
    assert "compressed" not in svg_text.lower()
    assert "blocked into S_k" in svg_text
    assert "S_k(0)=0" in svg_text

    persisted = json.loads((tmp_path / "method" / "method_overview_manifest.json").read_text(encoding="utf-8"))
    assert persisted["artifact_count"] == 2
