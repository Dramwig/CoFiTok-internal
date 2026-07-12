import json
from pathlib import Path

from PIL import Image

from scripts.make_visual_panels import (
    ORDER_ABLATION_ITEMS,
    PREFIX_COMPARISON_ITEMS,
    SAMPLING_ITEMS,
    SK_DIAGNOSTIC_ITEMS,
    make_visual_panels,
)


def _write_source_images(root: Path) -> None:
    all_items = PREFIX_COMPARISON_ITEMS + ORDER_ABLATION_ITEMS + SK_DIAGNOSTIC_ITEMS + SAMPLING_ITEMS
    for index, item in enumerate(all_items):
        path = root / item.relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        size = (360, 96) if "samples_prefix" in item.relative_path else (180, 220)
        color = ((index * 31) % 255, (index * 67) % 255, (index * 109) % 255)
        Image.new("RGB", size, color).save(path)


def test_make_visual_panels_writes_outputs(tmp_path) -> None:
    reports_root = tmp_path / "reports"
    output_dir = tmp_path / "figures"
    _write_source_images(reports_root)

    manifest = make_visual_panels(reports_root, output_dir, sample_crop_width=120)

    assert manifest["panel_count"] == 4
    assert (output_dir / "visual_panel_manifest.json").exists()
    for item in manifest["panels"]:
        path = Path(item["path"])
        assert path.exists()
        assert len(item["source_images"]) >= 3
        with Image.open(path) as image:
            assert image.width > 100
            assert image.height > 100

    persisted = json.loads((output_dir / "visual_panel_manifest.json").read_text(encoding="utf-8"))
    assert persisted["panel_count"] == 4
