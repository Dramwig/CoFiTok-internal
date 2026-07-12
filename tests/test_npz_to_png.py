from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from scripts.baselines.npz_to_png import export_npz_to_png


def test_export_npz_to_png_handles_nhwc_uint8(tmp_path: Path) -> None:
    input_path = tmp_path / "samples.npz"
    images = np.zeros((3, 4, 5, 3), dtype=np.uint8)
    images[0, :, :, 0] = 255
    np.savez(input_path, images)

    manifest = export_npz_to_png(input_path, tmp_path / "png", key="arr_0", limit=2, prefix="img")

    assert manifest["available_images"] == 3
    assert manifest["exported_images"] == 2
    assert (tmp_path / "png" / "img_000000.png").is_file()
    assert (tmp_path / "png" / "img_000001.png").is_file()
    assert not (tmp_path / "png" / "img_000002.png").exists()


def test_export_npz_to_png_handles_nchw_float_minus_one_to_one(tmp_path: Path) -> None:
    input_path = tmp_path / "samples.npz"
    images = np.full((1, 3, 2, 2), -1.0, dtype=np.float32)
    images[:, 1] = 1.0
    np.savez(input_path, images)

    export_npz_to_png(input_path, tmp_path / "png", key="arr_0", limit=0, prefix="sample")

    with Image.open(tmp_path / "png" / "sample_000000.png") as image:
        assert image.mode == "RGB"
        assert image.size == (2, 2)
        assert image.getpixel((0, 0)) == (0, 255, 0)
