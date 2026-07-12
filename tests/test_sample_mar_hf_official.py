import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "baselines" / "sample_mar_hf_official.py"
SPEC = importlib.util.spec_from_file_location("sample_mar_hf_official", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_balanced_class_labels_match_official_order() -> None:
    labels = MODULE.balanced_class_labels(50_000, 1_000)

    assert labels.shape == (50_000,)
    assert labels.dtype == np.int64
    assert np.all(labels[:50] == 0)
    assert np.all(labels[50:100] == 1)
    assert np.all(np.bincount(labels, minlength=1_000) == 50)


def test_balanced_class_labels_reject_unbalanced_protocol() -> None:
    with pytest.raises(ValueError, match="must be divisible"):
        MODULE.balanced_class_labels(128, 1_000)


def test_batch_completion_is_resume_safe(tmp_path: Path) -> None:
    for index in range(4):
        MODULE.sample_path(tmp_path, index).write_bytes(b"png")

    assert MODULE.batch_is_complete(tmp_path, 0, 4)
    assert not MODULE.batch_is_complete(tmp_path, 0, 5)
    assert list(MODULE.batch_ranges(10, 4)) == [(0, 4), (4, 8), (8, 10)]


def test_validate_and_pack_pngs(tmp_path: Path) -> None:
    sample_dir = tmp_path / "samples"
    sample_dir.mkdir()
    for index in range(2):
        image = np.full((256, 256, 3), index * 127, dtype=np.uint8)
        Image.fromarray(image, mode="RGB").save(MODULE.sample_path(sample_dir, index))

    output_npz = tmp_path / "samples.npz"
    manifest = MODULE.validate_and_pack_pngs(sample_dir, output_npz, 2)

    with np.load(output_npz) as payload:
        images = payload["arr_0"]
    assert images.shape == (2, 256, 256, 3)
    assert images.dtype == np.uint8
    assert int(images[1, 0, 0, 0]) == 127
    assert manifest["shape"] == [2, 256, 256, 3]
    assert len(manifest["sha256"]) == 64


def test_decode_batch_size_is_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(MODULE.sys, "argv", ["sample_mar_hf_official.py"])
    args = MODULE.parse_args()
    assert args.decode_batch_size == 16
