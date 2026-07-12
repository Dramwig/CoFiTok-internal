import json
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

from scripts.evaluate_official_fid_dirs import main as evaluate_main
from scripts.export_official_fid_dirs import official_fid_commands, prepare_image_dir, save_png_batch


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color=(17, 29, 43)).save(path)


def test_save_png_batch_writes_deterministic_names(tmp_path) -> None:
    images = torch.zeros(2, 3, 4, 4)
    paths = save_png_batch(images, tmp_path, start_index=7)

    assert [Path(path).name for path in paths] == ["00000007.png", "00000008.png"]
    assert (tmp_path / "00000007.png").is_file()
    assert (tmp_path / "00000008.png").is_file()


def test_prepare_image_dir_requires_overwrite_for_existing_images(tmp_path) -> None:
    _write_png(tmp_path / "00000000.png")

    with pytest.raises(FileExistsError):
        prepare_image_dir(tmp_path, overwrite=False)

    prepare_image_dir(tmp_path, overwrite=True)
    assert not list(tmp_path.glob("*.png"))


def test_official_fid_commands_point_to_external_protocol(tmp_path) -> None:
    commands = official_fid_commands(tmp_path / "real", tmp_path / "generated", tmp_path / "report", 64)

    assert "scripts/evaluate_official_fid_dirs.py" in commands["cofitok_wrapper"]
    assert "--require-pytorch-fid" in commands["cofitok_wrapper"]
    assert "python -m pytorch_fid" in commands["pytorch_fid_cli"]
    assert (tmp_path / "real").as_posix() in commands["pytorch_fid_cli"]
    assert (tmp_path / "generated").as_posix() in commands["pytorch_fid_cli"]


def test_evaluate_official_fid_dirs_dry_run_writes_report(tmp_path, monkeypatch) -> None:
    real_dir = tmp_path / "real"
    generated_dir = tmp_path / "generated"
    _write_png(real_dir / "00000000.png")
    _write_png(generated_dir / "00000000.png")
    output_dir = tmp_path / "report"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_official_fid_dirs.py",
            "--real-dir",
            str(real_dir),
            "--generated-dir",
            str(generated_dir),
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ],
    )
    evaluate_main()

    report = json.loads((output_dir / "official_fid_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "dry_run"
    assert report["metric"] == "fid"
    assert report["counts"]["real_image_count"] == 1
    assert report["counts"]["generated_image_count"] == 1
    assert report["metrics"]["fid"] is None
