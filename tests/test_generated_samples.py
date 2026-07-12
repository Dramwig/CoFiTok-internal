from pathlib import Path

from PIL import Image

from scripts.evaluate_generated_samples import find_image_files


def test_find_image_files_recurses_and_sorts(tmp_path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    Image.new("RGB", (2, 2)).save(nested / "b.png")
    Image.new("RGB", (2, 2)).save(tmp_path / "a.png")
    (tmp_path / "ignore.txt").write_text("nope", encoding="utf-8")

    files = find_image_files(tmp_path)

    assert [path.name for path in files] == ["a.png", "b.png"]
