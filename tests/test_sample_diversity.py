from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image

from cofitok.sample_diversity import (
    SampleRecord,
    analyze_cohort,
    build_balanced_real_records,
    build_generated_records,
    dhash64,
    nearest_hamming_distances,
)


def _write_image(path: Path, *, split: int, invert: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (16, 16), color=(0, 0, 0))
    pixels = image.load()
    for y in range(16):
        for x in range(16):
            light = x >= split
            if invert:
                light = not light
            value = 255 if light else 0
            pixels[x, y] = (value, value, value)
    image.save(path)


def _sample_set_sha(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def test_generated_records_and_cohort_exact_duplicates(tmp_path: Path) -> None:
    sample_dir = tmp_path / "samples"
    paths: list[Path] = []
    for index, split in enumerate((4, 6, 4, 8), start=10):
        path = sample_dir / f"{index:06d}.png"
        _write_image(path, split=split)
        paths.append(path)
    records = build_generated_records(
        sample_dir, start_index=10, sample_count=4, class_count=2
    )
    report = analyze_cohort(
        records,
        cohort_id="generated",
        cohort_kind="test",
        expected_size=(16, 16),
        expected_class_count=2,
        expected_samples_per_class=2,
        expected_sample_set_sha256=_sample_set_sha(paths),
        nearest_chunk_size=2,
    )
    assert [record.class_index for record in records] == [0, 1, 0, 1]
    assert report["cohort_digest"]["matches_declared_sample_set_sha256"] is True
    assert report["exact_duplicates"]["decoded_rgb_pixels"]["duplicate_image_count"] == 1
    assert report["dhash64"]["within_class"]["pixel_exact_duplicate_pair_count"] == 1


def test_balanced_real_selection_is_deterministic(tmp_path: Path) -> None:
    real_root = tmp_path / "real"
    for class_name, invert in (("b", False), ("a", True)):
        for index in range(4):
            _write_image(
                real_root / class_name / f"{index}.jpg",
                split=4 + index,
                invert=invert,
            )
    first, evidence = build_balanced_real_records(
        real_root,
        class_count=2,
        samples_per_class=2,
        expected_image_count=8,
        selection_salt="fixed",
    )
    second, _ = build_balanced_real_records(
        real_root,
        class_count=2,
        samples_per_class=2,
        expected_image_count=8,
        selection_salt="fixed",
    )
    assert [record.identifier for record in first] == [
        record.identifier for record in second
    ]
    assert {record.class_index for record in first} == {0, 1}
    assert evidence["selected_image_count"] == 4


def test_dhash_and_nearest_hamming_distances() -> None:
    left = Image.new("RGB", (16, 16), color="black")
    right = Image.new("RGB", (16, 16), color="white")
    assert dhash64(left) == dhash64(right) == 0
    assert nearest_hamming_distances([0, 1, (1 << 64) - 1], chunk_size=2) == [
        1,
        1,
        63,
    ]


def test_analyze_cohort_rejects_sample_set_sha_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "000000.png"
    other = tmp_path / "000001.png"
    _write_image(path, split=4)
    _write_image(other, split=8)
    records = [
        SampleRecord(path=path, identifier=path.name, class_index=0),
        SampleRecord(path=other, identifier=other.name, class_index=0),
    ]
    try:
        analyze_cohort(
            records,
            cohort_id="bad",
            cohort_kind="test",
            expected_size=(16, 16),
            expected_class_count=1,
            expected_samples_per_class=2,
            expected_sample_set_sha256="0" * 64,
        )
    except ValueError as error:
        assert "sample-set SHA mismatch" in str(error)
    else:
        raise AssertionError("expected sample-set SHA mismatch")
