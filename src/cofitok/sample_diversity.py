from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image


DHASH_WIDTH = 9
DHASH_HEIGHT = 8
DHASH_BITS = (DHASH_WIDTH - 1) * DHASH_HEIGHT
DHASH_THRESHOLDS = (0, 4, 8)
REAL_SELECTION_SALT = "cofitok_sample_diversity_real_reference_v1"
SUPPORTED_IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png", ".webp"}


@dataclass(frozen=True)
class SampleRecord:
    path: Path
    identifier: str
    class_index: int
    global_index: int | None = None


@dataclass(frozen=True)
class ImageFingerprint:
    identifier: str
    class_index: int
    encoded_sha256: str
    pixel_sha256: str
    dhash64: int


def _exact_nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def build_generated_records(
    root: str | Path,
    *,
    start_index: int,
    sample_count: int,
    class_count: int,
) -> list[SampleRecord]:
    root_path = Path(root)
    if root_path.is_symlink() or not root_path.is_dir():
        raise ValueError(f"generated sample root must be a real directory: {root_path}")
    exact_start = _exact_nonnegative_int(start_index, label="start_index")
    exact_count = _exact_nonnegative_int(sample_count, label="sample_count")
    exact_classes = _exact_nonnegative_int(class_count, label="class_count")
    if exact_count == 0 or exact_classes == 0:
        raise ValueError("sample_count and class_count must be positive")

    image_paths = sorted(
        path for path in root_path.iterdir() if path.suffix.lower() == ".png"
    )
    expected_names = [
        f"{global_index:06d}.png"
        for global_index in range(exact_start, exact_start + exact_count)
    ]
    actual_names = [path.name for path in image_paths]
    if actual_names != expected_names:
        missing = sorted(set(expected_names) - set(actual_names))[:5]
        unexpected = sorted(set(actual_names) - set(expected_names))[:5]
        raise ValueError(
            "generated sample names are not the exact contiguous window: "
            f"missing={missing}, unexpected={unexpected}, "
            f"actual_count={len(actual_names)}, expected_count={exact_count}"
        )
    records: list[SampleRecord] = []
    for path in image_paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"generated sample must be a real file: {path}")
        global_index = int(path.stem)
        records.append(
            SampleRecord(
                path=path,
                identifier=path.name,
                class_index=global_index % exact_classes,
                global_index=global_index,
            )
        )
    return records


def build_balanced_real_records(
    root: str | Path,
    *,
    class_count: int,
    samples_per_class: int,
    expected_image_count: int,
    selection_salt: str = REAL_SELECTION_SALT,
) -> tuple[list[SampleRecord], dict[str, Any]]:
    root_path = Path(root)
    if root_path.is_symlink() or not root_path.is_dir():
        raise ValueError(f"real-set root must be a real directory: {root_path}")
    exact_classes = _exact_nonnegative_int(class_count, label="class_count")
    exact_per_class = _exact_nonnegative_int(
        samples_per_class, label="samples_per_class"
    )
    exact_total = _exact_nonnegative_int(
        expected_image_count, label="expected_image_count"
    )
    if exact_classes == 0 or exact_per_class == 0 or exact_total == 0:
        raise ValueError("real reference counts must be positive")

    class_dirs = sorted(path for path in root_path.iterdir() if path.is_dir())
    if len(class_dirs) != exact_classes:
        raise ValueError(
            f"real set has {len(class_dirs)} class directories; expected {exact_classes}"
        )

    records: list[SampleRecord] = []
    observed_total = 0
    per_class_source_counts: list[int] = []
    for class_index, class_dir in enumerate(class_dirs):
        if class_dir.is_symlink():
            raise ValueError(f"real-set class directory must not be a symlink: {class_dir}")
        candidates = sorted(
            path
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        )
        if any(path.is_symlink() for path in candidates):
            raise ValueError(f"real-set class contains a symlink: {class_dir}")
        observed_total += len(candidates)
        per_class_source_counts.append(len(candidates))
        if len(candidates) < exact_per_class:
            raise ValueError(
                f"real-set class {class_dir.name} has only {len(candidates)} images"
            )

        def selection_key(path: Path) -> tuple[str, str]:
            relative = path.relative_to(root_path).as_posix()
            digest = hashlib.sha256(
                f"{selection_salt}\0{relative}".encode("utf-8")
            ).hexdigest()
            return digest, relative

        selected = sorted(candidates, key=selection_key)[:exact_per_class]
        for path in selected:
            records.append(
                SampleRecord(
                    path=path,
                    identifier=path.relative_to(root_path).as_posix(),
                    class_index=class_index,
                )
            )
    if observed_total != exact_total:
        raise ValueError(
            f"real set has {observed_total} images; expected {exact_total}"
        )
    return records, {
        "algorithm": "smallest_sha256_of_salt_nul_root_relative_path_per_class",
        "salt": selection_salt,
        "class_directory_order": "lexicographic_root_child_name",
        "class_count": exact_classes,
        "samples_per_class": exact_per_class,
        "selected_image_count": len(records),
        "source_image_count": observed_total,
        "source_images_per_class_min": min(per_class_source_counts),
        "source_images_per_class_max": max(per_class_source_counts),
    }


def dhash64(image: Image.Image) -> int:
    grayscale = image.convert("L").resize(
        (DHASH_WIDTH, DHASH_HEIGHT), Image.Resampling.LANCZOS
    )
    pixels = np.asarray(grayscale, dtype=np.int16)
    comparisons = (pixels[:, 1:] > pixels[:, :-1]).reshape(-1)
    value = 0
    for bit in comparisons:
        value = (value << 1) | int(bit)
    return value


def _pixel_sha256(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    digest = hashlib.sha256()
    digest.update(b"RGB\0")
    digest.update(rgb.width.to_bytes(8, byteorder="big"))
    digest.update(rgb.height.to_bytes(8, byteorder="big"))
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def _read_fingerprint(
    record: SampleRecord,
    *,
    expected_size: tuple[int, int],
) -> tuple[ImageFingerprint, bytes]:
    path = record.path
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"sample must be a real file: {path}")
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    if (
        len(raw) != before.st_size
        or after.st_size != before.st_size
        or after.st_mtime_ns != before.st_mtime_ns
    ):
        raise RuntimeError(f"sample changed while being read: {path}")
    try:
        with Image.open(BytesIO(raw)) as opened:
            opened.load()
            if opened.size != expected_size:
                raise ValueError(
                    f"sample {path} has size {opened.size}; expected {expected_size}"
                )
            rgb = opened.convert("RGB")
    except (OSError, SyntaxError) as error:
        raise ValueError(f"sample is not a decodable image: {path}") from error
    return (
        ImageFingerprint(
            identifier=record.identifier,
            class_index=record.class_index,
            encoded_sha256=hashlib.sha256(raw).hexdigest(),
            pixel_sha256=_pixel_sha256(rgb),
            dhash64=dhash64(rgb),
        ),
        raw,
    )


def _duplicate_summary(values: Iterable[str]) -> dict[str, int | float]:
    counts = Counter(values)
    total = sum(counts.values())
    unique = len(counts)
    duplicate_images = sum(count - 1 for count in counts.values())
    duplicate_pairs = sum(count * (count - 1) // 2 for count in counts.values())
    return {
        "unique_count": unique,
        "duplicate_image_count": duplicate_images,
        "duplicate_image_fraction": duplicate_images / total if total else math.nan,
        "duplicate_pair_count": duplicate_pairs,
    }


def _distribution(values: Sequence[int]) -> dict[str, float | int]:
    if not values:
        raise ValueError("distance distribution must not be empty")
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": int(array.min()),
        "p10": float(np.percentile(array, 10)),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "max": int(array.max()),
    }


def nearest_hamming_distances(
    hashes: Sequence[int], *, chunk_size: int = 128
) -> list[int]:
    if len(hashes) < 2:
        raise ValueError("at least two hashes are required")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    array = np.asarray(hashes, dtype=np.uint64)
    popcount = np.asarray([int(value).bit_count() for value in range(256)], dtype=np.uint8)
    nearest = np.full(array.size, DHASH_BITS + 1, dtype=np.uint8)
    for start in range(0, array.size, chunk_size):
        stop = min(start + chunk_size, array.size)
        xor = np.bitwise_xor(array[start:stop, None], array[None, :])
        byte_view = xor.view(np.uint8).reshape(stop - start, array.size, 8)
        distances = popcount[byte_view].sum(axis=2, dtype=np.uint16)
        distances[np.arange(stop - start), np.arange(start, stop)] = DHASH_BITS + 1
        nearest[start:stop] = distances.min(axis=1)
    return [int(value) for value in nearest]


def _threshold_summary(values: Sequence[int], *, denominator: int) -> dict[str, Any]:
    if denominator <= 0:
        raise ValueError("threshold denominator must be positive")
    result: dict[str, Any] = {}
    for threshold in DHASH_THRESHOLDS:
        count = sum(value <= threshold for value in values)
        result[f"le_{threshold}"] = {
            "count": count,
            "fraction": count / denominator,
        }
    return result


def _within_class_summary(
    fingerprints: Sequence[ImageFingerprint],
    *,
    expected_class_count: int,
    expected_samples_per_class: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    grouped: dict[int, list[ImageFingerprint]] = defaultdict(list)
    for fingerprint in fingerprints:
        grouped[fingerprint.class_index].append(fingerprint)
    if sorted(grouped) != list(range(expected_class_count)):
        raise ValueError("cohort does not cover every expected class exactly")

    all_pair_distances: list[int] = []
    all_nearest_distances: list[int] = []
    pixel_duplicate_pairs = 0
    class_rows: list[dict[str, Any]] = []
    for class_index in range(expected_class_count):
        members = grouped[class_index]
        if len(members) != expected_samples_per_class:
            raise ValueError(
                f"class {class_index} has {len(members)} samples; "
                f"expected {expected_samples_per_class}"
            )
        pair_distances: list[int] = []
        nearest = [DHASH_BITS + 1] * len(members)
        class_pixel_pairs = 0
        for left in range(len(members)):
            for right in range(left + 1, len(members)):
                distance = (members[left].dhash64 ^ members[right].dhash64).bit_count()
                pair_distances.append(distance)
                nearest[left] = min(nearest[left], distance)
                nearest[right] = min(nearest[right], distance)
                if members[left].pixel_sha256 == members[right].pixel_sha256:
                    class_pixel_pairs += 1
        pair_count = len(pair_distances)
        all_pair_distances.extend(pair_distances)
        all_nearest_distances.extend(nearest)
        pixel_duplicate_pairs += class_pixel_pairs
        class_rows.append(
            {
                "class_index": class_index,
                "sample_count": len(members),
                "pair_count": pair_count,
                "pixel_exact_duplicate_pair_count": class_pixel_pairs,
                "dhash_pair_fraction_le_0": sum(value <= 0 for value in pair_distances)
                / pair_count,
                "dhash_pair_fraction_le_4": sum(value <= 4 for value in pair_distances)
                / pair_count,
                "dhash_pair_fraction_le_8": sum(value <= 8 for value in pair_distances)
                / pair_count,
                "dhash_nearest_median": float(np.median(nearest)),
                "dhash_nearest_mean": float(np.mean(nearest)),
            }
        )
    return (
        {
            "class_count": expected_class_count,
            "samples_per_class": expected_samples_per_class,
            "pair_count": len(all_pair_distances),
            "pixel_exact_duplicate_pair_count": pixel_duplicate_pairs,
            "pair_distance": _distribution(all_pair_distances),
            "pair_thresholds": _threshold_summary(
                all_pair_distances, denominator=len(all_pair_distances)
            ),
            "nearest_distance": _distribution(all_nearest_distances),
            "nearest_thresholds": _threshold_summary(
                all_nearest_distances, denominator=len(all_nearest_distances)
            ),
        },
        class_rows,
    )


def analyze_cohort(
    records: Sequence[SampleRecord],
    *,
    cohort_id: str,
    cohort_kind: str,
    expected_size: tuple[int, int],
    expected_class_count: int,
    expected_samples_per_class: int,
    expected_sample_set_sha256: str | None = None,
    nearest_chunk_size: int = 128,
) -> dict[str, Any]:
    expected_count = expected_class_count * expected_samples_per_class
    if len(records) != expected_count:
        raise ValueError(
            f"cohort {cohort_id} has {len(records)} samples; expected {expected_count}"
        )
    identifiers = [record.identifier for record in records]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"cohort {cohort_id} contains duplicate identifiers")

    cohort_digest = hashlib.sha256()
    fingerprints: list[ImageFingerprint] = []
    for record in sorted(records, key=lambda item: item.identifier):
        fingerprint, raw = _read_fingerprint(record, expected_size=expected_size)
        identifier_bytes = record.identifier.encode("utf-8")
        cohort_digest.update(identifier_bytes)
        cohort_digest.update(b"\0")
        cohort_digest.update(raw)
        cohort_digest.update(b"\0")
        fingerprints.append(fingerprint)
    observed_digest = cohort_digest.hexdigest()
    if expected_sample_set_sha256 is not None and observed_digest != expected_sample_set_sha256:
        raise ValueError(
            f"cohort {cohort_id} sample-set SHA mismatch: "
            f"observed={observed_digest}, expected={expected_sample_set_sha256}"
        )

    encoded_duplicates = _duplicate_summary(
        fingerprint.encoded_sha256 for fingerprint in fingerprints
    )
    pixel_duplicates = _duplicate_summary(
        fingerprint.pixel_sha256 for fingerprint in fingerprints
    )
    global_nearest = nearest_hamming_distances(
        [fingerprint.dhash64 for fingerprint in fingerprints],
        chunk_size=nearest_chunk_size,
    )
    within_class, class_rows = _within_class_summary(
        fingerprints,
        expected_class_count=expected_class_count,
        expected_samples_per_class=expected_samples_per_class,
    )
    return {
        "cohort_id": cohort_id,
        "cohort_kind": cohort_kind,
        "sample_count": len(fingerprints),
        "image_size": [expected_size[0], expected_size[1]],
        "cohort_digest": {
            "algorithm": "sha256",
            "framing": "identifier_utf8_nul_file_bytes_nul",
            "sha256": observed_digest,
            "matches_declared_sample_set_sha256": (
                True if expected_sample_set_sha256 is not None else None
            ),
        },
        "exact_duplicates": {
            "encoded_file": encoded_duplicates,
            "decoded_rgb_pixels": pixel_duplicates,
        },
        "dhash64": {
            "algorithm": "grayscale_9x8_horizontal_difference_lanczos_64bit",
            "interpretation": "perceptual_hash_candidate_only_not_an_exact_duplicate_or_quality_metric",
            "global_nearest_distance": _distribution(global_nearest),
            "global_nearest_thresholds": _threshold_summary(
                global_nearest, denominator=len(global_nearest)
            ),
            "within_class": within_class,
        },
        "per_class": class_rows,
    }
