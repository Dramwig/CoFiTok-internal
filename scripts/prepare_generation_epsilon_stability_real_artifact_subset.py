from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from PIL import Image

from cofitok.generation import EPSILON_STABILITY_NON_AUTHORIZING_BOUNDARY
from cofitok.generation_gate_sources import gate_source_report_identity
from cofitok.image_integrity import image_tree_sha256, sample_set_sha256
from cofitok.reporting import write_json_report


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
SUBSET_SCHEMA = "cofitok_epsilon_stability_real_artifact_subset_v1"
SAMPLE_COUNT = 1_000
START_INDEX = 0


def _read_object(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON source is not an object: {path}")
    return payload


def _real_set(payload: dict[str, Any]) -> dict[str, Any]:
    candidate = payload.get("real_set", payload)
    if not isinstance(candidate, dict) or set(candidate) != {
        "digest_schema",
        "sha256",
        "root",
        "image_count",
    }:
        raise ValueError("epsilon-stability real-set contract is malformed")
    return candidate


def _find_images(root: Path) -> list[Path]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("epsilon-stability real directory is invalid")
    images = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if any(path.is_symlink() for path in images):
        raise ValueError("epsilon-stability real set contains a symlink")
    return images


def _decoded_source_image(path: Path) -> tuple[str, bytes]:
    with Image.open(path) as image:
        if (
            image.format not in {"JPEG", "PNG"}
            or image.mode != "RGB"
            or image.size != (256, 256)
        ):
            raise ValueError(f"real artifact source image contract differs: {path}")
        return str(image.format), image.tobytes()


def _decoded_pixel_sha256(pixels: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(b"cofitok_rgb_256x256_uint8_v1\0")
    digest.update(pixels)
    return digest.hexdigest()


def _materialize_png(
    source: Path,
    destination: Path,
    *,
    resume: bool,
) -> tuple[str, str]:
    source_format, source_pixels = _decoded_source_image(source)
    if destination.exists():
        if not resume or destination.is_symlink() or not destination.is_file():
            raise FileExistsError(f"real artifact subset destination exists: {destination}")
        with Image.open(destination) as image:
            if (
                image.format != "PNG"
                or image.mode != "RGB"
                or image.size != (256, 256)
                or image.tobytes() != source_pixels
            ):
                raise ValueError(
                    f"real artifact subset destination differs: {destination}"
                )
        return source_format, _decoded_pixel_sha256(source_pixels)

    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.part")
    try:
        Image.frombytes("RGB", (256, 256), source_pixels).save(
            temporary,
            format="PNG",
            optimize=False,
            compress_level=6,
        )
        with Image.open(temporary) as image:
            if (
                image.format != "PNG"
                or image.mode != "RGB"
                or image.size != (256, 256)
                or image.tobytes() != source_pixels
            ):
                raise ValueError(
                    f"real artifact subset PNG materialization differs: {source}"
                )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    with Image.open(destination) as image:
        if image.format != "PNG" or image.tobytes() != source_pixels:
            raise ValueError(f"real artifact subset destination differs: {destination}")
    return source_format, _decoded_pixel_sha256(source_pixels)


def build_subset(
    *,
    real_dir: str | Path,
    real_set_contract_path: str | Path,
    expected_real_set_contract_sha256: str,
    output_dir: str | Path,
    resume: bool,
) -> dict[str, Any]:
    contract_identity = gate_source_report_identity(real_set_contract_path)
    if contract_identity["sha256"] != expected_real_set_contract_sha256:
        raise ValueError("epsilon-stability real-set source SHA256 differs")
    real_set = _real_set(_read_object(contract_identity["path"]))
    root = Path(real_dir).resolve()
    if root.as_posix() != str(real_set["root"]):
        raise ValueError("epsilon-stability real directory differs from authorization")
    source_images = _find_images(root)
    if len(source_images) != int(real_set["image_count"]):
        raise ValueError("epsilon-stability real-set image count differs")
    if image_tree_sha256(source_images, root=root) != str(real_set["sha256"]):
        raise ValueError("epsilon-stability physical real-set digest differs")
    selected = source_images[:SAMPLE_COUNT]
    if len(selected) != SAMPLE_COUNT:
        raise ValueError("epsilon-stability real set has fewer than 1000 images")

    output = Path(output_dir)
    if output.is_symlink() or (output.exists() and not output.is_dir()):
        raise ValueError("epsilon-stability real subset output is invalid")
    output.mkdir(parents=True, exist_ok=True)
    expected_names = {f"{index:06d}.png" for index in range(SAMPLE_COUNT)}
    unexpected = {
        path.name for path in output.iterdir() if path.name not in expected_names
    }
    if unexpected:
        raise ValueError("epsilon-stability real subset contains unexpected files")

    rows = []
    destinations = []
    for index, source in enumerate(selected):
        source_identity = gate_source_report_identity(source)
        destination = output / f"{index:06d}.png"
        source_format, decoded_pixel_sha256 = _materialize_png(
            source,
            destination,
            resume=resume,
        )
        if gate_source_report_identity(source) != source_identity:
            raise ValueError(
                f"real artifact subset source changed during materialization: {source}"
            )
        destination_identity = gate_source_report_identity(destination)
        rows.append(
            {
                "index": index,
                "source": source_identity,
                "source_format": source_format,
                "decoded_pixel_sha256": decoded_pixel_sha256,
                "destination": destination_identity,
            }
        )
        destinations.append(destination)

    if image_tree_sha256(source_images, root=root) != str(real_set["sha256"]):
        raise ValueError(
            "epsilon-stability physical real set changed during materialization"
        )

    actual_names = {path.name for path in output.iterdir() if path.is_file()}
    if actual_names != expected_names:
        raise ValueError("epsilon-stability real subset is incomplete")
    return {
        "schema": SUBSET_SCHEMA,
        "status": "pass",
        "real_set_identity": contract_identity,
        "real_set": real_set,
        "selection": "first_1000_images_in_sorted_recursive_path_order",
        "materialization": "decoded_rgb_pixels_to_lossless_numbered_png_v1",
        "image_dir": output.resolve().as_posix(),
        "sample_count": SAMPLE_COUNT,
        "start_index": START_INDEX,
        "sample_set_sha256": sample_set_sha256(destinations),
        "source_images": rows,
        "authorization_boundary": dict(
            EPSILON_STABILITY_NON_AUTHORIZING_BOUNDARY
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create and physically replay the exact 1000-image real artifact "
            "reference subset for the non-authorizing epsilon-stability diagnostic."
        )
    )
    parser.add_argument("--real-dir", required=True)
    parser.add_argument("--real-set-contract", required=True)
    parser.add_argument("--expected-real-set-contract-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_subset(
        real_dir=args.real_dir,
        real_set_contract_path=args.real_set_contract,
        expected_real_set_contract_sha256=(
            args.expected_real_set_contract_sha256
        ),
        output_dir=args.output_dir,
        resume=args.resume,
    )
    output = Path(args.output)
    if output.is_file():
        if not args.resume or _read_object(output) != report:
            raise ValueError("existing real artifact subset manifest does not replay")
        print(f"reused {output}")
        return
    write_json_report(output, report)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
