"""Freeze/replay inputs for the pending loss-gradient probe, without model load.

This catalog alone cannot authorize execution. It preserves the prior screen,
checkpoint files, and every old diagnostic, and writes one new immutable JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from PIL import Image

from cofitok.generation_class_support_contingency import stable_file_identity


GENERATION = Path("/root/autodl-tmp/CoFiTok/checkpoints/generation")
DATASET = Path("/root/autodl-tmp/CoFiTok/datasets/imagenet_256")
OUTPUT = GENERATION / ".frozen_loss_gradient_attribution_v1.control/source_catalog.json"
RESULT_SHA = "9d1a5709a18f5beadf7a8654dd147906a269b35ce35ef0720a68ab2c484c0dc1"
RECEIPT_SHA = "6ad46a2cece75f17038ddb2c32fe08665db372909e728e956b34e02168855dae"
MANIFEST_SHA = "9a2eec642f0d56162bffaafed84a41267f22abfc9feff4cf41fed9f6881173f0"
CHECKPOINT_SHA = {
    "cofitok": "b36a92229ba2dd021db9c7585970ddda1d17d1919ca6b0eec0362b06d4bd462e",
    "dense_identity": "b6586cc906a9c38bbf9d592f5f6169b7a37a8d3aa485ca4879021fa840943d0b",
}
BOUNDARY = {
    "input_catalog_only": True, "execution_authorized": False,
    "model_deserialization_performed": False, "gpu_execution_allowed": False,
    "training_allowed": False, "sampling_allowed": False,
    "confirmation_allowed": False, "full_300k_allowed": False,
    "release_allowed": False, "paper_integration_allowed": False,
}


def read_bound(path, expected=None):
    path = Path(path)
    ident = stable_file_identity(path)
    if expected is not None and ident["sha256"] != expected:
        raise ValueError(f"source hash differs: {path}")
    value = json.loads(path.read_text())
    if stable_file_identity(path) != ident:
        raise ValueError(f"source changed during read: {path}")
    return value, ident


def select_rows(rows, excluded_wnids, *, count=32, seed=2031):
    by_class = {}
    seen = set()
    for row in rows:
        path = Path(row["path"])
        if path.as_posix() in seen:
            raise ValueError("duplicate validation path")
        seen.add(path.as_posix())
        if (path.is_absolute() or ".." in path.parts or len(path.parts) != 4
                or path.parts[:2] != ("extracted", "val")
                or path.parts[2] != row["wnid"]
                or row["height"] != 256 or row["width"] != 256
                or type(row["label"]) is not int or not 0 <= row["label"] < 1000):
            raise ValueError("validation image contract differs")
        if row["wnid"] not in excluded_wnids:
            by_class.setdefault(row["wnid"], []).append(row)
    if len(by_class) < count:
        raise ValueError("insufficient previously unused classes")
    score = lambda namespace, value: hashlib.sha256(f"{seed}/{namespace}/{value}".encode()).hexdigest()
    classes = sorted(by_class, key=lambda wnid: (score("class", wnid), wnid))[:count]
    return [min(by_class[wnid], key=lambda row: (score("image", row["path"]), row["path"])) for wnid in classes]


def census(root):
    found = []
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not d.startswith(("prefix_", "samples_"))
                         and d not in {"metrics", "eval_cache", "feature_cache"})
        if "conditioning_sensitivity_manifest.json" in files:
            found.append(Path(directory) / "conditioning_sensitivity_manifest.json")
    return sorted(found)


def checkout(root):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    if git("status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("catalog builder checkout must be completely clean")
    script = Path(__file__).resolve()
    relative = script.relative_to(root).as_posix()
    if subprocess.check_output(["git", "-C", str(root), "show", f"HEAD:{relative}"]) != script.read_bytes():
        raise ValueError("catalog builder differs from HEAD blob")
    return {"root": str(root), "revision": git("rev-parse", "HEAD"),
            "tree": git("rev-parse", "HEAD^{tree}"), "branch": git("branch", "--show-current"),
            "tracked_dirty": False, "script": stable_file_identity(script)}


def build_catalog(project_root):
    builder = checkout(project_root)
    result, result_id = read_bound(GENERATION / "class_support_contingency_v1/class_support_contingency_result.json", RESULT_SHA)
    receipt, receipt_id = read_bound(GENERATION / "class_support_contingency_v1/class_support_contingency_result.validation.json", RECEIPT_SHA)
    if receipt["status"] != "pass" or receipt["result"] != result_id:
        raise ValueError("class-support receipt does not bind result")
    methods, bound_sources = {}, [result_id, receipt_id]
    for method, source in result["source_evidence"]["frozen_methods"].items():
        sampling, sid = read_bound(source["identities"]["sampling_report"]["path"],
                                   source["identities"]["sampling_report"]["sha256"])
        cp = Path(sampling["checkpoint"])
        cp_id = stable_file_identity(cp)
        sidecar, side_id = read_bound(str(cp) + ".integrity.json")
        report, report_id = read_bound(cp.parent / "training_report.json")
        if (cp_id["sha256"] != CHECKPOINT_SHA[method]
                or cp_id["sha256"] != sidecar["checkpoint_sha256"]
                or cp_id["bytes"] != sidecar["checkpoint_bytes"]
                or sidecar["checkpoint"] != cp.name or sidecar["step"] != 100000
                or report["completed_steps"] != 100000 or not report["training_complete"]
                or report["latest_checkpoint"]["checkpoint_sha256"] != cp_id["sha256"]
                or report["git"]["revision"] != "cf0e5faa94bf4ab38d947b921935b3b765b5537a"
                or report["dataset_provenance"]["manifest"]["sha256"] != MANIFEST_SHA):
            raise ValueError(f"frozen checkpoint/report binding differs: {method}")
        methods[method] = {"checkpoint": cp_id, "sidecar": side_id,
                           "training_report": report_id, "sampling_report": sid,
                           "config": report["config"], "weights_to_measure": "model",
                           "teacher_weights": "original_checkpoint_ema"}
        bound_sources.extend([cp_id, side_id, report_id, sid])
    manifests = census(GENERATION)
    if len(manifests) < 16:
        raise ValueError("historical sensitivity census incomplete")
    excluded_wnids, excluded_hashes, excluded_names = set(), set(), set()
    exclusion_ids = []
    for path in manifests:
        value, ident = read_bound(path)
        if value["role"] != "generation_conditioning_sensitivity_manifest":
            raise ValueError("unexpected historical sensitivity schema")
        for row in value["dataset"]["samples"]:
            excluded_wnids.add(row["wnid"])
            excluded_hashes.add(row["image"]["sha256"])
            excluded_names.add(Path(row["image"]["path"]).name)
        exclusion_ids.append(ident)
    manifest_path = DATASET / "metadata/image_manifest.jsonl"
    manifest_id = stable_file_identity(manifest_path)
    if manifest_id["sha256"] != MANIFEST_SHA or manifest_id["bytes"] != 405484553:
        raise ValueError("ImageNet manifest identity differs")
    val_rows, counts = [], {}
    with manifest_path.open() as handle:
        for line in handle:
            row = json.loads(line)
            counts[row["split"]] = counts.get(row["split"], 0) + 1
            if row["split"] == "val":
                val_rows.append(row)
    if counts != {"train": 1281167, "val": 50000}:
        raise ValueError("dataset split counts differ")
    mapping = {wnid: i for i, wnid in enumerate(sorted({r["wnid"] for r in val_rows}))}
    if len(mapping) != 1000 or any(mapping[r["wnid"]] != r["label"] for r in val_rows):
        raise ValueError("validation labels differ from lexicographic WNID order")
    selected = []
    for row in select_rows(val_rows, excluded_wnids):
        image_path = DATASET / row["path"]
        image_id = stable_file_identity(image_path)
        if image_id["sha256"] in excluded_hashes or image_path.name in excluded_names:
            raise ValueError("selected image overlaps a previous sensitivity probe")
        with Image.open(image_path) as image:
            if image.size != (256, 256) or image.mode != "RGB":
                raise ValueError("selected image is not RGB 256x256")
            image.verify()
        selected.append({"label": row["label"], "wnid": row["wnid"], "image": image_id})
    bound_sources.extend(exclusion_ids + [manifest_id] + [r["image"] for r in selected])
    for ident in bound_sources:
        if stable_file_identity(ident["path"]) != ident:
            raise ValueError("source changed during catalog construction")
    if census(GENERATION) != manifests or checkout(project_root) != builder:
        raise ValueError("source census or builder changed")
    return {"schema_version": "cofitok_frozen_loss_gradient_source_catalog_v1",
            "status": "prepared_inputs_only", "boundary": BOUNDARY, "builder": builder,
            "class_support_result": result_id, "class_support_validation": receipt_id,
            "methods": methods, "dataset_manifest": manifest_id,
            "exclusion_inventory": exclusion_ids,
            "exclusion_scope": "all discoverable conditioning_sensitivity_manifest.json; generated-image and metric/cache subtrees pruned",
            "excluded_class_count": len(excluded_wnids), "excluded_classes": sorted(excluded_wnids),
            "excluded_image_hash_count": len(excluded_hashes),
            "selection": {"seed": 2031, "count": 32,
                          "algorithm": "ascending sha256('2031/class/'+wnid), then sha256('2031/image/'+relative_path)",
                          "samples": selected},
            "probe_protocol": {"timesteps": [100, 500, 700, 900], "unit": "single_selected_image",
                               "example_timestep_rows_per_method": 128, "not_original_minibatch_gradient": True},
            "remaining_before_execution": ["exact executable and source/runtime compatibility validation",
                                          "independent preparation replay", "distinct stage approval and execution authorization"]}


def exclusive_write(path, value):
    if path.resolve() != OUTPUT or path.is_symlink():
        raise ValueError("only the new exact source-catalog output is permitted")
    for ancestor in path.parents:
        if ancestor.is_symlink():
            raise ValueError("symlinked output parent")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(fd, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return stable_file_identity(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["build", "validate"])
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--catalog-sha256")
    args = parser.parse_args()
    report = build_catalog(args.project_root.resolve())
    if args.mode == "build":
        ident = exclusive_write(OUTPUT, report)
    else:
        if not args.catalog_sha256:
            raise ValueError("validation requires the exact catalog SHA256")
        existing, ident = read_bound(OUTPUT, args.catalog_sha256)
        if existing != report:
            raise ValueError("catalog differs from physical source replay")
    print(json.dumps({"status": "pass", "catalog": ident, "sample_count": 32,
                      "mode": args.mode, "execution_authorized": False}, sort_keys=True))
