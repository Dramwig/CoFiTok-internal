from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from cofitok.generation_sample_archive import audit_generation_sample_archive
from cofitok.image_integrity import sample_set_sha256
from cofitok.reporting import file_sha256


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _fixture(tmp_path: Path, *, link: bool = False, wrong_digest: bool = False) -> tuple[Path, str]:
    run = "test_run"
    root = f"{run}/samples_gate"
    source_root = f"/root/autodl-tmp/CoFiTok/checkpoints/generation/{root}"
    samples = {"000000.png": b"first", "000001.png": b"second"}
    physical = tmp_path / "physical"
    physical.mkdir()
    for name, payload in samples.items():
        (physical / name).write_bytes(payload)
    sample_sha = sample_set_sha256(physical.glob("*.png"))
    sampling = {
        "num_samples": 2,
        "start_index": 0,
        "prefix_budgets": [8],
        "sample_set_digest": {
            "algorithm": "sha256",
            "framing": "filename_utf8_nul_file_bytes_nul",
        },
    }
    common = {
        "git": {"revision": "a" * 40, "branch": "test", "tracked_dirty": False},
        "runtime_environment": {"python": "test"},
        "runtime_environment_sha256": "b" * 64,
        "checkpoint": "/checkpoints/model.pt",
        "checkpoint_sha256": "c" * 64,
        "checkpoint_integrity_manifest": "/checkpoints/model.pt.integrity.json",
        "checkpoint_step": 50_000,
        "weights": "ema",
        "sampling": sampling,
        "output_dirs": {"8": f"{source_root}/prefix_8"},
    }
    manifest = {"schema_version": 1, **common}
    manifest_payload = _json_bytes(manifest)
    manifest_sha = hashlib.sha256(manifest_payload).hexdigest()
    sample_sets = {
        "8": {"count": 2, "sha256": "0" * 64 if wrong_digest else sample_sha}
    }
    progress = {
        "status": "completed",
        "sampling_manifest_sha256": manifest_sha,
        "completed_samples": 2,
        "prefix_budgets": [8],
        "sample_sets": sample_sets,
    }
    report = {
        "schema_version": 1,
        "status": "completed",
        **common,
        "sampling_manifest_sha256": manifest_sha,
        "sampling_progress": f"{source_root}/sampling_progress.json",
        "sample_sets": sample_sets,
    }
    files = {
        f"{root}/sampling_manifest.json": manifest_payload,
        f"{root}/sampling_progress.json": _json_bytes(progress),
        f"{root}/sampling_report.json": _json_bytes(report),
        **{f"{root}/prefix_8/{name}": payload for name, payload in samples.items()},
    }
    archive_path = tmp_path / "samples.tar"
    with tarfile.open(archive_path, mode="w") as archive:
        for directory in (run, root, f"{root}/prefix_8"):
            info = tarfile.TarInfo(directory)
            info.type = tarfile.DIRTYPE
            archive.addfile(info)
        for name, payload in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        if link:
            info = tarfile.TarInfo(f"{root}/forbidden_link")
            info.type = tarfile.SYMTYPE
            info.linkname = "/tmp"
            archive.addfile(info)
    return archive_path, root


def test_sample_archive_replays_sampling_integrity(tmp_path: Path) -> None:
    archive, root = _fixture(tmp_path)
    report = audit_generation_sample_archive(
        archive_path=archive,
        expected_archive_sha256=file_sha256(archive),
        expected_roots=[root],
    )

    assert report["status"] == "pass"
    assert report["summary"]["sample_count"] == 2
    assert report["summary"]["regular_file_count"] == 5
    assert report["roots"][0]["sample_sets"]["8"]["count"] == 2
    assert report["remote_source_deletion_performed"] is False


def test_sample_archive_rejects_non_regular_members(tmp_path: Path) -> None:
    archive, root = _fixture(tmp_path, link=True)
    with pytest.raises(ValueError, match="non-regular member"):
        audit_generation_sample_archive(
            archive_path=archive,
            expected_archive_sha256=file_sha256(archive),
            expected_roots=[root],
        )


def test_sample_archive_rejects_sample_digest_mismatch(tmp_path: Path) -> None:
    archive, root = _fixture(tmp_path, wrong_digest=True)
    with pytest.raises(ValueError, match="sample-set digest differs"):
        audit_generation_sample_archive(
            archive_path=archive,
            expected_archive_sha256=file_sha256(archive),
            expected_roots=[root],
        )
