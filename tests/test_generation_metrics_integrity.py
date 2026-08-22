from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from cofitok import generation_metrics_integrity as integrity
from cofitok.inference_replay import file_identity


def _write(path: Path, content: str = "payload\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _cache_files(cache_root: Path, cache_name: str) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for role, spec in integrity.CACHE_SPECS.items():
        path = cache_root / f"{cache_name}{spec['suffix']}"
        _write(path, role)
        paths[role] = path.resolve()
    return paths


def test_torch_fidelity_distribution_identity_binds_package_trees(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "torch_fidelity"
    dist_info = tmp_path / "torch_fidelity-0.4.0.dist-info"
    _write(package / "__init__.py", "version = '0.4.0'\n")
    _write(package / "metric.py", "CACHE = True\n")
    _write(package / "__pycache__" / "ignored.pyc", "ignored\n")
    _write(dist_info / "METADATA", "Version: 0.4.0\n")
    monkeypatch.setattr(
        integrity.importlib.metadata,
        "version",
        lambda package_name: (
            integrity.EXPECTED_TORCH_FIDELITY_VERSION
            if package_name == "torch-fidelity"
            else None
        ),
    )

    before = integrity.torch_fidelity_distribution_identity(package, dist_info)
    assert before["version"] == "0.4.0"
    assert before["package_tree"]["file_count"] == 2
    assert before["dist_info_tree"]["file_count"] == 1

    _write(package / "metric.py", "CACHE = False\n")
    after = integrity.torch_fidelity_distribution_identity(package, dist_info)
    assert after["package_tree"]["sha256"] != before["package_tree"]["sha256"]


def test_expected_cache_paths_requires_exact_four_file_set(tmp_path: Path) -> None:
    cache_name = "imagenet256_val_50k__cofitok_0123456789abcdef"
    cache_root = tmp_path / "cache"
    expected = _cache_files(cache_root, cache_name)

    observed = integrity.expected_cache_paths(
        cache_root,
        real_cache_name=cache_name,
    )
    assert observed == expected

    _write(cache_root / f"{cache_name}-unexpected.pt")
    with pytest.raises(ValueError, match="cache file set differs"):
        integrity.expected_cache_paths(cache_root, real_cache_name=cache_name)


def test_cache_payload_semantics_replay_mean_and_covariance(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    paths = {role: cache_root / f"{role}.pt" for role in integrity.CACHE_SPECS}
    features = torch.tensor(
        [
            [0.0, 1.0, 2.0],
            [1.0, 2.0, 4.0],
            [2.0, 4.0, 8.0],
            [4.0, 8.0, 16.0],
        ],
        dtype=torch.float32,
    )
    torch.save(features, paths["inception_features_2048"])
    torch.save(
        torch.arange(8, dtype=torch.float32).reshape(4, 2),
        paths["inception_logits_unbiased"],
    )
    torch.save(
        {
            "mu": np.mean(features.numpy(), axis=0),
            "sigma": np.cov(features.numpy(), rowvar=False),
        },
        paths["inception_fid_statistics"],
    )
    torch.save(
        torch.arange(16, dtype=torch.float32).reshape(4, 4),
        paths["vgg_features_fc2_relu"],
    )
    specs = {
        "inception_features_2048": {
            "kind": "tensor",
            "shape_tail": [3],
            "dtype": "torch.float32",
        },
        "inception_logits_unbiased": {
            "kind": "tensor",
            "shape_tail": [2],
            "dtype": "torch.float32",
        },
        "inception_fid_statistics": {
            "kind": "fid_statistics",
            "mu_shape": [3],
            "mu_dtype": "float32",
            "sigma_shape": [3, 3],
            "sigma_dtype": "float64",
        },
        "vgg_features_fc2_relu": {
            "kind": "tensor",
            "shape_tail": [4],
            "dtype": "torch.float32",
        },
    }

    replay = integrity.validate_cache_payloads(
        paths,
        image_count=4,
        specs=specs,
        covariance_probe_indices=(0, 1, 2),
    )
    assert replay["fid_statistics_consistency"]["mean_exact"] is True
    assert replay["fid_statistics_consistency"]["covariance_probe_allclose"] is True

    statistics = torch.load(paths["inception_fid_statistics"], weights_only=False)
    statistics["mu"] = statistics["mu"].copy()
    statistics["mu"][0] += 1.0
    torch.save(statistics, paths["inception_fid_statistics"])
    with pytest.raises(ValueError, match="FID mean"):
        integrity.validate_cache_payloads(
            paths,
            image_count=4,
            specs=specs,
            covariance_probe_indices=(0, 1, 2),
        )


def _creation_fixture(tmp_path: Path) -> tuple[dict[str, Any], str, dict[str, Any]]:
    cache_root = (tmp_path / "cache").resolve()
    cache_name = "imagenet256_val_50k__cofitok_0123456789abcdef"
    cache_paths = _cache_files(cache_root, cache_name)
    real_dir = (tmp_path / "real").resolve()
    generated_dir = (tmp_path / "samples" / "png").resolve()
    real_set = {
        "digest_schema": "cofitok_image_tree_sha256_v1",
        "root": real_dir.as_posix(),
        "image_count": 2,
        "sha256": "0" * 64,
    }
    implementation = {"package": "torch_fidelity", "version": "0.4.0"}
    vgg_weight = {
        "path": (tmp_path / "weights" / "vgg16-397923af.pth").resolve().as_posix(),
        "bytes": 1,
        "sha256": "397923af" + "0" * 56,
    }
    report = {
        "status": "completed",
        "protocol": "torch_fidelity_directory_metrics",
        "implementation": implementation,
        "real_set": real_set,
        "parameters": {"real_cache_name": cache_name},
        "paths": {
            "real_dir": real_dir.as_posix(),
            "generated_dir": generated_dir.as_posix(),
        },
    }
    original_report = (
        generated_dir.parent / "metrics" / "generation_metrics_report.json"
    )
    events = [
        "Extracting features from input2",
        f'Looking for samples recursively in "{real_dir.as_posix()}"',
        "Found 2 samples",
        f"Caching {cache_paths['inception_features_2048'].as_posix()}",
        f"Caching {cache_paths['inception_logits_unbiased'].as_posix()}",
        f"Caching {cache_paths['inception_fid_statistics'].as_posix()}",
        "Creating feature extractor \"vgg16\" with features ['fc2_relu']",
        (
            'Downloading: "https://download.pytorch.org/models/'
            f'{Path(vgg_weight["path"]).name}" to {vgg_weight["path"]}'
        ),
        "Extracting features from input2",
        f'Looking for samples recursively in "{real_dir.as_posix()}"',
        "Found 2 samples",
        f"Caching {cache_paths['vgg_features_fc2_relu'].as_posix()}",
        f"wrote {original_report.as_posix()}",
    ]
    kwargs = {
        "creation_report": report,
        "cache_root": cache_root,
        "cache_paths": cache_paths,
        "real_cache_name": cache_name,
        "real_set": real_set,
        "implementation": implementation,
        "vgg_weight": vgg_weight,
    }
    return kwargs, "\n".join(events), report


def test_cache_creation_evidence_requires_ordered_first_creation_events(
    tmp_path: Path,
) -> None:
    kwargs, log_text, _ = _creation_fixture(tmp_path)
    verified = integrity.validate_cache_creation_evidence(
        creation_log_text=log_text,
        **kwargs,
    )
    assert verified["ordered_event_sequence_verified"] is True

    lines = log_text.splitlines()
    lines[3], lines[4] = lines[4], lines[3]
    with pytest.raises(ValueError, match="ordered evidence"):
        integrity.validate_cache_creation_evidence(
            creation_log_text="\n".join(lines),
            **kwargs,
        )


def test_replay_identity_rejects_physical_sha_drift(tmp_path: Path) -> None:
    payload = tmp_path / "cache.pt"
    _write(payload, "first\n")
    identity = file_identity(payload)
    assert integrity._replay_identity(identity, label="cache") == identity

    _write(payload, "second\n")
    with pytest.raises(ValueError, match="changed after attestation"):
        integrity._replay_identity(identity, label="cache")


def _generation_report(verified: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 3,
        "role": "generation_directory_metrics_report",
        "status": "completed",
        "protocol": "torch_fidelity_directory_metrics",
        "implementation": {
            "package": verified["torch_fidelity"]["package"],
            "version": verified["torch_fidelity"]["version"],
        },
        "runtime_environment_sha256": "1" * 64,
        "real_set": copy.deepcopy(verified["real_set"]),
        "parameters": {
            "cache_root": verified["cache_root"],
            "real_cache_name": verified["real_cache_name"],
            "precision_recall_enabled": True,
        },
    }


@pytest.mark.parametrize("drift_field", ["cache_root", "real_cache_name"])
def test_generation_report_binding_rejects_cache_input_drift(
    drift_field: str,
) -> None:
    verified = {
        "torch_fidelity": {"package": "torch_fidelity", "version": "0.4.0"},
        "real_set": {
            "digest_schema": "cofitok_image_tree_sha256_v1",
            "root": "/real",
            "image_count": 50_000,
            "sha256": "2" * 64,
        },
        "cache_root": "/cache",
        "real_cache_name": "imagenet256_val__cofitok_2222222222222222",
    }
    reports = {
        "cofitok": _generation_report(verified),
        "dense_identity": _generation_report(verified),
    }
    result = integrity.validate_generation_reports_against_metrics_trust(
        reports,
        verified_trust=verified,
    )
    assert result["matched_attested_metrics_inputs"] is True

    reports["dense_identity"]["parameters"][drift_field] += "-drift"
    with pytest.raises(ValueError, match="not bound to attested cache inputs"):
        integrity.validate_generation_reports_against_metrics_trust(
            reports,
            verified_trust=verified,
        )


def test_metrics_trust_boundary_is_permanently_non_authorizing() -> None:
    boundary = integrity.AUTHORIZATION_BOUNDARY
    assert boundary["diagnostic_non_authorizing"] is True
    assert boundary["cpu_only_physical_integrity"] is True
    for name in (
        "gpu_execution_allowed",
        "training_launch_allowed",
        "sampling_launch_allowed",
        "full_training_launch_allowed",
        "full_300k_launch_allowed",
        "promotion_authorization_allowed",
        "release_authorization_allowed",
        "process_signals_allowed",
    ):
        assert boundary[name] is False
