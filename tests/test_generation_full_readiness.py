from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from cofitok.configs import load_config
from cofitok.reporting import to_jsonable
from scripts import build_generation_full_readiness as readiness
from scripts.validate_generation_configs import validate_pair


ROOT = Path(__file__).resolve().parents[1]
COFITOK_CONFIG = (
    ROOT
    / "configs/generation/imagenet256_stability_rgbtail3_rollout_x0_u2_"
    "ema_teacher_k8_300k.json"
)
DENSE_CONFIG = (
    ROOT
    / "configs/generation/imagenet256_stability_rollout_x0_u2_"
    "ema_teacher_dense_300k.json"
)
REVISION = "a" * 40
BRANCH = "scale/generation-large-capacity"


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _storage_report(
    path: Path,
    *,
    multiplier: float = 4.0,
    sample_count: int = 16_384,
) -> dict:
    reference = 1_000
    checkpoint_bytes = math.ceil(reference * multiplier)
    checkpoint_reserve = 16 * checkpoint_bytes
    sample_reserve = sample_count * 256 * 1024
    additional = 16 * 1024**3
    safety = 64 * 1024**3
    required = checkpoint_reserve + sample_reserve + additional + safety
    free = required + 1024
    return {
        "schema_version": 2,
        "role": "generation_storage_capacity_preflight",
        "stage": "full_training",
        "status": "pass",
        "git": {
            "revision": REVISION,
            "branch": BRANCH,
            "tracked_dirty": False,
        },
        "filesystem": {
            "path": path.resolve().as_posix(),
            "total_bytes": free + 1024,
            "used_bytes": 1024,
            "free_bytes": free,
        },
        "plan": {
            "checkpoint_count": 16,
            "reference_checkpoint_bytes_each": reference,
            "checkpoint_size_multiplier": multiplier,
            "checkpoint_bytes_each": checkpoint_bytes,
            "checkpoint_reserve_bytes": checkpoint_reserve,
            "sample_count": sample_count,
            "estimated_sample_bytes_each": 256 * 1024,
            "sample_reserve_bytes": sample_reserve,
            "additional_bytes": additional,
            "safety_margin_bytes": safety,
            "required_free_bytes": required,
        },
        "headroom_bytes": free - required,
    }


def test_full_readiness_requires_exact_250m_config_contract() -> None:
    report = to_jsonable(
        validate_pair(
            load_config(COFITOK_CONFIG),
            load_config(DENSE_CONFIG),
            max_parameter_gap=0.02,
            stage="stability_full",
        )
    )

    evidence = readiness.validate_full_config_contract(
        report,
        cofitok_config_path=COFITOK_CONFIG,
        dense_config_path=DENSE_CONFIG,
    )

    assert evidence["cofitok_parameter_count"] == 250_153_763
    assert evidence["dense_parameter_count"] == 250_135_043
    assert evidence["base_channels"] == 256
    drifted = json.loads(json.dumps(report))
    drifted["cofitok"]["parameter_count"] -= 1
    with pytest.raises(ValueError, match="not reproducible"):
        readiness.validate_full_config_contract(
            drifted,
            cofitok_config_path=COFITOK_CONFIG,
            dense_config_path=DENSE_CONFIG,
        )


def test_full_readiness_storage_requires_four_x_and_exact_arithmetic(
    tmp_path: Path,
) -> None:
    report = _storage_report(tmp_path)
    evidence = readiness.validate_full_storage_capacity(
        report,
        expected_revision=REVISION,
        expected_branch=BRANCH,
        expected_path=tmp_path,
    )
    assert evidence["checkpoint_size_multiplier"] == 4.0

    report["plan"]["checkpoint_reserve_bytes"] += 1
    with pytest.raises(ValueError, match="checkpoint reserve arithmetic"):
        readiness.validate_full_storage_capacity(
            report,
            expected_revision=REVISION,
            expected_branch=BRANCH,
            expected_path=tmp_path,
        )

    with pytest.raises(ValueError, match="scaling was weakened"):
        readiness.validate_full_storage_capacity(
            _storage_report(tmp_path, multiplier=3.99),
            expected_revision=REVISION,
            expected_branch=BRANCH,
            expected_path=tmp_path,
        )


def test_full_launch_storage_requires_formal_completion_runway(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="sample_count was weakened"):
        readiness.validate_full_storage_capacity(
            _storage_report(tmp_path),
            expected_revision=REVISION,
            expected_branch=BRANCH,
            expected_path=tmp_path,
            minimum_sample_count=readiness.FULL_COMPLETION_SAMPLE_RESERVE,
        )

    evidence = readiness.validate_full_storage_capacity(
        _storage_report(
            tmp_path,
            sample_count=readiness.FULL_COMPLETION_SAMPLE_RESERVE,
        ),
        expected_revision=REVISION,
        expected_branch=BRANCH,
        expected_path=tmp_path,
        minimum_sample_count=readiness.FULL_COMPLETION_SAMPLE_RESERVE,
    )
    assert evidence["sample_count"] == 116_640


def test_full_readiness_runtime_contract_uses_fixed_candidates_and_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = {}

    def fake_validate(selection: dict, **kwargs: object) -> tuple[int, int]:
        observed.update(kwargs)
        return 4, 16

    monkeypatch.setattr(readiness, "validate_frozen_runtime_selection", fake_validate)
    selection = {
        "schema_version": 3,
        "baseline": {"micro_batch_size": 1},
        "runtime_environment_sha256": "b" * 64,
        "dataset_identity_sha256": "c" * 64,
        "selected": {"estimated_speedup_over_baseline": 1.25},
    }

    evidence = readiness.validate_full_runtime_selection(
        selection,
        cofitok_config_path=COFITOK_CONFIG,
        dense_config_path=DENSE_CONFIG,
        training_run_dirs=[tmp_path / "cofitok", tmp_path / "dense"],
        benchmark_root=tmp_path / "benchmarks",
        expected_revision=REVISION,
        expected_branch=BRANCH,
        project_root=ROOT,
        require_current_runtime_environment=False,
    )

    contract = observed["expected_contract"]
    assert contract["candidates"] == [
        {"micro_batch_size": 1, "gradient_accumulation_steps": 64},
        {"micro_batch_size": 2, "gradient_accumulation_steps": 32},
        {"micro_batch_size": 4, "gradient_accumulation_steps": 16},
        {"micro_batch_size": 8, "gradient_accumulation_steps": 8},
        {"micro_batch_size": 16, "gradient_accumulation_steps": 4},
    ]
    assert contract["baseline_candidate"] == {
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 64,
    }
    assert evidence["effective_batch_size"] == 64

    selection["schema_version"] = 2
    with pytest.raises(ValueError, match="baseline contract"):
        readiness.validate_full_runtime_selection(
            selection,
            cofitok_config_path=COFITOK_CONFIG,
            dense_config_path=DENSE_CONFIG,
            training_run_dirs=[tmp_path / "cofitok", tmp_path / "dense"],
            benchmark_root=tmp_path / "benchmarks",
            expected_revision=REVISION,
            expected_branch=BRANCH,
            project_root=ROOT,
            require_current_runtime_environment=False,
        )


def test_readiness_exact_replay_rejects_source_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_paths = {
        name: _write_json(tmp_path / f"{name}.json", {"name": name})
        for name in (
            "deployment_receipt",
            "promotion_gate",
            "cofitok_config",
            "dense_config",
            "config_validation",
            "storage_capacity",
            "runtime_selection",
        )
    }
    monkeypatch.setattr(
        readiness,
        "verify_deployment_receipt",
        lambda *args, **kwargs: {
            "role": "generation_large_capacity_isolated_deployment",
            "formal_repository": {
                "git": {
                    "revision": "f" * 40,
                    "branch": "scale/generative-system",
                    "tracked_dirty": False,
                }
            },
            "checkout": {
                "path": ROOT.resolve().as_posix(),
                "git": {
                    "revision": REVISION,
                    "branch": BRANCH,
                    "tracked_dirty": False,
                }
            },
            "validation": {
                "pytest": {"tests": 814},
                "runbook_syntax": {"checked_count": 97},
            },
            "readiness_execution_allowed": True,
            "readiness_executed": False,
            "full_training_launch_allowed": False,
        },
    )
    monkeypatch.setattr(
        readiness,
        "verify_generation_gate_source_reports",
        lambda gate: {"source_profile": "stability_scaling"},
    )
    monkeypatch.setattr(
        readiness,
        "capture_generation_gate_binding",
        lambda path, expected_stage: {
            "stage": expected_stage,
            "gate": Path(path).resolve().as_posix(),
        },
    )
    monkeypatch.setattr(
        readiness,
        "validate_full_config_contract",
        lambda *args, **kwargs: {"base_channels": 256},
    )
    monkeypatch.setattr(
        readiness,
        "validate_full_storage_capacity",
        lambda *args, **kwargs: {"checkpoint_size_multiplier": 4.0},
    )
    monkeypatch.setattr(
        readiness,
        "validate_full_runtime_selection",
        lambda *args, **kwargs: {"micro_batch_size": 4, "gradient_accumulation_steps": 16},
    )
    kwargs = {
        "source_paths": source_paths,
        "training_run_dirs": [tmp_path / "cofitok", tmp_path / "dense"],
        "benchmark_root": tmp_path / "benchmarks",
        "storage_path": tmp_path,
        "project_root": ROOT,
        "expected_revision": REVISION,
        "expected_branch": BRANCH,
        "require_current_runtime_environment": False,
        "require_current_git": False,
        "require_training_state_absent": True,
    }
    report = readiness.build_readiness_report(**kwargs)
    assert readiness.verify_readiness_report(report, **kwargs) == report
    assert report["training_state_absent_at_build"] is True
    assert report["deployment"]["pytest_tests"] == 814

    _write_json(source_paths["runtime_selection"], {"name": "changed"})
    with pytest.raises(ValueError, match="not reproducible"):
        readiness.verify_readiness_report(report, **kwargs)


def test_readiness_build_rejects_existing_training_state(tmp_path: Path) -> None:
    run_dir = tmp_path / "cofitok"
    run_dir.mkdir()
    (run_dir / "latest.json").write_text("{}\n", encoding="ascii")

    with pytest.raises(ValueError, match="absent training state"):
        readiness.require_absent_training_state([run_dir, tmp_path / "dense"])
