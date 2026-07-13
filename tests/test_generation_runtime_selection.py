from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from cofitok.data.provenance import (
    FORMAL_GENERATION_DATASETS,
    dataset_provenance_identity_sha256,
)
from cofitok.environment import runtime_environment_sha256
from cofitok.reporting import file_sha256
from scripts.select_generation_training_runtime import (
    _benchmark_matches,
    _expected_config,
    _selection_contract,
    parse_candidates,
    reuse_runtime_selection_after_training_start,
    select_runtime_candidate,
    validate_frozen_runtime_selection,
)


ROOT = Path(__file__).resolve().parents[1]
REVISION = "a" * 40


def _dataset_provenance() -> dict:
    spec = FORMAL_GENERATION_DATASETS["imagenet_256"]
    report = {
        "schema_version": 1,
        "status": "pass",
        "formal": True,
        "dataset": "imagenet_256",
        "dataset_root": "/root/autodl-tmp/CoFiTok/datasets/imagenet_256",
        "manifest": {
            "relative_path": "metadata/image_manifest.jsonl",
            "bytes": spec.manifest_bytes,
            "sha256": spec.manifest_sha256,
        },
        "splits": {"train": spec.train_images, "val": spec.val_images},
        "issues": [],
    }
    report["identity_sha256"] = dataset_provenance_identity_sha256(report)
    return report


def _method(seconds: float, *, peak_fraction: float = 0.5) -> dict:
    total = 100_000
    environment = {
        "schema_version": 1,
        "device": {"type": "cuda", "name": "GPU"},
    }
    return {
        "status": "completed",
        "runtime_environment": environment,
        "runtime_environment_sha256": runtime_environment_sha256(environment),
        "dataset_provenance": _dataset_provenance(),
        "effective_batch_size": 64,
        "mean_optimizer_step_seconds": seconds,
        "images_per_second": 64 / seconds,
        "peak_vram_bytes": int(total * peak_fraction),
        "device_total_memory_bytes": total,
    }


def _candidate(micro: int, accumulation: int, cofitok: dict, dense: dict) -> dict:
    return {
        "micro_batch_size": micro,
        "gradient_accumulation_steps": accumulation,
        "effective_batch_size": micro * accumulation,
        "methods": {"cofitok": cofitok, "dense_identity": dense},
    }


def _frozen_selection(tmp_path: Path) -> tuple[dict, dict, list[Path], Path, Path]:
    cofitok_config = ROOT / "configs/generation/imagenet256_cofitok_k8_300k.json"
    dense_config = ROOT / "configs/generation/imagenet256_dense_300k.json"
    run_dirs = [(tmp_path / "cofitok").resolve(), (tmp_path / "dense").resolve()]
    candidates = [(16, 4), (32, 2), (64, 1)]
    rows = []
    for index, (micro_batch, accumulation) in enumerate(candidates):
        methods = {}
        for method, config_path in (
            ("cofitok", cofitok_config),
            ("dense_identity", dense_config),
        ):
            report = _method(2.0 + index * 0.25)
            report.update(
                config=_expected_config(config_path, micro_batch, accumulation),
                git={
                    "revision": REVISION,
                    "branch": "scale/generative-system",
                    "dirty": False,
                },
                benchmark_steps=8,
                warmup_steps=2,
                checkpoint_written=False,
            )
            methods[method] = report
        rows.append(
            _candidate(
                micro_batch,
                accumulation,
                methods["cofitok"],
                methods["dense_identity"],
            )
        )
    selection = select_runtime_candidate(
        rows,
        expected_effective_batch=64,
        max_memory_fraction=0.9,
    )
    config_sha256 = {
        "cofitok": file_sha256(cofitok_config),
        "dense_identity": file_sha256(dense_config),
    }
    contract = _selection_contract(
        run_dirs=run_dirs,
        candidates=candidates,
        expected_effective_batch=64,
        benchmark_steps=8,
        warmup_steps=2,
        max_memory_fraction=0.9,
        target_steps=300_000,
        revision=REVISION,
        branch="scale/generative-system",
        config_sha256=config_sha256,
        benchmark_root=(tmp_path / "benchmarks").resolve(),
    )
    selection.update(
        git_revision=REVISION,
        config_sha256=config_sha256,
        benchmark_root=contract["benchmark_root"],
        selection_lock=contract,
    )
    return selection, contract, run_dirs, cofitok_config, dense_config


def test_parse_candidates_preserves_effective_batch_and_baseline() -> None:
    assert parse_candidates("16x4,32x2,64x1", expected_effective_batch=64) == [
        (16, 4),
        (32, 2),
        (64, 1),
    ]

    with pytest.raises(ValueError, match="changes effective batch"):
        parse_candidates("16x4,32x1", expected_effective_batch=64)
    with pytest.raises(ValueError, match="16x4 baseline"):
        parse_candidates("32x2,64x1", expected_effective_batch=64)


def test_selector_minimizes_worst_method_runtime() -> None:
    report = select_runtime_candidate(
        [
            _candidate(16, 4, _method(2.5), _method(2.4)),
            _candidate(32, 2, _method(1.9), _method(2.0)),
            _candidate(64, 1, _method(1.8), _method(2.2)),
        ],
        expected_effective_batch=64,
        max_memory_fraction=0.9,
    )

    assert report["selected"]["micro_batch_size"] == 32
    assert report["selected"]["gradient_accumulation_steps"] == 2
    assert report["selected"]["selection_score_seconds"] == 2.0
    assert report["selected"]["estimated_speedup_over_16x4"] == 1.25
    assert report["schema_version"] == 2
    assert len(report["runtime_environment_sha256"]) == 64
    assert len(report["dataset_identity_sha256"]) == 64


def test_selector_rejects_oom_or_insufficient_memory_headroom() -> None:
    failed = {"status": "failed", "failure_type": "cuda_oom"}
    report = select_runtime_candidate(
        [
            _candidate(16, 4, _method(2.5), _method(2.4)),
            _candidate(32, 2, _method(1.9, peak_fraction=0.95), _method(2.0)),
            _candidate(64, 1, failed, _method(1.7)),
        ],
        expected_effective_batch=64,
        max_memory_fraction=0.9,
    )

    assert report["selected"]["micro_batch_size"] == 16
    assert "cofitok_memory_headroom" in report["candidates"][1]["ineligible_reasons"]
    assert "cofitok_benchmark_incomplete" in report["candidates"][2]["ineligible_reasons"]


def test_selector_fails_closed_when_conservative_baseline_fails() -> None:
    with pytest.raises(ValueError, match="16x4 runtime baseline"):
        select_runtime_candidate(
            [
                _candidate(
                    16,
                    4,
                    {"status": "failed", "failure_type": "process_error"},
                    _method(2.4),
                ),
                _candidate(32, 2, _method(2.0), _method(2.0)),
            ],
            expected_effective_batch=64,
            max_memory_fraction=0.9,
        )


def test_selector_rejects_invalid_or_drifted_runtime_environment() -> None:
    invalid = _method(2.0)
    invalid["runtime_environment"]["device"]["name"] = "changed"

    with pytest.raises(ValueError, match="environment provenance failed"):
        select_runtime_candidate(
            [
                _candidate(16, 4, _method(2.5), _method(2.4)),
                _candidate(32, 2, _method(2.0), invalid),
            ],
            expected_effective_batch=64,
            max_memory_fraction=0.9,
        )

    changed_cofitok = _method(2.0)
    changed_dense = _method(2.0)
    for report in (changed_cofitok, changed_dense):
        report["runtime_environment"]["device"]["name"] = "another GPU"
        report["runtime_environment_sha256"] = runtime_environment_sha256(
            report["runtime_environment"]
        )
    with pytest.raises(ValueError, match="changed across benchmark candidates"):
        select_runtime_candidate(
            [
                _candidate(16, 4, _method(2.5), _method(2.4)),
                _candidate(32, 2, changed_cofitok, changed_dense),
            ],
            expected_effective_batch=64,
            max_memory_fraction=0.9,
        )

    changed_dense = _method(1.8)
    changed_dense["runtime_environment"]["device"]["name"] = "third GPU"
    changed_dense["runtime_environment_sha256"] = runtime_environment_sha256(
        changed_dense["runtime_environment"]
    )
    with pytest.raises(ValueError, match="changed across benchmark candidates"):
        select_runtime_candidate(
            [
                _candidate(16, 4, _method(2.5), _method(2.4)),
                _candidate(
                    64,
                    1,
                    {"status": "failed", "failure_type": "cuda_oom"},
                    changed_dense,
                ),
            ],
            expected_effective_batch=64,
            max_memory_fraction=0.9,
        )


def test_selector_rejects_dataset_identity_drift() -> None:
    changed_cofitok = _method(2.0)
    changed_cofitok["dataset_provenance"]["dataset_root"] += "_copy"
    changed_cofitok["dataset_provenance"]["identity_sha256"] = (
        dataset_provenance_identity_sha256(
            changed_cofitok["dataset_provenance"]
        )
    )
    changed_dense = copy.deepcopy(changed_cofitok)

    with pytest.raises(ValueError, match="dataset identity changed"):
        select_runtime_candidate(
            [
                _candidate(16, 4, _method(2.5), _method(2.4)),
                _candidate(32, 2, changed_cofitok, changed_dense),
            ],
            expected_effective_batch=64,
            max_memory_fraction=0.9,
        )


def test_benchmark_cache_requires_clean_git_and_valid_environment() -> None:
    expected_config = {"name": "benchmark"}
    report = _method(2.0)
    report.update(
        config=expected_config,
        git={"revision": "a" * 40, "dirty": False},
        benchmark_steps=8,
        warmup_steps=2,
        checkpoint_written=False,
    )

    assert _benchmark_matches(
        report,
        expected_config=expected_config,
        expected_revision="a" * 40,
        benchmark_steps=8,
        warmup_steps=2,
    )
    report["git"]["dirty"] = True
    assert not _benchmark_matches(
        report,
        expected_config=expected_config,
        expected_revision="a" * 40,
        benchmark_steps=8,
        warmup_steps=2,
    )
    report["git"]["dirty"] = False
    report["runtime_environment_sha256"] = "0" * 64
    assert not _benchmark_matches(
        report,
        expected_config=expected_config,
        expected_revision="a" * 40,
        benchmark_steps=8,
        warmup_steps=2,
    )


def test_frozen_selection_is_reused_without_rebenchmarking_after_training_starts(
    tmp_path: Path,
) -> None:
    selection, contract, run_dirs, cofitok_config, dense_config = _frozen_selection(
        tmp_path
    )
    selection_path = tmp_path / "runtime_selection.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    for run_dir in run_dirs:
        run_dir.mkdir()

    assert (
        reuse_runtime_selection_after_training_start(
            selection_path=selection_path,
            run_dirs=run_dirs,
            expected_contract=contract,
            cofitok_config=cofitok_config,
            dense_config=dense_config,
            current_runtime_environment_sha256=selection[
                "runtime_environment_sha256"
            ],
        )
        is None
    )

    (run_dirs[0] / "train_metrics.jsonl").write_text("{}\n", encoding="ascii")
    original = selection_path.read_bytes()
    assert reuse_runtime_selection_after_training_start(
        selection_path=selection_path,
        run_dirs=run_dirs,
        expected_contract=contract,
        cofitok_config=cofitok_config,
        dense_config=dense_config,
        current_runtime_environment_sha256=selection["runtime_environment_sha256"],
    ) == (16, 4)
    assert selection_path.read_bytes() == original


def test_training_state_without_selection_fails_before_benchmark(tmp_path: Path) -> None:
    _, contract, run_dirs, cofitok_config, dense_config = _frozen_selection(tmp_path)
    run_dirs[0].mkdir()
    (run_dirs[0] / "checkpoint_step_00050000.pt").write_bytes(b"checkpoint")

    with pytest.raises(FileNotFoundError, match="without a frozen runtime selection"):
        reuse_runtime_selection_after_training_start(
            selection_path=tmp_path / "missing.json",
            run_dirs=run_dirs,
            expected_contract=contract,
            cofitok_config=cofitok_config,
            dense_config=dense_config,
            current_runtime_environment_sha256="0" * 64,
        )


@pytest.mark.parametrize("drift", ["contract", "candidate", "environment"])
def test_frozen_selection_rejects_resume_drift(tmp_path: Path, drift: str) -> None:
    selection, contract, _, cofitok_config, dense_config = _frozen_selection(tmp_path)
    current_environment = selection["runtime_environment_sha256"]
    if drift == "contract":
        contract = copy.deepcopy(contract)
        contract["benchmark_steps"] = 9
    elif drift == "candidate":
        selection["candidates"][0]["micro_batch_size"] = 8
    else:
        current_environment = "f" * 64

    with pytest.raises(ValueError, match="frozen|environment"):
        validate_frozen_runtime_selection(
            selection,
            expected_contract=contract,
            cofitok_config=cofitok_config,
            dense_config=dense_config,
            current_runtime_environment_sha256=current_environment,
        )
