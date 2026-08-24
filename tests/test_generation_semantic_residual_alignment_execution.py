from __future__ import annotations

from pathlib import Path

from cofitok.generation.semantic_residual_alignment_probe import (
    CLAIM_BOUNDARY,
    EXECUTION_BOUNDARY,
)
from scripts.run_generation_semantic_residual_alignment_probe_supervisor import (
    AUTHORIZATION_BOUNDARY,
    parse_gpu_process_pids,
)


ROOT = Path(__file__).resolve().parents[1]
RUNBOOK = (
    ROOT
    / "artifacts"
    / "runbooks"
    / "generation_semantic_residual_alignment_four_arm_probe1k_v1.sh"
)


def test_gpu_parser_is_strict_and_deduplicated() -> None:
    assert parse_gpu_process_pids("") == []
    assert parse_gpu_process_pids("42\n7\n42\n") == [7, 42]


def test_supervisor_requires_five_idle_polls_and_cannot_promote() -> None:
    assert AUTHORIZATION_BOUNDARY["consecutive_idle_gpu_polls_required"] == 5
    for field in (
        "unrelated_process_signaling_allowed",
        "generation_sampling_allowed",
        "checkpoint_promotion_allowed",
        "followup_training_allowed",
        "full_training_launch_allowed",
        "full_300k_launch_allowed",
        "inference_export_allowed",
        "release_allowed",
    ):
        assert AUTHORIZATION_BOUNDARY[field] is False


def test_runbook_is_serial_and_physically_audits_all_checkpoints() -> None:
    source = RUNBOOK.read_text(encoding="utf-8")
    positions = [
        source.index(f'run_one {name} "')
        for name in EXECUTION_BOUNDARY["training_runs"]
    ]
    assert positions == sorted(positions)
    assert source.count("scripts/train_generation.py") == 1
    assert "--required-checkpoint-steps 500,750,1000" in source
    assert "--integrity-policy required" in source
    assert source.count("nvidia-smi --query-compute-apps=pid") >= 2
    assert "--wrong-label-offset 250" in source
    assert "--start-label 128" in source
    assert "--noise-seed 314159" in source


def test_all_claim_permissions_remain_false() -> None:
    assert CLAIM_BOUNDARY["generation_advantage_proven"] is False
    for field in (
        "training_quality_claim_allowed",
        "sample_quality_claim_allowed",
        "cofitok_specific_advantage_claim_allowed",
        "checkpoint_promotion_allowed",
        "followup_training_allowed",
        "full_training_launch_allowed",
        "full_300k_launch_allowed",
        "inference_export_allowed",
        "release_allowed",
        "process_signal_allowed",
    ):
        assert CLAIM_BOUNDARY[field] is False
