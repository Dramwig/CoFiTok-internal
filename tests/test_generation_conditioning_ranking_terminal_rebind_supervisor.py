from __future__ import annotations

import pytest

from scripts.run_generation_conditioning_ranking_terminal_rebind_supervisor import (
    AUTHORIZATION_BOUNDARY,
    parse_gpu_process_pids,
)


def test_gpu_process_parser_is_strict_and_deduplicated() -> None:
    assert parse_gpu_process_pids("") == []
    assert parse_gpu_process_pids("42\n7\n42\n") == [7, 42]
    with pytest.raises(ValueError, match="unparseable"):
        parse_gpu_process_pids("not-a-pid")


def test_supervisor_cannot_expand_beyond_four_arm_probe() -> None:
    assert AUTHORIZATION_BOUNDARY["consecutive_idle_gpu_polls_required"] == 5
    for field in (
        "unrelated_process_signaling_allowed",
        "sampling_launch_allowed",
        "checkpoint_promotion_allowed",
        "followup_training_allowed",
        "full_training_launch_allowed",
        "full_300k_launch_allowed",
        "inference_export_allowed",
        "release_allowed",
    ):
        assert AUTHORIZATION_BOUNDARY[field] is False
