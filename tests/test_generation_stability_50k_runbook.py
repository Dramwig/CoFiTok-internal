from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNBOOK = (
    ROOT
    / "artifacts/runbooks/generation_stability_ema_teacher_matched_50k_after_gate.sh"
)


def test_stability_50k_runbook_is_gate_bound_and_non_promoting() -> None:
    source = RUNBOOK.read_text(encoding="utf-8")

    gate_index = source.index(
        "scripts/validate_generation_stability_scaling_decision.py"
    )
    config_index = source.index("scripts/validate_generation_configs.py")
    runtime_index = source.index("scripts/select_generation_training_runtime.py")
    training_index = source.index('start_monitor\nrun_training "$COFITOK_CONFIG"')
    assert gate_index < config_index < runtime_index < training_index
    assert "EXPECTED_STABILITY_DECISION_SHA256=${" in source
    assert (
        "EXPECTED_SOURCE_REVISION="
        "59db142fc45d69dc92bb0333be5ac2d0162d9dc4"
    ) in source
    assert "--expected-next-stage fresh_matched_50k_preparation" in source
    assert "--stage stability_scaling" in source
    assert "--checkpoint-integrity-policy required" in source
    assert '--expected-checkpoint-revision "$EXPECTED_TARGET_REVISION"' in source
    assert "scripts/check_generation_storage_capacity.py" in source
    assert "scripts/build_generation_stability_50k_summary.py" in source
    assert "300k" not in source.lower()


def test_stability_50k_runbook_uses_the_new_matched_configs() -> None:
    source = RUNBOOK.read_text(encoding="utf-8")

    assert (
        "imagenet256_10pct_stability_rgbtail3_rollout_x0_u2_"
        "ema_teacher_k8_50k.json"
    ) in source
    assert (
        "imagenet256_10pct_stability_rollout_x0_u2_"
        "ema_teacher_dense_50k.json"
    ) in source
    assert "--expected-steps 50000" in source
    assert "--checkpoint-interval 5000" in source
    assert "--candidates 16x4,32x2,64x1" in source
