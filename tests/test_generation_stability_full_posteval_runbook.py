from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNBOOK = (
    ROOT
    / "artifacts/runbooks/"
    "generation_stability_ema_teacher_full_posteval_50k.sh"
)


def test_stability_full_posteval_is_training_and_evaluation_identity_bound() -> None:
    source = RUNBOOK.read_text(encoding="utf-8")

    assert "EXPECTED_SCALING_GATE_SHA256=${" in source
    assert "EXPECTED_TRAINING_REVISION=${" in source
    assert "EXPECTED_TRAINING_BRANCH=${" in source
    assert "EXPECTED_TARGET_REVISION=${" in source
    assert "EXPECTED_TARGET_BRANCH=${" in source
    assert '--expected-training-revision "$EXPECTED_TRAINING_REVISION"' in source
    assert '--expected-evaluation-revision "$EXPECTED_TARGET_REVISION"' in source


def test_stability_full_posteval_uses_formal_protocol_and_profile() -> None:
    source = RUNBOOK.read_text(encoding="utf-8")

    assert source.count("--num-samples 50000") == 2
    assert source.count("--sample-steps 250") == 3
    assert source.count("--weights ema") >= 5
    assert source.count("--min-samples 50000") >= 3
    assert "--source-profile stability_full" in source
    assert "--max-absolute-fid 20.0" in source
    assert "--min-precision 0.30" in source
    assert "--min-recall 0.30" in source


def test_stability_full_posteval_reaudits_training_and_gate_sources() -> None:
    source = RUNBOOK.read_text(encoding="utf-8")

    assert "--expected-recipe-stage stability_full" in source
    assert source.count("scripts/audit_generation_training_progress.py") == 2
    assert source.count("--integrity-policy required") == 2
    assert "--required-checkpoint-steps 50000,100000,200000,300000" in source
    assert "--sources-only" in source
    assert "scripts/validate_generation_gate_report.py" in source


def test_stability_full_comparison_keeps_official_methods_contextual() -> None:
    source = RUNBOOK.read_text(encoding="utf-8")

    assert "scripts/build_large_scale_generation_comparison.py" in source
    assert "--official-related" in source
    assert "--source-profile stability_full" in source
    assert "generation_stability_ema_teacher_full_matched_300k_after_gate.sh" not in source
