from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNBOOK = (
    ROOT
    / "artifacts/runbooks/"
    "generation_stability_ema_teacher_50k_posteval_after_training.sh"
)


def test_stability_50k_posteval_is_decision_and_revision_bound() -> None:
    source = RUNBOOK.read_text(encoding="utf-8")

    assert "EXPECTED_STABILITY_DECISION_SHA256" in source
    assert "EXPECTED_TARGET_REVISION" in source
    assert "EXPECTED_TARGET_BRANCH" in source
    assert "EXPECTED_TRAINING_REVISION" in source
    assert "EXPECTED_TRAINING_BRANCH" in source
    assert "scripts/validate_generation_stability_scaling_decision.py" in source
    assert "scripts/build_generation_stability_50k_summary.py" in source
    assert source.index(
        "scripts/build_generation_stability_50k_summary.py"
    ) < source.index("scripts/select_generation_sampling_batch.py")


def test_stability_50k_posteval_uses_formal_ema_gate_protocol() -> None:
    source = RUNBOOK.read_text(encoding="utf-8")

    assert source.count("--num-samples 10000") == 2
    assert source.count("--sample-steps 100") == 5
    assert source.count("--weights ema") >= 5
    assert source.count("scripts/evaluate_generation_checkpoint.py") == 2
    assert source.count("scripts/evaluate_generation_metrics.py") == 2
    assert source.count("--min-samples 10000 \\\n  --resume") == 2
    assert (
        "--random-orders 16 \\\n"
        "  --weights ema \\\n"
        "  --precision bf16 \\\n"
        "  --resume"
    ) in source
    assert (
        "--random-orders 0 \\\n"
        "  --weights ema \\\n"
        "  --precision bf16 \\\n"
        "  --resume"
    ) in source
    assert "--source-profile stability_scaling" in source
    assert '--expected-training-revision "$EXPECTED_TRAINING_REVISION"' in source
    assert '--expected-training-branch "$EXPECTED_TRAINING_BRANCH"' in source
    assert '--expected-evaluation-revision "$EXPECTED_TARGET_REVISION"' in source
    assert '--expected-evaluation-branch "$EXPECTED_TARGET_BRANCH"' in source
    assert "--min-coarse-token-energy-ratio 0.05" in source
    assert "--sources-only" in source
    assert "scripts/validate_generation_gate_report.py" in source


def test_stability_50k_posteval_binds_matched_ema_rollout_diagnostics() -> None:
    source = RUNBOOK.read_text(encoding="utf-8")

    assert source.count("scripts/evaluate_generation_rollout_stability.py") == 2
    assert source.count("--num-images 64") >= 2
    assert source.count("--seed 2029") == 2
    assert "scripts/build_generation_stability_qualification.py" in source
    qualification = source.index("scripts/build_generation_stability_qualification.py")
    assert source.index("--weights ema", qualification) < source.index(
        '--output-dir "$EMA_ROLLOUT_QUALIFICATION_ROOT"', qualification
    )
    assert (
        '--expected-evaluation-revision "$EXPECTED_TARGET_REVISION"' in source
    )
    assert '--expected-evaluation-branch "$EXPECTED_TARGET_BRANCH"' in source
    assert (
        '--rollout-stability-qualification "$EMA_ROLLOUT_QUALIFICATION"'
        in source
    )
    assert source.index("scripts/evaluate_generation_checkpoint.py") < source.index(
        "scripts/evaluate_generation_rollout_stability.py"
    )
    assert source.index("scripts/build_generation_stability_qualification.py") < source.index(
        "scripts/build_generation_gate_report.py"
    )


def test_stability_50k_posteval_cannot_launch_full_training() -> None:
    source = RUNBOOK.read_text(encoding="utf-8")

    assert "generation_full_matched_300k_after_gate.sh" not in source
    assert "scripts/train_generation.py" not in source


def test_stability_50k_posteval_audits_each_run_against_its_config() -> None:
    source = RUNBOOK.read_text(encoding="utf-8")

    cofitok_run = source.index('--run-dir "$COFITOK_RUN"')
    cofitok_config = source.index('--config "$COFITOK_CONFIG"', cofitok_run)
    dense_run = source.index('--run-dir "$DENSE_RUN"')
    dense_config = source.index('--config "$DENSE_CONFIG"', dense_run)

    assert cofitok_run < cofitok_config < source.index(
        "--expected-steps", cofitok_run
    )
    assert dense_run < dense_config < source.index("--expected-steps", dense_run)


def test_stability_50k_posteval_is_exclusive_before_writing_evidence() -> None:
    source = RUNBOOK.read_text(encoding="utf-8")

    lock = source.index('exec 8>"$POSTEVAL_LOCK"')
    acquisition = source.index("flock -n 8", lock)
    first_evidence_write = source.index(
        "scripts/validate_generation_stability_scaling_decision.py"
    )
    first_gpu_work = source.index("scripts/select_generation_sampling_batch.py")
    assert "command -v flock >/dev/null" in source
    assert 'POSTEVAL_LOCK="$OUTPUT_ROOT/posteval.lock"' in source
    assert lock < acquisition < first_evidence_write < first_gpu_work
    assert "exit 75" in source
