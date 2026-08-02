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
    assert source.count("--sample-steps 250") == 5
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


def test_stability_full_posteval_binds_matched_ema_rollout_diagnostics() -> None:
    source = RUNBOOK.read_text(encoding="utf-8")

    assert source.count("scripts/evaluate_generation_rollout_stability.py") == 2
    assert source.count("--num-images 64") >= 2
    assert source.count("--seed 2029") == 2
    assert "scripts/build_generation_stability_qualification.py" in source
    assert '--rollout-stability-qualification "$EMA_ROLLOUT_QUALIFICATION"' in source
    assert '--input-file "$EMA_ROLLOUT_QUALIFICATION"' in source
    assert '--expected-evaluation-revision "$EXPECTED_TARGET_REVISION"' in source
    assert '--expected-evaluation-branch "$EXPECTED_TARGET_BRANCH"' in source


def test_stability_full_comparison_keeps_official_methods_contextual() -> None:
    source = RUNBOOK.read_text(encoding="utf-8")

    assert "scripts/build_large_scale_generation_comparison.py" in source
    assert "--official-related" in source
    assert 'TRAINING_CONTENTION="$OUTPUT_ROOT/pair_monitor.json"' in source
    assert '--training-contention "$TRAINING_CONTENTION"' in source
    assert '--input-file "$TRAINING_CONTENTION"' in source
    assert "--source-profile stability_full" in source
    assert "generation_stability_ema_teacher_full_matched_300k_after_gate.sh" not in source


def test_stability_full_posteval_audits_each_run_against_its_config() -> None:
    source = RUNBOOK.read_text(encoding="utf-8")

    assert (
        "COFITOK_CONFIG=configs/generation/"
        "imagenet256_stability_rgbtail3_rollout_x0_u2_ema_teacher_k8_300k.json"
    ) in source
    assert (
        "DENSE_CONFIG=configs/generation/"
        "imagenet256_stability_rollout_x0_u2_ema_teacher_dense_300k.json"
    ) in source
    cofitok_run = source.index('--run-dir "$COFITOK_RUN"')
    cofitok_config = source.index('--config "$COFITOK_CONFIG"', cofitok_run)
    dense_run = source.index('--run-dir "$DENSE_RUN"')
    dense_config = source.index('--config "$DENSE_CONFIG"', dense_run)

    assert cofitok_run < cofitok_config < source.index(
        "--expected-steps", cofitok_run
    )
    assert dense_run < dense_config < source.index("--expected-steps", dense_run)


def test_stability_full_posteval_receipts_high_cost_partial_stages() -> None:
    source = RUNBOOK.read_text(encoding="utf-8")

    assert source.count("scripts/run_generation_stage_once.py") == 10
    for receipt in (
        "cofitok_checkpoint_eval.json",
        "dense_checkpoint_eval.json",
        "cofitok_rollout_stability.json",
        "dense_rollout_stability.json",
        "ema_rollout_stability_qualification.json",
        "cofitok_generation_metrics.json",
        "dense_generation_metrics.json",
        "visual_audit.json",
        "final_gate.json",
        "comparison.json",
    ):
        assert receipt in source
    assert source.count("--input-file \"$DATASET_MANIFEST\"") == 6
    assert source.count('--input-tree "$DATA"') == 2
    assert '--input-tree "$COFITOK_RUN/samples_50k_ddim250_cfg15/prefix_8"' in source
    assert '--input-tree "$DENSE_RUN/samples_50k_ddim250_cfg15/prefix_1"' in source


def test_stability_full_posteval_is_exclusive_before_writing_evidence() -> None:
    source = RUNBOOK.read_text(encoding="utf-8")

    lock = source.index('exec 8>"$POSTEVAL_LOCK"')
    acquisition = source.index("flock -n 8", lock)
    first_evidence_write = source.index("scripts/validate_generation_gate_report.py")
    first_gpu_work = source.index("scripts/select_generation_sampling_batch.py")
    assert "command -v flock >/dev/null" in source
    assert 'POSTEVAL_LOCK="$OUTPUT_ROOT/posteval.lock"' in source
    assert lock < acquisition < first_evidence_write < first_gpu_work
    assert "exit 75" in source
