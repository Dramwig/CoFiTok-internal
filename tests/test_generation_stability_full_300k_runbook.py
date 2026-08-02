from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNBOOK = (
    ROOT
    / "artifacts/runbooks/"
    "generation_stability_ema_teacher_full_matched_300k_after_gate.sh"
)
READINESS_RUNBOOK = (
    ROOT
    / "artifacts/runbooks/"
    "generation_stability_ema_teacher_full_readiness_after_gate.sh"
)


def test_stability_full_runbook_is_gate_and_identity_bound() -> None:
    source = RUNBOOK.read_text(encoding="utf-8")

    gate_hash = source.index('sha256sum "$GATE"')
    gate_validation = source.index("scripts/validate_generation_gate_report.py")
    readiness_validation = source.index("scripts/validate_generation_full_readiness_bridge.py")
    storage_validation = source.index("scripts/check_generation_storage_capacity.py")
    training = source.index("start_monitor")
    assert (
        gate_hash
        < gate_validation
        < readiness_validation
        < storage_validation
        < training
    )
    assert "EXPECTED_SCALING_GATE_SHA256=${" in source
    assert "EXPECTED_STABILITY_SUPPLEMENTAL_SHA256=${" in source
    assert "EXPECTED_DEPLOYMENT_RECEIPT_SHA256=${" in source
    assert "EXPECTED_READINESS_SHA256=${" in source
    assert "EXPECTED_READINESS_BRIDGE_SHA256=${" in source
    assert "EXPECTED_TARGET_REVISION=${" in source
    assert "EXPECTED_TARGET_BRANCH=${" in source
    assert '--expected-checkpoint-revision "$EXPECTED_TARGET_REVISION"' in source
    assert "--expected-recipe-stage stability_full" in source
    assert "scripts/select_generation_training_runtime.py" not in source
    assert "--deployment-receipt" in source
    assert '--stability-supplemental "$STABILITY_SUPPLEMENTAL"' in source
    assert "--expected-stability-supplemental-sha256" in source
    assert '--bridge "$READINESS_BRIDGE"' in source
    assert "--source-deployment-receipt" in source
    assert "--target-deployment-receipt" in source
    assert "--expected-source-revision" in source
    assert "--expected-source-branch" in source
    assert "--expected-target-revision" in source
    assert "--expected-target-branch" in source
    assert "storage_capacity_launch.json" in source


def test_stability_full_readiness_runbook_builds_evidence_without_training() -> None:
    source = READINESS_RUNBOOK.read_text(encoding="utf-8")

    gpu_idle = source.index("nvidia-smi --query-compute-apps=pid")
    gate = source.index("scripts/validate_generation_gate_report.py")
    config = source.index("scripts/validate_generation_configs.py")
    storage = source.index("scripts/check_generation_storage_capacity.py")
    selector = source.index("scripts/select_generation_training_runtime.py")
    builder = source.index("scripts/build_generation_full_readiness.py")
    assert gpu_idle < gate < config < storage < selector < builder
    assert 'find "$run_dir" -mindepth 1 -print -quit' in source
    assert "--stage stability_full" in source
    assert "validate_generation_large_capacity_deployment.py" in source
    assert "--expected-deployment-receipt-sha256" in source
    assert "--candidates 1x64,2x32,4x16,8x8,16x4" in source
    assert "--baseline-candidate 1x64" in source
    assert "--checkpoint-size-multiplier 4.0" in source
    assert "scripts/train_generation.py" not in source
    assert "start_monitor" not in source
    assert (
        "deployment/large_capacity/deployments/"
        "$EXPECTED_TARGET_REVISION/deployment_receipt.json"
    ) in source
    assert "deployment/large_capacity/deployment_receipt.json" not in source


def test_stability_full_runbook_uses_new_paths_and_configs() -> None:
    source = RUNBOOK.read_text(encoding="utf-8")

    assert "stability_full_300k_ema_teacher" in source
    assert (
        "imagenet256_stability_rgbtail3_rollout_x0_u2_"
        "ema_teacher_k8_300k.json"
    ) in source
    assert (
        "imagenet256_stability_rollout_x0_u2_"
        "ema_teacher_dense_300k.json"
    ) in source
    assert "imagenet256_full_cofitok_k8_300k" not in source
    assert "imagenet256_full_dense_300k" not in source


def test_stability_full_runbook_protects_and_evaluates_all_milestones() -> None:
    source = RUNBOOK.read_text(encoding="utf-8")

    assert "for milestone in 50000 100000 200000 300000" in source
    assert "--required-checkpoint-steps 50000,100000,200000,300000" in source
    assert "generation_full_milestone_eval.sh" in source
    assert "--checkpoint-integrity-policy required" in source
    assert "--integrity-policy required" in source
    assert "--authorization-gate \"$GATE\"" in source


def test_stability_full_runbook_does_not_self_authorize_or_postevaluate() -> None:
    source = RUNBOOK.read_text(encoding="utf-8")

    assert "generation_stability_ema_teacher_50k_posteval_after_training.sh" not in source
    assert "generation_full_posteval_50k.sh" not in source
    assert "formal_300k_allowed" not in source


def test_shared_milestone_runner_accepts_isolated_project_identity() -> None:
    source = (
        ROOT / "artifacts/runbooks/generation_full_milestone_eval.sh"
    ).read_text(encoding="utf-8")

    assert "PROJECT=${PROJECT:-" in source
    assert "PYTHON=${PYTHON:-python}" in source
    assert '"$PYTHON" scripts/preflight_generation_sampling.py' in source
    assert '"$PYTHON" scripts/evaluate_generation_checkpoint.py' in source
    assert "--weights ema --precision bf16 --resume" in source
    assert '"$PYTHON" scripts/evaluate_generation_metrics.py' in source
    assert "--min-samples 2048 --skip-prc --resume" in source


def test_stability_full_runbook_audits_each_run_against_its_config() -> None:
    source = RUNBOOK.read_text(encoding="utf-8")

    cofitok_run = source.index('--run-dir "$COFITOK_RUN"')
    cofitok_config = source.index('--config "$COFITOK_CONFIG"', cofitok_run)
    dense_run = source.index('--run-dir "$DENSE_RUN"')
    dense_config = source.index('--config "$DENSE_CONFIG"', dense_run)

    assert cofitok_run < cofitok_config < source.index(
        "--expected-steps", cofitok_run
    )
    assert dense_run < dense_config < source.index("--expected-steps", dense_run)


def test_stability_full_runbook_uses_large_capacity_runtime_and_storage_budget() -> None:
    source = RUNBOOK.read_text(encoding="utf-8")
    readiness = READINESS_RUNBOOK.read_text(encoding="utf-8")

    assert "--candidates 1x64,2x32,4x16,8x8,16x4" in readiness
    assert "--baseline-candidate 1x64" in readiness
    assert "--checkpoint-size-multiplier 4.0" in source


def test_stability_full_runbook_uses_immutable_launch_receipt_for_resume() -> None:
    source = RUNBOOK.read_text(encoding="utf-8")

    receipt = source.index("scripts/build_generation_full_launch_receipt.py")
    monitor = source.index("start_monitor")
    assert receipt < monitor
    supplemental_hash = source.index('sha256sum "$STABILITY_SUPPLEMENTAL"')
    assert supplemental_hash < receipt
    assert "EXPECTED_FULL_LAUNCH_RECEIPT_SHA256=${" in source
    assert "scripts/validate_generation_full_launch_receipt.py" in source
    assert '[[ -f "$FULL_LAUNCH_RECEIPT" ]]' in source
    assert "refusing unreceipted stability full training state" in source
    assert "storage_capacity_current.json" in source
    assert 'rm -f -- "$LAUNCH_STORAGE_CAPACITY"' in source
    assert "--allow-later-formal-repository" not in source
    assert '--readiness-bridge "$READINESS_BRIDGE"' in source
    assert source.count('storage_preflight "$LAUNCH_STORAGE_CAPACITY"') == 1
    assert source.count('storage_preflight "$CURRENT_STORAGE_CAPACITY"') == 1
    assert (
        "deployment/large_capacity/deployments/"
        "$EXPECTED_TARGET_REVISION/deployment_receipt.json"
    ) in source
    assert "deployment/large_capacity/deployment_receipt.json" not in source
