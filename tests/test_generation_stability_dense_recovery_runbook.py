from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNBOOK = (
    ROOT
    / "artifacts/runbooks"
    / "generation_stability_ema_teacher_dense_recovery_after_transition_failure.sh"
)


def test_dense_recovery_keeps_controller_and_training_identities_separate() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")

    assert "EXPECTED_CONTROL_REVISION" in text
    assert "EXPECTED_CONTROL_BRANCH" in text
    assert "EXPECTED_TRAINING_REVISION" in text
    assert "EXPECTED_TRAINING_BRANCH" in text
    assert "2c2c1f5166b73d4f28df93b276901671ac1a7836" in text
    assert "scale/generation-stability-50k-preflight" in text
    assert 'git -C "$CONTROL_PROJECT"' in text
    assert 'git -C "$TRAINING_PROJECT"' in text
    assert 'PYTHONPATH="$CONTROL_PROJECT/src"' in text
    assert 'PYTHONPATH="$TRAINING_PROJECT/src"' in text


def test_dense_recovery_revalidates_completed_cofitok_and_frozen_runtime() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")

    assert "validate_generation_training_completion.py" in text
    assert '--expected-branch "$EXPECTED_TRAINING_BRANCH"' in text
    assert "verify_training_checkpoint" in text
    assert "ec7b9a0981f1d45420a9a86cdb80339d6d87b87fa77891c234db3d1b84376c2a" in text
    assert "f7befbcdbc6644fc066a71b85cb1df5cae5b922c526f09fe0000ae252b324663" in text
    assert "frozen runtime selection does not preserve effective batch 64" in text
    assert "validate_generation_configs.py" in text
    assert "check_generation_storage_capacity.py" in text


def test_dense_recovery_launches_only_dense_and_refuses_resource_conflicts() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")

    training_call = (
        '"$PYTHON" scripts/train_generation.py \\\n'
        '        --config "$DENSE_CONFIG"'
    )
    assert text.count("scripts/train_generation.py") == 1
    assert training_call in text
    assert '--config "$COFITOK_CONFIG" \\\n        --output-dir "$COFITOK_RUN"' not in text
    assert "build_generation_stability_50k_summary.py" not in text
    assert 'PAIR_SUMMARY="$REPORT_ROOT/pair_summary.json"' not in text
    assert "refusing dense recovery while another stability 50K queue process exists" in text
    assert "refusing dense recovery while the GPU is busy" in text
    assert "RECOVERY_EXECUTION_ALLOWED" in text
    assert 'write_status prepared "dense recovery preflight passed; no process was launched"' in text
    assert text.index("check_generation_storage_capacity.py") < text.index("start_monitor\n")
    assert '"full_training_launch_allowed": False' in text
    assert "300000" not in text


def test_dense_recovery_holds_a_nonblocking_single_controller_lock() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")

    assert 'RECOVERY_LOCK="$OUTPUT_ROOT/dense_recovery.lock"' in text
    assert "command -v flock >/dev/null" in text
    assert 'exec 6>"$RECOVERY_LOCK"' in text
    assert "flock -n 6" in text
    assert "refusing concurrent dense recovery controller" in text
    assert text.index("flock -n 6") < text.index(
        'write_status running "validating completed CoFiTok and frozen matched runtime"'
    )
    assert text.index("flock -n 6") < text.index("start_monitor\n")
