from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNBOOK = (
    ROOT
    / "artifacts/runbooks/"
    "generation_stability_ema_teacher_export_inference_artifacts.sh"
)


def test_stability_export_requires_exact_gate_and_checkout_identity() -> None:
    source = RUNBOOK.read_text(encoding="utf-8")

    gate_hash = source.index('sha256sum "$FINAL_GATE"')
    gate_validation = source.index("scripts/validate_generation_gate_report.py")
    export = source.index("scripts/export_generation_inference_artifact.py")
    assert gate_hash < gate_validation < export
    assert "EXPECTED_FINAL_GATE_SHA256=${" in source
    assert "EXPECTED_TARGET_REVISION=${" in source
    assert "EXPECTED_TARGET_BRANCH=${" in source
    assert "--stage full" in source


def test_stability_export_verifies_and_smoke_tests_both_methods() -> None:
    source = RUNBOOK.read_text(encoding="utf-8")

    assert source.count("export_generation_inference_artifact.py") == 2
    assert source.count("preflight_generation_sampling.py") == 2
    assert source.count("infer_generation.py") == 2
    assert source.count("--require-release-authorization") == 4
    assert "cofitok_k8_ema_inference.pt" in source
    assert "dense_identity_ema_inference.pt" in source
    assert "--prefix-budgets 1,8" in source
    assert "--sample-steps 10" in source


def test_stability_export_uses_new_paths_and_cannot_train() -> None:
    source = RUNBOOK.read_text(encoding="utf-8")

    assert "stability_full_300k_ema_teacher" in source
    assert "imagenet256_full_cofitok_k8_300k" not in source
    assert "scripts/train_generation.py" not in source
    assert "generation_stability_ema_teacher_full_matched_300k_after_gate.sh" not in source


def test_inference_documentation_uses_release_authorized_stability_paths() -> None:
    documentation = (ROOT / "docs/INFERENCE.md").read_text(encoding="utf-8")
    runbook = RUNBOOK.read_text(encoding="utf-8")

    assert (
        "stability_full_300k_ema_teacher/"
        "cofitok_rgbtail3_rollout_x0_u2_ema_teacher/"
        "checkpoint_step_00300000.pt"
    ) in documentation
    assert (
        "exports/stability_full_300k_ema_teacher/"
        "cofitok_k8_ema_inference.pt"
    ) in documentation
    assert "dense_identity_ema_inference.pt" in documentation
    assert "--require-release-authorization" in documentation
    assert "imagenet256_full_cofitok_k8_300k" not in documentation
    assert "exports/imagenet256_full_300k" not in documentation


def test_stability_export_receipts_every_partial_success_boundary() -> None:
    source = RUNBOOK.read_text(encoding="utf-8")

    assert source.count("scripts/run_generation_stage_once.py") == 6
    for receipt in (
        "cofitok_export.json",
        "dense_export.json",
        "cofitok_export_preflight.json",
        "dense_export_preflight.json",
        "cofitok_export_smoke.json",
        "dense_export_smoke.json",
    ):
        assert receipt in source
    assert source.count("--output-tree") == 2
    assert "--overwrite" not in source
