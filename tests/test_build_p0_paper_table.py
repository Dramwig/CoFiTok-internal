import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "build_p0_paper_table.py"
SPEC = importlib.util.spec_from_file_location("build_p0_paper_table", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_dense_and_endpoint_only_controls_are_separate() -> None:
    aliases = {row[0]: row[2] for row in MODULE.INTERNAL_METHODS}

    assert aliases["same_backbone_dense"] == "dense_monolithic"
    assert aliases["endpoint_only_factorized"] == "epsilon_only"


def test_dense_monolithic_selects_direct_head_token_count() -> None:
    rows = [
        {
            "dataset": "tiny_imagenet_200",
            "variant": "dense_monolithic",
            "steps": 5_000,
            "token_count": 1,
            "config_name": "train_tiny_densehead_5k_cuda",
            "report_dir": "dense",
        },
        {
            "dataset": "tiny_imagenet_200",
            "variant": "dense_monolithic",
            "steps": 5_000,
            "token_count": 8,
            "config_name": "train_tiny_densehead_wrong_5k_cuda",
            "report_dir": "wrong",
        },
    ]

    selected = MODULE.select_internal(
        rows,
        "tiny_imagenet_200",
        "dense_monolithic",
        5_000,
        8,
    )

    assert selected["report_dir"] == "dense"


def test_quality_selection_rejects_seeded_and_wrong_count_rows() -> None:
    rows = [
        {
            "dataset": "cifar10",
            "variant": "light_denoise_path",
            "steps": 3_000,
            "token_count": 8,
            "config_name": "train_cifar10_light_3k_seed2_cuda",
            "image_count": 256,
            "fixed_timestep": 500,
            "prefix_budgets": list(range(1, 9)),
            "denoise_path_mse_auc": 0.2,
            "component_order": "ordered",
            "report_dir": "seeded",
        },
        {
            "dataset": "cifar10",
            "variant": "light_denoise_path",
            "steps": 3_000,
            "token_count": 8,
            "config_name": "train_cifar10_light_3k_cuda",
            "image_count": 32,
            "fixed_timestep": 500,
            "prefix_budgets": list(range(1, 9)),
            "denoise_path_mse_auc": 0.2,
            "component_order": "ordered",
            "report_dir": "smoke",
        },
        {
            "dataset": "cifar10",
            "variant": "light_denoise_path",
            "steps": 3_000,
            "token_count": 8,
            "config_name": "train_cifar10_light_3k_cuda",
            "image_count": 256,
            "fixed_timestep": 500,
            "prefix_budgets": list(range(1, 9)),
            "denoise_path_mse_auc": 0.2,
            "component_order": "ordered",
            "report_dir": "canonical",
        },
    ]

    selected = MODULE.select_internal(
        rows,
        "cifar10",
        "light_denoise_path",
        3_000,
        8,
        section="quality",
        quality_count=256,
    )

    assert selected["report_dir"] == "canonical"


def test_generation_selection_requires_locked_ddim50_counts() -> None:
    rows = [
        {
            "dataset": "tiny_imagenet_200",
            "variant": "light_denoise_path",
            "steps": 5_000,
            "token_count": 8,
            "config_name": "train_tiny_light_5k_cuda",
            "sample_image_count": 64,
            "real_image_count": 256,
            "sample_steps": 20,
            "prefix_budget": 8,
            "report_dir": "smoke",
        },
        {
            "dataset": "tiny_imagenet_200",
            "variant": "light_denoise_path",
            "steps": 5_000,
            "token_count": 8,
            "config_name": "train_tiny_light_5k_cuda",
            "sample_image_count": 1024,
            "real_image_count": 4096,
            "sample_steps": 50,
            "prefix_budget": 8,
            "report_dir": "formal",
        },
    ]

    selected = MODULE.select_internal(
        rows,
        "tiny_imagenet_200",
        "light_denoise_path",
        5_000,
        8,
        section="generated_quality",
        sample_count=1024,
        real_count=4096,
    )

    assert selected["report_dir"] == "formal"


def test_external_selection_rejects_eval_only_or_incomplete_rows() -> None:
    common = {
        "dataset": "cifar10",
        "baseline": "edm",
        "train_steps": 3_000,
        "sample_image_count": 1_024,
        "real_image_count": 4_096,
        "sampler_nfe": 79,
    }
    rows = [
        {**common, "status": "completed", "eval_only": True, "run_name": "eval_only"},
        {**common, "status": "running", "eval_only": False, "run_name": "running"},
        {**common, "status": "completed", "eval_only": False, "run_name": "formal"},
    ]

    selected = MODULE.select_baseline(
        rows,
        "cifar10",
        "edm",
        steps=3_000,
        sample_count=1_024,
        real_count=4_096,
    )

    assert selected["run_name"] == "formal"


def test_quality_selection_rejects_permuted_component_order() -> None:
    common = {
        "dataset": "cifar10",
        "variant": "light_denoise_path",
        "steps": 3_000,
        "token_count": 8,
        "config_name": "train_cifar10_light_3k_cuda",
        "image_count": 512,
        "fixed_timestep": 500,
        "prefix_budgets": list(range(1, 9)),
        "denoise_path_mse_auc": 0.2,
    }
    rows = [
        {**common, "component_order": "reverse", "report_dir": "z_reverse"},
        {**common, "component_order": "ordered", "report_dir": "a_ordered"},
    ]

    selected = MODULE.select_internal(
        rows,
        "cifar10",
        "light_denoise_path",
        3_000,
        8,
        section="quality",
        quality_count=512,
    )

    assert selected["report_dir"] == "a_ordered"


def test_factorization_contract_rejects_parameter_or_batch_mismatch(monkeypatch) -> None:
    monkeypatch.setattr(MODULE, "DATASETS", [{"dataset": "demo"}])
    rows = [
        {
            "dataset": "demo",
            "method": "cofitok_light",
            "parameter_count": 100_000,
            "train_batch_size": 8,
        },
        {
            "dataset": "demo",
            "method": "same_backbone_dense",
            "parameter_count": 80_000,
            "train_batch_size": 4,
        },
        {
            "dataset": "demo",
            "method": "endpoint_only_factorized",
            "parameter_count": 99_000,
            "train_batch_size": 8,
        },
    ]

    violations = MODULE.factorization_contract_violations(rows)

    assert any("endpoint-only params" in value for value in violations)
    assert any("exceeds 2%" in value for value in violations)
    assert any("train batches differ" in value for value in violations)
