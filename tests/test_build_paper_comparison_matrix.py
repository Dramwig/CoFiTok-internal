import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "build_paper_comparison_matrix.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("build_paper_comparison_matrix", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_rows_marks_completed_partial_and_external_adapter(tmp_path: Path) -> None:
    module = load_script_module()
    matrix = {
        "datasets": [{"alias": "tiny_imagenet_200", "tier": "formal_64", "resolution": 64}],
        "methods": [
            {
                "alias": "cofitok_light",
                "display_name": "CoFiTok full",
                "type": "ours",
                "priority": "P0",
                "summary_variant": "light_denoise_path",
                "required_sections": ["train", "quality"],
            },
            {
                "alias": "same_backbone_dense",
                "display_name": "Dense",
                "type": "internal_baseline",
                "priority": "P0",
                "summary_variant": "epsilon_only",
                "required_sections": ["train", "quality"],
            },
            {
                "alias": "edm",
                "display_name": "EDM",
                "type": "external_baseline",
                "priority": "P0",
                "summary_variant": None,
                "required_sections": ["baseline_train", "baseline_eval"],
            },
        ],
    }
    summary = {
        "train": [
            {"dataset": "tiny_imagenet_200", "variant": "light_denoise_path"},
            {"dataset": "tiny_imagenet_200", "variant": "epsilon_only"},
        ],
        "quality": [{"dataset": "tiny_imagenet_200", "variant": "light_denoise_path"}],
    }
    baseline_registry = {
        "entries": [
            {"alias": "edm", "repo_type": "external", "clone_on_setup": True},
            {"alias": "same_backbone_dense", "repo_type": "internal", "clone_on_setup": False},
        ]
    }

    rows = module.build_rows(matrix, summary, baseline_registry, tmp_path)
    status_by_method = {row["method"]: row["status"] for row in rows}

    assert status_by_method == {
        "cofitok_light": "completed",
        "same_backbone_dense": "partial",
        "edm": "needs_adapter",
    }


def test_build_rows_uses_external_baseline_reports(tmp_path: Path) -> None:
    module = load_script_module()
    matrix = {
        "datasets": [{"alias": "ffhq_64", "tier": "formal_64", "resolution": 64}],
        "methods": [
            {
                "alias": "improved_diffusion",
                "display_name": "Improved DDPM",
                "type": "external_baseline",
                "priority": "P0",
                "summary_variant": None,
                "required_sections": ["baseline_train", "baseline_eval"],
            },
        ],
    }
    summary = {}
    baseline_registry = {
        "entries": [
            {
                "alias": "improved_diffusion",
                "repo_type": "external",
                "clone_on_setup": True,
                "adapter_status": "smoke_passed",
            },
        ]
    }
    report_dir = tmp_path / "improved_diffusion" / "run"
    report_dir.mkdir(parents=True)
    (report_dir / "baseline_train_report.json").write_text(
        json.dumps(
            {
                "report_type": "baseline_train",
                "baseline": "improved_diffusion",
                "dataset": "ffhq_64",
                "status": "completed",
            }
        ),
        encoding="utf-8",
    )
    (report_dir / "baseline_eval_report.json").write_text(
        json.dumps(
            {
                "report_type": "baseline_eval",
                "baseline": "improved_diffusion",
                "dataset": "ffhq_64",
                "status": "completed",
            }
        ),
        encoding="utf-8",
    )

    rows = module.build_rows(matrix, summary, baseline_registry, tmp_path)

    assert rows[0]["status"] == "completed"
    assert rows[0]["present_sections"] == "baseline_train,baseline_eval"


def test_build_rows_marks_feasibility_only_baseline_protocol_blocked() -> None:
    module = load_script_module()
    matrix = {
        "datasets": [{"alias": "ffhq_64", "tier": "formal_64", "resolution": 64}],
        "methods": [
            {
                "alias": "d_ar",
                "display_name": "D-AR",
                "type": "external_baseline",
                "priority": "P0",
                "summary_variant": None,
                "required_sections": ["baseline_train", "baseline_eval"],
            }
        ],
    }
    baseline_registry = {
        "entries": [
            {
                "alias": "d_ar",
                "repo_type": "external",
                "clone_on_setup": True,
                "adapter_status": "feasibility_only",
            }
        ]
    }

    rows = module.build_rows(matrix, {}, baseline_registry)

    assert rows[0]["status"] == "protocol_blocked"
    assert rows[0]["baseline_repo_status"] == "feasibility_only"

    baseline_registry["entries"][0]["adapter_status"] = "official_hf_safetensors_50k_running"
    rows = module.build_rows(matrix, {}, baseline_registry)
    assert rows[0]["status"] == "protocol_blocked"

    baseline_registry["entries"][0]["adapter_status"] = "official_pth_ema_50k_running"
    rows = module.build_rows(matrix, {}, baseline_registry)
    assert rows[0]["status"] == "protocol_blocked"


def test_build_rows_keeps_eval_only_tokenizer_evidence_separate(tmp_path: Path) -> None:
    module = load_script_module()
    matrix = {
        "datasets": [{"alias": "cifar10", "tier": "smoke", "resolution": 32}],
        "methods": [
            {
                "alias": "titok_1d_tokenizer",
                "display_name": "TiTok",
                "type": "external_baseline",
                "priority": "P1",
                "summary_variant": None,
                "required_sections": ["baseline_train", "baseline_eval"],
            },
        ],
    }
    baseline_registry = {
        "entries": [
            {
                "alias": "titok_1d_tokenizer",
                "repo_type": "external",
                "clone_on_setup": True,
                "adapter_status": "smoke_passed_eval_only",
            }
        ]
    }
    report_dir = tmp_path / "titok_1d_tokenizer" / "smoke"
    report_dir.mkdir(parents=True)
    (report_dir / "baseline_train_report.json").write_text(
        json.dumps(
            {
                "report_type": "baseline_train",
                "baseline": "titok_1d_tokenizer",
                "dataset": "cifar10",
                "status": "completed",
                "parameters": {"eval_only": True},
            }
        ),
        encoding="utf-8",
    )
    (report_dir / "baseline_eval_report.json").write_text(
        json.dumps(
            {
                "report_type": "baseline_eval",
                "baseline": "titok_1d_tokenizer",
                "dataset": "cifar10",
                "status": "completed",
                "parameters": {"sampler": "tokenizer_reconstruction"},
            }
        ),
        encoding="utf-8",
    )

    rows = module.build_rows(matrix, {}, baseline_registry, tmp_path)

    assert rows[0]["status"] == "completed_eval_only"
    assert rows[0]["baseline_evidence_modes"] == "eval_only,eval_only_tokenizer_reconstruction"
