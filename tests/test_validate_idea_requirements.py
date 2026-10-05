import json
from pathlib import Path

from scripts.validate_idea_requirements import (
    check_restricted_synthesis_code,
    render_markdown,
    validate_requirements,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_source_tree(root: Path) -> None:
    _write(
        root / "src/cofitok/models/cofitok.py",
        """
tokens: list[torch.Tensor]
components: list[torch.Tensor]
prefix_epsilons: list[torch.Tensor]
components = self.synthesis(tokens)
running = running + component
prefix_epsilons.append(running)
epsilon=prefix_epsilons[-1]
""",
    )
    _write(
        root / "src/cofitok/models/synthesis.py",
        """
Maps one denoising token to one dense noise component
def forward(self, token: torch.Tensor)
bias=False
self.local(self.proj(token))
zero_components_like
deep synthesis
""",
    )
    _write(
        root / "src/cofitok/training/losses.py",
        """
monotonic_loss
zero_token_loss
component_decorrelation
denoise_path_prefix
denoise_path_component
""",
    )
    _write(
        root / "src/cofitok/diagnostics.py",
        """
random_tokens
shuffle_tokens_across_batch
zero_components
shuffled_component_relative_mse
""",
    )


def _summary_payload() -> dict:
    train = []
    for dataset in ["cifar10", "tiny_imagenet_200", "imagenet_1k_64x64_hf"]:
        train.append(
            {
                "dataset": dataset,
                "variant": "light_denoise_path",
                "token_count": 8,
                "steps": 20000,
                "seed": 139,
                "synthesis_mode": "restricted",
                "path_auc": 0.05,
                "final_clean_mse": 0.1,
                "effective_tokens": 6.0,
                "zero_token_ratio": 0.0,
                "random_token_ratio": 0.2,
                "shuffled_final_ratio": 2.0,
            }
        )
    for dataset in ["tiny_imagenet_200", "imagenet_1k_64x64_hf"]:
        train.append(
            {
                "dataset": dataset,
                "variant": "epsilon_only",
                "token_count": 8,
                "steps": 20000,
                "seed": 139,
                "synthesis_mode": "restricted",
                "path_auc": 1.0,
                "final_clean_mse": 0.1,
                "effective_tokens": 5.0,
                "zero_token_ratio": 0.0,
                "random_token_ratio": 0.2,
                "shuffled_final_ratio": 2.0,
            }
        )
    for seed in [103, 139]:
        train.append(
            {
                "dataset": "imagenet_1k_64x64_hf",
                "variant": "deep_synthesis_ablation",
                "token_count": 8,
                "steps": 5000,
                "seed": seed,
                "synthesis_mode": "deep",
                "zero_token_ratio": 0.05,
            }
        )

    order_eval = []
    for token_count in [4, 8, 16]:
        for seed in [103, 139]:
            for order, auc in [("ordered", 0.1), ("random", 0.4), ("reverse", 0.6)]:
                order_eval.append(
                    {
                        "dataset": "imagenet_1k_64x64_hf",
                        "variant": "light_denoise_path",
                        "token_count": token_count,
                        "seed": seed,
                        "component_order": order,
                        "path_auc": auc,
                    }
                )
    for seed in [103, 139]:
        for order, auc in [("ordered", 0.1), ("random", 0.4), ("reverse", 0.6)]:
            order_eval.append(
                {
                    "dataset": "imagenet_1k_64x64_hf",
                    "variant": "simultaneous_predictor",
                    "token_count": 8,
                    "seed": seed,
                    "component_order": order,
                    "path_auc": auc,
                }
            )

    quality = [
        {
            "dataset": "imagenet_1k_64x64_hf",
            "variant": "light_denoise_path",
            "token_count": 8,
            "image_count": 256,
            "lpips_available": True,
            "inception_available": True,
        }
    ]
    sampling = [{"dataset": dataset} for dataset in ["cifar10", "tiny_imagenet_200", "imagenet_1k_64x64_hf"]]
    generated_quality = []
    for dataset in ["tiny_imagenet_200", "imagenet_1k_64x64_hf"]:
        for variant in ["epsilon_only", "light_denoise_path"]:
            generated_quality.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "token_count": 8,
                    "steps": 20000,
                    "sample_image_count": 1024,
                    "real_image_count": 4096,
                }
            )
    return {
        "train": train,
        "order_eval": order_eval,
        "quality": quality,
        "sampling": sampling,
        "generated_quality": generated_quality,
    }


def _queue_manifest() -> dict:
    return {
        "run_count": 10,
        "quality_count": 10,
        "order_eval_count": 24,
        "sampling_count": 6,
        "generated_quality_count": 6,
        "sample_count": 2048,
        "sample_steps": 50,
        "real_count": 8192,
    }


def test_validate_requirements_accepts_mvp_and_marks_publication_pending(tmp_path: Path) -> None:
    _write_source_tree(tmp_path)
    summary = tmp_path / "summary.json"
    queue = tmp_path / "queue.json"
    _write_json(summary, _summary_payload())
    _write_json(queue, _queue_manifest())

    result = validate_requirements(tmp_path, summary, queue)

    assert result["status"] == "mvp_ok_publication_pending"
    assert result["missing_count"] == 0
    assert result["pending_count"] == 1
    markdown = render_markdown(result)
    assert "CoFiTok Idea Requirements Matrix" in markdown
    assert "publication_readiness" in markdown


def test_validate_requirements_reports_missing_factorization_code(tmp_path: Path) -> None:
    _write_source_tree(tmp_path)
    _write(tmp_path / "src/cofitok/models/cofitok.py", "tokens: list[torch.Tensor]\n")
    summary = tmp_path / "summary.json"
    queue = tmp_path / "queue.json"
    _write_json(summary, _summary_payload())
    _write_json(queue, _queue_manifest())

    result = validate_requirements(tmp_path, summary, queue)

    assert result["status"] == "missing"
    factorization = next(check for check in result["checks"] if check["name"] == "dense_noise_factorization_code")
    assert factorization["status"] == "missing"


def test_restricted_synthesis_accepts_named_projection_intermediate(tmp_path: Path) -> None:
    path = tmp_path / "src/cofitok/models/synthesis.py"
    _write(
        path,
        """
Maps one denoising token to one dense noise component
def forward(self, token: torch.Tensor)
bias=False
projected = self.proj(token)
return self.local(projected)
zero_components_like
deep synthesis
""",
    )

    result = check_restricted_synthesis_code(tmp_path)

    assert result.status == "ok"
    assert result.evidence["token_projection_local_pattern"] == "named_projection_intermediate"
