from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

import pytest

import scripts.infer_released_generation as released_cli
from scripts.infer_generation import build_parser


ROOT = Path(__file__).resolve().parents[1]


def test_released_entrypoint_has_standalone_fail_closed_cli() -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/infer_released_generation.py"),
            "--help",
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--completion-receipt" in result.stdout
    assert "--require-release-authorization" not in result.stdout
    assert "--require-completion-authorization" not in result.stdout


def test_generic_inference_parser_remains_explicitly_research_capable() -> None:
    args = build_parser().parse_args(
        [
            "--checkpoint",
            "training_checkpoint.pt",
            "--output-dir",
            "outputs",
            "--weights",
            "model",
        ]
    )

    assert args.weights == "model"
    assert args.completion_receipt == ""
    assert args.require_release_authorization is False
    assert args.require_completion_authorization is False


def test_released_parser_requires_receipt_and_enforces_ema_authorization() -> None:
    parser = build_parser(released=True)
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--checkpoint",
                "artifact.pt",
                "--output-dir",
                "outputs",
            ]
        )
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--checkpoint",
                "artifact.pt",
                "--output-dir",
                "outputs",
                "--completion-receipt",
                "release_receipt.json",
                "--weights",
                "model",
            ]
        )

    args = parser.parse_args(
        [
            "--checkpoint",
            "artifact.pt",
            "--output-dir",
            "outputs",
            "--completion-receipt",
            "release_receipt.json",
        ]
    )
    assert args.weights == "ema"
    assert args.require_release_authorization is True
    assert args.require_completion_authorization is True


def test_released_runner_forces_both_authorization_boundaries(monkeypatch) -> None:
    observed = {}

    def fake_run(args):
        observed.update(vars(args))
        return {"status": "completed", "output_count": 1}

    monkeypatch.setattr(released_cli, "run_inference", fake_run)
    report = released_cli.run_released_inference(
        argparse.Namespace(
            completion_receipt="release_receipt.json",
            weights="ema",
            require_release_authorization=False,
            require_completion_authorization=False,
        )
    )

    assert report["status"] == "completed"
    assert observed["require_release_authorization"] is True
    assert observed["require_completion_authorization"] is True
    assert observed["completion_receipt"] == "release_receipt.json"
    assert observed["weights"] == "ema"


@pytest.mark.parametrize(
    ("receipt", "weights", "message"),
    [
        ("", "ema", "requires a completion receipt"),
        ("release_receipt.json", "model", "EMA weights only"),
    ],
)
def test_released_runner_rejects_weakened_programmatic_requests(
    receipt,
    weights,
    message,
) -> None:
    with pytest.raises(ValueError, match=message):
        released_cli.run_released_inference(
            argparse.Namespace(
                completion_receipt=receipt,
                weights=weights,
            )
        )


def test_inference_documentation_uses_the_released_entrypoint() -> None:
    documentation = (ROOT / "docs/INFERENCE.md").read_text(encoding="utf-8")
    assert "python scripts/infer_released_generation.py" in documentation
    assert "do not use that permissive research entrypoint" in documentation
