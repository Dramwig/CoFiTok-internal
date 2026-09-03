from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

import scripts.capacity_confirmation_result_cli as result_cli
import scripts.validate_generation_capacity_confirmation_stage_authorization as stage_validator_cli
from cofitok.generation.capacity_screen import ARM_NAMES
from cofitok.reporting import file_sha256
from scripts.capacity_confirmation_arm_cli import arm_kwargs, parse_build_args


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def test_arm_cli_maps_exact_physical_sources(tmp_path: Path) -> None:
    args = parse_build_args(
        [
            "--arm",
            "base256_cofitok",
            "--launch-receipt",
            str(tmp_path / "launch.json"),
            "--expected-launch-receipt-sha256",
            "a" * 64,
            "--screen-arm-validation",
            str(tmp_path / "screen.json"),
            "--expected-screen-arm-validation-sha256",
            "b" * 64,
            "--sampling-report",
            str(tmp_path / "sampling.json"),
            "--metrics-report",
            str(tmp_path / "metrics.json"),
            "--class-fidelity-report",
            str(tmp_path / "class.json"),
            "--output",
            str(tmp_path / "arm.json"),
        ]
    )
    kwargs = arm_kwargs(args)
    assert kwargs["arm"] == "base256_cofitok"
    assert kwargs["expected_launch_receipt_sha256"] == "a" * 64
    assert kwargs["expected_screen_arm_validation_sha256"] == "b" * 64
    assert kwargs["sampling_report_path"] == tmp_path / "sampling.json"
    assert kwargs["metrics_report_path"] == tmp_path / "metrics.json"
    assert kwargs["class_fidelity_report_path"] == tmp_path / "class.json"


def test_result_cli_loads_all_four_bound_arm_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preparation = tmp_path / "preparation.json"
    launch = tmp_path / "launch.json"
    _write(preparation, {"kind": "preparation"})
    _write(launch, {"kind": "launch"})
    paths: dict[str, Path] = {}
    argv = [
        "--project-root",
        str(tmp_path),
        "--preparation",
        str(preparation),
        "--expected-preparation-sha256",
        file_sha256(preparation),
        "--launch-receipt",
        str(launch),
        "--expected-launch-receipt-sha256",
        file_sha256(launch),
    ]
    for arm in ARM_NAMES:
        path = tmp_path / f"{arm}.json"
        _write(path, {"arm": arm})
        paths[arm] = path
        option = arm.replace("_", "-")
        argv.extend(
            [
                f"--{option}-validation",
                str(path),
                f"--expected-{option}-validation-sha256",
                file_sha256(path),
            ]
        )
    argv.extend(["--output", str(tmp_path / "result.json")])
    monkeypatch.setattr(
        result_cli,
        "checkout_identity",
        lambda _: {
            "revision": "a" * 40,
            "tree": "b" * 40,
            "branch": "test",
            "tracked_dirty": False,
        },
    )
    args = result_cli.parse_build_args(argv)
    kwargs = result_cli.result_kwargs(args)
    assert set(kwargs["arm_validations"]) == set(ARM_NAMES)
    assert set(kwargs["arm_validation_identities"]) == set(ARM_NAMES)
    for arm in ARM_NAMES:
        assert kwargs["arm_validations"][arm] == {"arm": arm}
        assert kwargs["arm_validation_identities"][arm]["sha256"] == file_sha256(
            paths[arm]
        )


def test_stage_validation_cli_maps_validator_keyword_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "stage.json"
    _write(path, {})
    args = argparse.Namespace(
        stage_authorization=path,
        expected_stage_authorization_sha256="c" * 64,
    )
    captured: dict[str, object] = {}
    monkeypatch.setattr(stage_validator_cli, "parse_validate_stage_args", lambda: args)
    monkeypatch.setattr(
        stage_validator_cli,
        "reject_symlink_chain",
        lambda value, **_: value,
    )
    monkeypatch.setattr(stage_validator_cli, "file_sha256", lambda _: "c" * 64)
    monkeypatch.setattr(stage_validator_cli, "read_object", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        stage_validator_cli,
        "stage_kwargs",
        lambda _: {
            "preparation_identity": {"sha256": "d" * 64},
            "execution_checkout": {"revision": "e" * 40},
            "output_root": "/root/capacity/confirmation_10000",
        },
    )

    def _validate(_report: object, **kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(
        stage_validator_cli,
        "validate_capacity_confirmation_stage_authorization",
        _validate,
    )
    stage_validator_cli.main()
    assert captured == {
        "preparation_identity": {"sha256": "d" * 64},
        "execution_checkout": {"revision": "e" * 40},
        "expected_output_root": "/root/capacity/confirmation_10000",
    }
    assert json.loads(capsys.readouterr().out)["status"] == "pass"


def test_confirmation_cli_entrypoints_import() -> None:
    import scripts.build_generation_capacity_confirmation_arm  # noqa: F401
    import scripts.build_generation_capacity_confirmation_result  # noqa: F401
    import scripts.validate_generation_capacity_confirmation_arm  # noqa: F401
    import scripts.validate_generation_capacity_confirmation_result  # noqa: F401
