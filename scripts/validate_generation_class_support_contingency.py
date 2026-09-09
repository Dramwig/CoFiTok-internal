"""Independently replay a frozen class-support contingency result.

This validator is source-bound and non-authorizing.  It never loads a model or
classifier; it rehashes the frozen trees and prediction JSONL files, rebuilds
all CPU statistics, and optionally writes one adjacent immutable receipt.
"""

# Parsed artifact type mismatches are schema-value failures, not API misuse.
# ruff: noqa: TRY004

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from cofitok.generation_class_support_contingency import (
    METHODS,
    build_analysis,
    build_validation_receipt,
    physical_sample_tree_identity,
    stable_file_identity,
    validate_execution_authorization,
    validate_git_identity,
    validate_identity,
    validate_preparation,
    validate_result,
    validate_validation_receipt,
)


def _read_json(
    path: Path, *, expected_sha256: str | None, name: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    identity = stable_file_identity(path)
    if expected_sha256 is not None and identity["sha256"] != expected_sha256:
        raise ValueError(f"{name} SHA256 differs")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain an object")
    if stable_file_identity(path) != identity:
        raise RuntimeError(f"{name} changed while it was read")
    return value, identity


def _read_prediction_rows(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    identity = stable_file_identity(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(
                    f"prediction JSONL contains a blank line: {path}:{line_number}"
                )
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"prediction JSONL is malformed: {path}:{line_number}"
                ) from error
            if not isinstance(value, dict):
                raise ValueError(
                    f"prediction JSONL row is not an object: {path}:{line_number}"
                )
            rows.append(value)
    if stable_file_identity(path) != identity:
        raise RuntimeError(f"prediction JSONL changed while it was read: {path}")
    return rows, identity


def _exclusive_json(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite validation receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            payload, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False
        )
        + "\n"
    ).encode("ascii")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return stable_file_identity(path)


def _checkout_identity(project_root: Path, script: Path) -> dict[str, Any]:
    root = project_root.resolve(strict=True)
    status = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all"],
        text=True,
    )
    if status:
        raise ValueError("validator checkout must be completely clean")
    resolved_script = script.resolve(strict=True)
    relative = resolved_script.relative_to(root).as_posix()
    subprocess.run(
        ["git", "-C", str(root), "ls-files", "--error-unmatch", "--", relative],
        check=True,
        capture_output=True,
    )
    if (
        subprocess.check_output(["git", "-C", str(root), "show", f"HEAD:{relative}"])
        != resolved_script.read_bytes()
    ):
        raise ValueError("validator script differs from the exact HEAD blob")
    return validate_git_identity(
        {
            "revision": subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD^{commit}"], text=True
            ).strip(),
            "tree": subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD^{tree}"], text=True
            ).strip(),
            "branch": subprocess.check_output(
                ["git", "-C", str(root), "branch", "--show-current"], text=True
            ).strip(),
            "tracked_dirty": False,
        },
        "validator Git",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--preparation-sha256", required=True)
    parser.add_argument("--execution-authorization", type=Path, required=True)
    parser.add_argument("--execution-authorization-sha256", required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--result-sha256", required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    preparation, preparation_id = _read_json(
        args.preparation,
        expected_sha256=args.preparation_sha256,
        name="class-support preparation",
    )
    prepared = validate_preparation(preparation)
    authorization, authorization_id = _read_json(
        args.execution_authorization,
        expected_sha256=args.execution_authorization_sha256,
        name="class-support execution authorization",
    )
    authorized = validate_execution_authorization(
        authorization,
        preparation=prepared,
        preparation_identity=preparation_id,
    )
    classifier_identity = stable_file_identity(
        Path(prepared["source_evidence"]["fixed_classifier"]["path"])
    )
    if classifier_identity != prepared["source_evidence"]["fixed_classifier"]:
        raise ValueError("fixed classifier differs from the preparation")
    stage_approval, stage_approval_id = _read_json(
        Path(authorized["stage_approval"]["path"]),
        expected_sha256=authorized["stage_approval"]["sha256"],
        name="class-support stage approval",
    )
    if (
        stage_approval_id != authorized["stage_approval"]
        or stage_approval != authorized["stage_approval_record"]
    ):
        raise ValueError("execution authorization stage approval differs")
    result, result_id = _read_json(
        args.result,
        expected_sha256=args.result_sha256,
        name="class-support result",
    )
    validated = validate_result(
        result,
        preparation=prepared,
        preparation_identity=preparation_id,
    )
    if validated["execution_authorization"] != authorization_id:
        raise ValueError("result execution authorization binding differs")
    if validated["evaluator_git"] != authorized["evaluator_git"]:
        raise ValueError("result evaluator Git differs from the authorization")
    expected_result_path = (
        Path(prepared["output_contract"]["output_root"])
        / prepared["output_contract"]["result_file"]
    ).resolve()
    if args.result.resolve() != expected_result_path:
        raise ValueError("result path differs from the preparation")
    prediction_rows: dict[str, list[dict[str, Any]]] = {}
    prediction_ids: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        expected = validate_identity(
            validated["prediction_files"][method], f"{method} result prediction"
        )
        path = Path(expected["path"])
        rows, identity = _read_prediction_rows(path)
        if identity != expected:
            raise ValueError(f"{method} prediction identity differs from result")
        prediction_rows[method] = rows
        prediction_ids[method] = identity
    recall = {
        method: float(validated["existing_generation_recall"][method])
        for method in METHODS
    }
    replayed_analysis = build_analysis(
        prediction_rows,
        existing_recall=recall,
    )
    if replayed_analysis != validated["analysis"]:
        raise ValueError("result analysis differs from independent prediction replay")
    for method in METHODS:
        tree = prepared["source_evidence"]["frozen_methods"][method]["sample_tree"]
        observed_tree = physical_sample_tree_identity(
            tree["path"],
            expected_image_sha256=[
                row["image_sha256"] for row in prediction_rows[method]
            ],
        )
        if observed_tree != tree:
            raise ValueError(f"{method} frozen sample tree changed after evaluation")
    if stable_file_identity(args.preparation) != preparation_id:
        raise RuntimeError("preparation changed during validation")
    if stable_file_identity(args.execution_authorization) != authorization_id:
        raise RuntimeError("execution authorization changed during validation")
    if (
        stable_file_identity(Path(authorized["stage_approval"]["path"]))
        != stage_approval_id
    ):
        raise RuntimeError("stage approval changed during validation")
    if stable_file_identity(args.result) != result_id:
        raise RuntimeError("result changed during validation")
    if (
        stable_file_identity(
            Path(prepared["source_evidence"]["fixed_classifier"]["path"])
        )
        != classifier_identity
    ):
        raise RuntimeError("fixed classifier changed during validation")
    for method in METHODS:
        if (
            stable_file_identity(Path(prediction_ids[method]["path"]))
            != prediction_ids[method]
        ):
            raise RuntimeError(f"{method} predictions changed during validation")
    receipt_id = None
    if args.receipt is not None:
        expected_receipt_path = (
            Path(prepared["output_contract"]["output_root"])
            / prepared["output_contract"]["validation_file"]
        ).resolve()
        if args.receipt.resolve() != expected_receipt_path:
            raise ValueError("validation receipt is not adjacent to the result")
        validator_git = _checkout_identity(args.project_root, args.script)
        receipt = build_validation_receipt(
            result=validated,
            result_identity=result_id,
            validator_git=validator_git,
            replayed_analysis=replayed_analysis,
        )
        if args.receipt.exists():
            existing, receipt_id = _read_json(
                args.receipt,
                expected_sha256=None,
                name="class-support validation receipt",
            )
            validate_validation_receipt(
                existing,
                result=validated,
                result_identity=result_id,
            )
            if existing != receipt:
                raise ValueError("existing validation receipt differs from replay")
        else:
            receipt_id = _exclusive_json(args.receipt, receipt)
            persisted, _ = _read_json(
                args.receipt,
                expected_sha256=receipt_id["sha256"],
                name="class-support validation receipt",
            )
            validate_validation_receipt(
                persisted,
                result=validated,
                result_identity=result_id,
            )
    print(
        json.dumps(
            {
                "status": "pass",
                "scientific_status": validated["scientific_status"],
                "result": result_id,
                "validation_receipt": receipt_id,
                "physical_prediction_replay": True,
                "execution_authorization_replayed": True,
                "authorization_created": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
