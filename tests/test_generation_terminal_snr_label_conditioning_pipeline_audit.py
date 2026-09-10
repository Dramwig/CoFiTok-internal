from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "scripts"
    / "build_generation_terminal_snr_label_conditioning_pipeline_audit.py"
)
SPEC = importlib.util.spec_from_file_location("label_pipeline_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def _mapping(count: int = 3) -> dict[str, object]:
    return {
        "label_to_wnid": {
            str(index): f"n{index + 1:08d}" for index in range(count)
        }
    }


def _identity(path: Path, character: str = "a") -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": 123,
        "sha256": character * 64,
    }


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _manifest_rows() -> list[dict[str, object]]:
    rows = []
    for split in ("train", "val"):
        for label, wnid in enumerate(("n00000001", "n00000002")):
            rows.append(
                {
                    "split": split,
                    "label": label,
                    "wnid": wnid,
                    "path": f"extracted/{split}/{wnid}/{split}_{label}.jpg",
                    "width": 256,
                    "height": 256,
                }
            )
    return rows


def _audit_fixture(tmp_path: Path) -> dict[str, object]:
    builder_git = {
        "branch": "analysis/test",
        "revision": "1" * 40,
        "tracked_dirty": False,
        "tree": "2" * 40,
    }
    next_stage = {
        "route": "hold",
        "frozen_confirmation_preparation_allowed": False,
        "frozen_confirmation_launch_allowed": False,
        "full_training_launch_allowed": False,
        "full_300k_launch_allowed": False,
        "promotion_allowed": False,
        "export_allowed": False,
        "release_allowed": False,
        "paper_integration_allowed": False,
    }
    return {
        "schema_version": audit.SCHEMA,
        "role": audit.ROLE,
        "status": "completed",
        "operational_status": "pass",
        "scientific_status": "hold",
        "terminal_status": "hold",
        "decision": (
            "class_index_mapping_mismatch_ruled_out_support_collapse_unexplained"
        ),
        "generation_advantage_proven": False,
        "builder_git": builder_git,
        "builder_script": _identity(tmp_path / "builder.py"),
        "execution_git": {
            "branch": audit.EXECUTION_BRANCH,
            "revision": audit.EXECUTION_REVISION,
            "tracked_dirty": False,
            "tree": audit.EXECUTION_TREE,
        },
        "source_evidence": {"source": _identity(tmp_path / "source.json", "b")},
        "dataset_label_pipeline": {},
        "execution_code_pipeline": {},
        "real_classifier_calibration": {},
        "arm_replays": {arm: {} for arm in audit.ARM_SPECS},
        "screen_judgment": {
            "recall_by_arm": {arm: 0.0 for arm in audit.ARM_SPECS}
        },
        "causal_context": {},
        "scientific_judgment": {
            "class_index_mapping_mismatch": False,
            "mapping_can_explain_support_collapse": False,
            "shared_support_collapse_consistent": True,
            "common_cause_proven": False,
            "interpretation": "test",
        },
        "physical_replay": {},
        "next_stage": next_stage,
        "authorization_boundary": copy.deepcopy(audit.AUTHORIZATION_BOUNDARY),
    }


def test_normalize_mapping_requires_contiguous_lexicographic_indices() -> None:
    assert audit.normalize_label_mapping(_mapping(), expected_classes=3) == [
        "n00000001",
        "n00000002",
        "n00000003",
    ]
    missing = _mapping()
    del missing["label_to_wnid"]["1"]
    with pytest.raises(ValueError, match="not contiguous"):
        audit.normalize_label_mapping(missing, expected_classes=3)
    reordered = _mapping()
    reordered["label_to_wnid"]["0"], reordered["label_to_wnid"]["1"] = (
        reordered["label_to_wnid"]["1"],
        reordered["label_to_wnid"]["0"],
    )
    with pytest.raises(ValueError, match="not lexicographic"):
        audit.normalize_label_mapping(reordered, expected_classes=3)


def test_categories_and_synsets_must_match_loader_order() -> None:
    mapping = audit.normalize_label_mapping(_mapping(), expected_classes=3)
    categories = b"first,n00000001\nsecond, with comma,n00000002\nthird,n00000003\n"
    result = audit.audit_categories(categories, mapping)
    assert result["class_count"] == 3
    assert result["first_wnid"] == "n00000001"
    assert audit.audit_synsets(
        b"n00000001\nn00000002\nn00000003\n", mapping
    )["class_order_sha256"] == audit._canonical_sha256(mapping)
    with pytest.raises(ValueError, match="WNID order differs"):
        audit.audit_categories(
            b"first,n00000002\nsecond,n00000001\nthird,n00000003\n", mapping
        )


def test_manifest_stream_replays_label_path_and_balance(tmp_path: Path) -> None:
    path = tmp_path / "image_manifest.jsonl"
    _write_manifest(path, _manifest_rows())
    mapping = audit.normalize_label_mapping(_mapping(2), expected_classes=2)
    summary, identity = audit.audit_dataset_manifest(
        path.resolve(),
        mapping,
        expected_split_counts={"train": 2, "val": 2},
        expected_validation_per_class=1,
    )
    assert summary["row_count"] == 4
    assert summary["label_matches_wnid_for_every_row"] is True
    assert summary["path_wnid_matches_for_every_row"] is True
    assert identity["bytes"] == path.stat().st_size


@pytest.mark.parametrize("tamper", ["label", "wnid", "path", "resolution"])
def test_manifest_stream_rejects_semantic_tamper(
    tmp_path: Path, tamper: str
) -> None:
    rows = _manifest_rows()
    if tamper == "label":
        rows[0]["label"] = 1
    elif tamper == "wnid":
        rows[0]["wnid"] = "n00000002"
    elif tamper == "path":
        rows[0]["path"] = "extracted/train/n00000002/wrong.jpg"
    else:
        rows[0]["width"] = 128
    path = tmp_path / "image_manifest.jsonl"
    _write_manifest(path, rows)
    mapping = audit.normalize_label_mapping(_mapping(2), expected_classes=2)
    with pytest.raises(ValueError):
        audit.audit_dataset_manifest(
            path.resolve(),
            mapping,
            expected_split_counts={"train": 2, "val": 2},
            expected_validation_per_class=1,
        )


def test_execution_code_contract_matches_locked_pipeline() -> None:
    raw = {key: (ROOT / path).read_bytes() for key, path in audit.CODE_FILES.items()}
    result = audit.audit_code_contract(raw)
    assert result["all_required_semantics_present"] is True
    assert result["synthesis_operator_receives_tokens_not_class_labels"] is True


def test_execution_code_contract_rejects_label_formula_tamper() -> None:
    raw = {key: (ROOT / path).read_bytes() for key, path in audit.CODE_FILES.items()}
    raw["code_generate_samples"] = raw["code_generate_samples"].replace(
        b"% num_classes", b"% (num_classes - 1)", 1
    )
    with pytest.raises(ValueError, match="execution code contract differs"):
        audit.audit_code_contract(raw)


def _metric_rows() -> list[dict[str, object]]:
    steps = [1, *range(50, 10_001, 50)]
    rows = []
    for step in steps:
        row: dict[str, object] = {
            "step": step,
            "samples_seen": step * 64,
            "loss": 1.0 / step,
        }
        if step % 1000 == 0:
            row["validation_epsilon_mse"] = 0.03
        rows.append(row)
    return rows


def test_metrics_stream_requires_strict_finite_sample_accounting(tmp_path: Path) -> None:
    path = tmp_path / "train_metrics.jsonl"
    _write_manifest(path, _metric_rows())
    summary, identity = audit.audit_metrics_jsonl(path.resolve())
    assert summary["row_count"] == 201
    assert summary["validation_event_count"] == 10
    assert summary["final_samples_seen"] == 640_000
    assert identity["bytes"] == path.stat().st_size


def test_metrics_stream_rejects_sample_accounting_tamper(tmp_path: Path) -> None:
    rows = _metric_rows()
    rows[20]["samples_seen"] = 1
    path = tmp_path / "train_metrics.jsonl"
    _write_manifest(path, rows)
    with pytest.raises(ValueError, match="sample accounting differs"):
        audit.audit_metrics_jsonl(path.resolve())


def test_audit_validation_is_fail_closed(tmp_path: Path) -> None:
    value = _audit_fixture(tmp_path)
    assert audit.validate_audit(value) == value
    tampered = copy.deepcopy(value)
    tampered["authorization_boundary"]["full_300k_launch_allowed"] = True
    with pytest.raises(ValueError, match="contract differs"):
        audit.validate_audit(tampered)
    tampered = copy.deepcopy(value)
    tampered["scientific_judgment"]["common_cause_proven"] = True
    with pytest.raises(ValueError, match="scientific judgment differs"):
        audit.validate_audit(tampered)


def test_validation_receipt_is_reproducible(tmp_path: Path) -> None:
    value = _audit_fixture(tmp_path)
    audit_identity = _identity(tmp_path / "audit.json", "c")
    validator_git = {
        "branch": "analysis/validator",
        "revision": "3" * 40,
        "tracked_dirty": False,
        "tree": "4" * 40,
    }
    receipt = audit.build_validation(
        audit=value,
        audit_identity=audit_identity,
        validator_git=validator_git,
        validator_script=_identity(tmp_path / "validator.py", "d"),
    )
    assert audit.validate_validation(
        receipt, audit=value, audit_identity=audit_identity
    ) == receipt
    receipt["scientific_judgment"]["mapping_can_explain_support_collapse"] = True
    with pytest.raises(ValueError, match="not reproducible"):
        audit.validate_validation(receipt, audit=value, audit_identity=audit_identity)


def test_exclusive_write_refuses_overwrite(tmp_path: Path) -> None:
    path = (tmp_path / "immutable.json").resolve()
    identity = audit._write_exclusive(path, {"status": "pass"})
    assert identity["bytes"] > 0
    with pytest.raises(FileExistsError):
        audit._write_exclusive(path, {"status": "different"})
