from __future__ import annotations

import copy
from pathlib import Path

import pytest

from scripts import (
    run_generation_conditioning_ranking_training_confirmation_posteval_supervisor as supervisor,
)


ROOT = Path(__file__).resolve().parents[1]


def _identity(path: str, character: str) -> dict:
    return {"path": path, "bytes": 100, "sha256": character * 64}


def _training_status(output_root: Path) -> dict:
    root = output_root.resolve().as_posix()
    runs = {}
    for index, run in enumerate(supervisor.RUN_NAMES):
        run_dir = f"{root}/{run}"
        runs[run] = {
            "run_dir": run_dir,
            "training_report": _identity(
                f"{run_dir}/training_report.json", f"{index + 1:x}"
            ),
            "training_audit": _identity(
                f"{root}/reports/{run}_training_audit.json", f"{index + 5:x}"
            ),
            "checkpoint": _identity(
                f"{run_dir}/checkpoint_step_00005000.pt", f"{index + 9:x}"
            ),
            "integrity_manifest": _identity(
                f"{run_dir}/checkpoint_step_00005000.pt.integrity.json",
                ("d", "e", "f", "a")[index],
            ),
            "final_metrics": {"step": 5_000, "samples_seen": 320_000},
        }
    return {
        "schema_version": 1,
        "role": supervisor.TRAINING_STATUS_ROLE,
        "status": "completed",
        "stage": supervisor.TRAINING_STAGE,
        "revision": supervisor.TRAINING_GIT["revision"],
        "branch": supervisor.TRAINING_GIT["branch"],
        "output_root": root,
        "runs": runs,
        "execution_boundary": copy.deepcopy(
            supervisor.TRAINING_STATUS_EXECUTION_BOUNDARY
        ),
        "claim_boundary": copy.deepcopy(supervisor.TRAINING_STATUS_CLAIM_BOUNDARY),
    }


def test_training_process_match_is_exact_to_the_confirmation_root(tmp_path: Path) -> None:
    output_root = tmp_path / "confirmation"
    marker = output_root.resolve().as_posix()
    rows = "\n".join(
        [
            f"101 python scripts/train_generation.py --output-dir {marker}/control_cofitok",
            "102 python scripts/train_generation.py --output-dir /other/project/run",
            f"103 python scripts/evaluate_generation_conditioning_sensitivity.py {marker}",
        ]
    )

    matches = supervisor._matching_training_processes(
        rows,
        output_root=output_root,
    )

    assert matches == [
        {
            "pid": 101,
            "command": (
                f"python scripts/train_generation.py --output-dir "
                f"{marker}/control_cofitok"
            ),
        }
    ]


def test_training_status_requires_exact_completed_four_arm_evidence(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "confirmation"
    status = _training_status(output_root)

    validated = supervisor._validate_training_status(
        status,
        output_root=output_root,
    )

    assert set(validated["runs"]) == set(supervisor.RUN_NAMES)

    drifted = copy.deepcopy(status)
    drifted["runs"]["ranked_cofitok"]["checkpoint"]["path"] = (
        f"{output_root.resolve().as_posix()}/ranked_cofitok/"
        "checkpoint_step_00002500.pt"
    )
    with pytest.raises(ValueError, match="run differs"):
        supervisor._validate_training_status(drifted, output_root=output_root)


def test_launch_receipt_is_one_shot_cpu_only_and_non_authorizing(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "confirmation"
    posteval_root = output_root / supervisor.POSTEVAL_RELATIVE_ROOT
    evaluator_git = {
        "revision": "a" * 40,
        "branch": "scale/generation-label-ranking-5k-heldout-evaluation-v1",
        "tracked_dirty": False,
    }
    identities = {
        "runbook_identity": _identity("/checkout/runbook.sh", "1"),
        "preparation_identity": _identity("/reports/preparation.json", "2"),
        "training_status_identity": _identity("/reports/training_status.json", "3"),
        "training_execution_receipt_identity": _identity(
            "/preparations/execution_receipt.json", "4"
        ),
        "training_replay_identity": _identity(
            "/preparations/execution_replay.json", "5"
        ),
    }

    receipt = supervisor._build_launch_receipt(
        evaluator_git=evaluator_git,
        output_root=output_root,
        posteval_root=posteval_root,
        **identities,
    )

    assert receipt["status"] == "launch_reserved"
    assert receipt["authorization_boundary"]["cpu_only"] is True
    assert receipt["authorization_boundary"]["runbook_launch_count_maximum"] == 1
    for field in (
        "training_allowed",
        "sampling_allowed",
        "checkpoint_promotion_allowed",
        "followup_training_allowed",
        "full_100k_or_300k_launch_allowed",
        "release_authorization_allowed",
    ):
        assert receipt["authorization_boundary"][field] is False


def test_heldout_runbook_is_cpu_only_source_bound_and_creates_no_training() -> None:
    runbook = (
        ROOT
        / supervisor.RUNBOOK_RELATIVE_PATH
    ).read_text(encoding="utf-8")

    assert "export CUDA_VISIBLE_DEVICES=-1" in runbook
    assert "nice -n 19 ionice -c 3" in runbook
    assert "--num-samples 16" in runbook
    assert "--start-label 192" in runbook
    assert "--wrong-label-offset 500" in runbook
    assert "--timesteps 100 500 900" in runbook
    assert "--noise-seed 304060" in runbook
    assert "--threads 2" in runbook
    assert "checkpoint_step_00005000.pt" in runbook
    assert runbook.count('sha256sum "$TRAINING_STATUS"') >= 2
    assert runbook.count('sha256sum "$TRAINING_EXECUTION_REPLAY"') >= 2
    assert '[[ ! -e "$POSTEVAL_ROOT" ]]' in runbook
    assert runbook.index("pgrep -af 'scripts/train_generation.py'") < runbook.index(
        'mkdir "$POSTEVAL_ROOT"'
    )
    assert "generate_samples.py" not in runbook
    assert "train_generation.py --config" not in runbook
    assert "kill " not in runbook
    assert "pkill" not in runbook
