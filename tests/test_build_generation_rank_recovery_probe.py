from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts import build_generation_rank_recovery_probe as probe


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _candidate(run_dir: Path, *, endpoint: float, rank: int, fid: float) -> None:
    _write(
        run_dir / "training_report.json",
        {
            "training_complete": True,
            "completed_steps": 5000,
            "target_steps": 5000,
            "git": {"revision": "a" * 40, "branch": "scale/generative-system"},
        },
    )
    _write(
        run_dir / probe.CHECKPOINT_EVAL,
        {
            "status": "completed",
            "checkpoint_sha256": "b" * 64,
            "metrics": {
                "ordered_rank_by_path_auc": rank,
                "component_energy": [0.25, 0.75],
                "orders": {
                    "ordered": {
                        "endpoint_clean_mse": endpoint,
                        "endpoint_clean_psnr": 20.0,
                        "prefix_path_mse_auc": 0.1,
                    }
                },
            },
        },
    )
    _write(
        run_dir / probe.SAMPLING_REPORT,
        {"status": "completed", "sampling": {"num_samples": 512}},
    )
    _write(
        run_dir / probe.METRICS_REPORT,
        {
            "status": "completed",
            "metrics": {
                "frechet_inception_distance": fid,
                "inception_score_mean": 2.0,
            }
        },
    )


def test_probe_requires_visual_review_before_a_formal_rerun(
    tmp_path: Path, monkeypatch
) -> None:
    denoise = tmp_path / "denoise"
    epsilon_band = tmp_path / "epsilon_band"
    _candidate(denoise, endpoint=0.08, rank=2, fid=300.0)
    _candidate(epsilon_band, endpoint=0.07, rank=1, fid=250.0)
    legacy = tmp_path / "legacy.json"
    _write(
        legacy,
        {
            "metrics": {
                "ordered_rank_by_path_auc": 3,
                "component_energy": [0.01, 0.99],
                "orders": {"ordered": {"endpoint_clean_mse": 0.17}},
            }
        },
    )
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_generation_rank_recovery_probe.py",
            "--candidate",
            f"denoise_path={denoise}",
            "--candidate",
            f"epsilon_band={epsilon_band}",
            "--legacy-checkpoint-eval",
            str(legacy),
            "--output",
            str(output),
        ],
    )

    probe.main()

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["selection"] == {
        "automatic_50k_or_300k_launch_allowed": False,
        "best_mechanism_candidate": "epsilon_band",
        "decision": "manual_visual_review_required",
        "mechanism_ready_for_visual_review": True,
    }
    assert report["formal_claim_allowed"] is False
