from __future__ import annotations

from cofitok.monitoring import build_monitor_report, inspect_run

try:
    from scripts.monitor_generation_pair import main as _main
except ModuleNotFoundError:
    from monitor_generation_pair import main as _main


DEFAULTS = {
    "monitor_name": "generation_10pct_matched_pair",
    "cofitok_run": "imagenet256_10pct_cofitok_k8_50k_2026-07-12",
    "dense_run": "imagenet256_10pct_dense_50k_2026-07-12",
    "expected_steps": 50_000,
    "training_process_pattern": r"[s]cripts/train_generation.py.*imagenet256_10pct_",
    "runbook_process_pattern": r"[g]eneration_10pct_matched_50k_2026-07-12.sh",
    "checkpoint_interval": 5_000,
    "checkpoint_grace_steps": 250,
}

__all__ = ["build_monitor_report", "inspect_run"]


if __name__ == "__main__":
    _main(DEFAULTS)
