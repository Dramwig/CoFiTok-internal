from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_quality_bridge_milestone_transition_is_serial_and_resumable() -> None:
    execute = (
        ROOT
        / "artifacts/runbooks/generation_stability_full_data_quality_bridge_100k_execute.sh"
    ).read_text(encoding="utf-8")
    milestone = (
        ROOT / "artifacts/runbooks/generation_full_milestone_eval.sh"
    ).read_text(encoding="utf-8")

    loop = execute.index("for milestone in 50000 100000; do")
    ordered_steps = (
        'train_to_milestone "$COFITOK_CONFIG" "$COFITOK_RUN" "$milestone"',
        "snapshot_monitor",
        'evaluate_milestone cofitok "$COFITOK_RUN" "$milestone" 8 4',
        'train_to_milestone "$DENSE_CONFIG" "$DENSE_RUN" "$milestone"',
        "snapshot_monitor",
        'evaluate_milestone dense_identity "$DENSE_RUN" "$milestone" 1 0',
        'build_paired_milestone "$milestone"',
    )
    positions = []
    cursor = loop
    for step in ordered_steps:
        cursor = execute.index(step, cursor)
        positions.append(cursor)
        cursor += len(step)
    assert positions == sorted(positions)

    for command in (
        ordered_steps[0],
        ordered_steps[2],
        ordered_steps[3],
        ordered_steps[5],
        ordered_steps[6],
    ):
        line = next(line for line in execute.splitlines() if command in line)
        assert not line.rstrip().endswith("&")

    milestone_commands = (
        '"$PYTHON" scripts/preflight_generation_sampling.py',
        '"$PYTHON" scripts/evaluate_generation_checkpoint.py',
        '"$PYTHON" scripts/generate_samples.py',
        '"$PYTHON" scripts/evaluate_generation_metrics.py',
    )
    milestone_positions = [milestone.index(command) for command in milestone_commands]
    assert milestone_positions == sorted(milestone_positions)
    assert milestone.count("--resume") == 3
    assert "set -euo pipefail" in milestone
    assert "refusing to overwrite invalid quality bridge milestone report" in execute
