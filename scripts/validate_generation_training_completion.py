from __future__ import annotations

import argparse
import json

from cofitok.training.completion import validate_completed_generation_training


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate whether one generation run is safe to skip as complete."
    )
    parser.add_argument("--training-report", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--expected-steps", type=int, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--expected-micro-batch-size", type=int)
    parser.add_argument("--expected-gradient-accumulation-steps", type=int)
    args = parser.parse_args()
    report = validate_completed_generation_training(
        report_path=args.training_report,
        config_path=args.config,
        expected_steps=args.expected_steps,
        expected_revision=args.expected_revision,
        expected_branch=args.expected_branch,
        expected_micro_batch_size=args.expected_micro_batch_size,
        expected_gradient_accumulation_steps=(
            args.expected_gradient_accumulation_steps
        ),
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "completed_steps": report["completed_steps"],
                "revision": report["git"]["revision"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
