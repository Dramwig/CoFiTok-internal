# Terminal training-exposure waiter deployment

The CPU-only waiter at commit `e6f083d01062d594c32b1d595a77f0feb81a0ac4`
will atomically freeze and replay the final matched 100K training exposure only
after the active quality-bridge runbook has verified its terminal result.

It was deployed as PID `414533` and initially reported
`waiting_for_cofitok_training_milestone`. It has no GPU or experiment-launch
authority. See `deployment_receipt.json` for machine-readable identities and
`docs/records/2026-08-19_generation_terminal_training_exposure_binding_waiter.md`
for the full rationale.

