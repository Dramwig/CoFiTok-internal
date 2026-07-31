# Large-capacity isolated deployment execution (2026-07-31)

## Deployment identity

- implementation branch: `scale/generation-large-capacity`;
- deployed target:
  `7bab083b8d303c03d19074af4c1ed3200b1ad445`;
- formal repository remained:
  `1ebcc15210e63a776a2ba448481cbd8bb94a4066` on
  `scale/generative-system`, tracked clean;
- isolated checkout:
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/deployment/large_capacity/checkout-7bab083`.

## Bundle

`/root/autodl-tmp/CoFiTok/checkpoints/generation/deployment_bundles/
cofitok-generation-large-capacity-7bab083-from-1ebcc15.bundle`

- bytes: `39,369,166`;
- SHA256:
  `0369f8e304b25eda0837f177ddaa3339f380edcce6564ea6102b5cd13d630b86`;
- advertised head: `7bab083b8d303c03d19074af4c1ed3200b1ad445`;
- prerequisites:
  `1ebcc15210e63a776a2ba448481cbd8bb94a4066` and
  `58d83bfce2770eab2565b8c89a5f9a06201a0c86`.

The formal repository successfully ran `git bundle verify` without fetching
or moving its HEAD.

## Validation

The deployment runbook ran with `CUDA_VISIBLE_DEVICES=-1`. Its authoritative
evidence is under:

`/root/autodl-tmp/CoFiTok/checkpoints/generation/deployment/large_capacity`

- JUnit: `819 tests / 0 failures / 0 errors / 2 skipped`;
- runbook syntax: `97/97` tracked shell runbooks, zero failures;
- checkout branch/revision/tracked state: exact and clean;
- deployment receipt SHA256:
  `d07f4149376f8b6f757f607471dd9bac731d9927787e2e3d8526c72116205dda`.

The receipt replay validator passed after creation. The receipt states:

`readiness_execution_allowed=true`
`readiness_executed=false`
`full_training_launch_allowed=false`
`formal_generation_completion_claimed=false`

## GPU and training boundary

GPU process membership and memory were identical immediately before and after
deployment: trainer PID `319202` at approximately `84,122 MiB`. No
readiness benchmark ran. No `full_training_readiness.json` exists, and neither
the CoFiTok nor dense 300K formal run directory exists.

The deployment therefore proves only that the validated large-capacity checkout
is ready to execute the separately gated readiness stage. The readiness runbook
still requires the future source-bound 50K promotion gate SHA, this deployment
receipt SHA, an idle GPU, and absent full-training state. It does not authorize
full 300K training.
