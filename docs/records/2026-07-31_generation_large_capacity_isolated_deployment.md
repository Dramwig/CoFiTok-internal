# Large-capacity isolated deployment boundary (2026-07-31)

## Purpose

The existing post-training deployer is coupled to the legacy completion
supervisor: it fast-forwards the formal repository and launches the old full
pipeline. That behavior is incompatible with the new 250M three-stage chain.

The new chain is deliberately split into:

1. CPU-only isolated deployment;
2. source-gated, GPU-only runtime readiness;
3. separately authorized full 300K training launch.

No stage automatically invokes the next one.

## Isolated deployment

`artifacts/runbooks/generation_deploy_large_capacity_readiness_checkout.sh`
requires the exact formal-repository revision and branch, bundle bytes/SHA256,
target revision/branch, and both bundle prerequisites. It verifies the bundle
from the formal repository without fetching it there, then clones a temporary
checkout under:

`/root/autodl-tmp/CoFiTok/checkpoints/generation/deployment/large_capacity`

The temporary checkout is promoted atomically only after:

- all tracked generation shell runbooks pass `bash -n`;
- the complete CPU test suite passes with at least 800 passed tests;
- failures and errors are zero;
- skipped tests are at most five;
- target revision, branch, and tracked-clean state are exact.

Failure cleanup resolves both the temporary path and deployment root and only
allows recursive removal for the expected `checkout-<sha>.tmp.<pid>` child.
The formal repository is never checked out, fetched, merged, or modified.

## Receipt

`scripts/build_generation_large_capacity_deployment_receipt.py` writes a
deterministic receipt binding:

- formal repository path and deployment-time Git identity;
- isolated checkout path and target Git identity;
- bundle path, bytes, SHA256, advertised head, and prerequisites;
- complete JUnit bytes/SHA and pass/skip counts;
- complete runbook-syntax report bytes/SHA and enumerated count.

The receipt states:

`readiness_execution_allowed=true`
`readiness_executed=false`
`full_training_launch_allowed=false`
`formal_generation_completion_claimed=false`

Readiness must run from the exact receipt-bound checkout and adds the receipt
as an immutable source in `full_training_readiness.json`. Full launch replays
that readiness artifact. The terminal completion audit also reopens the
receipt, but permits the formal repository to have advanced normally after
deployment while still rehashing the fixed bundle, checkout, JUnit, and
runbook-syntax evidence.

## Current boundary

This implementation does not itself deploy, benchmark, or train. Deployment
may run while the active trainer owns the GPU because it sets
`CUDA_VISIBLE_DEVICES=-1`. The runtime readiness runbook remains forbidden
until the source-bound 50K promotion gate exists and the GPU is idle. Full 300K
still requires the separately supplied readiness SHA and deployment receipt
SHA and must never be launched by the deployment runbook.
