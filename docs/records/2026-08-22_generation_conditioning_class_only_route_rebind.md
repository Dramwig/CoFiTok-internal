# Conditioning probe exact class-only route rebind

Date: 2026-08-22

## Outcome

The first-stage four-arm 1K conditioning-ranking supervisor was rebound to the
current quality-bridge terminal chain. It remains non-authorizing and can create
GPU work only when the terminal follow-up route is exactly a class-fidelity-only
failure.

The implementation rejects mixed class-plus-matched, class-plus-absolute, and
class-plus-mechanism failure sets. It also replays and binds the current
exposure-aware decision builder, terminal-system guard waiter, requested-class
visual-audit waiter/report, quality-bridge result, standing authorization, probe
preparation, exact checkout revision/tree/branch, output root, and five
consecutive idle-GPU polls.

## Locked implementation

```text
branch: analysis/generation-conditioning-exact-route-rebind-v2-20260822
revision: ccb93c9d17401d41d221173f6d4ca9564ab1b8ea
tree: 34f2b5e7111d14c68cbb217b6bcc1d56a7a1da31
remote checkout: /root/autodl-tmp/CoFiTok/checkouts/conditioning-exact-route-rebind-ccb93c9/CoFiTok-internal
decision builder revision: cd78a348769f0efad0d42de063e5b0943444a29b
```

Incremental deployment bundle:

```text
D:/cofitok-bundles/conditioning-exact-class-route-rebind-ccb93c9.bundle
bytes: 14948
sha256: a37ac567f450410eaaeec1d0fb5cb374e65c2df2a1d043ce1cb49d5cca2ce437
prerequisite: cd78a348769f0efad0d42de063e5b0943444a29b
```

## Validation

Local and remote CPU-only validation both passed:

```text
113 passed
bash -n artifacts/runbooks/generation_conditioning_ranking_four_arm_probe1k_v1.sh
```

Negative coverage includes:

- class + matched-quality failure;
- class + absolute-quality failure;
- class + mechanism failure;
- wrong exposure-aware decision-builder Git;
- missing or mismatched requested-class visual audit;
- wrong terminal-system guard waiter Git.

## Deployment

Preparation:

```text
path: /root/autodl-tmp/CoFiTok/checkpoints/generation/preparations/conditioning_ranking_four_arm_probe1k_standing_auth_v1/ccb93c9d17401d41d221173f6d4ca9564ab1b8ea/preparation.json
bytes: 8944
sha256: 5f06bd1c3ef39a5cf567d7e3a0986d44075be3b0029ddad40f7ae605ea5f3fe2
status: pass
valid: true
```

Supervisor deployment snapshot:

```text
pid at deployment: 657177
ppid: 1
nice: 10
ionice: idle
CUDA_VISIBLE_DEVICES: -1
OMP_NUM_THREADS: 1
MKL_NUM_THREADS: 1
OPENBLAS_NUM_THREADS: 1
status: waiting
detail: waiting_for_quality_bridge_followup_decision
```

The PID is only a deployment snapshot and must be rediscovered and verified from
`/proc` before future use. The single supervisor owns an advisory lock and was
absent from the GPU compute-process list; the only GPU owner at deployment was
the active dense quality-bridge trainer.

Deployment receipt:

```text
path: /root/autodl-tmp/CoFiTok/checkpoints/generation/preparations/conditioning_ranking_four_arm_probe1k_standing_auth_v1/ccb93c9d17401d41d221173f6d4ca9564ab1b8ea/deployment_receipt.json
bytes: 8535
sha256: f4f2816553460cd4a494358a0a9aac4398fb091492e4a5ae09463bab5038f9a6
```

At deployment, the execution authorization, probe output, and probe output lock
were absent. The supervisor therefore had not authorized or launched training.

## Authorization boundary

```text
generation_advantage_proven=false
full_training_launch_allowed=false
full_300k_launch_allowed=false
promotion_or_release_allowed=false
mixed_failure_route_allowed=false
unrelated_process_signaling_allowed=false
```

The conditioning chain may proceed only if the completed terminal evidence routes
to exactly `failed_checks == ["class_fidelity"]`. Any mixed, hold, malformed, or
source-mismatched route fails closed.
