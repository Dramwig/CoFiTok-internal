# Conditioning-ranking postevaluation binding hardening

Date: 2026-08-19

## Finding

After repairing and launching the standing-authorization supervisor, a second
runtime audit found that
`scripts/build_generation_conditioning_ranking_probe_posteval.py` still bound
the postevaluation to obsolete identities:

```text
training revision: 7d5ff8c661ee3f17c99b2c5fb17525051b27a92f
training branch: scale/generation-label-ranking-probe-v1
evaluator branch: analysis/generation-label-ranking-posteval-v1
```

The current runbook intentionally performs the four 1K training arms and their
CPU-only held-out postevaluation from one exact, clean, source-bound checkout.
Its active branch is
`scale/generation-label-ranking-standing-authorization-v1`.  Consequently, the
legacy constants would have allowed all four GPU training arms to finish and
then forced the postevaluation to fail on Git identity.

No probe child had started when this was discovered.  The prior supervisor was
still waiting for the quality-bridge follow-up decision, with `child_pid=null`,
zero idle-GPU polls, no execution authorization, and no probe output root.

## Repair

The postevaluation now obtains the exact revision and branch from the immutable
preparation report, validates all four training reports and checkpoints against
that contract, and then requires its own evaluator Git provenance to equal the
same exact clean checkout:

```text
evaluator Git == {
  revision: preparation/training revision,
  branch: preparation/training branch,
  tracked_dirty: false
}
```

This removes the stale hard-coded revision and split evaluator branch without
weakening source binding.  The supervisor already verifies the preparation
SHA256, standing authorization, exact revision/branch, terminal decision,
terminal-system visual evidence, output root, and idle GPU state before the
runbook can start.

Implementation:

```text
branch: scale/generation-label-ranking-standing-authorization-v1
revision: 64f85fe3a34aebd664c083d43bfb38d1c61aaeb7
tree: e445020ed50580bbae9958cd9586e5a8fa0bffab
subject: Bind ranking postevaluation to probe checkout
```

Incremental bundle:

```text
D:/cofitok-bundles/conditioning-ranking-posteval-binding-64f85fe.bundle
bytes: 4,280
SHA256: 4ea923d269dd2d49fc22c480d8e2f6ab6bc6a288fe3f0288fae7030d5e40727e
prerequisite: fc47dbfda9758bf0aa440f1c23dd058e446c751d
advertised revision: 64f85fe3a34aebd664c083d43bfb38d1c61aaeb7
```

The bundle passed local and server `git bundle verify` and was applied only to:

```text
/tmp/cofitok-label-ranking-standing-auth-64f85fe
```

The active quality-bridge checkout, trainer, monitors, checkpoints, and formal
checkout were not modified.

## Validation

The focused local suite passed:

```text
21 passed
```

It covers the preparation, standing-authorization supervisor, source-bound
postevaluation, shared-repair decision, method asymmetry, correct-MSE
regression, request/checkpoint drift, schedule and audit drift, non-finite raw
metrics, derived-field tampering, cross-method row identity, paired sign-test
aggregation, and exact evaluator/training Git equality.

The exact Linux checkout independently passed:

```text
focused pytest: 21 passed
Python compileall: pass
probe runbook bash -n: pass
postevaluation runbook bash -n: pass
legacy Git constants absent: verified
post-test Git porcelain: empty
```

## New immutable preparation

The preparation was generated twice byte-identically from the repaired Linux
checkout.  The postevaluation's preparation validator and the live Git
provenance were then compared directly and were exactly equal.

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/preparations/conditioning_ranking_four_arm_probe1k_standing_auth_v1/64f85fe3a34aebd664c083d43bfb38d1c61aaeb7/preparation.json
bytes: 8,937
SHA256: 5872be0fe5b910fd20955c3ac4041cd64bc107447e5d47289f296e98fb769bba
```

The matched parameter counts remain:

```text
CoFiTok: 62,834,083
dense identity: 62,824,707
control/ranked parameter counts: identical within each method
```

## Supervisor replacement

The obsolete supervisor PID `205564` was identity-checked by PID receipt,
status receipt, `/proc` command line, start ticks, and working directory.  It
was confirmed to have no child process, no execution authorization, and no GPU
work.  It was terminated with SIGTERM and exited normally.  No trainer or other
controller was signaled.

Replacement receipt:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/preparations/conditioning_ranking_four_arm_probe1k_standing_auth_v1/64f85fe3a34aebd664c083d43bfb38d1c61aaeb7/supervisor_replacement_receipt.json
bytes: 1,399
SHA256: 2dbd663f88acfe66532b928b96a2b52d74398853bd5b8913417f9563e3e0eac4
status: replaced_before_child_launch
```

The repaired supervisor was launched once as PID `210203`:

```text
status: waiting
detail: waiting_for_quality_bridge_followup_decision
child_pid: null
idle_gpu_polls: 0
probe output root: absent
execution authorization: absent
```

Initial live artifact identities:

```text
supervisor.pid.json
bytes: 322
SHA256: 1e670ffadcbbaea6f90b7a341fa4786b70786ebcfbcd048572cab0175f64248a

supervisor_status.json
bytes: 1,582
initial SHA256: 2135b30d0e46fe53102c6c27a0d29cbe59b18219a35569cb761cef0a07971536
```

The status report is intentionally mutable while waiting, so the status hash
identifies only the initial captured snapshot.

At replacement verification time, the active dense quality-bridge trainer had
advanced to step `9,750`, the GPU remained at 99% utilization with 77,983 MiB
allocated, and the probe output root was still absent.  The replacement did not
interrupt or change the active matched training trajectory.

## Authorization boundary

This hardening and replacement do not authorize the four-arm GPU work by
themselves.  The supervisor can create its execution receipt only if the exact
full-data terminal follow-up decision selects
`run_class_conditioning_fidelity_diagnostic`, the terminal-system guard and
requested-class visual evidence are complete, the bound output root is absent,
the exact checkout remains clean, and five consecutive GPU-idle polls pass.

The probe and postevaluation remain permanently non-authorizing for checkpoint
promotion, follow-up training, full 300K training, release, or a CoFiTok-specific
advantage claim.
