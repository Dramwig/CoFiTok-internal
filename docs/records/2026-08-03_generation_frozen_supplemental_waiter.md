# Frozen post-evaluation supplemental waiter

Date: 2026-08-03

## Gap

The immutable stability 50K post-evaluation waiter and the already queued
large-capacity readiness waiter both consume the same GPU after dense training.
The later frozen supplemental runbook also needs the GPU for matched checkpoint
and EMA free-rollout diagnostics, but previously had no source-bound waiter.
Launching it directly when post-evaluation passed could race the readiness CUDA
qualification, fail on a transient busy GPU, or be forgotten after the older
waiters completed.

This is a sample-quality evidence gap, not a training-authorization gap. The
supplemental remains required as a negative quality prerequisite, but it cannot
authorize or launch full training.

## Implementation

The new entry points are:

```text
scripts/run_generation_stability_frozen_supplemental_waiter.py
artifacts/runbooks/generation_stability_frozen_50k_supplemental_waiter.sh
```

The waiter holds one nonblocking OS lock for its status output and then:

1. Requires one exact clean supplemental checkout revision and branch.
2. Continuously validates the frozen post-evaluation waiter's training and
   evaluation identities, terminal detail, child exit, non-authorizing flag,
   and freshness while it is nonterminal.
3. Validates the existing large-capacity readiness waiter's training,
   evaluation, readiness, and non-authorizing identities. It waits for that
   already queued GPU stage to become terminal, whether it passed or failed,
   so the supplemental cannot race it.
4. Waits until `nvidia-smi` reports no compute process. The child supplemental
   runbook independently repeats the busy-GPU check, closing the final check-to-
   launch race.
5. Launches only
   `generation_stability_frozen_50k_supplemental_after_posteval.sh` with the
   exact source revisions and checkpoint root. Exit `9` (new GPU contention)
   and exit `75` (supplemental lock contention) return to the wait loop; other
   failures are terminal and recorded.
6. Revalidates the produced combined report's clean builder identity,
   provenance contract, four named checks, and non-authorizing boundary. A
   scientific `hold` is recorded as a successfully completed execution, not
   converted into a pass. A scientific pass additionally replays all direct
   and nested source bindings through the standalone frozen-supplemental
   verifier.

Every status record fixes:

```text
supplemental_non_authorizing=true
full_training_launch_allowed=false
```

The waiter contains no trainer or full-training runbook and receives no launch
receipt or full-training authorization input.

## Verification

- The new waiter passes `py_compile`.
- The focused waiter, supplemental, entrypoint-contract, readiness-waiter, and
  post-evaluation-waiter suite passes (`35 passed`).
- The complete local suite passes (`1,018 passed, 6 skipped` in `260.82s`).
- Tests prove that a terminal readiness pass or failure permits later
  supplemental diagnosis, authorization or identity drift fails closed, a
  stale source status fails closed, a scientific hold remains non-authorizing,
  and no child starts until the queued readiness GPU stage is terminal.
- The runbook contract test proves that neither `train_generation.py` nor a
  full-300K runbook is referenced.

## Active-run boundary

This change is local-only. It does not deploy the supplemental checkout, start
the waiter, run a GPU diagnostic, modify the active training/post-evaluation/
readiness processes, create a bridge or launch receipt, or authorize full 300K.
The waiter may be deployed only as a separately attested clean evaluation
checkout and remains dormant until the current dense 50K training, frozen
post-evaluation, and queued readiness GPU stage have finished.

## Follow-up checkout attestation

The required clean evaluation checkout was prepared later on 2026-08-03 at
`/tmp/cofitok-stability-frozen-supplemental-c212b9e/CoFiTok-internal`, fixed to
`c212b9e2b64d1b302b17a9d4e30a296d773d4215` on
`scale/generation-large-capacity`. Linux CPU-only verification passed `70`
focused tests and all `104/104` runbook syntax checks. The checkout remains
dormant: the waiter and supplemental evaluation were not started, the active
GPU trainer was untouched, and full-training authorization remains false. See
`docs/records/2026-08-03_generation_frozen_supplemental_checkout_attestation.md`
and its machine-readable receipt for exact hashes.
