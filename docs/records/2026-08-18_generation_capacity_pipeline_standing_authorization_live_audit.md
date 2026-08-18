# Capacity pipeline standing-authorization live audit

Date: 2026-08-18 (Asia/Shanghai)

## Outcome

The deployed capacity progression is consistent with the user's active standing
experiment authorization while preserving exact stage, revision, source, and
output bindings. The authorization is not a bypass around the scientific or
operational gates: every future GPU stage remains dormant until its own
source-bound decision and receipt exist.

At the live snapshot, every capacity, full-300K, post-evaluation, and
finalization supervisor was `waiting`, all reported `child_pid=null`, the full
300K output root did not exist, and the only GPU compute process was the active
base128 quality-bridge CoFiTok trainer PID `619775`.

No process was signalled, restarted, replaced, or launched by this audit.

## Standing authorization

The active source is:

```text
/tmp/cofitok-quality-bridge-execution-cf0e5fa/standing_authorization.json
bytes:  865
SHA256: 5fe64a0941acb55a21cb6479a726987c6259e93a772c47290f2d6883752de4df
```

It binds the user's instruction:

```text
之后不要我授权你直接运行需要的实验
```

Its recorded interpretation authorizes necessary future experiments for the
active CoFiTok objective without repeated per-stage prompts. It explicitly
preserves clean isolated checkouts, exact revision/stage/output binding,
formal-checkout immutability, locked-evidence preservation, unrelated-process
non-interference, and the non-authorizing status of protocols that declare
themselves diagnostic.

## Exact progression

The deployed stages remain separated as follows:

1. The 250M/10K capacity probe requires the source-bound quality-bridge
   follow-up decision and preparation. Its result is diagnostic and cannot
   directly authorize 50K, 100K, full 300K, release, or promotion.
2. Capacity 50K requires the exact probe result plus a physically replayed
   scaling decision. It may resume only the matching 10K checkpoints and cannot
   start fresh training or authorize 100K/full 300K by itself.
3. Capacity 100K requires the exact completed 50K result plus a new completion
   decision. It may resume only from the matching 50K pair. Its terminal result
   is explicitly not a promotion gate.
4. Full-300K readiness requires a selected source-compatible 100K result, a new
   readiness decision, a fresh output root, storage validation, a bounded GPU
   runtime benchmark, and an immutable readiness receipt. The readiness
   supervisor cannot launch training.
5. The separate full-training supervisor may launch only a fresh matched 300K
   pair after the exact passed readiness receipt. It cannot resume the capacity
   100K checkpoints, claim formal completion, or authorize release.
6. Post-evaluation waits for passed exact full training; finalization waits for
   the final quality gate, verified exports, and the terminal completion audit.
   Only that terminal chain can produce a release receipt.

Thus standing authorization removes repeated prompting, but it does not remove
any source, scientific, storage, idle-GPU, checkpoint, provenance, or completion
gate.

## Live control state

All current stage supervisors were waiting with attempt zero and no child:

```text
capacity preparation              waiting
250M/10K probe execution          waiting
250M/50K scaling                  waiting
250M/100K completion              waiting
full-300K readiness               waiting
full-300K training                waiting
full-300K post-evaluation         waiting
full-300K finalization            waiting
```

The original lineage observer remains in a historical failed state because its
formal-checkout snapshot predates the server recovery. It is not the current
authority. The source-bound recovery-aware observer v2 was `running`, reported
`issues=[]`, and correctly classified the active exact-resume quality bridge as
the current stage.

## Evidence boundary

This audit verifies control-plane scope and live dormancy. It does not prove
sample quality, completion of any future training stage, a generation-quality
advantage, release readiness, or a usable final model. Those claims still
require the corresponding physical training, sampling, statistical,
quality-gate, export, and terminal-completion evidence.

Machine-readable snapshot:

```text
artifacts/reports/generation/
capacity_pipeline_standing_authorization_live_audit_2026-08-18/audit.json
bytes:  5,851
SHA256: 9e9ec13c06b5a1d86337425f84b852e539c466b27cee643c1db99330dffe33b3
```
