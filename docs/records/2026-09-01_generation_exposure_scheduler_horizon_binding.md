# Exposure continuation scheduler horizon binding (2026-09-01)

## Scope

This record covers CPU-only hardening of the bounded 100K to 110K exposure
continuation. It does not authorize training, sampling, promotion, export,
release, or 300K escalation, and it does not modify the remote 100K source
checkout or locked quality evidence.

## Finding

Rebuilding the warmup-cosine scheduler with the target 110K horizon would
silently change the optimization trajectory at the source boundary. With the
current 100K source configuration, a scheduler evaluated at step 100,000 has
learning rate `1e-5`; the same scheduler formula configured for 110,000 steps
would produce approximately `1.1856e-5` at that step. That is a learning-rate
increase despite the candidate being scoped as training-exposure-only.

## Contract change

The continuation now records a versioned `horizon_extension` object in the
run manifest, every continuation checkpoint's exact-resume extra state, and
the training report. It binds:

- source and target horizons and the additional step count;
- the source checkpoint payload, integrity sidecar, and their physical
  identities;
- canonical source/target resolved-config SHA256 values; and
- the explicit scheduler policy
  `preserve_source_scheduler_horizon`.

Under this policy the continuation constructs the scheduler with the source
horizon, restores its state, and keeps the source schedule's terminal learning
rate floor during the additional exposure. The result validator requires the
same policy, source checkpoint identities, target-config digest, and restored
scheduler epoch before accepting a bounded continuation result.

## Verification

- Focused exact-resume, exposure-authorization, and continuation tests pass
  (`1` environment skip).
- The complete CPU suite excluding the pre-existing outer-paper-path tests
  exits `0`; platform/environment skips remain unchanged.
- `compileall` and `git diff --check` pass.
- No remote process, GPU stage, formal checkout, or locked artifact was
  changed.

The implementation is committed on the isolated branch
`analysis/generation-exposure-continuation-v1` and remains dormant until a
separate source-compatible execution gate is valid.
