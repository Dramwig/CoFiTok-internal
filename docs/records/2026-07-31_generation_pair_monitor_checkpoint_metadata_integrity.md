# Generation pair monitor checkpoint metadata integrity

Date: 2026-07-31

Branch: `scale/generation-stability`

Revision: `b7d87158e4fa20ce06988262aff6bfe3ac0ae509`

## Motivation

The pair monitor previously checked metric finiteness, process liveness,
checkpoint cadence, and non-empty checkpoint files. It did not inspect the
adjacent integrity sidecar or prove that `latest.json` still bound the newest
stable checkpoint. A damaged metadata chain could therefore remain invisible
until a later training or sampling gate.

## Implementation

`cofitok.monitoring.inspect_run` now accepts:

```text
checkpoint_integrity_policy = optional | required
```

For every observed checkpoint it reads only file stat and JSON metadata. It
checks:

- checkpoint filename, byte count, and step against the adjacent sidecar;
- a syntactically valid lowercase SHA256 declaration;
- `latest.json` filename, byte count, step, sidecar filename, and declared
  SHA256 binding for the newest stable checkpoint;
- malformed or non-object sidecar and latest JSON;
- missing required metadata after the configured checkpoint grace window.

The monitor explicitly records:

```text
verification: metadata_only_no_payload_hash
```

It never loads a checkpoint and never computes the checkpoint payload hash.
The existing post-training trust boundary remains responsible for recomputing
and validating payload SHA256 before promotion, sampling, or release.

At the exact checkpoint step, metadata may still be completing atomically.
Required enforcement therefore begins only after
`checkpoint_grace_steps`, or immediately when the training report declares
exact completion. This prevents a transient checkpoint-write boundary from
terminating a healthy queue.

The true-compressed 10% matched 50K and full matched 300K runbooks now pass:

```text
--checkpoint-integrity-policy required
```

The currently active stability 5K run remains pinned to its original
`59db142fc45d69dc92bb0333be5ac2d0162d9dc4` monitor and was not hot-switched.

## Verification

Local targeted tests passed with two existing skips. Local full pytest passed
with three existing skips. Coverage includes:

- grace-window behavior;
- fail-closed missing sidecar and latest binding;
- valid sidecar/latest metadata;
- sidecar byte-count mismatch;
- existing cadence, liveness, and transition behavior.

The incremental rehearsal bundle was:

```text
bytes: 22,715
SHA256: c802a52ade38ccd149ffb663f759c39d17958730ed4b0575b42534ba14824539
```

The isolated Linux checkout
`/tmp/cofitok-monitor-integrity-b7d8715` passed the same targeted monitor and
transition suite (`27 passed`, two existing skips), and both modified formal
runbooks passed `bash -n`.

The rehearsal did not move either protected checkout:

```text
active stability 5K:
59db142fc45d69dc92bb0333be5ac2d0162d9dc4
official remote repository:
1ebcc15210e63a776a2ba448481cbd8bb94a4066
```

At the end of the rehearsal, the active 5K monitor remained
`running / cofitok_training / issues=[]` at step 250 (16,000 images).

## Multi-checkpoint grace transition

Follow-up revision `fcd2434b3821ed5f2f4501da68a9bb09debafd37`
fixes a transition boundary found before deploying the observer. When a new
milestone checkpoint first appears, `latest.json` may already point to it while
that checkpoint is still inside the write-grace window. The initial
implementation selected the previous stable checkpoint for the latest binding
and could therefore compare two different milestones.

The monitor now treats the newest observed checkpoint as the only candidate
for `latest.json`:

- before grace, a missing sidecar or not-yet-updated latest pointer is
  `pending_checkpoint_grace` and is not a health failure;
- a complete sidecar/latest pair may verify immediately;
- once grace expires, missing or mismatched metadata fails closed;
- the previous checkpoint is never compared with the new latest pointer.

Two regression tests cover both the valid in-grace transition and an incorrect
latest pointer after grace. Local full pytest again passed with three existing
skips. The isolated Linux suite passed `29` relevant tests with two existing
skips, and both formal runbooks passed `bash -n`.

The follow-up incremental bundle was:

```text
bytes: 4,365
SHA256: a764c98629bd505135b6767078e3c7097cd14c6d85df6568493c6eb4c849e60a
```

The active 5K and official remote repository remained pinned to `59db142` and
`1ebcc15`, respectively. The active 5K monitor was still healthy at step 400.

## Checkpoint code-revision binding

Revision `e22e1784ee5040167eeff029a357334990ebcc33` extends the required
policy to bind every checkpoint sidecar to an expected clean Git revision.
For formal same-checkout monitors, the expected revision defaults to the
monitor checkout's current HEAD. A detached observer may instead pass:

```text
--expected-checkpoint-revision <training revision>
```

After a sidecar appears, a different `git_revision` or any value other than
`git_dirty=false` is a health failure. The report includes the expected
checkpoint revision independently from the observer-code revision, so an
external observer cannot blur the two identities.

Local targeted and full tests passed, with two and three existing skips
respectively. The isolated Linux suite passed `30` relevant tests with two
existing skips; both formal runbooks again passed `bash -n`. The incremental
bundle was:

```text
bytes: 2,811
SHA256: f4a5f54048151f8373ac9ea8889e7560fe6f07257067962f8bf37da5b3fc7abd
```

A second read-only observer now watches the active EMA-teacher 5K pair:

```text
observer checkout revision:
e22e1784ee5040167eeff029a357334990ebcc33
expected checkpoint revision:
59db142fc45d69dc92bb0333be5ac2d0162d9dc4
PID: 36703
report:
/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_probe_2026-07-29/pair5k_rollout_x0_u2_ema_teacher_integrity_observer_e22e178.json
```

The initial report was `running / cofitok_training / issues=[]` at step 550.
It uses required sidecar/latest checks but does not participate in the
authoritative training monitor or post-evaluation waiter. The superseded
metadata-only observer PID 21289 was terminated only after its `/proc` cwd and
monitor name were both verified; no training, authoritative monitor, waiter,
or GPU process was signaled.

## Run-manifest and schedule binding

Revision `63af805b7ec1fe622c7a58aa65ed1d136097142e` extends the required
monitor policy to validate `run_manifest.json` before accepting live metrics.
The monitor now binds:

- configured target steps and checkpoint interval;
- clean training Git revision;
- runtime-environment SHA256;
- formal dataset provenance status and dataset identity SHA256;
- every logged rollout-consistency and EMA-teacher-consistency scale against
  the manifest schedule using the same float32 scalar semantics as training.

Missing positive-weight scale fields and any scale drift are health failures.
Legacy optional-policy runs remain compatible and report the manifest check as
`not_enforced`.

Local targeted and full tests passed, with the same existing skips. The
incremental rehearsal bundle was:

```text
bytes: 4,637
SHA256: 171686331d62437df54473d5c4646d6badfea27c0295906bc133e5e5815a0ea8
```

The isolated Linux checkout
`/tmp/cofitok-monitor-manifest-63af805` passed all 20 targeted monitor tests,
all 88 tracked shell runbooks passed `bash -n`, and the checkout remained
clean.

A replacement read-only observer now watches the active EMA-teacher 5K pair:

```text
observer checkout revision:
63af805b7ec1fe622c7a58aa65ed1d136097142e
expected training/checkpoint revision:
59db142fc45d69dc92bb0333be5ac2d0162d9dc4
PID: 37643
report:
/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_probe_2026-07-29/pair5k_rollout_x0_u2_ema_teacher_manifest_observer_63af805.json
```

Its first accepted snapshot was `running / issues=[]` at CoFiTok step 800.
The manifest was `verified`; all 33 metric rows matched both configured
consistency schedules, and the runtime-environment and dataset-identity SHA256
values were present and valid. The superseded `e22e178` observer PID 36703 was
terminated only after the replacement PID, checkout, command, and healthy
report were verified.

The authoritative monitor, training runbook, post-evaluation waiter, active
training checkout, and official repository were not changed:

```text
active training revision:
59db142fc45d69dc92bb0333be5ac2d0162d9dc4
official remote repository:
1ebcc15210e63a776a2ba448481cbd8bb94a4066
```

This is observability hardening only. The active run remains a fresh matched
5K gate, and formal 50K training remains unauthorized until the bounded raw
n=8 and two-seed n=64 post-evaluation decision passes.

## Malformed-provenance fail-closed follow-up

Revision `164c96e71dd97f0ac85a4f906c5a86cf768b7390` closes two remaining
parser boundaries. Required monitoring now rejects an absent or non-formal
dataset-provenance declaration, and valid JSON values that are not metric
objects are reported as health failures. Invalid step or consistency-scale
types can no longer terminate the monitor while schedule validation runs.

The local targeted suite passed all 22 tests, the full suite passed with three
existing skips, and the isolated Linux suite also passed all 22 targeted
tests. The incremental bundle was:

```text
bytes: 2,903
SHA256: bbec81d902a4d08e4f62a3a3100d264d5bf83948449f75358a2fb2c2c94398d1
```

The replacement observer is:

```text
checkout:
/tmp/cofitok-monitor-provenance-164c96e
PID: 38141
report:
/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_probe_2026-07-29/pair5k_rollout_x0_u2_ema_teacher_provenance_observer_164c96e.json
```

Its first accepted snapshot was `running / issues=[]` at step 925, with all
38 metric rows matching both schedules and the formal dataset identity
verified. The previous observer PID 37643 was stopped only after the
replacement process, checkout, command, and report passed. Training,
authoritative monitoring, and post-evaluation control remain unchanged.
