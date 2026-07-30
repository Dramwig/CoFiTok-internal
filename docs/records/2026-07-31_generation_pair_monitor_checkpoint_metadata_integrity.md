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
