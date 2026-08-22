# Append-only resume history integration for matched trajectory audits

Date: 2026-08-23 CST

## Purpose

`build_generation_matched_training_trajectory.py` previously required repeated
`--cofitok-reconciliation` / `--dense-reconciliation` arguments to recover more
than the latest metrics-resume boundary. That path could physically bind the
files explicitly named on the command line, but it could not prove that an
earlier resume event had not been omitted.

This change makes the trainer's append-only `metrics_resume_history` the primary
source whenever it is present in a run manifest. The CLI reports remain a legacy
fallback and an optional corroborating source; they are no longer treated as
proof of a complete recovery chain by themselves.

## Locked implementation identity

- Branch:
  `integration/generation-resume-history-trajectory-v1-20260823`
- Code revision:
  `3c6dc320f93d7f2071e13e5b5d87ad618ca02bbf`
- Code tree:
  `e54ebeb93754b91d44114921af791f6dfafe6482`
- Direct parent (append-only trainer history):
  `0227da0007f5b04a7dff69e25ff182f5ae261688`

Changed files:

- `scripts/build_generation_matched_training_trajectory.py`
- `tests/test_build_generation_matched_training_trajectory.py`

## Fail-closed behavior

When `metrics_resume_history` is embedded in the manifest, the schema-v5
trajectory builder now:

1. validates the canonical `history_sha256`;
2. requires the independent `metrics_resume_history.json` journal to be
   physically present and byte-semantically equal to the manifest copy;
3. physically verifies every known reconciliation report's path, byte count,
   SHA256, schema, resume step, metrics path and retained/orphan row counts;
4. physically verifies every known orphan archive's path, byte count and SHA256;
5. verifies each recorded canonical metrics prefix by exact byte count and
   SHA256 against the current append-only JSONL;
6. verifies checkpoint-sidecar identity and, when the checkpoint payload is
   still present, its physical byte count;
7. requires the current manifest `resume` checkpoint and full
   `metrics_resume_reconciliation` document to equal the latest checkpoint-bound
   history event;
8. automatically supplies all verified resume boundaries to the exact logging
   schedule audit;
9. requires manually supplied reconciliation reports to match both the full
   normalized event sequence and the exact report paths from manifest history;
10. marks `complete_recovery_chain_claim_allowed=false` whenever the run has
    `legacy_history_complete=false`, an unresolved legacy reconciliation, only
    CLI reports, only a current reconciliation, or no append-only history.

The trajectory audit deliberately does not rehash multi-gigabyte checkpoint
payloads. Its report states `payload_sha256_recomputed=false`; full physical
payload SHA verification remains the responsibility of the separately pinned
checkpoint-integrity waiters and replay auditors.

## Test evidence

Local Windows project environment:

```text
tests/test_build_generation_matched_training_trajectory.py: 39 passed
combined trajectory + metrics history + exact resume: 88 passed
```

The added coverage includes:

- automatic two-resume and three-resume history consumption;
- successful CLI corroboration and CLI/history divergence rejection;
- history digest drift;
- bound report path drift and report payload drift;
- orphan archive drift;
- canonical metrics-prefix drift;
- current manifest/latest-event divergence;
- checkpoint-bound `absent` reconciliation;
- explicit incomplete-legacy claim downgrade;
- the real `main()` path with no reconciliation CLI arguments.

## Bundle and Linux rehearsal

- Bundle:
  `D:/cofitok-bundles/resume-history-trajectory-3c6dc32.bundle`
- Bundle bytes: `60,651`
- Bundle SHA256:
  `de346cff39873491c3d70949a48357069e5bbad0f765fa54f7c8b3b28b2754e9`
- Required prerequisite:
  `cf0e5faa94bf4ab38d947b921935b3b765b5537a`
- Advertised ref: exactly
  `integration/generation-resume-history-trajectory-v1-20260823` at
  `3c6dc320f93d7f2071e13e5b5d87ad618ca02bbf`
- Uploaded copy:
  `/tmp/resume-history-trajectory-3c6dc32.bundle`
- Isolated Linux checkout:
  `/root/autodl-tmp/CoFiTok/checkouts/resume-history-trajectory-3c6dc32/CoFiTok-internal`
- Linux CPU-only rehearsal:
  `88 passed in 32.56s`

The rehearsal used `CUDA_VISIBLE_DEVICES=''`, `OMP_NUM_THREADS=1` and
`MKL_NUM_THREADS=1`. Before and after the rehearsal, the active quality-bridge
checkout remained exactly:

```text
revision = cf0e5faa94bf4ab38d947b921935b3b765b5537a
tree     = 6cef27723196fd363379bca2e7b85b1678ebd777
branch   = scale/generation-stability-quality-bridge-100k
```

The only GPU owner remained the existing dense trainer. This integration was
not deployed into the active training checkout and did not create a controller,
trainer, sampler or GPU evaluation process.

## Claim and authorization boundary

This is provenance/audit hardening for future exact-resume runs. It does not
change the current quality-bridge result, does not prove generation advantage,
does not authorize a factorization or conditioning diagnostic, and does not
authorize 300K training, promotion or release. The active scientific state
remains `generation_advantage_proven=false` until matched 100K terminal sampling
and all claim guards finish.
