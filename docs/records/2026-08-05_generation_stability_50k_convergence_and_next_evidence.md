# Stability 50K convergence audit and ordered next evidence

Date: 2026-08-05

## Outcome

The immutable stability 50K pair is finite, matched, exactly resumable, and
checkpoint-auditable. It does not show a CoFiTok-only optimization failure.
The decision-grade problem is instead a shared sample-quality and distribution-
support failure: the frozen promotion gate remains `fail/hold`, with only
`absolute_fid_quality` failing, while both methods have formal recall below
`0.01`.

The late epsilon stream is descriptively flat at the cosine learning-rate floor,
but this is deliberately classified under an explicitly post-hoc, non-gating
5% band. It is not evidence that 50K training is sufficient, and the changing
validation batches do not form a temporal convergence curve. Existing frozen
evidence therefore cannot causally distinguish sampling-policy failure from
checkpoint undertraining.

The ordered next action is:

1. run the smaller matched 1,000-sample-per-case sampling-recovery diagnostic;
2. run an independently authorized matched 10K confirmation only if one shared
   non-baseline case strictly improves both methods;
3. otherwise run the fresh full-data matched 100K quality bridge;
4. never infer or authorize a 300K launch from this audit.

No sampling, training, approval sentinel, deployment, or GPU evaluation was
performed while producing this evidence.

## Machine-readable evidence

The local evidence receipt is:

```text
artifacts/reports/generation/stability_50k_convergence_audit_2026-08-05.json
bytes: 13315
sha256: e08efa919a14e12b1fab255c02a94713fb2d42d42e4834afc31be4e9b793c72b
```

The authoritative physical report is:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_scaling_50k_ema_teacher/reports/frozen_training_convergence_audit/convergence_audit.json
bytes: 57848
sha256: c3f362910fce56d6de8568a676ba378820fb07cf5bd0b4180fc22b20cab04eb6
```

The report was rebuilt independently from the same frozen sources and was
byte-identical. The builder refuses an existing output, validates the complete
metrics and manifest identities, and records all authorization flags as false.

## Exact audit identity and validation

The code execution target is:

```text
branch: scale/generation-stability-50k-convergence-audit-v1
revision: f3606dcac217debe558dea64581f63f0cbbd991c
tree: f5ec513846771104a2e019ddae0db3ebbdf87fc2
```

Its incremental bundle was `40,255,023` bytes with SHA256
`a67f0f885f1cd31e9719bbb7d123fcf35b04353893340a53ba2507a214fd0307`.
It advertised one exact head, verified locally and against the formal remote
repository, and was exercised only in an isolated `/tmp` checkout with CUDA
hidden.

Validation evidence:

- local targeted tests: `8 passed`;
- local expanded relevant tests: `112 passed`;
- Linux targeted tests: `8 passed`;
- Python compilation: pass;
- runbook `bash -n`: pass;
- direct entrypoint `--help`: pass;
- report replay: byte-exact;
- GPU work: none.

The formal checkout stayed at
`scale/generative-system@1ebcc15210e63a776a2ba448481cbd8bb94a4066`.
Its tracked state remained clean. Its full porcelain inventory was nonempty and
unchanged during the actual audit, with before/after SHA256
`774cfad9f88f6e4fa953556e466b02b16212184930ca9fcbf4e8c233ab1cb79c`.
A later read-only recheck found the same HEAD and no tracked/staged changes, but
an externally changed untracked inventory (`296` paths, SHA256
`51b3f5402a691968e7b8b31aaefd24f59744fbd7c5b88a036ec97ffa813100a2`).
This work did not modify that checkout.

## Training and checkpoint integrity

Both methods are bound to:

```text
training revision: 2c2c1f5166b73d4f28df93b276901671ac1a7836
training branch: scale/generation-stability-50k-preflight
dataset: imagenet_256_10pct
dataset identity: 97cfec247a6991d3fcda6ff14bc75a89c07063836fd9cbe99fa58a41ab867741
runtime identity: d5bfcd085ea467ee5d24dfccc6e147da06dd7a0a0efdcdab355882b8547c985e
steps: 50000 per method
effective batch: 64
images seen: 3200000 per method
```

The actual immutable stability manifests report `62,834,083` CoFiTok parameters
and `62,824,707` dense parameters, a `+0.014924%` gap. This pair-specific value
must not be silently replaced by an older full-configuration handoff count.

Checkpoint evidence:

| method | checkpoint bytes | checkpoint SHA256 | sidecar SHA256 |
|---|---:|---|---|
| CoFiTok | 1,006,325,418 | `ec7b9a0981f1d45420a9a86cdb80339d6d87b87fa77891c234db3d1b84376c2a` | `4f3f6f3f401f34131f902b016baf41897b35f95d242bb023981feff4a36226a8` |
| dense identity | 1,006,120,214 | `325da25f9fd228ab224abd977da7f9e1d000ba36e3e8136b55edae9c9f47c716` | `31b1cf05a0bafaf06e3fb434ebabe3fbed41800687e28e447140565b8f4cafff` |

The CoFiTok canonical metrics contain `1,002` rows and exactly one valid
resume-only extra row at step `36,546`; dense contains `1,001` rows. Both streams
are finite and strictly ordered, satisfy `samples_seen = step * 64`, and contain
50 matched validation events. The convergence audit verified physical bytes,
`latest.json`, required integrity-sidecar provenance, and locked checkpoint SHA
bindings without rereading the two approximately 1 GB checkpoint payloads.

## Matched validation boundary

Across the 50 scheduled paired validation events:

- Pearson correlation is `0.9975195919`;
- the CoFiTok/dense ratio-of-means delta is `-0.0004224801`, or approximately
  `-0.04225%`;
- CoFiTok has lower epsilon MSE in 18 events and dense in 32.

This supports a matched-method comparison and rejects a large CoFiTok-only
optimization defect. It does not support a temporal convergence claim because
each event advances to another deterministic validation batch.

## Late-training evidence

| method | epsilon mean 40-45K | epsilon mean 45-50K | late ratio | linear change 40-50K | epsilon mean 49-50K |
|---|---:|---:|---:|---:|---:|
| CoFiTok | 0.02736173 | 0.02766882 | 1.011223 | -0.3024% | 0.02606965 |
| dense identity | 0.02731535 | 0.02762335 | 1.011275 | -0.3008% | 0.02601692 |

Both finish at learning rate `1e-5`. Both signatures lie inside the explicitly
post-hoc 5% descriptive flat band. This is useful experiment-selection evidence,
not a preregistered threshold, quality gate, or proof that more training cannot
help.

Per-method total loss is intentionally not compared across methods because the
CoFiTok total contains factorization-specific terms. Epsilon and shared matched
validation evidence are the fair comparison channels.

## Frozen quality interpretation

The frozen gate is `31,870` bytes with SHA256
`2f763913d5a07ba5563f5bc75548780678fcdf9da4e119e03262f10daf84dd90`.

| method | FID | precision | recall |
|---|---:|---:|---:|
| CoFiTok | 138.2970 | 0.75050 | 0.00868 |
| dense identity | 151.4477 | 0.78330 | 0.00978 |

The frozen sample-support audit, SHA256
`6b3acfa62780b951b0adb45f817335b8086bafbed9dc7dcb90ba0afa07d0d9eb`,
rejects decoded-pixel and dHash duplicate collapse. It instead shows shared
excessive local variation and weak coarse class/distribution support, with
CoFiTok generally less affected than dense. This supports neither a
CoFiTok-specific factorization failure nor a causal sampler-versus-training
diagnosis.

The supplemental waiter's stale-heartbeat failure is a coordination defect and
must not be counted as model-quality evidence. It remains non-authorizing, and
the readiness waiter still has `full_training_launch_allowed=false`.

## Ordered next evidence and authorization boundary

Stage 1 is the matched 1,000-sample-per-case sampling-recovery diagnostic:

```text
revision: 11d8f954030915f1d8848594683ac70518ce39bc
branch: scale/generation-large-capacity
approval text: Approve the non-authorizing matched 1000-sample sampling-recovery diagnostic only.
```

It uses five shared cases for both methods and changes only shared sampling
settings. Its purpose is to test the cheaper sampling/CFG hypothesis before a
new matched training run. It remains permanently non-authorizing.

Stage 2 is a separate matched 10K confirmation, allowed only if one non-baseline
case strictly improves both methods in stage 1. Stage-1 approval does not approve
stage 2.

Stage 3 is the full-data matched 100K quality bridge at exact revision
`cf0e5faa94bf4ab38d947b921935b3b765b5537a` on
`scale/generation-stability-quality-bridge-100k`. It requires separate exact user
approval and should be used only if the sampling sweep does not recover both
methods or the independent confirmation fails.

FieldScope PID `433140` remains the only GPU compute process, with cwd
`/root/autodl-tmp/FieldScope/FieldScope-internal` and approximately `15,412 MiB`
allocated. It was not signaled or modified. Current storage free space is
`296,698,707,968` bytes. No CoFiTok GPU stage may start while this process is
active, and no stage has exact user execution approval.

All temporary convergence-audit bundle and `/tmp` checkout copies were removed
after verification. No unrelated worktree, dirty file, process, checkpoint, or
report was modified.
