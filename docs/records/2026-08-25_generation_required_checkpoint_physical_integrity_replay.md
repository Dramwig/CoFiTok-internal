# Required-checkpoint physical-integrity replay (2026-08-25)

## Scope

This change closes a narrow audit gap discovered after the completed matched
four-arm 1K conditioning-ranking screen.  The existing
`audit_generation_training_progress.py` required protected checkpoint files to
exist, but physically hashed only the newest checkpoint.  Steps 500 and 750
were therefore not physically verified by each original training audit.

This work is permanently non-authorizing.  It did not train, sample, load a
checkpoint into a model, promote a checkpoint, export an artifact, signal a
process, or authorize full training / 300K execution.

## Source

- Implementation commit:
  `17eaa4a2d37521281e5abcbc4050ad0e8b11ba94`
- Tree: `92c766900b27574e9768e728fedc63c080513315`
- Branch:
  `analysis/generation-required-checkpoint-physical-integrity-v1-20260825`
- Parent training/evaluation source:
  `d019a17b742c5da867e71fa72b2d67f75cb0bc7c`
- Auditor SHA256:
  `597cbea1ede17a7e98d1c8c7db397527f7bab26e40967c027b39b0900d4bc803`
- Replay builder SHA256:
  `6715bb6165401a8a0c599ae797c67bf3929c82a7892b42c9906e53fac4229721`
- Incremental bundle:
  `D:/cofitok-bundles/required-checkpoint-physical-integrity-17eaa4a.bundle`
- Bundle bytes / SHA256:
  `11,871` / `90f5e4340dbebe32548f3b8e7048c6658b8ab7d2823e743e1c75896bac80500c`

The bundle advertises only `17eaa4a` and requires the exact `d019a17`
prerequisite.  Local and remote `git bundle verify` passed.

## Implementation

The progress auditor now physically verifies every reached
`--required-checkpoint-steps` entry under the selected integrity policy.  The
report retains the existing schema version and adds
`checkpoint.required_integrity`, containing the requested/reached steps and a
per-checkpoint physical result.  The latest checkpoint is hashed once and
retains strict `latest.json` binding; historical protected checkpoints verify
their payload, sidecar, step, bytes, SHA256, and available Git/data/runtime
provenance without pretending to be the latest pointer.

The new screen-specific replay builder additionally checks:

- all four metrics streams are strictly increasing;
- every row satisfies `samples_seen == step * 64`;
- final metrics, training report, and `latest.json` agree;
- all 12 sidecars bind the expected clean `d019a17` training lineage, dataset,
  and runtime;
- all small source files and payload stat identities remain unchanged across
  the replay;
- preparation, authorization, training status, and postevaluation hashes match
  the completed screen;
- every execution and claim permission remains false.

## Deployment and replay

Remote isolated checkout:

```text
/root/autodl-tmp/CoFiTok/checkouts/required-checkpoint-physical-integrity-17eaa4a
```

Canonical report:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/conditioning_ranking_four_arm_probe1k_terminal_rebind_v2/reports/checkpoint_integrity_replay_v1/replay.json
```

Report bytes / SHA256:

```text
21,653
81b32088c6584ecc6c0ea8478bd4f2f06ad38bffc473c7ecec7107e4f8542cc1
```

The canonical builder ran with `CUDA_VISIBLE_DEVICES=""`, `OMP_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`, `nice=10`, and idle I/O priority.  The first wrapper's
final diagnostic `stat` received a carriage-return suffix and returned 1 only
after the builder had completed and the report SHA had been printed.  A
structured read verified the report, and a second complete physical replay
used the immutable-output path and returned byte-identical SHA256
`81b32088...`; the canonical report was not overwritten.

All 12 checkpoint payloads passed:

| Run | Step 500 | Step 750 | Step 1000 |
|---|---|---|---|
| `control_cofitok` | `b70f5f9ce88a185869657291034ff687fcfe72c9c843e23a9cc73789356c7f25` | `10d531263a55a2c71f373fe70f2d1071586a02ff83bc36bb7071233299400bae` | `2a9745315e931da44bbb036c02d7c779d126ae20ce15a9f58b6ffc8128928dc6` |
| `ranked_cofitok` | `0479fe05979698b1f8c1f51c08f9b7dc6be928557cab617e863d78a2cbbe4052` | `b7650a54be9a76216c1b70d100c791ec0a5a142e17bad47dc0b32d9d49a37547` | `1aaa81ea704022c83b2a00c2a2e918d66a55804c58f569463473630be4a77896` |
| `control_dense_identity` | `e02e1720883312c50a7b108c5fda427aad8a448e1db625f7e7794083b6ddb848` | `2cba245ccb07329b7568f01df5e6e931cfddc0b8fe698f0c3f4bf0be7f30abea` | `e02e907dcc1306e4fea601582161a24dd04215395aee33a59c6b3b19e5f4fc9e` |
| `ranked_dense_identity` | `2efb1910269c3abe1e1b1de926a71ee40ce1d32a65b98f61d4fae6861964e6d9` | `dc3840e48bb6cc91262b7da78e66d449bf0bfc48a4073204af3ef62367e2a49e` | `b12c3e95d093c052530676fb1197b6fda15d2ca2d4a2a6c24913d7f4c46207c9` |

## Verification

- Local auditor + builder focused tests: `32 passed`.
- Local downstream-consumer suite: `80 passed in 95.48s`.
- Remote exact-commit focused tests: `32 passed`.
- Remote exact-commit full suite: `1,272 passed, 2 skipped`, no failures.
- Remote collected test count: `1,274`.
- `compileall` and `git diff --check`: pass.
- Remote checkout tracked state: clean.
- GPU before, during, and after replay: zero compute processes and zero MiB
  used by compute applications.

## Scientific boundary

The replay preserves the completed postevaluation decision exactly:

```text
shared_semantic_alignment_recovery_supported=false
cofitok_specific_advantage_claim_allowed=false
recommended_next_action=revise_training_time_semantic_alignment_objective
generation_advantage_proven=false
```

This integrity result strengthens the trustworthiness of the completed screen;
it does not change the scientific outcome and does not authorize another
experiment.
