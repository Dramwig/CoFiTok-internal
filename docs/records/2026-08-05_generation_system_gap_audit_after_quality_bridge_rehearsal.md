# Generation-system gap audit after the quality-bridge rehearsal

Date: 2026-08-05

## Outcome

The large-scale generation objective is not complete. The codebase now has a
substantial operational foundation, but the evidence required to call CoFiTok a
usable large-model generation system is still missing at the model-quality and
training-scale layers.

The machine-readable audit is:

```text
artifacts/reports/generation/generative_system_gap_audit_2026-08-05/gap_audit.json
```

The audit is `11,770` bytes with SHA256
`9ea3dea4e0aae5065f1463de5865ff821d01def1e0384af3a9aed25dbff9245f`.
It maps the objective into five independent evidence pillars and deliberately
distinguishes implemented infrastructure from completed scientific evidence.

## Current decision-grade quality evidence

The frozen 50K EMA post-evaluation remains source-valid. Its promotion gate is
`31,870` bytes with SHA256
`2f763913d5a07ba5563f5bc75548780678fcdf9da4e119e03262f10daf84dd90`.
The gate is `fail/hold`; the only failed row is `absolute_fid_quality`.

| method | FID | precision | recall |
|---|---:|---:|---:|
| CoFiTok | 138.2970 | 0.75050 | 0.00868 |
| dense identity | 151.4477 | 0.78330 | 0.00978 |

CoFiTok is better than the matched dense predictor in FID and still passes the
endpoint, ordered-prefix, coarse-token-utilization, exact-zero, and shuffle
checks. That evidence supports the scoped mechanism claim. It directly
contradicts a usable-generation claim because both absolute FID and recall are
far from the required release regime.

The official D-AR, MAR, and ReTok 50K rows remain contextual eval-only evidence,
not compute-matched rows. Their FIDs are approximately `2.63 / 2.34 / 2.22`.
They cannot be numerically merged into the matched panel, but they make the
remaining usability gap visible and reinforce that passing only the dense
relative comparison would be insufficient.

## Five-pillar status

### 1. Sample quality: failed current gate

The current gate is decision-grade and fails absolute quality. The prepared
full-data 100K bridge is the next controlled experiment, but it has not been
deployed or launched and its output root is absent. Completion still requires a
full-data bridge result and eventually the full 50K-sample DDIM-250 release gate.

### 2. Training scale: incomplete

The exact matched 50K pair used `imagenet_256_10pct`, 50,000 optimizer steps,
and 3.2M images per method. No full-data 100K pair exists, and the 250M full
root contains no 300K checkpoint. The only files in that root are the failed
readiness waiter and its empty log/PID state.

### 3. Strong-baseline fairness: partially complete

The role separation is correct:

- CoFiTok K8 and dense identity are the only direct matched-training methods.
- D-AR, MAR, and ReTok remain official-pretrained contextual rows.
- cross-tier numeric ranking remains forbidden.

The missing evidence is a terminal full-scale comparison built from the actual
matched training, generation, cost, VRAM, checkpoint, sample-set, evaluator,
and runtime reports.

### 4. Stable inference: implementation ready, unreleased

`GenerationSession`, integrity-checked EMA artifacts, completion-bound release
receipts, pre-deserialization policy checks, and terminal completion auditing are
implemented and covered by the clean `1077`-test Linux collection. There is no
release-authorized EMA artifact or release receipt because no model has passed
the full scientific gate. The API implementation is therefore ready, but a
usable released model is not.

### 5. Reproducible checkpoints: partially complete

The source-valid 50K checkpoints are:

```text
CoFiTok: 1,006,325,418 bytes
SHA256: ec7b9a0981f1d45420a9a86cdb80339d6d87b87fa77891c234db3d1b84376c2a

dense: 1,006,120,214 bytes
SHA256: 325da25f9fd228ab224abd977da7f9e1d000ba36e3e8136b55edae9c9f47c716
```

New-run integrity sidecars, exact-resume metrics reconciliation, runtime/Git/
dataset binding, retention audits, and terminal physical rehash are implemented.
The final requirement remains unproven because no physical step-300K checkpoint,
four-milestone set, release artifact, or terminal completion receipt exists.

## Immediate execution boundary

The next executable stage remains the base-128 full-data matched 100K quality
bridge at code target
`cf0e5faa94bf4ab38d947b921935b3b765b5537a`. It is blocked before deployment by
two independent conditions:

1. no explicit user execution approval exists for that exact target;
2. FieldScope PID `433140` is still the only GPU compute process, using about
   `15,412 MiB` with GPU utilization at `100%`.

The formal checkout remains tracked-clean at
`scale/generative-system@1ebcc15210e63a776a2ba448481cbd8bb94a4066`, the bridge
output root is absent, and no quality-bridge process exists. No process was
signaled or modified while producing this audit.

The required order remains:

1. obtain explicit approval for exact target `cf0e5fa`;
2. wait for unrelated GPU compute to disappear;
3. recheck the gate SHA, formal checkout, GPU idleness, and storage;
4. deploy only to a dedicated clean checkout and build immutable preparation,
   approval, runtime, storage, and launch receipts;
5. execute and verify the 100K matched bridge;
6. make a new scientific decision without auto-authorizing 300K;
7. require another explicit user decision before any 250M matched 300K launch.

All launch and release flags remain false.
