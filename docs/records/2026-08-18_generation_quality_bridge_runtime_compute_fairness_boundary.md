# Full-data quality-bridge runtime and compute fairness boundary

Date: 2026-08-18

## Outcome

The active full-data ImageNet-256 100K quality bridge has a source-bound,
parameter-matched runtime contract, and the currently running CoFiTok method is
physically observed using the selected runtime. The final observed CoFiTok/dense
runtime and compute comparison is not yet available because dense training has
not started.

The exact current distinction is:

- **contract parity is proven**;
- **CoFiTok actual runtime is observed**;
- **dense actual runtime parity is pending**;
- **final elapsed time, throughput, and VRAM comparison is pending**;
- **no training-speed or memory advantage is established**.

## Matched launch contract

The immutable launch receipt selected:

```text
micro batch: 64
gradient accumulation: 1
effective batch: 64
runtime environment SHA256:
d5bfcd085ea467ee5d24dfccc6e147da06dd7a0a0efdcdab355882b8547c985e
```

Receipt identity:

```text
bytes: 8,588
SHA256: 4a9fd8577c24a92583d9538846dcab35c2d73e6066d164050f49b976985a3a21
```

The active runbook reads this pair once and passes the same variables through
the shared `train_to_milestone` function for both CoFiTok and dense. It also
requires both terminal training reports to validate against the same selected
micro-batch and accumulation values before running the matched-pair validator.

Runbook identity:

```text
bytes: 27,943
SHA256: f0531763b4b888964a75923a8d59cb8cf8d1c79bd7782f845006c39e1a57fe73
```

The config validation passed with no mismatches. Physical parameter counts are:

```text
CoFiTok:        62,834,083
dense_identity: 62,824,707
relative gap:   +0.0149240648%
```

## Observed runtime

The CoFiTok run manifest records the resolved overrides:

```text
data.batch_size: 64
optimization.gradient_accumulation_steps: 1
effective batch: 64
```

The source-bound 31K schedule audit independently replayed this run manifest and
verified `samples_seen == step * 64` across the canonical metrics prefix. Its
report identity is:

```text
bytes: 31,913
SHA256: f044ab66d45f3c585126c3c375753f98bf5d129ba063f7efd557523ff437051e
```

Dense had not started at this audit boundary. Its resolved training report does
not yet exist, so observed runtime parity must not be inferred solely from the
runbook contract.

## Recovery compute that must remain visible

The pin-memory recovery rolled CoFiTok back from step 20,200 to the trusted step
20,000 checkpoint. Canonical metrics correctly describe the exact resumed
trajectory, but the discarded physical GPU work still counts toward cost:

```text
orphaned physical compute: 560.6116147041321 seconds
orphaned optimizer steps: 200
orphaned images: 12,800
```

The final CoFiTok physical lower-bound cost is therefore:

```text
training_report.elapsed_seconds + 560.6116147041321
```

Throughput must be recomputed from this adjusted elapsed time. Adjustment
identity:

```text
SHA256: 7e4c2362ee38b4b973034dbe492a8ac64044c17f805e6f3cd66b7421c48af740
```

If dense later resumes over orphaned canonical rows, dense must receive its own
source-bound adjustment. A zero adjustment may be used only when its final
training report and run directory prove that no orphan archive exists.

## Terminal pipeline gap

The active quality-bridge runbook already validates exact completion, actual
runtime overrides, effective batch, Git identity, checkpoint binding, and the
matched training-pair contract. However, the current quality-bridge result does
not carry training elapsed time, corrected throughput, peak VRAM, or a
resume-compute adjustment identity.

The audit-only hardening at commit
`2b3c3975255f92cc9613f9990592f37ba2cfa968` makes later generation gates,
comparison reports, and completion audits fail closed when recovery compute is
omitted. It is not invoked by the active quality-bridge terminal runbook itself.
No dedicated terminal runtime/compute fairness waiter was present on the server
at this audit boundary.

Therefore a CPU-only terminal waiter is required. It must wait for both 100K
training reports, replay their exact runtime identity, include any method-specific
recovery adjustment, and publish corrected training hours, images, throughput,
and peak VRAM without making a performance claim in advance.

## Claim boundary

This audit establishes matched configuration and launch-runtime parity, plus
observed CoFiTok `64x1` execution. It does not establish observed dense parity,
training-speed advantage, memory advantage, sample quality, promotion readiness,
release readiness, or authority for full 300K training.

The compact evidence receipt is:

```text
artifacts/reports/generation/quality_bridge_runtime_compute_fairness_boundary_2026-08-18.json
bytes: 7,561
SHA256: 98693b85025e790e24eae938980a17a2dce54ff9bb9504d2ab44193579ebef1c
Git blob OID: aa3202f1267e0610a87c6caf319fd8a186dcad1c
```
