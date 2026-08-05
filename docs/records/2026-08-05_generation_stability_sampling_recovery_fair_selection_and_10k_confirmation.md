# Stability sampling-recovery fair selection and matched 10K confirmation

Date: 2026-08-05

## Outcome

The frozen 50K sampling-recovery path now has a predeclared fair selection
rule and a dormant, source-bound matched 10K confirmation stage. Neither stage
can change the frozen promotion gate, issue a launch receipt, or authorize full
300K training.

The live scientific state is unchanged:

- frozen CoFiTok FID: `138.29702495267782`;
- frozen dense-identity FID: `151.4476773464495`;
- required absolute CoFiTok FID: `<= 100.0`;
- promotion decision: `hold`;
- `full_training_launch_allowed=false`.

## Shared matched selection policy

The first recovery builder draft reported each method's lowest diagnostic FID
separately. Those rankings are useful exploratory information, but selecting a
different CFG protocol for CoFiTok and dense would break the matched formal
comparison.

The recovery plan and builder now require one shared protocol selected by the
following predeclared policy:

```text
baseline case: cfg150_r000
objective: minimize the worst method's FID ratio to that shared baseline
eligibility: a non-baseline case must strictly improve both methods
tie-break: worst ratio, prefer formal baseline, mean ratio, case id
per-method formal protocol selection: forbidden
automatic formal protocol change: forbidden
```

The comparison is entirely within the same 512-sample sweep. It does not
compare small-sample FID numerically against the frozen 10K FID. If no
non-baseline case strictly improves both methods, the formal CFG-1.5/rescale-0
baseline remains selected and no new 10K confirmation is eligible.

Per-method rankings remain in the report with the explicit field:

```text
per_method_protocol_selection_allowed=false
```

The recovery summary is also deterministic: it no longer embeds wall-clock
creation time. Rebuilding from the same plan, reports, Git identity, and ten
physical cases produces the same content and SHA256.

## Dormant matched 10K confirmation

Builder:

```text
scripts/build_generation_stability_sampling_confirmation.py
```

Runbook:

```text
artifacts/runbooks/generation_stability_frozen_50k_sampling_confirmation_10k.sh
```

The confirmation preflight requires:

- the recovery and confirmation code to use the same exact clean Git revision
  and branch;
- a completed recovery summary whose shared selection is non-baseline and
  eligible;
- an independent rebuild of the recovery summary from all ten physical
  sampling and metrics reports;
- unchanged frozen promotion-gate, formal-metrics, real-set, checkpoint, and
  integrity-sidecar identities;
- one shared CFG/rescale case for both methods;
- exact DDIM-100, EMA, bf16, balanced-modulo classes, seed/start index, and
  batch-invariant per-sample random stream.

Clean checkout relocation is allowed only when all content identities remain
exact. Absolute source paths inside content-addressed evidence do not by
themselves invalidate an otherwise identical clean checkout.

The runbook writes to the independent root:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_scaling_50k_sampling_confirmation_10k_v1
```

For both CoFiTok and dense it generates 10,000 samples with the same selected
protocol and evaluates full FID, Inception Score, precision, and recall. It
does not use `--skip-prc`. Before each GPU stage it checks live compute
processes and exits `75` without launching when the GPU is busy.

## Confirmation decision boundary

The confirmation report can return a non-authorizing quality `pass` only when:

- CoFiTok FID meets the frozen absolute threshold;
- CoFiTok remains within the frozen relative tolerance against dense under the
  same candidate protocol;
- both methods strictly improve their own frozen formal FID.

Even that pass means only:

```text
candidate_quality_confirmed_non_authorizing
```

It explicitly retains:

```text
candidate_protocol_formalized=false
confirmation_report_is_promotion_gate=false
new_gate_required=true
full_training_launch_allowed=false
full_300k_launch_allowed=false
```

A later schema-current gate would have to ingest and independently validate the
confirmation source. This work does not create that gate, bridge, receipt, or
launch path.

## Verification

```text
targeted recovery/confirmation/runbook tests: 26 passed
both runbooks streamed to pro6000 bash -n: pass
full repository suite: 1086 passed, 6 skipped
git diff --check: pass
```

At the last live read, unrelated FieldScope PID `433140` still used about
`15412 MiB` with 100% GPU utilization. No recovery sweep, matched 10K
confirmation, training process, waiter replacement, or other-project process
was launched or modified.
