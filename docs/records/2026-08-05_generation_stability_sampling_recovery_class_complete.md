# Stability sampling recovery class-complete evidence

Date: 2026-08-05

## Superseded by the distribution-support candidate

Revision `aed68853914dc0ecf4b3f15d5d090ddee740d1df` remains the historical
class-coverage and metrics-window correction, but it must not be executed. Its
10K confirmation could pass on FID alone despite collapsed precision or
recall. The exact replacement is
`11d8f954030915f1d8848594683ac70518ce39bc`, documented in
`2026-08-05_generation_stability_sampling_confirmation_distribution_support.md`.

## Outcome

The frozen sampling-recovery selector and its dormant 10K confirmation now
form an executable, source-bound evidence chain rather than a plan-only one.
The exact code candidate is:

```text
development branch: scale/generation-stability-sampling-recovery-v1
execution branch identity: scale/generation-large-capacity
revision: aed68853914dc0ecf4b3f15d5d090ddee740d1df
tree: b8d7db6f550e1ed924aa8b93816ebb4b26c0847b
parent: f27dba44cd8708e8dfb58b44538dd7de0dbbae58
```

This candidate supersedes `0a1b88630e2b24e27b3f084dfc1490381e906655`
and all earlier sampling candidates. None has been deployed to the formal
checkout or authorized for execution.

## Corrected scientific and evidence contract

The former 512-sample selector used global indices `[0,512)`. The sampler maps
labels as `global_index mod num_classes`, so that interval covered only classes
0--511 of ImageNet-1000. It was matched across methods but was not a
class-complete protocol selector.

The replacement fixes:

```text
selection samples per method/case: 1000
selection global indices:           [0, 1000)
selection class coverage:           exactly one sample per each of 1000 classes
frozen formal global indices:       [0, 10000)
confirmation global indices:        [10000, 20000)
confirmation class coverage:        exactly ten samples per each class
weights:                             EMA
sampler:                             DDIM-100
```

The plan validator requires exact sample, class, sampling-batch, metrics-batch,
metrics-seed, CFG, precision, EMA, and PRC-disabled selector fields. The
sampler now records its physical checkpoint configuration's `num_classes` in
the immutable sampling manifest/report. Recovery and confirmation builders
require that value to be exactly `1000`, bind the corresponding preflight
protocol, and reject sampling or metrics drift.

The prior confirmation runbook correctly declared the disjoint interval
`[10000,20000)`, but `evaluate_generation_metrics.py` still rejected every
sample set whose `start_index` was nonzero. The evaluator now verifies the
complete declared global-index filename window. For confirmation this is
`010000.png` through `019999.png`; no renumbering to zero is allowed. Existing
formal zero-based reports remain valid.

This keeps selection and confirmation statistically independent while
retaining identical seeds, labels, CFG cases, EMA weights, and random-stream
semantics for CoFiTok and dense identity. Following the `ml-training-recipes`
EMA/checkpoint guidance, the change preserves EMA as the only inference weight
path and keeps checkpoint bytes, integrity sidecar, Git revision, runtime, and
sample-set SHA256 inside the trust boundary.

## Local verification

At the exact code candidate:

- recovery/confirmation/metrics/runbook focused suite: `67 passed`;
- full repository excluding four parent-layout paper tests:
  `1,130 collected / 1,124 passed / 6 skipped / 0 failed`;
- normalized paper-test Git blob:
  `3fe62a15e0df581e9aacc3e8130b0792e142b36c`, followed by `4/4` passes in
  the real sibling `paper/` layout;
- five changed CLI entrypoints passed Python compilation and `--help`;
- the physical plan parsed and validated, and `git diff --check` passed.

## Isolated Linux rehearsal

The prerequisite-aware bundle was:

```text
bytes: 40197168
sha256: a4d70751584392af9c4529b47c247f2beaefb206470fac134ce487d9cd5d930e
advertised refs: one, HEAD=aed68853914dc0ecf4b3f15d5d090ddee740d1df
prerequisites: 1ebcc15210e63a776a2ba448481cbd8bb94a4066,
               58d83bfce2770eab2565b8c89a5f9a06201a0c86
```

In the temporary checkout
`/tmp/cofitok-sampling-recovery-class-complete-aed68853914d/CoFiTok-internal`,
under `scale/generation-large-capacity` with `CUDA_VISIBLE_DEVICES` empty:

- revision/tree/branch were exact and the checkout remained clean;
- `torch.cuda.is_available()==false` and CUDA device count was zero;
- both runbooks passed native Linux `bash -n`;
- all five changed Python entrypoints compiled and passed `--help`;
- the same focused suite passed `67/67`;
- direct replay produced exactly one occurrence of every class in the
  1,000-sample selector and all four global-index/derived-seed disjointness
  checks were true;
- post-test porcelain SHA256 was the empty digest
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

The remote temporary checkout and bundle were removed after identity checks.
The formal checkout remained
`scale/generative-system@1ebcc15210e63a776a2ba448481cbd8bb94a4066`, with
tracked status empty. Its pre-existing untracked set was neither read as
source nor modified; full porcelain SHA256 remained
`774cfad9f88f6e4fa953556e466b02b16212184930ca9fcbf4e8c233ab1cb79c`
before and after rehearsal.

The machine-readable receipt is:

```text
artifacts/reports/generation/sampling_recovery_class_complete_rehearsal_2026-08-05.json
```

## Live and authorization boundary

The frozen gate remains SHA256
`2f763913d5a07ba5563f5bc75548780678fcdf9da4e119e03262f10daf84dd90`,
`fail/hold`, with sole failed gate `absolute_fid_quality`; CoFiTok/dense FID is
`138.29702495267782 / 151.4476773464495`. The recovery, confirmation, and
full-data 100K bridge output roots remain absent.

Unrelated FieldScope PID `433140` remains active from
`/root/autodl-tmp/FieldScope/FieldScope-internal`, using approximately
`15,412 MiB`. It was not modified or signaled. Available generation-filesystem
storage was `298,641,821,696` bytes.

No approval sentinel was created. No recovery, confirmation, training, 100K
bridge, release, or full-300K stage was launched. A future recovery execution
requires all of an idle GPU, this exact code revision in a separate clean
checkout, `SAMPLING_RECOVERY_EXECUTION_ALLOWED=true`, and a user-created
sentinel with exact text:

```text
Approve the non-authorizing matched 1000-sample sampling-recovery diagnostic only.
```

That approval would authorize only the bounded selector. A later 10K
confirmation still requires its own approval bound to the completed physical
recovery summary. Neither stage replaces the frozen gate, authorizes training,
or authorizes full 300K.
