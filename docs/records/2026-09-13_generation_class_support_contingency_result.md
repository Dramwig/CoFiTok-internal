# Frozen class-support diagnostic: completed, not a generation pass

## Outcome

The separately authorized ResNet-50 evaluation of **20,000 existing images**
completed once. The exact independent validator rebuilt its statistics,
verified every prediction row and physical PNG binding, and created an adjacent
immutable receipt. Its September 13 replay passed without changing that receipt's
mtime (`1789204501873599445` ns). The terminal-SNR screen remains **hold**.

This evidence supports a narrower interpretation than the machine decision label
`shared_unconditional_support_collapse`: **weak detectable requested-label
association, no held-out relabeling rescue, and shared support concentration**.
It does not prove that conditioning is completely ignored, a common training-time
cause, an effective repair, or generation superiority.

## Exact execution and artifacts

The user's instruction `继续推进让结果有效支撑`, following the explanation and
approval request for this particular frozen-image diagnostic, was recorded as
contextual approval. `approval_context_20260912.json` explicitly records Codex as
the recorder on the user's behalf; it does not claim the user typed a preparation
hash. This approval covers only the fixed classifier and existing-image
statistics, not new samples, training, confirmation, or later stages.

The four separate preparation/authorization/evaluator/validator checkouts remain
clean at commit `c38291b14058746712dc0934937539a86bb3800e`, tree
`ae5d00d450cee99777ed1f663e92d179511204ed`. The evaluator root is
`/tmp/cofitok-class-support-contingency-evaluator-c38291b`; the validator root is
`/tmp/cofitok-class-support-contingency-validator-c38291b`.

Control artifacts are under
`/root/autodl-tmp/CoFiTok/checkpoints/generation/.class_support_contingency_v1.control/`:

| Artifact | Bytes | SHA256 |
| --- | ---: | --- |
| preparation.json | 19,469 | `4a1aebfcb2ee5b32586d97a2a1d39ec619048fcd99840fef7dfce9033dabc2bc` |
| approval_context_20260912.json | 1,167 | `3dcfc1eb15a28c67dab533e244fa56b5e2305f8d5f7e9289618a3d735250666a` |
| stage_approval.json | 1,277 | `3a79f1b34b3d75bcbdd7dad943f8fe43fba2ac35d37ebbfd3310682337ceb3ec` |
| execution_authorization.json | 20,432 | `2ee0bc6fece60404c54c3d505903ed166423d8c38f912091cb46635e23e8937c` |

The immutable output root is
`/root/autodl-tmp/CoFiTok/checkpoints/generation/class_support_contingency_v1/`:

| Artifact | Bytes | SHA256 |
| --- | ---: | --- |
| class_support_contingency_result.json | 2,535,030 | `9d1a5709a18f5beadf7a8654dd147906a269b35ce35ef0720a68ab2c484c0dc1` |
| class_support_contingency_result.validation.json | 2,209 | `6ad46a2cece75f17038ddb2c32fe08665db372909e728e956b34e02168855dae` |
| cofitok_predictions.jsonl | 4,521,151 | `b429c8672657d8065ffd66110e2d59b608a3860c9e5c1c0eef6411c650edd139` |
| dense_identity_predictions.jsonl | 4,524,855 | `90903aaf7620f65610f72914dccdac7856ab0cd6d5e4ede28fe1ef1a48302902` |

All above files have mode 0444. Small control files, the receipt, and a compact
independent arithmetic audit are archived at
`artifacts/reports/generation/class_support_contingency_2026-09-13/`.
Images, checkpoints, full prediction rows and the full result stay on pro6000.

The evaluation used the fixed ResNet-50 V2 checkpoint SHA
`11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca`, CUDA,
classifier batch 64, eight workers, torch 2.7.1+cu128, torchvision 0.22.1+cu128,
and scipy 1.15.3. Reported elapsed time was 31.1942434348166 seconds.
The observed evaluator handle exited zero. GPU was observed idle before/after;
the short-lived compute PID was **not** directly observed during inference.

This did not change the original sampling protocol: each tree has 10K images
from a frozen 100K EMA checkpoint, DDIM-100, seed 0, CFG 1.5, bf16, original
sampling batch 32. The classifier batch is not a resampling change.

## Findings and limits

| Diagnostic on frozen 100K samples | CoFiTok | Dense identity |
| --- | ---: | ---: |
| Top-1 correct / 10,000 | 24 (0.24%) | 17 (0.17%) |
| Top-5 correct / 10,000 | 106 (1.06%) | 81 (0.81%) |
| Adjusted mutual information (AMI) | 0.0039196470 | 0.0030333642 |
| AMI permutation p-value, 512 replicates | 0.0019493 | 0.0019493 |
| Held-out one-to-one mapping accuracy | 0.14% | 0.09% |
| Best cyclic offset / accuracy | 0 / 0.24% | 65 / 0.22% |
| Effective predicted class count | 218.97 | 178.01 |
| Mass in common top predicted class 885 | 8.54% | 7.70% |

Both AMIs are statistically distinguishable from the declared permutation null,
but both are below the predeclared practical floor 0.01. The frozen field
`dependence_detected=false` combines effect size and significance; it does **not**
mean independence was demonstrated. The global requested-label/seed assignment
also was not a randomized label-swap intervention at fixed noise.

An in-sample fitted label map is overfit (8.11% / 7.47%); held-out mapping does
not improve the original 0.24% / 0.17% accuracies. With only ten samples per
requested class, this rules out a demonstrated relabeling rescue, not every
possible mapping. The cyclic-offset search also fails its multiplicity-adjusted
significance and practical thresholds.

Histogram overlap is 0.6731 and Jensen-Shannon divergence is 0.1383668 bits.
Same-index predictions agree 1,441/10,000 times, versus a within-requested-class
permutation mean of 0.0271182. Both methods use the same seed/index noise stream;
that coupling makes agreement descriptive evidence, not proof of a common
training-time mechanism. There are no exact PNG duplicates within or across the
two frozen sets. Classifier labels remain a semantic proxy, not direct human
assessment of image quality.

The original binomial tails assume iid trials; balanced labels and correlated
outputs limit that interpretation. The operational validator recomputed the
512-replicate permutation nulls; the second observer independently recomputed
counts, entropy, MI/AMI, cyclic scores, held-out map application, overlap/JSD,
and duplicate counts, but did not rerun classifier inference or permutation
replicates. It uses scipy hypergeometric probabilities rather than the producer's
log-combinatorial expected-MI routine. AMI differences were below 2.5e-11.
This is the scope of the independent arithmetic check, not a second formal gate.

Observer command (read-only, no GPU or remote output writes):

```text
/root/autodl-tmp/conda/envs/pf-vlm/bin/python3.10 \
  /tmp/cofitok-class-support-readout-ff62fc4988b9.py \
  --root /root/autodl-tmp/CoFiTok/checkpoints/generation/class_support_contingency_v1
```

The tracked source is `scripts/audit_generation_class_support_readout.py`,
11,093 bytes, SHA256
`ff62fc4988b913d905648fc8f06c06ccd49def5988ec9bfea1c2086d10b8fef3`.
The archived observer JSON SHA256 is
`e4412e007f3cc10a78492f32b9dd9e17aaf86772e06dd7c0b553c617b1274482`.

## Next discriminator: loss gradients, not another label-mapping run

Source inspection identified two concrete properties worth measuring before
selecting another shared intervention. Four new CPU characterization cases use
the production conditioning and consistency helpers:

- Identical student/EMA weights can yield positive teacher loss when the student
  drops a class label but the eval-mode teacher uses the original label. In the
  isolated conditioning toy, loss is 1 and null-embedding gradient is -2; with
  dropout disabled both are zero. This can also be intentional consistency
  regularization, so the test is not sufficient to call the recipe defective.
- In two strictly saturated clipped-x0 rollout steps, loss can be 1 while the
  relevant prediction-parameter gradient is zero. The corresponding epsilon
  objective gives nonzero gradient. A boundary-touching toy was corrected to use
  strictly saturated values; no production code or tolerance was changed.

These are possibility proofs on toys, not measurements of real-checkpoint
prevalence or causality. The three inspected production files are byte-identical
to `89bcd9a`: `training/rollout.py`, `models/scalable_unet.py`, and
`diffusion/schedule.py`. Crucially, the terminal screen stops at 10K, before
teacher activation at 30K. EMA teacher behavior cannot be the sole explanation
of collapse already present at that screen endpoint.

The next bounded diagnostic is specified in
`docs/experiment_conditions/frozen_loss_gradient_attribution_v1.md`. It must
measure actual weighted-gradient contributions and clipping prevalence using
frozen checkpoints and held-out images before any repair/pilot is selected.
Its source-bound preparation, executable, exact authorization, and execution
are **not yet complete**; the 20K classifier approval cannot be reused for it.

## Verification and unchanged gates

The observer tests, four new semantic characterizations, existing rollout tests,
contingency tests, and label-pipeline tests pass: **50 passed**. Production
training/model/sampling code is unchanged. This targeted suite does not claim
the entire repository was rerun.

The four terminal-SNR arm reports were physically replayed again. All 201-row
metrics are finite, strictly increasing, end at 10K, and satisfy
`samples_seen=step*64`. All eight 5K/10K checkpoints match sidecars; arm replay
checks latest/report, sampling manifests/progress/digests, distribution,
class-fidelity, checkpoint diagnostics and rollout reports. Result and adjacent
receipt replay pass. All six pinned prelaunch SHA256 bindings match.

The immutable screen still fails CoFiTok relative FID improvement
`-0.11363547384116164` and dense `-0.09408838680587096`, versus required `>=0.05`;
all four screen recalls remain zero. These are distinct from the older frozen
100K recalls 0.00832 / 0.01000 analyzed above. Exact execution/validator Git
identities, controller status/log and result/receipt hashes remain unchanged.
There is no screen/controller/evaluator process left; GPU is idle. The first
September 13 host check reported 189,696,655,360 free filesystem bytes.

No confirmation, full-training/300K, promotion, export, release, or paper
integration was authorized. The historical exposure controller, current screen,
and locked paper evidence were not modified. The data-validation skill prompted
the explicit distinction between significance, practical effect size and causal
support, as well as the independent arithmetic check.
