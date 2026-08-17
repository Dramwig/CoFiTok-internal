# 2026-08-17 Generation matched uncertainty audit

## Motivation

The frozen 10%-data stability pair currently has two matched 10K observations in
the same direction:

- frozen formal stream: CoFiTok FID `138.2970`, dense FID `151.4477`;
- disjoint confirmation stream: CoFiTok FID `138.9917`, dense FID `150.5660`.

These point estimates are encouraging, but neither report contains a
generation-level confidence interval. Repeated 2048-dimensional FID bootstrap
covariance square roots would be unnecessarily expensive and would still need a
careful paired design.

## Added audit

`scripts/audit_generation_matched_uncertainty.py` adds a permanently
non-authorizing paired uncertainty report. It:

1. revalidates both physical PNG sets, their immutable sampling manifests,
   progress reports, sample-set SHA256 values, checkpoint identities, and bound
   FID reports;
2. requires exact CoFiTok/dense sampling equality except for the expected prefix
   budget (`K=8` versus the dense direct head);
3. extracts the same torch-fidelity `inception-v3-compat/2048` features used by
   the existing FID evaluator, with content-addressed feature caches;
4. partitions a 10K matched stream into 20 disjoint 500-sample generated blocks;
5. pairs every generated block with five disjoint 500-sample real folds, so the
   default ImageNet-256 audit uses all 50K validation images exactly once;
6. computes the paired unbiased polynomial-KID difference. The shared real-only
   KID term cancels algebraically, so only the two candidate self terms and two
   candidate-real cross terms are required;
7. reports a paired block-bootstrap 95% interval and an exact one-sided sign
   test over the 20 disjoint block differences.

The scoped relative advantage passes only when:

- the bound FID point estimate is lower for CoFiTok;
- the paired block-bootstrap upper bound for `CoFiTok - dense` is below zero;
- the exact one-sided sign-test p-value is at most `0.05`.

`scripts/build_generation_matched_uncertainty_summary.py` combines two or more
disjoint global-index windows without pooling their confidence intervals. It
distinguishes exact-protocol replication from cross-protocol repetition. The
existing frozen stream (`guidance_rescale=0`) and confirmation stream
(`guidance_rescale=1`) therefore can support only a repeated cross-protocol
relative direction, not exact-protocol replication.

## Claim boundary

Both reports state that they:

- do not replace FID point estimates or the frozen promotion gate;
- do not repair poor absolute FID, recall, class fidelity, or visual quality;
- do not authorize additional training, full 300K, or release;
- do not support a broad generation-superiority or SOTA claim.

The intended claim is limited to uncertainty around the relative matched
CoFiTok-versus-dense direction for the exact bound checkpoints, sample windows,
and protocols.

## Source-bound execution manifests

The two existing 10K streams are now frozen by separate schema-v1 execution
manifests:

```text
configs/generation/diagnostics/matched_uncertainty_formal_10k_execution_v1.json
configs/generation/diagnostics/matched_uncertainty_confirmation_10k_execution_v1.json
```

Each manifest binds all seven input paths, the four small source-report
bytes/SHA256 identities, the real-set tree identity, both sample-set and
checkpoint SHA256 values, checkpoint step, matched global-index window, complete
sampling-signature SHA256, FID point estimates, evaluator runtime identity,
output path, cache root, and all uncertainty parameters. The formal stream is
bound to `[0, 10000)` with `guidance_rescale=0`; the confirmation stream is bound
to `[10000, 20000)` with `guidance_rescale=1`. Their windows are therefore
physically disjoint, while their protocol difference remains explicit.

`audit_generation_matched_uncertainty.py` accepts a manifest-only invocation and
hydrates its scientific arguments from that immutable source. Before feature
extraction, it fails closed if a CLI override, report path, report byte count,
report SHA256, physical real/sample tree, checkpoint identity, global-index
window, sampling signature, FID source value, or evaluator runtime identity has
drifted. The verified manifest identity is embedded in the resulting audit
report.

The eventual safe-slot commands are intentionally minimal:

```bash
python scripts/audit_generation_matched_uncertainty.py \
  --execution-manifest configs/generation/diagnostics/matched_uncertainty_formal_10k_execution_v1.json

python scripts/audit_generation_matched_uncertainty.py \
  --execution-manifest configs/generation/diagnostics/matched_uncertainty_confirmation_10k_execution_v1.json
```

They must still run from the exact clean evaluator checkout after confirming a
safe evaluator slot. A manifest pass is an input-integrity precondition, not GPU
authorization and not a scientific advantage result.

## Validation

The implementation lives in the isolated worktree:

```text
C:/qbstats
branch: analysis/generation-matched-uncertainty-v1
base: 24cce1ee3b464b0b46776fb7e2c9f56cec514f4b
```

Current local validation:

```text
20 targeted uncertainty tests passed
110 uncertainty + generation-metrics + sampling-confirmation + distribution-support tests passed
Python compile passed
git diff --check passed
```

The tests include a direct numerical comparison with torch-fidelity 0.4.0,
verification of the shared-reference cancellation identity, a strong synthetic
advantage case, and an identical-method negative control that must remain on
hold.

Locked code identity:

```text
revision: f70a15d0ba992e63368cddd24c4f2022b15697c6
tree: a2613f2db4e18a7b9332759810e75749400c7077
incremental bundle bytes: 22,085
incremental bundle SHA256: 02acfdfc99a551168a28702df36b18796e75d77d31a4e7fa4bec15d0ba585c9c
bundle prerequisite: 1ff6bb3db932ef9e43ddfafc067db779dab797d9
```

The bundle was verified on `pro6000` and fetched only into a disposable isolated
checkout under `/tmp`. With CUDA hidden and `OMP_NUM_THREADS=1` /
`MKL_NUM_THREADS=1`, the same 105-test group passed on Linux. Python compilation
and `git diff --check` passed, and the isolated checkout remained tracked-clean.
The rehearsal did not modify the formal checkout, observer checkout, active
training checkout, checkpoints, samples, or GPU process.

## Execution boundary

No feature extraction or new GPU evaluation was launched while the full-data
100K quality bridge was training. The audit should be executed from a clean,
isolated Linux checkout after a safe evaluator slot is available. It must write
new reports and caches under a dedicated generation report root and must not
alter either frozen sample set or the active training checkout.
