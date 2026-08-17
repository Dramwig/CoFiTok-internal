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
report. The repeated-stream summary also requires a distinct verified execution
manifest on every input audit; missing, malformed, or duplicate manifest
identities are rejected before any repeated-direction conclusion is built.

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
21 targeted uncertainty tests passed
111 uncertainty + generation-metrics + sampling-confirmation + distribution-support tests passed
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

Source-bound execution hardening identity:

```text
revision: d6e822ead81c93f9feb96c5f226efacaa10e10a0
tree: de332ac56fdac7cfe87fafaa4e7b33baef0f984f
incremental bundle bytes: 48,263,720
incremental bundle SHA256: f6131697300240351d786aa6fcf549cbef8a82825e038733d3fea482db669ee4
bundle prerequisites:
  1ebcc15210e63a776a2ba448481cbd8bb94a4066
  58d83bfce2770eab2565b8c89a5f9a06201a0c86
```

Repeated-summary manifest enforcement identity:

```text
revision: 1c8ef207cb6d79850d73a45abc345fc421e6aa7f
tree: a09f14a0eca44af6db8e6781863a463163ddf646
incremental bundle bytes: 3,004
incremental bundle SHA256: 25ae45be5f597540a7b623c6ac21e204cec3444263b26554782fcd5d50106ef1
bundle prerequisite: d6e822ead81c93f9feb96c5f226efacaa10e10a0
```

The bundle was verified on `pro6000` and fetched only into a disposable isolated
checkout under `/tmp`. With CUDA hidden and `OMP_NUM_THREADS=1` /
`MKL_NUM_THREADS=1`, the same 105-test group passed on Linux. Python compilation
and `git diff --check` passed, and the isolated checkout remained tracked-clean.
The rehearsal did not modify the formal checkout, observer checkout, active
training checkout, checkpoints, samples, or GPU process.

The source-bound hardening bundle was separately verified against the formal
repository and checked out detached under:

```text
/tmp/cofitok-uncertainty-manifest-rehearsal-mRqd3i/CoFiTok-internal
```

With CUDA hidden and one CPU thread per BLAS runtime, the expanded related suite
passed `110/110`, Python compilation passed, and the detached checkout remained
tracked-clean. Both checked-in manifests then passed their small-source
bytes/SHA256 preflight against the live server:

```text
formal manifest SHA256:
2561b76113eb81c4288694302cc7fd683b806fad6fb420c2a0f1336b0ea3faac

confirmation manifest SHA256:
e13518e421970fc7c2fe9ce12d22cb6636f77b421ba5af1b355e5623692a510a
```

The dedicated uncertainty output root was still absent after rehearsal. The
formal checkout remained at `1ebcc15210e63a776a2ba448481cbd8bb94a4066`, and
the active quality-bridge checkout remained at
`cf0e5faa94bf4ab38d947b921935b3b765b5537a`.

The same detached rehearsal checkout was then advanced only through the
3,004-byte incremental summary-hardening bundle. At revision `1c8ef20`, the
expanded Linux suite passed `111/111`, both source-report manifest preflights
passed again, Python compilation passed, and tracked status remained empty.
The formal and active training checkout revisions remained unchanged.

## Execution boundary

No feature extraction or new GPU evaluation was launched while the full-data
100K quality bridge was training. The audit should be executed from a clean,
isolated Linux checkout after a safe evaluator slot is available. It must write
new reports and caches under a dedicated generation report root and must not
alter either frozen sample set or the active training checkout.

## Existing feature-cache inventory

A read-only server inventory was completed while the quality bridge retained
exclusive use of the GPU. Machine-readable evidence is stored at:

```text
artifacts/reports/generation/matched_uncertainty_cache_inventory_2026-08-17.json
SHA256: 7e310cdc8a9dd3cb9dae1819ab3ce909767259a8fe227c0ede321cb889dce941
```

The scan found exactly three `inception-v3-compat/2048` feature files under the
generation root. All three are copies of the same ImageNet-256 50K validation
feature matrix:

```text
bytes: 409,601,577
SHA256: 20103588dca9ce47bfceef6b68b473fdf4be720f149d1b8bdd96341d27c10dcd
real-set tree SHA256: 19ace4e37bee2fcaeac2b7cbd26ed785aa0e6e01015c90b4955027d163f6e44f
```

The three files have distinct inodes and are not hardlinks, but their SHA256
values are identical. No CoFiTok or dense generated `features-2048.pt` cache was
present anywhere below `/root/autodl-tmp/CoFiTok/checkpoints/generation`, and
the dedicated uncertainty output root remained absent.

Consequently the eventual audit may cryptographically reuse one exact real-set
feature file, but it must still extract both generated feature matrices for
each stream. This inventory is not an uncertainty result and cannot support a
relative-generation claim by itself. No remote file, sample tree, checkout, or
GPU process was modified during the inspection.
