# Capacity-screen hold recovery (2026-09-07)

## Scope

This change adds an immutable validation receipt for the completed four-arm
capacity screen and a source-bound, non-authorizing synthesis of the terminal
evidence after that screen held. It does not change a model, objective,
training or sampling protocol, scientific threshold, or locked result. It does
not authorize training, sampling, evaluation, capacity confirmation, full
300K, promotion, export, release, or process signals.

The implementation was developed in the isolated worktree
`C:/cofitok-capacity-hold-recovery-20260907`, based on
`b15d979d8c03d50f29d1e6e460d3ce3988152fbe`. The main dirty worktree and the
remote authoritative experiment checkout were not modified.

## Authoritative evidence

The exposure continuation is bound to revision
`5c23141a24a3385101ed6081c1ed656aad955a99`, tree
`ffe66d2d2fc666884a21856bbc84a23fe667f88b`. Both methods completed 110K;
each metrics stream has 201 strictly increasing finite rows from 100001 to
110000 and satisfies `samples_seen = step * 64`. The final checkpoint SHA256
values are:

- CoFiTok: `994fc9ceae8cd24a67d9f35d0c4f565b74cee55b231b8d823c5908518e7620c5`.
- dense identity: `5a5bfc1d794dabfa96ee27ee9c092a748b22fa91245c9f6ba1e2741053acf28d`.

The recovered exposure result, adjacent validation receipt, scientific
decision, and decision validation have SHA256 values
`d35995f503a395b047f88c979fc3c9773eff9132b7cabc751016ff5a0dc726db`,
`8ca77a8d67f903cd04b6e03ab91750b0d33f00e95bfc2f0939b893f37835acd8`,
`f7af487108d1cfcd26d27887b9dbc0a97cbf00cf22564b5183b3345eede131c1`,
and `6bc62c2f1cfb2c90ae0f6bd1e9874d466d9456f7a635551ef208cdb039b3e989`.
The controller's historical failed status is retained: its old result builder
incorrectly required the method-specific runtime fingerprints to be equal.
The later immutable replay does not overwrite that history.

The capacity screen is bound to revision
`3370e68727259fb9144216d5c50ec9262645dc86`, tree
`5dc3e72beb969e574aec60e652f761602c2ae31b`. All four arms completed the
authorized 10K training boundary. Each metrics stream has 201 strictly
increasing finite rows from 1 to 10000 and satisfies
`samples_seen = step * 64`. The partial-stage reports intentionally preserve
`target_steps=100000` and `training_complete=false`; they are not evidence for
continuation beyond 10K. The capacity result SHA256 is
`1ec16a2ac7dbf247f04be6b14bd980bc85a6f933791b37964767b1fea9998e45`.
It reports `operational_status=pass` and `scientific_status=hold`.

The predeclared failures are all-zero recall improvement for both methods,
CoFiTok precision delta `-0.14499998092651367` against a `-0.05` floor, and
dense relative FID change `+0.029602188156230724` against a strict improvement
threshold. The 1K-per-arm recall estimate has limited directional power, but
that limitation cannot override the independent precision and FID failures.
No threshold was relaxed.

The sampling-only recovery result SHA256
`d6b5c3190a205f5be886c01e8543b9e2511988d5f0b9f2e642a33d0e8be1249f`
selects `no_shared_sampling_recovery_candidate`. The matched Min-SNR result
SHA256 `dde62640e84a461be3b571b228e7903785e42f118dd7103941e20592bd0059fc`
and physical guard SHA256
`3ff6ad8314e0ea9e10c88aee785564d6713d2e93ddfd9f83f33e1ed33f0b99ed`
select `no_shared_min_snr_candidate_at_50k` and preserve a terminal hold.

All six matched free-sampling rollout reports begin at `t=999`. Their first
raw-x0 clipping fractions exceed 0.99. This is recorded only as a common
terminal low-SNR or distribution-support observation, not as a proven causal
root cause.

## Decision contract

The resulting decision is fixed to:

```text
decision=hold_for_source_bound_training_objective_reassessment
selected_intervention=null
preparation_allowed=false
gpu_execution_allowed=false
full_training_launch_allowed=false
full_300k_launch_allowed=false
```

A future experiment requires a separate source-bound objective reassessment
that selects exactly one bounded matched intervention. Only after that new
decision may separate preparation, stage/execution authorization, live GPU,
runtime and storage evidence, and a launch receipt be considered.

## Verification

The initial focused local suite reported 16 passed. After the missing-core-denial
guard was added, the complete capacity/exposure/sampling-recovery focused set
passed. The full local suite reported `1207 passed, 12 skipped in 419.18s`.
Compilation of `src`, `scripts`, and `tests`, `git diff --check`, all `110/110`
tracked runbook `bash -n` checks, and all three changed/new CLI `--help` imports
passed.

On the live server, the six final checkpoint payloads were physically rehashed.
Every byte count and SHA256 matched `latest.json`; each adjacent integrity
sidecar was exactly the corresponding `latest.json` object without its
`integrity_manifest` pointer; and every training report embedded the exact
`latest.json` object. At the snapshot boundary, no matching training, sampling,
or evaluation process owned the GPU, GPU memory use was 0 MiB, and
`/root/autodl-tmp` had more than 640 GB free.

## Linux rehearsal and immutable outputs

The implementation commit is
`302f9213e3e04adb4e76eae3f71948322c440995`, tree
`c48c5a59f9198e31fc655e80a14a1c3df957fbbf`, on branch
`analysis/generation-capacity-screen-hold-recovery-v1-20260907`. Its parent is
the clean exposure-result recovery revision
`b15d979d8c03d50f29d1e6e460d3ce3988152fbe`. The first incremental bundle was
25,445 bytes with SHA256
`974c1d51e551391a5166b249a8ea67cf54c6132758aef242eca3f367b1a0207c`;
it advertised only the implementation branch head and required exactly
`b15d979`. The authoritative main repository correctly rejected that missing
prerequisite. The exact clean remote recovery checkout at `b15d979` accepted
the bundle, and only then was the isolated checkout
`/tmp/cofitok-capacity-hold-recovery-302f921` created. Neither the source nor
the formal repository moved.

The isolated Linux checkout reported `torch.cuda.is_available()==false` and
completed `1213 passed, 6 skipped in 245.60s`. Its JUnit SHA256 is
`44fd935d248bab4672bee4ec8a64a8d0b19f755be94c35150f24abea939cbe24`.
Compilation, all `110/110` runbook syntax checks, the three CLI imports, and
tracked-clean verification passed.

The exact clean `3370e68` producer checkout first replayed the existing
capacity result byte-for-byte and returned `status=pass`. The new validator
then repeated that physical replay with `--project-root` still pointing to the
exact producer checkout and created the adjacent immutable receipt:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/capacity_qualification_v1/capacity_screen_result.validation.json
bytes=3311
sha256=213685d6d6f38232c83272250d0d211cfdc2f390a38b79f1f3bee0ab383b44ba
```

The receipt records producer Git `3370e68.../5dc3e72...` and validator Git
`302f921.../c48c5a5...`, preserves `scientific_status=hold`, and has no enabled
authorization boundary.

The joint source replay then created and validated:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/capacity_qualification_v1/capacity_screen_hold_recovery.json
bytes=10905
sha256=585840359db244c51cbfddc6c425f8703ea8ed1b12557fbfc9b6b441812ae013

/root/autodl-tmp/CoFiTok/checkpoints/generation/capacity_qualification_v1/capacity_screen_hold_recovery.validation.json
bytes=8513
sha256=79baa341bbf9017d53775af3dc6c2c14405dbfbb3b80c4332400bb26bed15880
```

The verifier physically replayed all eleven source groups: both exposure
artifacts and validations, capacity result and validation, four capacity arm
validations, sampling recovery, Min-SNR result and guard, and six rollout
reports. A second validation returned the same identities; all three new file
mtimes were unchanged, demonstrating the existing-receipt path was read-only.

At final evidence capture the GPU had 0 MiB allocated, no compute application
was present, and no experiment trainer, sampler, or evaluator remained. The
isolated checkout occupied about 359 MB while the project filesystem retained
more than 500 GB free. No process was signaled, no threshold or locked result
was changed, and no GPU stage was launched.
