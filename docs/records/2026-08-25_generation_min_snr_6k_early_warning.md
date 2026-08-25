# Matched Min-SNR pilot 6K fixed-validation early warning

Date: 2026-08-25 CST

## Scope

The fresh gamma-5 CoFiTok arm reached its sixth scheduled fixed-validation
event at step 6,000. This record binds a reproducible, CPU-only comparison
against the corresponding prefixes of the physically completed gamma-zero
CoFiTok and dense identity controls.

This is a training epsilon early warning. It contains no generated samples,
FID, precision/recall, or class-fidelity result and cannot substitute for the
matched 50K terminal evaluation.

## Controlled treatment

The live gamma-5 manifest and the immutable gamma-zero CoFiTok training report
were recursively compared. After accounting for the experiment name, the only
resolved training difference is:

```text
loss.min_snr_gamma: 0.0 -> 5.0
```

The parameter count remains `62,834,083`. Dataset identity
`6ec1d96ac3cd8a41fc66c40d424bf8e005c6a08bf9f580f5379c93772c8fe659`
and runtime identity
`d5bfcd085ea467ee5d24dfccc6e147da06dd7a0a0efdcdab355882b8547c985e`
are unchanged. The legacy gamma-zero CoFiTok/dense pair contract also replayed
with `valid=true`.

## Reproducible builder

```text
local branch: analysis/generation-min-snr-checkpoint-integrity-monitor-v1-20260825
builder revision: b6e300820663ace2377fe51ccdc120e94cf3cb36
builder tree: 4c327900b2796ef7105e1ab75eb0d7ee765094d1
remote checkout: /tmp/cofitok-min-snr-early-warning-b6e3008/CoFiTok-internal
builder bytes: 18267
builder SHA256: 809afc266116449d0a3803b7ad93e95838980188c1273c3babad8e2a3f2f73db
test SHA256: cd7471ed57076878711975ac5a712662f0165340bab020b6162b3ab568735fd0
bundle bytes: 17387
bundle SHA256: 1a7bd0139448e4060b0e9605c4bb47cbbd5e839fd57adb936a7119ce67226f9c
```

Local focused tests passed `6/6`. The remote isolated checkout passed `30/30`
tests covering the new builder, the existing matched-trajectory builder, the
Min-SNR pilot contract, and the checkpoint-integrity waiter. Python compilation
and tracked-clean checkout verification also passed. The remote process used
`CUDA_VISIBLE_DEVICES` empty, one OMP thread, one MKL thread, `nice=10`, and
idle I/O priority; it exited before the final process replay.

## Immutable metric prefixes

All three prefixes contain 121 strictly increasing rows through exact step
6,000 and bind 384,000 images per source with `samples_seen == step * 64`.
Numeric metrics are finite. The gamma-5 prefix contains valid
`min_snr_weight_mean` and `epsilon_unweighted` fields in every row, applies
downweighting in all 121 rows, and always has weighted epsilon no greater than
unweighted epsilon.

```text
gamma5 CoFiTok: 124645 bytes
SHA256: b4449152d6eb52eaca35bab1903c83005e43538bc180431c6420d64020fd8de2

gamma0 CoFiTok: 114180 bytes
SHA256: 003fd8735931103a862b7ca942e07c3b4fd375f50ce3b70a7295bdf65a699ad7

gamma0 dense identity: 106351 bytes
SHA256: 6fab3ffd5ec242ff9ba998005cc08aae5aa71d59507f893d9ffa5984f5b3138b
```

Every validation event pairs the same step, event index, validation batch
index, 64-image count, and noise seed `102030`.

## Result

Across the six fixed-validation events at steps 1K through 6K:

```text
gamma5 CoFiTok mean epsilon MSE: 0.040233038986722626
gamma0 CoFiTok mean epsilon MSE: 0.03513677356143793
relative delta: +14.504079085046584%
gamma5 lower-event count: 0/6
step-6K relative delta: +16.98029344151082%

gamma0 dense mean epsilon MSE: 0.03510591387748718
relative delta: +14.604733342445142%
gamma5 lower-event count: 0/6
step-6K relative delta: +17.895817182928542%
```

The direction is consistently unfavorable on this fixed epsilon diagnostic.
Min-SNR intentionally changes timestep weighting, so this signal is not a
sample-quality measurement and cannot by itself reject or select the treatment.

Canonical report:

```text
path: /root/autodl-tmp/CoFiTok/checkpoints/generation/min_snr_gamma5_matched_50k_pilot_v1/reports/min_snr_early_warning_v1/cofitok_step_00006000/early_warning.json
bytes: 14001
SHA256: 20b06b611a5e6de5a40fe24c9835bdf3d7aa6d4389b9d670cf0198faf5f552f1
status: pass
```

An independent parser rehashed the builder and all snapshots, replayed the
event provenance and summaries, and verified that every authorization field is
false. At that replay the live training had advanced to step 6,300 with
`health_status=running`, `issues=[]`; the only GPU process remained the existing
trainer. The unique 10K checkpoint waiter remained
`waiting/checkpoint_missing` with its original process identity.

## Claim and execution boundary

`generation_advantage_proven=false`, `quality_claim_allowed=false`, and
`formal_50k_result_substitute=false`. Training, evaluation, sampling,
continuation beyond 50K, full 300K, promotion, export, release, and
process-signal permissions remain false in the report. Required evidence is
still matched 50K completion followed by all four 10K DDIM-100 terminal arms.
