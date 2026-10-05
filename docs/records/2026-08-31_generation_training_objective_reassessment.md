# Generation training-objective reassessment (2026-08-31)

## Scope

This is a CPU-only decision record. It does not launch training or sampling,
create an execution gate, modify the remote checkout, promote a checkpoint,
export an inference artifact, or replace the terminal quality hold.

The machine-readable contract is
`artifacts/reports/generation/training_objective_reassessment_2026-08-31.json`.

## Rechecked evidence

The full-data 100K quality bridge remains source-bound to
`cf0e5faa94bf4ab38d947b921935b3b765b5537a`. Its terminal result is `hold`:
CoFiTok K8 is FID `115.2622`, precision `0.7556`, recall `0.00832`; dense
identity is FID `123.0210`, precision `0.6653`, recall `0.01000`. Matched FID,
precision, endpoint, and mechanism checks pass, but the absolute FID ceiling,
recall floor, and class-fidelity qualification fail. The result keeps
`generation_advantage_proven=false`.

The completed matched 1,000-sample sampling-recovery screen selected
`no_shared_sampling_recovery_candidate`. The Min-SNR gamma-5 50K pilot also
selected `no_shared_min_snr_candidate_at_50k`: CoFiTok FID improved only
`1.33%` and recall fell, while dense improved FID more but lost precision.
Neither result is formal quality evidence or a replacement for the terminal
hold.

The final matched validation epsilon MSEs are `0.0292560` (CoFiTok) and
`0.0292407` (dense), with both methods at 100,000 steps and 6,400,000 images
seen. This does not show a large primary-loss divergence that would justify
changing the factorized representation.

## Decision

The qualified CoFiTok token layout and restricted synthesis objective are
preserved. Another scalar timestep weighting is deferred because the completed
Min-SNR screen is negative shared evidence. A factorization/layout change is
also deferred because ordered-prefix, zero-token, shuffle, and compressed-energy
checks pass. A conditioning-only route is not selected because the terminal
failure is mixed absolute quality, support, and class fidelity rather than a
class-only failure.

Insufficient exposure or capacity remains a live scientific hypothesis, but it
is not itself a new objective intervention. Any test of that hypothesis needs a
new source-compatible preparation, a separately scoped execution gate, and
exact stage authorization. The contract therefore records
`preserve_qualified_objective_defer_new_intervention` with every training,
sampling, full-training, 300K, promotion, export, release, and process-signal
permission set to `false`.

The current evidence supports only the scoped ordered restricted dense-noise
factorization and prefix-controllable denoising claims. It does not prove a
broad generation advantage or a released usable generation system.

## Source identities

The JSON contract records the rehashed quality bridge, reconciliation,
sampling-recovery, Min-SNR, pair-monitor, training-report, and JSONL metric
identities used for this decision. The standing authorization hash remains
`5fe64a0941acb55a21cb6479a726987c6259e93a772c47290f2d6883752de4df`; it does
not bypass the missing successor execution gate.
