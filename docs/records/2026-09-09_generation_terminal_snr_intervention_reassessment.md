# Terminal-SNR intervention reassessment (2026-09-09)

This record documents a read-only, source-bound reassessment after the fresh
four-arm terminal-SNR endpoint screen. It does not authorize training, sampling,
evaluation, checkpoint promotion, export, release, or paper integration.

## Authoritative result

- Active screen: `/root/autodl-tmp/CoFiTok/checkpoints/generation/terminal_snr_endpoint_screen_v1/`
- Screen result SHA256: `b740c8b21aabf640c156aea076058c73350d26c110e2cc967576a26492a52dea`
- Existing result-validation receipt SHA256: `3be043d852bd61d6e79005dc9d14f77113f1250bef967ca4e2a8ac74bc715f95`
- Controller state: `completed`, `complete`, `terminal_status=hold`
- Scientific state: `scientific_status=hold`, `screen_pass=false`, `generation_advantage_proven=false`
- Failed scientific checks: `cofitok.relative_fid_improvement` and `dense_identity.relative_fid_improvement`

The endpoint intervention reduced terminal raw-x0 clipping for both methods
(CoFiTok `0.6902521550655365`; dense `0.7379159331321716`) but produced
negative relative FID changes for both (CoFiTok `-0.11363547384116164`; dense
`-0.09408838680587096`). Mechanism checks remained passing, so the intervention
is a symptom mitigation, not a demonstrated generation repair.

## Reassessment decision

- Decision: `no_defensible_shared_intervention_selected`
- Reassessment SHA256: `6da8e3b61e9cd632ee16bcace16e6f32f0d3a3c011847c573603e79e3f7d1075`
- Validation receipt SHA256: `1e3b82f40385d2489d3c59404f9df027573c5c8b30d7d98eec258c3cb409c48e`
- Scientific and terminal status: `hold`

The reassessment records the following dispositions: endpoint `0.975` rejected
by the fresh screen; exposure/capacity held by completed screens; sampling-only
recovery rejected because no shared candidate exists; Min-SNR/scalar weighting
rejected or deferred without a shared candidate; semantic residual alignment
rejected by post-evaluation; conditioning ranking insufficiently supported and
confounded; velocity/x0 prediction incompatible with the dense-epsilon-component
boundary; and architecture/layout changes deferred without discriminating
evidence.

## Reproducibility and safety

The builder is commit `7ec178f3f1bf1d3e23ab899750754b47bbc4626c`, tree
`c060ad94e8fa348aaf75ed67c0bf125b583bf3dd`, derived from required screen
revision `89bcd9adb2a1e4625a9cd48dc2be82cbee8c6430`. A separate clean validator
checkout at the same commit replayed the decision and receipt. The incremental
bundle is `/tmp/cofitok-generation-terminal-snr-intervention-reassessment-7ec178f-from-89bcd9a.bundle`
(SHA256 `9f2c3c8ed269f309ad49cd2a4d7a01ccffc0a4a45ee5a617f25f42cc3359e223`,
10,960 bytes).

The old exposure controller and its failed builder status remain immutable
historical evidence. Frozen confirmation, large-capacity readiness, full
training, 300K, promotion, export, release, process-signal, and paper
integration permissions all remain `false`.
