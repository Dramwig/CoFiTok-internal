# CoFiTok Min-SNR 1K treatment-delivery evidence

Date: 2026-08-25 CST

## Scope

This record freezes a single-arm diagnostic through CoFiTok step 1,000 in the
fresh matched Min-SNR pilot. It answers only whether the configured gamma-5
epsilon weighting was delivered and whether the logged trajectory remained
finite and exposure-consistent through that cutoff.

It is not a CoFiTok-versus-dense comparison, a convergence result, a sample
quality result, a class-fidelity result, or an authorization artifact.

## Training identity

- Run root:
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/min_snr_gamma5_matched_50k_pilot_v1`
- Method: `cofitok`
- Training revision: `842a34130e82f241330707118a05bf6ed01e263e`
- Training tree: `3fd4c4538d15b85233b1f8b582dce0f185dedba2`
- Training branch: `scale/generation-min-snr-matched-pilot-v1-20260825`
- Dataset: `imagenet_256`
- Seed: `2027`
- Scheduler horizon: `100000`
- Bounded physical stop: `50000`
- Effective batch: `64`
- Controlled shared treatment: `loss.min_snr_gamma=5.0`

## Builder identity

- Builder revision: `39251b797de0e2b4dab7c827e952b5e660b89672`
- Builder tree: `a6579d13250c47ed88c4453a94f20360dbc4ff82`
- Builder branch: `analysis/generation-min-snr-treatment-delivery-v1-20260825`
- Remote checkout:
  `/root/autodl-tmp/CoFiTok/checkouts/min-snr-treatment-delivery-39251b7/CoFiTok-internal`
- Builder SHA256:
  `57ca5d75c22e29795e0cb35b46bc6781965453c12e2643274ebdc509917b0400`
- Incremental bundle: `D:/cofitok-bundles/min-snr-treatment-delivery-39251b7.bundle`
- Bundle bytes: `8274`
- Bundle SHA256:
  `a253db4675c36efd4a1dbf97b82f45db15456d316053323152fd7e731ee607f4`

The builder verified that the training revision is its ancestor and that
`scripts/train_generation.py`, `src/cofitok/diffusion/schedule.py`, and
`src/cofitok/training/losses.py` are unchanged from the training revision.

## Canonical evidence

Remote directory:

`/root/autodl-tmp/CoFiTok/checkpoints/generation/min_snr_gamma5_matched_50k_pilot_v1/reports/min_snr_treatment_delivery_v1/cofitok_step_00001000`

- `treatment_delivery.json`: `6560` bytes,
  SHA256 `99bbe63ade18a7ce1b4a6d45a0ed7ad54af63334dcb2d3bbc300549eeaca3261`
- `cofitok_train_metrics_through_step_00001000.jsonl`: `21430` bytes,
  SHA256 `450c5033e4a55b73b317671cd2d2c2f6f957984105e74445eadbcdf749c2127a`
- Bound run manifest SHA256:
  `5965ebd2f0d28213f04a695ec178b6d7cab714b3dd8c22b1c3a407f4540d9268`

The same two small artifacts are archived under
`artifacts/reports/generation/min_snr_treatment_delivery_2026-08-25/`.

## Verified result

- Exact logged steps: `1, 50, 100, ..., 1000` (`21` rows).
- Exposure: `64000` images, with `samples_seen == step * 64` for every row.
- All numeric fields are finite.
- Cumulative elapsed time is strictly increasing.
- All 21 rows have `0 < min_snr_weight_mean < 1`.
- All 21 rows have `epsilon < epsilon_unweighted`.
- Min-SNR weight mean across logged rows: `0.8307189771107265`.
- Min/max Min-SNR weight mean: `0.7591409087181091 / 0.8910196423530579`.
- Mean weighted/unweighted epsilon ratio: `0.6741174893726368`.
- Step-1,000 weighted epsilon: `0.016866452991962433`.
- Step-1,000 unweighted epsilon: `0.030121920630335808`.
- Step-1,000 Min-SNR weight mean: `0.8910196423530579`.

Validation used CUDA-hidden, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, nice 10,
and idle I/O priority. The new and existing Min-SNR focused tests passed
`13/13`. A temporary real-source replay passed before canonical publication,
and an independent post-publication replay verified source hashes, exact rows,
exposure, finiteness, treatment delivery, claim boundaries, and process exit.

## Scientific boundary

This proves only that gamma-5 Min-SNR weighting was active at every logged point
through CoFiTok step 1,000 and that the logged trajectory was numerically finite.
It does not prove improved optimization, matched advantage, sample quality, or
generation advantage. A genuinely matched trajectory requires dense identity to
reach the same cutoff under the same pilot contract.

At publication, the unique training controller and trainer remained healthy,
the independent monitor reported no issues, and the sole GPU owner remained the
CoFiTok trainer. `generation_advantage_proven=false`; all continuation, 300K,
promotion, export, release, and process-signal permissions remain false.
