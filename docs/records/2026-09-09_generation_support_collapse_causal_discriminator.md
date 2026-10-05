# Support-collapse causal discriminator (2026-09-09)

This record documents a read-only, source-bound causal-discriminator replay
after the completed four-arm terminal-SNR endpoint screen. It does not alter
the screen, controller, arm outputs, historical exposure evidence, model,
objective, protocol, configuration, runtime, checkpoint, or generated sample.
It does not authorize training, sampling, evaluation, confirmation, promotion,
export, release, or paper integration.

## Active terminal-SNR state preserved

- Active screen: `/root/autodl-tmp/CoFiTok/checkpoints/generation/terminal_snr_endpoint_screen_v1/`
- Controller: `completed`, `stage=complete`, `terminal_status=hold`
- Result: `scientific_status=hold`, `screen_pass=false`, `generation_advantage_proven=false`
- Controller status SHA256: `dfba8a85c6aa0b530e27874c5a0187f18ace3ac2624ab0cfedd92b8f4dd45e14`
- Controller log SHA256: `2167f99c0c899c7e79896dc2924849e9a5bd8638e9c985777f13f97edaca084d`
- Result SHA256: `b740c8b21aabf640c156aea076058c73350d26c110e2cc967576a26492a52dea`
- Result-validation SHA256: `3be043d852bd61d6e79005dc9d14f77113f1250bef967ca4e2a8ac74bc715f95`

The exact four arm validators and the result validator were replayed again.
All returned `status=pass`, preserving the arm-validation hashes
`2d499716...`, `f66e2439...`, `68dfe993...`, and `5a1c90e1...`, and the
result-validation hash above. The execution checkout remains clean at commit
`89bcd9adb2a1e4625a9cd48dc2be82cbee8c6430`, tree
`46efd20cff489bccd799bb13c4155a0cc79e7649`.

An independent physical audit re-read all four `train_metrics.jsonl` files,
`latest.json`, partial `training_report.json`, both available checkpoints and
their sidecars, sampling manifest/progress/report, all 1,000 generated PNGs per
arm, generation metrics, class fidelity, checkpoint diagnostics, rollout
reports, and arm validations. Every arm passed:

- 201 finite, strictly increasing metric rows ending at step 10,000.
- `samples_seen = step * 64` for every row and 640,000 at the screen endpoint.
- Exact checkpoint bytes/SHA256 matched `latest.json` and the integrity sidecar.
- Sampling was completed at exactly 1,000 images and each physical sample-set
  digest matched progress, report, metrics, and class-fidelity provenance.
- All report Git identities remained bound to the clean execution checkout.

No terminal-SNR training, sampling, or evaluation process is active. The RTX
PRO 6000 was idle with 0 MiB reported in use and approximately 176.68 GiB was
free on `/root/autodl-tmp`. Historical shell PID `87723` is unrelated to the
controller/GPU workload and was not signaled.

## Immutable discriminator and validation receipt

- Remote report:
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/support_collapse_causal_discriminator_v1/support_collapse_causal_discriminator.json`
- Report bytes/SHA256: `420423` / `6b0887dec10d6908550bee8e90c3a6e242dbddb3cb62c97935eac841dd262bc0`
- Adjacent validation receipt:
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/support_collapse_causal_discriminator_v1/support_collapse_causal_discriminator.validation.json`
- Validation bytes/SHA256: `33690` / `0043680c6109de3adffae743517ef87b4efd81f3e87f6d540a144aa804354292`
- Both files were exclusively created with mode `0444` and re-read after
  creation.

The builder and separate validator use identical tracked script bytes
(SHA256 `1e7fb3412556eeb1c279b68a71adc4aa57eb0cb914a3b3c290c61049d75b5fca`)
from commit `94f7dd1a5833f76eda66ef8880a704e9730f4602`, tree
`7a0d8bdc6a0ab626d8b8a9a595d2b63ce1886a4b`, derived from parent
`7ec178f3f1bf1d3e23ab899750754b47bbc4626c`. The separate clean branch names
are:

- `analysis/generation-support-collapse-causal-discriminator-v1-20260909`
- `validation/generation-support-collapse-causal-discriminator-v1-20260909`

The incremental recovery bundle is
`/tmp/cofitok-generation-support-collapse-causal-discriminator-94f7dd1-from-7ec178f.bundle`.
It is `20786` bytes, has SHA256
`9bd5c397f04c55c90387c2ea1dff305e4221a1688b9a350702210c7032c5be0a`,
advertises only commit `94f7dd1...`, requires `7ec178f...`, and passes
`git bundle verify`.

The validator independently re-read and recomputed all 70 content-addressed
sources. Its receipt records `physical_source_replay.performed=true`, source
count 70, and the reproduced canonical decision SHA256
`544421289905625f1fe88b92c2af1e7ebf959362d3b8da8f4c463d0ed3cd95f6`.

## Scientific judgment

The deterministic outcome is:

- `scientific_status=hold`
- `selected_candidate=null`
- `decision=no_defensible_shared_intervention_selected`
- `shared_support_collapse_consistent=true`
- `common_cause_proven=false`

Conditioning ranking does not qualify. Its correct-MSE ratios are non-regressive
(CoFiTok `0.9872239995`, dense `0.9918479541`) and paired changes are positive,
but the absolute correct-versus-null margins remain negative for both methods
(CoFiTok `-0.0027841256`, dense `-0.0030336429`), the required positive-sample
fractions fail, and distribution-support quality is not non-regressive.

Semantic residual alignment does not qualify. Correct-MSE ratios regress to
CoFiTok `1.5986143347` and dense `1.3278585554`; absolute
correct-versus-null margins remain negative, and the paired-versus-null and
support-quality gates fail.

Exposure replay confirms that correct denoising MSE improves from step 1,250
to 5,000, while correct-versus-wrong semantic advantage worsens for both
methods (CoFiTok `-0.0014246419`, dense `-0.0010517309`). This is descriptive
cross-method consistency, not proof of a shared cause.

The 100K support evidence remains severely recall-limited:

| method | FID | precision | recall | predicted-class fraction | normalized entropy | top-1 | top-5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| CoFiTok | 115.2622 | 0.7556 | 0.00832 | 0.734 | 0.7801 | 0.0024 | 0.0106 |
| dense identity | 123.0210 | 0.6653 | 0.01000 | 0.696 | 0.7501 | 0.0017 | 0.0081 |

The shared low-recall/class-fidelity pattern is consistent with support
collapse. The available evidence does not identify or prove a common cause,
and therefore does not support selecting a new shared intervention.

## Authorization boundary

All confirmation, large-capacity readiness, full-training, 300K, promotion,
export, release, process-signal, and paper-integration permissions remain
`false`. The frozen-checkpoint confirmation must not be prepared or launched
because the active terminal-SNR screen and this discriminator are both held.

Small local mirrors are stored at:

- `artifacts/reports/generation/support_collapse_causal_discriminator_v1/support_collapse_causal_discriminator.json`
- `artifacts/reports/generation/support_collapse_causal_discriminator_v1/support_collapse_causal_discriminator.validation.json`
- `.codex-bundles/cofitok-generation-support-collapse-causal-discriminator-94f7dd1-from-7ec178f.bundle`
