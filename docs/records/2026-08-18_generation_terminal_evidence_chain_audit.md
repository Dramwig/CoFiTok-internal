# Generation terminal evidence-chain audit (2026-08-18)

## Outcome

The deployed full-data 100K terminal evidence chain is implementation-ready and
fail-closed for the intended scoped claim. No code path audited here permits a
broad generation-superiority, SOTA, release, export, or later-training claim.

Scientific qualification remains **pending**, because the active matched 100K
training pair has not completed and the terminal quality, uncertainty, claim,
and runtime reports do not yet exist.

## Exact deployed revisions

| Role | Revision | Tree |
|---|---|---|
| Paired uncertainty evaluator | `1c8ef207cb6d79850d73a45abc345fc421e6aa7f` | `a09f14a0eca44af6db8e6781863a463163ddf646` |
| Terminal 100K uncertainty waiter | `e9a424b36bc5b951a9a27d7e1dc7f7afc650f706` | `4f4cf5015ceb4be426e314bbd224778860e98d00` |
| Statistical claim qualification | `c42ac96c6628ff71f7c67c8957c86ea0523aea0b` | `f94f2deb974b0a347b3a647e68dc3dcba22a0d63` |
| Statistical claim-language guard | `5fab079a7b48024b05e2a306b95040baf839e276` | `3f5ae4b7f79a26bb17183caa04394070c5e9dcd8` |
| Runtime/compute fairness waiter | `85bf1d6ae5b44f401153d0d1e390fc3973ff4bed` | `7900c5617c28c18e328c93aba4c547a3889f1aa7` |
| Runtime claim guard | `6fdeebe7c0ca6b2f7fbe593b3690f14d36f98e46` | `bc41a4df45b83cef3c81cc63aad1668c49b85a75` |

The tested descendants had no changes to the audited source or test files
relative to these deployed revisions.

## Statistical contract verified

- CoFiTok and `dense_identity` must use the same physical global-index filenames
  and the same sampling signature, except for the method-specific prefix budget.
- The default terminal audit uses 10,000 matched generated samples, 20 disjoint
  generated blocks of 500 samples, five disjoint real folds, 10,000 bootstrap
  repetitions, and seed 2027.
- Each block computes the CoFiTok-minus-dense polynomial KID contrast against a
  shared real reference. The shared real-only KID term cancels exactly.
- Statistical support requires both the paired bootstrap 95% interval upper
  bound to be below zero and the one-sided exact sign-test p-value to be at most
  0.05.
- A terminal uncertainty pass additionally requires the bound CoFiTok FID point
  estimate to be lower than the bound dense FID point estimate.
- The final quality-bridge claim qualification additionally requires the
  absolute quality screen itself to pass. Positive paired KID evidence cannot
  override a failed absolute-quality screen.
- The claim-language guard explicitly labels FID as a point estimate without a
  FID confidence interval or FID significance test. Statistical support is
  attributed only to paired block-KID.

## Provenance and fail-closed checks verified

- Execution manifests bind source paths, bytes, SHA256 values, parameters,
  sample-set SHA256 values, checkpoint SHA256 values and steps, real-set tree
  SHA256, evaluator/runtime identity, FID point estimates, and the matched
  sample-index window before feature extraction.
- Bound sources are rechecked before feature extraction and again across waiter
  transitions. Source drift, Git drift, protocol drift, FID drift, report drift,
  or claim-policy drift raises an error rather than weakening the decision.
- The terminal waiter waits for the quality bridge, the preceding uncertainty
  chain, process exit, and a verified idle GPU slot. It does not signal trainers
  or unrelated processes.
- Runtime comparison is direct only when terminal pair-monitor evidence proves
  complete exclusive GPU observation coverage. Otherwise elapsed time,
  throughput, and cost values remain observational lower bounds.
- D-AR, MAR, and ReTok remain contextual eval-only evidence; cross-tier numeric
  ranking is explicitly forbidden.

## CPU-only validation

Using `C:/qbfd5/.venv/Scripts/python.exe` with `CUDA_VISIBLE_DEVICES=-1`:

- Terminal uncertainty, manifest, qualification, and language-guard tests:
  `61 passed, 1 skipped`.
- Runtime fairness and runtime-claim tests: `14 passed`.

The skipped test is the optional torch-fidelity reference comparison when its
test-only dependency is unavailable; the independent explicit polynomial-MMD
formula and shared-reference cancellation tests passed.

Machine-readable receipt:
`artifacts/reports/generation/terminal_evidence_chain_audit_2026-08-18/audit_receipt.json`
(`3,595` bytes, SHA256
`7b630818fcd117ec5f659ee2954816e0e254f7e4a296adc033f107c98e2cd1bf`).

## Live server snapshot

Read-only snapshot from `pro6000` on 2026-08-18 around 10:36 CST:

- output root:
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_base128_quality_bridge_v1`
- pair monitor: `running`, stage `cofitok_training`, issues `[]`
- latest observed training metric: CoFiTok step `41,750 / 100,000`,
  `2,672,000` images seen; dense has not started
- latest bound checkpoint: step 40,000, bytes `1,010,937,514`, SHA256
  `f6b0b3d285d4565d7d163dedc4727d5d9580581cf857e4856df6801dfff283a2`,
  `latest.json` binding status `metadata_verified`
- the physical 50K checkpoint waiter remains `waiting / checkpoint_missing`
- controller identity guard remains active and the terminal report chain remains
  in its expected waiting states

## Required terminal evidence still absent

- `reports/quality_bridge_result.json`
- `terminal_100k/matched_uncertainty.json`
- `reports/statistical_claim_qualification.json`
- `reports/statistical_claim_language_guard.json`
- `reports/runtime_compute_fairness/final_report.json`
- `reports/runtime_compute_claim_guard_v1/runtime_compute_claim_guard.json`

Until those exact bound artifacts exist and pass, the allowed conclusion remains:
CoFiTok has established mechanism and prefix-control advantages plus repeated
directional FID evidence, but it has not yet established the terminal matched
large-scale generation-quality claim.
