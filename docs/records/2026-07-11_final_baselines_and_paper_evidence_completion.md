# Final baselines and paper evidence completion (2026-07-11)

Status: `completed_defined_protocols`.

## Coverage

- Matched dataset/step training: `80/80` completed.
- Tokenizer reconstruction eval-only: `16/16` completed.
- Official ImageNet-256 related-method eval-only: `3/3` completed.
- Guarded cross-task cells: `24 protocol_blocked`; these are fairness boundaries, not missing runs.
- All five final runbook status files are `completed` on `pro6000`.
- All seven pinned external repositories have zero tracked modifications.

The final matrix is:

```text
artifacts/reports/paper_comparison_matrix_2026-07-11_final_guard/
```

## Main findings

- CoFiTok beats the endpoint-only factorized control in path AUC on `8/8` short-budget datasets.
- The matched K8 20k two-seed repeat passes `4/4` path-AUC pairs, with `96.18%` mean reduction and `+4.20%` mean endpoint x0-MSE change.
- The ImageNet-256 K4 20k confirmatory protocol passes `7/7` gates. The learned order ranks `1/24` for both seeds and mean endpoint MSE is `6.73%` below the parameter-matched direct dense control.
- Restricted synthesis has zero-token ratio zero on `8/8`; deep synthesis leaks on `8/8`.
- CoFiTok is best on `0/8` lowres and `1/8` Inception-style generation rows. Broad generation SOTA is not supported.

The evidence supports the scoped claim of ordered restricted dense-noise factorization with prefix-controllable denoising. It does not guarantee venue acceptance.

## Official secondary rows

| Method | FID | sFID | IS | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| D-AR | 2.6281 | 6.7195 | 285.8914 | 0.8029 | 0.5884 |
| MAR PTH `model_ema` | 2.3385 | 4.6999 | 62.8979 | 0.8216 | 0.5716 |
| ReTok | 2.2189 | 5.8956 | 245.9392 | 0.8159 | 0.5994 |

These rows use the common ADM evaluator and remain outside the P0 matched-training table.

## Final artifacts

```text
artifacts/reports/paper_evidence_report_2026-07-11_final/
artifacts/reports/top_tier_claim_audit_2026-07-11_final/
artifacts/reports/imagenet256_confirmatory_2026-07-11/
artifacts/figures/final_visual_claim_2026-07-11/
paper/venues/aaai27/main.pdf
```

Local verification: 215 tests passed; generated tables compile to 2 pages; the AAAI draft compiles to 7 pages with no undefined references, citation warnings, overfull boxes, LaTeX warnings, or errors. All rendered pages were visually checked.

SHA256:

```text
main.pdf                         ec18cb271acd5fca58e8c65599a794670329b6949717012e6e232b1768e960b8
paper_tables_compile_check.pdf  a803744a9bb95d52bb267336b0506cdec98847a8e3b72c2091a4b192797be072
cofitok_final_visual_claim.png   5f27ccd270affdef3b9178d44ac6b5e19b1145e9a241df230c5b073b761f7da6
top_tier_claim_audit.json        9aec29575d8581e0037497bfc0cc08f8d10699ebeb80b5784ce5041cec3b7b9e
```
