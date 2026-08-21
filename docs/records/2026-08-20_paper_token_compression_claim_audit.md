# Paper token-compression claim audit

Date: 2026-08-20 (Asia/Shanghai)

## Outcome

The locked 2026-07-11 paper evidence proves ordered restricted dense-noise
factorization, prefix controllability, ordering sensitivity, exact zero-token
behavior, and shuffle mismatch. It does **not** prove that the reported K4/K8
predictor outputs are compressed denoising tokens.

All 14 bound locked-evidence reports audited here use full-resolution
`tiny_conv` token fields with `token_channels=16` and no
`token_channel_schedule` or `token_spatial_strides`. The
`synthesis_active_token_channels` list masks inputs inside `S_k`; it does not
reduce the token emitted by the predictor, and the unmasked token also remains
available to predictor feedback. Therefore the locked evidence supports this
wording:

> ordered restricted denoising components

It does not independently support this wording:

> compressed denoising tokens

This is a claim-boundary correction, not a rejection of the locked metric
results. No locked report, table, checkpoint, or generated sample was modified.

## Locked evidence layout

The audit binds the eight-dataset P0 CoFiTok rows, the four Tiny/ImageNet-64
20K repeated rows, and the two ImageNet-256 confirmatory seed rows.

| evidence family | emitted token shape | dense field | one token / dense | K-token total / dense |
| --- | --- | ---: | ---: | ---: |
| CIFAR-10 K8 | `16 x 32 x 32` | `3 x 32 x 32` | `5.3333x` | `42.6667x` |
| 64x64 K8 | `16 x 64 x 64` | `3 x 64 x 64` | `5.3333x` | `42.6667x` |
| ImageNet-256 K4 | `16 x 256 x 256` | `3 x 256 x 256` | `5.3333x` | `21.3333x` |

Thus none of the 14 locked reports passes per-token compression, and none
passes aggregate sequence compression.

## Current true-compressed generation layout

The active full-data quality bridge uses the later `rgbtail3` layout:

```text
token channels:       [4, 4, 8, 8, 8, 1, 1, 1]
token spatial strides:[16, 16, 8, 8, 4, 1, 1, 1]
token scalar counts:  [1024, 1024, 8192, 8192, 32768, 65536, 65536, 65536]
dense RGB field:      196608
K8 total:             247808
```

Every current token is individually smaller than the dense field, so the 50K
layout evidence supports **individually compressed denoising tokens**. The K8
sequence totals `1.2604167x` the dense field, so it does not support aggregate
sequence compression. The current 50K layout evidence alone also does not
substitute for the still-pending terminal 100K quality and uncertainty reports.

## Paper-language finding

The current venue-neutral main text, AAAI-27 main text, and citation audit use
unsupported compressed-token wording at 15 matched locations in total:

- `paper/latex/main.tex`: 7 locations;
- `paper/venues/aaai27/main.tex`: 7 locations;
- `paper/citation_audit.md`: 1 location.

The paper tree is outside this clean code worktree and the metric package is
locked. A wording-only correction was subsequently applied to the three paper
sources: all unsupported `compressed ... token/component` phrases were changed
to the scoped `denoising token/component` wording. The method-overview generator
and its text-bearing SVG were also audited. Its subtitle now says `ordered
restricted denoising components`, and the token boxes say `ordered denoising
token`; the PNG, SVG, and manifest were regenerated from that source. No metric,
table, checkpoint, locked result artifact, or locked Figure 2 panel was changed.
The post-edit language guard passes across all five text-bearing sources.

Before submission, the prose should either:

1. describe the locked experiments as ordered restricted denoising components
   and separately identify the later true-compressed generation-system evidence;
   or
2. add matched true-compressed paper experiments before attributing the locked
   mechanism metrics to compressed tokens.

The first option is the immediate evidence-preserving correction now applied.
The second option is scientifically stronger but requires new matched
experiments and must not overwrite the locked 2026-07-11 artifacts.

## Automated guard

Two fail-closed audit tools were added:

- `scripts/audit_token_compression_claim.py` computes actual emitted token
  scalar counts from resolved configs. Synthesis-only masking or resampling
  never counts as predictor-token compression.
- `scripts/audit_paper_token_compression_language.py` checks paper prose, the
  method-figure generator, and text-bearing SVG output against the bound layout
  decision and reports exact source lines. Figure sources fail closed on any
  residual `compressed` wording when per-token compression is unsupported.

Focused CPU validation:

```text
14 passed
```

## Evidence artifacts

```text
artifacts/reports/generation/token_compression_claim_audit_2026-08-20/
  locked_paper_evidence_audit.json
  current_quality_bridge_50k_layout_audit.json
  paper_language_audit.json                 (pre-edit, incompatible)
  paper_language_audit_after_wording_fix.json (post-edit, compatible)
```

Artifact identities:

```text
locked_paper_evidence_audit.json
  bytes:  23290
  SHA256: 147781a93e15d4420c125421cd0ceebf036f55675fcb197e7dc3212bb840331e

current_quality_bridge_50k_layout_audit.json
  bytes:  2528
  SHA256: 5ad5b143b4feb60fff1f27d447c97dd9b14d245c1ded196898701f1a05935683

paper_language_audit.json
  bytes:  7094
  SHA256: 43c07cd1cfe8f41dd983f66b4783f449e6e93db219c83c70c3f05e010f5657f7

paper_language_audit_after_wording_fix.json
  bytes:  2981
  SHA256: a4d1e307ad93f64b4a532b4615c3ca51d7d632dc72bee540aa84f67c2d215e00

method_overview.png
  bytes:  107831
  SHA256: a512d37a23bf4c0285ba73fca6ef6c2791ea10e1a754961b27b36fdc690c67d5

method_overview.svg
  bytes:  16244
  SHA256: 1b0a1539faa7014b4a1fe16a18a7dbfe57c651a9e1b388974e9cf8d92a8a8c65

method_overview_manifest.json
  bytes:  1208
  SHA256: 01f4e9daca2255f968cec79c0982e40ccedd81f4206242f46be85dfcea73fb94
```

The JSON reports bind every inspected source by path, byte count, and SHA256.
The audit used the live `pro6000` report files as authoritative inputs and did
not use the GPU, signal any process, or alter the active 100K controller chain.
