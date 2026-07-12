# Dense monolithic baseline protocol (2026-07-11)

## Correction

The existing `epsilononly` runs retain K token heads and restricted synthesis.
They are endpoint-only **factorized** controls and were incorrectly labeled as
the same-backbone dense predictor in early tables. They remain valid mechanism
controls but must be renamed `endpoint_only_factorized`.

The corrected `same_backbone_dense` registry alias denotes a parameter-matched
direct dense baseline using:

```text
one full-resolution 3-channel epsilon head
token_count=1, token_channels=3
synthesis_mode=dense_identity (interface-only identity, no S_k)
predictor_use_feedback=false
epsilon endpoint loss only
```

The predictor trunk keeps the same depth and convolutional design. Width is
selected by enumerating dense-head widths against the actual source model; a
test requires every generated config to remain within 2% of its paired CoFiTok
parameter count:

| comparison | CoFiTok params | dense params | relative gap |
|---|---:|---:|---:|
| K8 / 32-64px | 81,808 | 80,707 | -1.35% |
| K4 / 256px | 15,776 | 15,690 | -0.55% |

## Runs

The corrected dense baseline is trained and evaluated on all eight P0 datasets
with the same steps, optimizer, data split, quality protocol, and DDIM-50 sample
counts used by the current internal P0 rows. ImageNet-256 also receives 20K
seeds 103/139 for confirmatory endpoint comparison.

The locked evaluation contract is:

| dataset group | quality at t=500 | generated / real | sampler |
|---|---:|---:|---|
| CIFAR-10 and all 64px datasets | 512 | 1,024 / 4,096 | DDIM-50 |
| ImageNet-256 10% and full | 256 | 512 / 2,048 | DDIM-50 |

Legacy 32/64-image and DDIM-20 smoke reports do not satisfy this contract. The
same runbook repairs the missing CoFiTok/endpoint generation rows, aligns the
early-dataset quality evaluations, and trains the missing canonical
ImageNet-64-HF simultaneous control before building the final table.

CoFiTok and endpoint-only path metrics use every prefix (`1..8` or `1..4`) in
these cross-batch quality passes. The older path AUC stored in a training
visualization summary is not accepted by the final P0 builder. The same queue
also evaluates all eight Tiny/ImageNet-64-HF 20k seed-method checkpoints on
1,024 images with full K8 prefixes before rebuilding the two-seed table.

Config generator:

```text
scripts/make_dense_monolithic_configs.py
```

Runbook:

```text
artifacts/runbooks/dense_monolithic_p0_2026-07-11.sh
```

Interpretation:

- Compare endpoint denoising/sample quality against `same_backbone_dense`.
- Compare prefix ordering/path AUC against `endpoint_only_factorized` and
  channel-mask controls; a monolithic head has no native multi-prefix interface.
