# 2026-07-19 generation rank-deficit recovery

## Decision

The first true-compressed ImageNet-256 10% matched 50K pair is complete and
protocol-valid, but its scientific promotion gate is `hold`. Full ImageNet-256
300K training was not started. The failed CoFiTok run remains immutable failure
evidence at revision `04a738c637555b2c108e5c468d7931ebdbb59cb9`; it must not be
overwritten or relabeled as the rank-complete architecture.

Authoritative local gate copy:

```text
artifacts/reports/generation/imagenet256_10pct_compressed_matched_50k_failed_04a_2026-07-19/promotion_gate.json
```

The revision/date suffix is intentional: the completion auditor reserves the
unsuffixed path for the next current same-revision promotion attempt.
The six gate-bound source JSON files are archived in its `source_reports/`
subdirectory; every local byte count and SHA256 matches the binding in the gate.

## Completed evidence

Both methods completed exactly 50,000 optimizer steps and 3,200,000 training
images with effective batch 64. Checkpoint integrity, same-revision provenance,
matched data/diffusion/U-Net/optimizer protocol, parameter parity, sample-set
integrity, evaluator identity, runtime identity, and deterministic DDIM-100
sampling all passed.

| metric | CoFiTok K8 | dense identity |
|---|---:|---:|
| 10K FID | 422.5906 | 114.8773 |
| Inception Score | 1.3042 | 9.2107 |
| endpoint clean MSE at t=500 | 0.169522 | 0.016330 |
| endpoint clean PSNR | 13.7283 | 23.8906 |
| ordered path-AUC rank | 3 / 18 | n/a |
| training images/s | 28.0290 | 31.2806 |

The failed gates were `fid_within_tolerance`, `absolute_fid_quality`,
`endpoint_within_tolerance`, and `ordered_prefix_path`. CoFiTok fixed samples
are saturated high-frequency color texture rather than recognizable ImageNet
content. Its prefix sheet moves from low-frequency fields to the same terminal
texture. Dense samples are still weak at this budget but contain recognizable
image structure, matching the quantitative gap.

## Root cause

The failed layout used channels `[4,4,8,8,8,8,4,2]` at strides
`[16,16,8,8,4,4,2,1]`. Only the last token was full resolution, and it had two
channels. Because every `S_k` is condition-free and linear/local, the
highest-frequency RGB output subspace therefore had rank at most two. No larger
`T_k`, longer training run, or loss reweighting can span three independent RGB
high-frequency directions under that layout.

The checkpoint audit supports this diagnosis. Component energy was
`[0.000213, 0.000459, 0.001109, 0.000987, 0.016798, 0.018991, 0.350718,
0.733049]`: almost all prediction energy collapsed into the last two tokens,
while the endpoint remained 10.38 times the dense MSE.

The replacement layout uses channels `[4,4,8,8,8,8,1,2]` at strides
`[16,16,8,8,4,4,1,1]`. It preserves every per-token scalar capacity and the
exact aggregate 280,576-scalar budget, keeps every token strictly smaller than
a 196,608-scalar dense RGB field, and allocates three aggregate full-resolution
channels across the last two tokens. Config validation now rejects restricted
layouts whose full-resolution channel sum is below the image channel count.

## Operational recovery

The first post-evaluation attempt also exposed two independent operational
defects:

- The cached torch-fidelity Inception weight was truncated to 12,992,512 bytes.
  It was quarantined and atomically replaced by the official 95,628,359-byte
  file with SHA256
  `6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2`.
  The local repair receipt is
  `artifacts/reports/generation/dependency_repair_2026-07-19/torch_fidelity_inception_dependency_repair.json`.
- Exact resume loaded the serialized CPU RNG state with `map_location=cuda`,
  then passed a CUDA byte tensor to `torch.set_rng_state`. RNG restoration now
  validates uint8 state and explicitly moves CPU and per-device CUDA generator
  states to CPU before calling PyTorch restore APIs.

The 10% training runbook now treats an already complete, internally consistent
training report as terminal success. Supervisor retry can therefore re-enter
post-evaluation without deserializing a completed checkpoint or mutating its
training history.

## Recovery protocol

Before another matched 50K pair, run the non-formal rank-recovery probe:

```bash
bash artifacts/runbooks/generation_rank_recovery_probe_2026-07-19.sh
```

The first launch under commit `05572b1` was rejected at step 260 because the
probe JSON omitted three legacy-default fields (`prefix_weight=0.25`,
`monotonic_weight=0.05`, and `zero_token_weight=0.01`). The trainer handled
SIGTERM by atomically writing its incomplete checkpoint and report. That run is
kept only as operational failure evidence; it is not resumed or compared.
Its small reports are archived at
`artifacts/reports/generation/rank_recovery_probe_rejected_05572b1_2026-07-20/`;
the 1 GB recovery checkpoint remains server-only.
The corrected launch explicitly sets all three fields to zero and uses fresh
`*_probe5k_v2` run directories plus a distinct v2 monitor/report identity.

It trains two same-backbone, same-seed, rank-complete 5K candidates:

1. denoise-path prefix/component supervision;
2. epsilon-band prefix/component supervision aligned directly to token spatial
   bandwidth.

Both use the first-5K-like learning-rate range, EMA, bf16, effective batch 64,
a 512-image t=500 mechanism audit, and a deterministic 512-sample DDIM-50 probe
at prefix budgets 1/2/4/8. The aggregate report is explicitly non-formal and
cannot authorize 50K or 300K automatically. Candidate selection requires
mechanism improvement plus direct visual inspection. The winner must then run a
fresh same-revision matched 50K pair and pass the unchanged 10K scaling gate
before full ImageNet-256 training is allowed.

The fresh formal pair is isolated from the failed `04a` assets through the
versioned run identities `imagenet256_10pct_rankcomplete_cofitok_k8_50k_v2`
and `imagenet256_10pct_rankcomplete_dense_50k_v2`; its report root is
`imagenet256_10pct_rankcomplete_matched_50k_v2`. Training, post-evaluation,
full-training authorization, gate source verification, and terminal audit all
resolve these names from `cofitok.generation_paths`. The legacy directories are
therefore never resumed, overwritten, or accepted as current completion.
