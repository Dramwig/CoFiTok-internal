# 100K cross-protocol reconciliation

The completed full-data 100K bridge contains a matched ranking reversal on the
same terminal checkpoints:

- the existing DDIM-50, 2,048-image milestone favors `dense_identity`;
- the existing DDIM-100, 10,000-image terminal evaluation favors CoFiTok;
- both runs use seed 0, start index 0, the per-global-index random stream,
  CFG 1.5, no guidance rescale, eta 0, bf16, and balanced-modulo classes.

The versioned authoritative follow-up decision therefore routes to
`reconcile_100k_cross_protocol_evidence`. The reconciliation evaluates only
the already-existing DDIM-100 PNGs at indices `000000..002047`. It performs no
model load and no sampling. A hardlink directory view gives
torch-fidelity 0.4.0 the exact first 2,048 files without copying or modifying
the source sample set.

The report compares:

1. existing DDIM-50 at 2,048 images;
2. DDIM-100 at the same first 2,048 global indices;
3. existing DDIM-100 at 10,000 images.

This separates the sampler-step contrast from the nested sample-count
contrast. The new metric computation is CPU-only, nice 10, ionice idle, and
uses the already content-addressed ImageNet-256 validation cache. The script
physically reopens and hashes the decision, terminal result, milestone,
training exposure, checkpoints and sidecars, sampling reports/manifests/
progress, PNG sets, source metric reports, and real image tree.

The report is permanently non-authorizing. It cannot launch training or
sampling, use a GPU, promote or export a model, authorize scaling, release an
artifact, or signal any process. The terminal quality status remains `hold`
and `generation_advantage_proven=false` regardless of the reconciliation
outcome.

## Executed result

The locked evaluator commit was
`c1a27c557e81e89d4e015648e9c065924725ce6c` with tree
`c97007957138ae890352672e82b33cfead0e88d4`. Local targeted tests and the
Linux rehearsal both passed `41/41` tests. The remote evaluator ran with CUDA
hidden, torch-fidelity `0.4.0`, nice 10, ionice idle, and eight OMP/MKL
threads. GPU memory and utilization stayed at zero.

The matched FID rows are:

| protocol | CoFiTok | dense identity | lower FID |
|---|---:|---:|---|
| DDIM-50, 2,048 existing images | 219.860534 | 134.684115 | dense identity |
| DDIM-100, first 2,048 existing images | 128.534946 | 137.412779 | CoFiTok |
| DDIM-100, 10,000 existing images | 115.262173 | 123.021031 | CoFiTok |

At fixed 2,048 indices, raising the sampler from 50 to 100 steps changes
CoFiTok FID by `-91.325589`, while dense changes by `+2.728664`. At fixed
DDIM-100, increasing the evaluated set from 2,048 to 10,000 changes FID by
`-13.272773` for CoFiTok and `-14.391748` for dense without changing the
matched ranking. The observed ranking reversal is therefore assigned to the
sampler-step contrast rather than the sample-count contrast.

This resolves the evidence conflict but does not prove a usable-generation
advantage. Absolute FID, recall, and class fidelity still fail the terminal
quality screen. The authoritative scientific state remains `hold` with
`generation_advantage_proven=false`; 300K, promotion, export, and release all
remain forbidden.

Remote canonical versioned report:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_base128_quality_bridge_v1/reports/100k_cross_protocol_reconciliation_v1_20260824/cross_protocol_reconciliation.json
SHA256 6fc639caec0320225c2e8d26316490cde2d765d49c6062a18e0e671f79384d19
```
