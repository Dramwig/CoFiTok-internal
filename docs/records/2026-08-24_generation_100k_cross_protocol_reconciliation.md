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
