# D-AR Formal64 Feasibility

Date: 2026-07-09

Scope: inspect the cloned D-AR repository as the nearest diffusion-as-AR threat
and decide whether it can enter the current formal64 comparison as a fair
completed baseline.

External repo:

```text
/root/autodl-tmp/CoFiTok/baselines/repos/d_ar
commit: cd921b892c14ef52a06133b21a73257a6399ad52
```

CoFiTok-side files:

```text
baselines/adapters/d_ar/README.md
configs/baselines/d_ar/formal64_feasibility.json
scripts/baselines/probe_dar_adapter.py
artifacts/runbooks/d_ar_feasibility_2026-07-09.sh
```

Generated report:

```text
artifacts/reports/baselines/d_ar/adapter_feasibility_2026-07-09.json
```

Validation:

```text
pytest tests/test_dar_adapter_probe.py tests/test_build_paper_comparison_matrix.py tests/test_baseline_registry.py
```

Result: `8 passed`.

## Decision

D-AR should not be counted as a completed formal64 baseline in the current
paper table. It is recorded as `feasibility_only` in `baselines/registry.json`
and as `protocol_blocked` in the matrix audit.

Blockers:

- AR train/sample image-size choices exclude 64.
- Tokenizer train image-size choices exclude 64.
- Official tokenizer and D-AR checkpoints target ImageNet-256.
- AR training path is class-conditional ImageFolder/ImageNet-style; `ffhq_64`
  is unlabeled under the current protocol.
- A fair run requires training or adapting the D-AR tokenizer before AR
  training, which is not equivalent to the current 5k-step formal64 pixel
  diffusion baseline budget.

Updated matrix:

```text
artifacts/reports/paper_comparison_matrix_2026-07-09_p0_formal64_plus_improved_diffusion_edm_dar_feasibility/paper_comparison_matrix_audit.md
```

Status counts:

```text
completed=29
missing=42
needs_adapter=32
partial=1
protocol_blocked=8
```

Next use: discuss D-AR as a nearest related/threat method for mechanism
positioning, or revisit it under a separate ImageNet-256 protocol with explicit
tokenizer-training budget.
