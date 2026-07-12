# D-AR Adapter Feasibility

This adapter keeps `showlab/D-AR` as an external repository and records only
CoFiTok-side feasibility glue.

External repo:

```text
/root/autodl-tmp/CoFiTok/baselines/repos/d_ar
```

Current status: feasibility-only for the formal64 table.

D-AR is a nearest-method threat because it recasts diffusion as autoregressive
token generation. It is not currently a drop-in fair formal64 baseline because
the public code and checkpoints target ImageNet-256 with a learned sequential
diffusion tokenizer plus class-conditional Llama-style AR models.

Formal64 blockers recorded by `probe_dar_adapter.py`:

- AR train/sample image-size choices exclude 64.
- Tokenizer train image-size choices exclude 64.
- Official pretrained tokenizer and D-AR checkpoints are ImageNet-256.
- AR training is class-conditional ImageFolder; `ffhq_64` is unlabeled and
  would require a protocol change.
- A fair run would require training or adapting the D-AR tokenizer before AR
  training, which is not equivalent to the 5k-step monolithic pixel baselines.

Recommended use:

- Keep D-AR in related/nearest-method discussion for the MVP formal64 table.
- Revisit as an ImageNet-256 or eval-only baseline if the paper expands to a
  higher-resolution protocol with explicit resource accounting.
