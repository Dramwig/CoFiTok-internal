# CoFiTok Code Architecture

This scaffold keeps high-cohesion modules with narrow interfaces so the first
MVP can evolve into stronger predictors or larger diffusion backbones without
rewriting the whole training path.

## Boundaries

- `data` owns dataset lookup and dataloader construction only.
- `diffusion` owns schedules and noising / denoising algebra only.
- `models.predictors` owns `T_k`, the expressive next-token predictor.
- `models.synthesis` owns `S_k`, the restricted condition-free token-to-noise operator.
- `models.cofitok` wires predictors and synthesis into ordered prefix outputs.
- `training` owns losses and step assembly, without dataset or CLI concerns.
- `diagnostics` owns zero / random / shuffle checks for non-degeneration.
- `scripts` are thin entrypoints.

## Sampling

`scripts/sample_checkpoint.py` runs prefix-aware DDIM sampling from a trained
checkpoint. It starts from Gaussian noise, predicts ordered CoFiTok components
at each reverse step, and can use either the full token budget or a prefix
budget `m`. With `--save-images`, it also writes individual PNG files under
`samples_prefix_<m>/` for generated-sample distribution checks.

```bash
PYTHONPATH=src python scripts/sample_checkpoint.py \
  --config configs/train_cifar10_k8_denoisepath_p150_light_3k_cuda.json \
  --checkpoint /root/autodl-tmp/CoFiTok/checkpoints/<run>/checkpoint_final.pt \
  --output-dir /root/autodl-tmp/CoFiTok/checkpoints/<sample-run> \
  --num-samples 16 \
  --sample-steps 20 \
  --prefix-budgets 1,4,8 \
  --save-images
```

`scripts/evaluate_generated_samples.py` compares those generated PNGs with a
dataset split using the low-res Frechet proxy and optional torchvision
Inception Frechet features.

This sampler is a method-completeness and smoke-test tool for the MVP. The
current small-model checkpoints and generated-sample smoke metrics should not
be presented as final unconditional generation quality.

## Reporting

`scripts/make_method_figure.py` renders the paper-facing Figure 1 schematic
from code, producing both PNG and SVG outputs plus a manifest:

```bash
PYTHONPATH=src python scripts/make_method_figure.py \
  --output-dir artifacts/figures/method_overview_2026-07-08
```

The figure explicitly records the default `S_k` boundary: current token only,
no `x_t`, timestep, label, prompt, previous tokens, skip features, learned
constants, or final VAE-style decoder.

`scripts/summarize_experiments.py` aggregates report JSON files into the
deterministic `experiment_summary.json` and CSV/Markdown tables. Treat those
summary artifacts as the source for MVP tables before hand-copying values into
docs.

`scripts/validate_summary_consistency.py` checks that `experiment_summary.json`,
summary CSV files, and selected Markdown documents agree on row counts. Use it
after regenerating summaries and before copying values into reports:

```bash
PYTHONPATH=src python scripts/validate_summary_consistency.py \
  --summary-dir artifacts/reports/summary_2026-07-08 \
  --doc docs/reports/cofitok_mvp_report_2026-07-08.md \
  --doc docs/records/2026-07-08_completion_audit.md
```

`scripts/make_report_figures.py` renders draft PNG figures from the summary:

```bash
PYTHONPATH=src python scripts/make_report_figures.py \
  --summary artifacts/reports/summary_2026-07-08/experiment_summary.json \
  --output-dir artifacts/figures/summary_2026-07-08
```

Current figure package:

```text
artifacts/figures/summary_2026-07-08/figure_manifest.json
```

Generated-sample Frechet figures are smoke evidence only unless the sample
count and protocol are upgraded to a publication-scale FID setup.

`scripts/make_visual_panels.py` composes existing PNG report artifacts into
paper-facing qualitative panels:

```bash
PYTHONPATH=src python scripts/make_visual_panels.py \
  --reports-root artifacts/reports \
  --output-dir artifacts/figures/visual_panels_2026-07-08
```

These panels are for visual inspection and draft paper assembly. They do not
replace quantitative tables or formal FID/LPIPS evaluation.

## Swappable Components

To replace the predictor, implement a module that returns:

```python
list[torch.Tensor]  # length K, each [B, C_token, H, W]
```

`ModelConfig.predictor_use_feedback=false` disables token-to-token feedback
inside the tiny predictor. This is the simultaneous-prediction ablation: all
token heads read the same hidden state, while `S_k` remains unchanged and
restricted.

To replace the synthesis operator, preserve this contract:

```python
component = S_k(token_k)  # no condition, no timestep, no prompt, no previous tokens
```

`S_k(0) = 0` should remain true by construction, preferably with bias-free
linear / convolutional layers.

`ModelConfig.synthesis_mode="deep_decoder"` is an ablation-only escape hatch.
It keeps the same token-only interface, but uses biased nonlinear convolutions,
so it can violate `S_k(0)=0` and should not be reported as the default CoFiTok
operator.

## Training Objectives

`LossConfig` keeps the idea-document losses configurable. The default path
uses epsilon, prefix, monotonic, and zero-token terms for the lightweight
baseline. Additional terms are available for ablations or second-stage runs:

- `residual_component_weight` supervises each component against the target
  prefix residual.
- `energy_budget_weight` with `energy_target` shapes per-token component
  energy.
- `component_decorrelation_weight` penalizes batch-level cosine correlation
  between dense component outputs. `component_decorrelation_start_step` and
  `component_decorrelation_warmup_steps` optionally delay and ramp this term so
  prefix/order supervision can settle before component separation pressure is
  applied.
- `denoise_path_prefix_weight` and `denoise_path_component_weight` supervise
  the pixel-space denoise path used by the current main runs.
- `sampled_*`, `group_residual_*`, `epsilon_band_*`, and `tail_*` terms support
  targeted tail-utilization and prefix-ordering ablations.

## First Smoke Target

The first server-side validation is random data only:

```bash
PYTHONPATH=src python scripts/smoke_forward.py \
  --config configs/smoke_random_cpu.json \
  --output /root/autodl-tmp/CoFiTok/checkpoints/smoke/random_forward_report.json
```

The report should include losses, prefix shapes, component energy, and synthesis
diagnostics.
