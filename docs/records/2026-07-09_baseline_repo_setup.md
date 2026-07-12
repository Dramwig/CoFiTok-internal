# Baseline Repo Setup

Date: 2026-07-09

Purpose: create an auditable first-pass baseline code inventory without mixing
third-party repositories into `CoFiTok-internal`.

Policy:

- External GitHub clones go to `/root/autodl-tmp/CoFiTok/baselines/repos/<alias>`.
- CoFiTok-owned adapters, runner glue, configs, patches, and manifests stay in
  `CoFiTok-internal`.
- No pretrained weights, datasets, generated samples, or large logs are pulled
  during this step.
- Full fairness experiments are separate follow-up work and must record data
  alias, preprocessing, train budget, sampling budget, metrics, and patches.

First-pass clone set:

| Alias | Priority | Role |
|---|---|---|
| `edm` | P0 | monolithic pixel-space diffusion |
| `improved_diffusion` | P0 | DDPM-style pixel-space diffusion |
| `d_ar` | P0 | nearest diffusion-as-AR baseline |
| `ml_flextok` | P1 | ordered/flexible visual tokenizer |
| `mar` | P1 | continuous-token AR with diffusion loss |
| `titok_1d_tokenizer` | P1 | compact 1D visual tokenizer |
| `retok` | P1 | flexible tokenizer / early-token concentration |

P2 repos (`latent_diffusion`, `dit`, `var`, `selftok`) are registered but not
cloned in the first pass.

Remote command:

```bash
cd /root/autodl-tmp/CoFiTok/CoFiTok-internal
python scripts/baselines/clone_baseline_repos.py \
  --registry baselines/registry.json \
  --repo-root /root/autodl-tmp/CoFiTok/baselines/repos \
  --manifest-output docs/experiment_conditions/baselines_repo_manifest_2026-07-09.json
```

Expected manifest:

```text
docs/experiment_conditions/baselines_repo_manifest_2026-07-09.json
```

Result:

- Manifest copied back to `CoFiTok-internal/docs/experiment_conditions/baselines_repo_manifest_2026-07-09.json`.
- `failed_count = 0`.
- External `.git` directories were found only under `/root/autodl-tmp/CoFiTok/baselines/repos/*`, not inside `CoFiTok-internal`.

Pinned repositories:

| Alias | Status | Commit | License candidates |
|---|---|---|---|
| `edm` | cloned | `008a4e5316c8e3bfe61a62f874bddba254295afb` | `LICENSE.txt` |
| `improved_diffusion` | cloned | `1bc7bbbdc414d83d4abf2ad8cc1446dc36c4e4d5` | `LICENSE` |
| `d_ar` | cloned | `cd921b892c14ef52a06133b21a73257a6399ad52` | `LICENSE` |
| `ml_flextok` | cloned | `28e0ffc24e590bdab0099d4a317b601cdc674a5b` | `LICENSE`, `LICENSE_WEIGHTS` |
| `mar` | cloned | `c6d53f7fa6427634b5850ebed771b7c2d19ea21f` | `LICENSE` |
| `titok_1d_tokenizer` | cloned | `942a96fbdd873780179d1b78d5462911528bf8c8` | `LICENSE` |
| `retok` | cloned | `ffaa8a7b4ea7e39c09f78010dfcc239c2a054ed7` | none found at top level |

Next baseline work:

1. Inspect each repo README/license/environment file before running anything.
2. Build P0 adapters/configs first: same-backbone dense, EDM or improved-diffusion, and D-AR.
3. Keep `imagenet_1k_64x64_hf` and strict-source `downsampled_imagenet_64` as separate result aliases.
