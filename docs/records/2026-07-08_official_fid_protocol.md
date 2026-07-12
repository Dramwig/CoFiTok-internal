# 2026-07-08 Official FID Protocol

## Purpose

The generated-quality rows are streamed DDIM/Inception-Frechet evidence. They
are useful for internal comparisons, but they do not produce the image-directory
artifacts expected by external FID tools.

This record adds and executes a reproducible official-FID protocol without
changing the paper claim. It prepares `real/` and `generated/` PNG directories
and runs `pytorch-fid` through a wrapper script.

## Artifacts

- `scripts/export_official_fid_dirs.py`
- `scripts/evaluate_official_fid_dirs.py`
- `artifacts/runbooks/official_fid_protocol_2026-07-08.sh`
- `artifacts/reports/official_fid_protocol_2026-07-09/`
- `artifacts/reports/summary_2026-07-08/official_fid_summary.csv`

## Protocol

Remote execution:

```bash
ssh pro6000
cd /root/autodl-tmp/CoFiTok/CoFiTok-internal
export PYTHONPATH=src
/root/autodl-tmp/conda/envs/pf-vlm/bin/python -m pip install -i https://pypi.org/simple pytorch-fid
source /etc/network_turbo || true
bash artifacts/runbooks/official_fid_protocol_2026-07-08.sh
```

The runbook exports matched multiscale 20k controls:

| dataset | variant | images |
|---|---:|---:|
| Tiny ImageNet-200 | epsilon-only multiscale 20k | 10000 real / 10000 generated |
| Tiny ImageNet-200 | K8 light multiscale 20k | 10000 real / 10000 generated |
| ImageNet-family 64x64 HF | epsilon-only multiscale 20k | 50000 real / 50000 generated |
| ImageNet-family 64x64 HF | K8 light multiscale 20k | 50000 real / 50000 generated |

Each export writes:

```text
official_fid_export_manifest.json
real/*.png
generated/*.png
official_fid_report/official_fid_report.json
```

The manifest includes a direct `python -m pytorch_fid real generated` command
and the CoFiTok wrapper command.

## Results

The full protocol completed on `pro6000` on 2026-07-09. Reports were pulled
back to:

```text
artifacts/reports/official_fid_protocol_2026-07-09/
```

| dataset | variant | generated / real | official FID |
|---|---|---:|---:|
| Tiny ImageNet-200 | epsilon-only multiscale 20k | 10000 / 10000 | 148.8414 |
| Tiny ImageNet-200 | K8 light multiscale 20k | 10000 / 10000 | 152.2474 |
| ImageNet-family 64x64 HF | epsilon-only multiscale 20k | 50000 / 50000 | 140.7213 |
| ImageNet-family 64x64 HF | K8 light multiscale 20k | 50000 / 50000 | 134.0818 |

Interpretation:

- HF ImageNet-family official FID favors K8 light over epsilon-only on this
  matched multiscale 20k protocol.
- Tiny ImageNet-200 official FID favors epsilon-only over K8 light.
- This closes the strict `official_generation_quality_protocol` evidence gap.
- It does not justify a broad generation-quality superiority claim. The stable
  paper claim remains ordered prefix-controllable dense noise factorization with
  restricted `S_k`.

## Current Status

Protocol implementation, full-scale exports, and four official `pytorch-fid`
reports now exist. After the AAAI-27 anonymous-submission adaptation and the
2026-07-09 strict `downsampled_imagenet_64` source record, the current strict
completion audit is closed.
