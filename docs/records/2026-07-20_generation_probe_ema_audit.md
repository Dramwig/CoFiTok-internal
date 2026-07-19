# Short-probe EMA lag audit

Date: 2026-07-20

Branch: `scale/generative-system`

## Motivation

The rank-recovery candidates train for only 5,000 optimizer steps but use the
production EMA recipe (`decay=0.9999`, linear warmup over 2,000 updates). At the
5K checkpoint, the resulting shadow has a weighted mean source step of about
2,367, a median source step of 1,961, and only about 9.52% of its mass on the
latest 1,000 updates. EMA remains the required formal inference weight, but a
short-horizon mechanism comparison based on EMA alone can confuse objective
quality with averaging lag.

## Implementation

`scripts/audit_generation_probe_ema.py` binds each candidate's training report,
EMA checkpoint evaluation, and raw-model checkpoint evaluation by checkpoint
SHA256, full config, Git provenance, 5K step, 512-image count, timestep 500,
and the same 18-order protocol. It reports endpoint MSE/PSNR, ordered path AUC,
ordered rank, component energy, and an analytical EMA weight-age profile.

The rank-recovery runbook now adds a raw-model checkpoint evaluation after the
existing EMA evaluation and writes `ema_lag_audit.json`. It does not duplicate
DDIM sampling. The existing EMA prefix samples remain the visual selection
source and all formal 50K/300K sampling remains EMA-only.

The audit is explicitly non-formal and cannot authorize 50K or 300K. Candidate
selection must use the raw-model view only to detect EMA lag; the chosen recipe
must still pass the fresh 50K EMA promotion gate.
