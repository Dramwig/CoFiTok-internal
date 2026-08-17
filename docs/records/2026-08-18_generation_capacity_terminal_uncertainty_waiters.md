# Capacity terminal matched-uncertainty waiters

Date: 2026-08-18

## Outcome

Two source-bound terminal matched-uncertainty waiters are active on `pro6000`.
They cover the capacity-completion 100K result and the future capacity-full 300K
final gate. Both deployments are permanently non-authorizing: they cannot launch
training, launch the full 300K stage, authorize release, signal trainers, or
signal unrelated processes.

The machine-readable deployment summary is:

```text
artifacts/reports/generation/capacity_terminal_uncertainty_2026-08-18/deployment_summary.json
```

## Control and evaluation identities

The running waiters use the clean control checkout:

```text
revision: b0714052424065c18a21384e1d998c529724d098
tree:     f9f5f9e6f763ef1063c1a598b7a6e882e0f92cf5
branch:   analysis/generation-capacity-terminal-uncertainty-v1
path:     /root/autodl-tmp/CoFiTok/checkouts/capacity-terminal-uncertainty-b071405/CoFiTok-internal
```

The shared uncertainty evaluator remains clean and detached at:

```text
revision: 1c8ef207cb6d79850d73a45abc345fc421e6aa7f
tree:     a09f14a0eca44af6db8e6781863a463163ddf646
path:     /root/autodl-tmp/CoFiTok/checkouts/matched-uncertainty-1c8ef20/CoFiTok-internal
```

The immutable receipt builder is clean and detached at:

```text
revision: f7e3f9fdcdf7238eefaa3efac9a8215df6122e03
tree:     70c20a57c9adf0315e12505468db142da39a07f9
path:     /root/autodl-tmp/CoFiTok/checkouts/capacity-terminal-receipt-f7e3f9f/CoFiTok-internal
```

Its validation rejects an authorizing claim boundary, PID or command drift,
dirty checkouts, incorrect bundle ancestry, an unexpected GPU allocation, or a
source anchor that already existed before the deployment receipt was written.

## Capacity-completion 100K waiter

```text
PID:          858118
status:       waiting
phase:        source
detail:       waiting_for_capacity_terminal_source_anchor
output root:  /root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_capacity_terminal_matched_uncertainty_v1
source:       /root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_capacity_probe_250m_10k_v1/reports/capacity_completion_100k_result.json
```

The source contract binds the exact completion execution revision `892c1d6`,
training revision `3a7dc9d`, and result-waiter revision `b9f940d`. The immutable
deployment receipt is 13,142 bytes with SHA256:

```text
d070c3feb76cb67466205a89776202bef32c502d1ef2632696d84aab8d420f26
```

When the exact source exists, the waiter builds a 10,000-sample matched
manifest, evaluates twenty 500-sample generated blocks against five real folds,
and runs 10,000 bootstrap repetitions. A relative-advantage claim is permitted
only if both the source quality gate and the matched uncertainty audit pass.

## Capacity-full 300K waiter

```text
PID:          859186
status:       waiting
phase:        source
detail:       waiting_for_capacity_terminal_source_anchor
output root:  /root/autodl-tmp/CoFiTok/checkpoints/generation/stability_capacity_full_300k_terminal_matched_uncertainty_v1
source:       /root/autodl-tmp/CoFiTok/checkpoints/generation/stability_capacity_full_300k_v1/reports/final_generation_gate.json
```

The source contract binds training revision `d75dfea` and postprocessing
revision `5e04dbe`. The immutable deployment receipt is 12,578 bytes with
SHA256:

```text
43945c01037db2242a65529753ac257a84fcbf778d00d79b98de62401ef44e61
```

If the full stage is eventually reached and its exact final gate exists, this
waiter evaluates the matched 50,000-sample sets using one hundred 500-sample
blocks and 10,000 bootstrap repetitions. The waiter does not authorize that
training stage; it only audits a result produced by the separately gated
capacity pipeline.

## Verification

- Latest Windows targeted suite: `18 passed`.
- Windows capacity suite: `220 passed`; the only failure was the known
  CRLF-sensitive frozen-SHA test.
- Linux capacity suite at the first receipt-builder commit: `221 passed` with
  CUDA hidden.
- Latest Linux targeted suite at `f7e3f9f`: `18 passed`; Python compile passed
  with `CUDA_VISIBLE_DEVICES=-1`.
- Both waiter PIDs remained alive after their receipts were written and neither
  PID appeared in the GPU compute table.

## Live training boundary

At `2026-08-18T05:24:36+08:00`, the active full-data quality-bridge CoFiTok leg
was at step `35,150/100,000` (`2,249,600` images seen). Dense had not started.
PID `619775` remained the only GPU compute process and used about `85,286 MiB`.
The read-only EMA-teacher full-warmup audit was still waiting for step 40,000.

These deployments strengthen provenance only. They do not change the current
scientific conclusion: ordered prefix-denoising advantages are already
supported, while a formal large-scale relative generation advantage remains
unproven until the matched terminal source, quality gate, and bootstrap audit
all complete successfully.
