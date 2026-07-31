# Large-capacity stability-full contract (2026-07-31)

## Scope

This change prepares the dormant ImageNet-256 `stability_full` matched pair
for a genuinely larger generation model. It does not modify the active 50K
qualification checkout, move the formal server repository, authorize 300K
training, or claim a completed CUDA feasibility result.

## Capacity tier

- CoFiTok config:
  `configs/generation/imagenet256_stability_rgbtail3_rollout_x0_u2_ema_teacher_k8_300k.json`
- Dense config:
  `configs/generation/imagenet256_stability_rollout_x0_u2_ema_teacher_dense_300k.json`
- Shared U-Net base channels: `256`
- CoFiTok parameters: `250,153,763`
- Dense parameters: `250,135,043`
- Relative parameter gap: `+0.00748396%`

The qualified `stability_scaling` pair remains unchanged at 128 base
channels and approximately 62.8M parameters. The recipe contract requires 256
channels only for `stability_full`; legacy `scaling`/`full` and
`stability_scaling` continue to require 128.

## Runtime contract

The training runtime selector now separates the candidate grid from an explicit
baseline candidate. Existing callers retain the `16x4` default and legacy
schema-v2 selection compatibility. The large stability-full runbook binds:

```text
candidates: 1x64,2x32,4x16,8x8,16x4
baseline:   1x64
effective batch: 64
```

New schema-v3 reports store the baseline, use
`estimated_speedup_over_baseline`, and bind the baseline in the frozen
selection lock. Both matched methods must complete the baseline below the VRAM
limit before any faster candidate can be selected.

## Storage contract

The 50K reference checkpoints come from the 128-channel pair, so using their raw
bytes would under-budget the 256-channel run by roughly four times. The full
runbook therefore passes `--checkpoint-size-multiplier 4.0`. Storage report
schema v2 records:

- measured reference checkpoint bytes;
- the multiplier;
- rounded-up planned checkpoint bytes;
- reserve bytes for 16 checkpoint slots;
- sample, additional, safety, required, and headroom bytes.

The stability completion audit requires this real full-training report and
rejects a multiplier below `4.0` or inconsistent arithmetic.

The first-launch storage report is stronger than the readiness-only report.
Readiness retains the historical `16,384` training-time sample reserve so the
already queued CUDA qualification artifact remains replayable. Before a full
launch receipt can be written, the launch runbook must instead reserve
`116,640` samples: the same `16,384` training-time allowance plus the exact
`100,256` images required by matched formal 50K post-evaluation and prefix
diagnostics. The terminal completion audit revalidates this aggregate runway.
This prevents a 300K training launch that fits its checkpoints but can only
discover after training that formal evaluation no longer fits on disk.

## Authorization boundary

The full runbook still requires a source-bound passing stability 50K promotion
gate before runtime benchmarking or training. At the time of this change the GPU
is occupied by the active 50K CoFiTok-to-dense queue, so CUDA runtime feasibility
for the 250M pair remains pending. No full 300K run is authorized or launched.

## Verification

Local Windows verification at commit
`fc6a077576e58357564a0d7e6ac39bdc548308b3`:

- full pytest: `804 passed, 3 skipped`;
- `git diff --check`: pass;
- changed Python entry points: `py_compile` pass;
- direct config validation: pass with the exact parameter counts above.

An incremental bundle from prerequisite
`c1efb12c6640f2d2d62ac7e9982c8804d96e7289` was verified and transferred
to the server:

- bundle bytes: `21,055`;
- bundle SHA256:
  `f5e6beabdfcdf12a6ca4a52b501341da81e00a0dc35180124b2b743d9dae2bc7`;
- isolated checkout:
  `/tmp/cofitok-generation-large-capacity-fc6a077`;
- Linux code suite excluding sibling-paper layout tests: `803 passed`;
- the four paper-layout tests in a real
  `CoFiTok/CoFiTok-internal` plus sibling `paper/` layout: `4 passed`;
- tracked runbooks: `95/95` passed `bash -n`;
- isolated checkout tracked status: clean.

The server-side config validator independently reproduced CoFiTok
`250,153,763`, dense `250,135,043`, and relative gap
`7.483957375776412e-05`. A read-only storage preflight using the existing
50K checkpoint measured:

- reference checkpoint: `1,006,321,770` bytes;
- multiplier: `4.0`;
- planned checkpoint: `4,025,287,080` bytes;
- required free bytes: `154,598,906,496`;
- observed free bytes: `219,438,235,648`;
- headroom: `64,839,329,152` bytes;
- result: `pass`.

CUDA feasibility was intentionally not run. At the verification snapshot the
active 50K CoFiTok job was at step `8,450/50,000`, monitor status
`running` with `issues=[]`, and trainer PID `319202` held
`84,122 MiB`. Dense had not started. The formal server repository remained
at `1ebcc15210e63a776a2ba448481cbd8bb94a4066`, and waiter v4 PID
`281834` was alive with no child process.
