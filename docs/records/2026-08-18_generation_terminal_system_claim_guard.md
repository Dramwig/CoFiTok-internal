# Terminal generation system claim guard (2026-08-18)

## Outcome

The full-data 100K quality-bridge pipeline already had strict, source-bound
reports for absolute/matched quality, paired statistical uncertainty,
requested-class visual panels, and runtime/compute fairness. Those reports were
individually fail-closed, but no single terminal artifact required all four
evidence families to refer to the same physical quality-bridge run before
publishing the final scoped claim policy.

This change adds:

- `scripts/build_generation_terminal_system_claim_guard.py`
- `scripts/run_generation_terminal_system_claim_guard_waiter.py`
- focused builder and waiter tests

The new guard is CPU-only and non-authorizing. It does not modify the quality
bridge result, the statistical decision, the visual audit, the runtime decision,
or any training/sampling process.

## Exact bound sources

The builder requires exact bytes/SHA256 identities for:

1. `reports/quality_bridge_result.json` under the full-data 100K output root;
2. the terminal statistical claim-language guard;
3. the completed requested-class visual-audit waiter status and its embedded
   visual report/panel identities;
4. the terminal runtime/compute claim guard.

It then replays the statistical claim guard from its embedded qualification and
paired-uncertainty sources, rechecks the quality-result identity used by that
qualification, rechecks every visual panel and terminal sampling/checkpoint
binding, and rechecks the runtime guard's fairness and pair-monitor source
identities. The runtime sources must resolve to the same quality-bridge output
root, Git revision, dataset, and 100K training budget.

## Aggregate decision

The unified guard reports `pass` only when:

- the bound quality bridge absolute screen passes;
- the bound FID point estimate is lower for CoFiTok than `dense_identity`;
- paired block-KID has a bootstrap 95% interval with upper bound below zero;
- the one-sided exact sign-test satisfies `p <= 0.05`;
- the requested-class visual panels are present and source-revalidated; and
- the runtime guard is valid for the same matched training pair.

If the absolute quality screen passes but paired statistical support does not,
the unified result is `hold`, not an exception and not a positive claim. A
positive statistical claim is never allowed to override an absolute-quality
hold.

Runtime direct-comparison permission remains independent: incomplete exclusive
GPU observation coverage yields observational lower-bound language without
invalidating a separately qualified sample-quality claim.

## Claim boundary

Even on `pass`, the guard permits only the exact matched full-data ImageNet-256
100K distribution-quality statement. It explicitly forbids:

- FID confidence-interval or FID-significance language;
- treating requested-class panels as a quantitative metric;
- an automated absolute-usability claim;
- equal wall-clock, GPU-hour, or FLOP budget language;
- cross-tier ranking against D-AR, MAR, or ReTok;
- broad generation superiority or SOTA language;
- training, sampling, export, release, promotion, or process signaling.

## Local CPU validation

Using `C:/qbfd5/.venv/Scripts/python.exe` with
`CUDA_VISIBLE_DEVICES=-1`:

- new terminal-system builder/waiter tests: `7 passed`;
- combined quality/statistical/visual/new-guard regression subset: passed;
- both new scripts passed `py_compile`.

The existing Linux-only visual waiter imports `fcntl`, so its direct waiter test
is not collectable under native Windows. The new waiter has a Windows-safe local
test path while retaining an exclusive `fcntl` lock on Linux.

An isolated Linux checkout at commit
`8ec9a09ddcd981c1ffbd06b6bda81386b33321de` then passed:

- the focused quality/statistical/visual/new-guard subset: `44 passed`;
- the complete terminal uncertainty/claim/visual/new-guard subset: `100 passed`.

The second run used the real non-symlink executable
`/root/autodl-tmp/conda/envs/pf-vlm/bin/python3.10`. Running the same tests
through the conda `python` symlink correctly triggered four pre-existing
symlink-rejection tests; this was an invocation issue rather than a code failure.

## CPU waiter deployment

The source-bound waiter was deployed from the isolated checkout:

```text
/root/autodl-tmp/CoFiTok/checkouts/terminal-system-guard-8ec9a09/CoFiTok-internal
revision: 8ec9a09ddcd981c1ffbd06b6bda81386b33321de
tree: 344f6365d2e962e350b73f5ce4bc18c005f6dff9
PID: 965918
```

Authoritative outputs:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/
  stability_full_data_100k_base128_quality_bridge_v1/reports/
  terminal_system_claim_guard_v1/
```

The deployment receipt is `deployment_receipt.json`, `5,455` bytes, SHA256
`28fbdae892f8072f63849fd29d82bbb16de4c02076c46410172f276dea29d9ad`.
It binds the incremental bundle, builder/waiter bytes and SHA256 values, exact
checkout identity, process start ticks and argv, target paths, formal-checkout
snapshot, and GPU snapshot.

The compact local deployment summary is
`artifacts/reports/generation/terminal_system_claim_guard_waiter_deployment_2026-08-18.json`.

Initial waiter status is `waiting` for the exact quality result, statistical
claim guard, runtime claim guard, and terminal visual-audit status. This is the
expected state while the matched 100K pair remains incomplete.

## Live-process boundary

No trainer, controller, monitor, upstream waiter, or GPU process was signaled,
paused, restarted, or replaced while implementing or deploying this guard. At
the post-deployment snapshot, CoFiTok training remained active at step `42,600`,
the pair monitor remained `running / cofitok_training / issues=[]`, and the only
GPU compute process remained trainer PID `619775`. The formal remote checkout
remained at `1ebcc15210e63a776a2ba448481cbd8bb94a4066` and was not modified.
