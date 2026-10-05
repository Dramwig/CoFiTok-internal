# Quality bridge server reverification (2026-08-20)

## Scope

This is a read-only live-state reverification after the `pro6000` connection
was updated to the current server. It does not modify, signal, pause, restart,
or authorize any GPU process. It does not change the frozen paper evidence or
the formal repository checkout.

Observation time: `2026-08-20T03:05:29+08:00`.

## Server and active execution

- Hostname: `autodl-container-scvpc4sj5x-0e706d96`.
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition.
- The only GPU compute process was trainer PID `79894`, using `77,970 MiB`.
- Free bytes on `/root/autodl-tmp`: `313,187,487,744`.
- Quality-bridge root:
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_base128_quality_bridge_v1`.
- Active execution checkout:
  `scale/generation-stability-quality-bridge-100k@cf0e5faa94bf4ab38d947b921935b3b765b5537a`.
- Active execution tree: `6cef27723196fd363379bca2e7b85b1678ebd777`.
- Tracked execution status was clean.
- Runbook PID `618821`, pair-monitor PID `619509`, dense watchdog PID
  `79830`, and dense trainer PID `79894` were alive.
- Pair status/stage was `running / dense_identity_training`, with no reported
  health issues or unrelated current GPU process.
- CoFiTok was at step `50,000/100,000`.
- Dense was at step `47,150/100,000`, with `3,017,600` images seen and finite
  latest metrics.

The formal checkout remained
`scale/generative-system@1ebcc15210e63a776a2ba448481cbd8bb94a4066`.
Its full porcelain identity remained exactly `89` rows with SHA256
`18e5981f22a2ac255c7f60343429daa6ceea86412b1d7bd09bc474f2faf74004`.
No cleanup or mutation was performed.

## Bound live sources

| source | bytes | SHA256 |
|---|---:|---|
| `pair_monitor.json` | 40,440 | `b43b36ac5213ab4afe304fb9ed57c0c62d704f4f08740cf2d05694d385581025` |
| `dense_rollout_x0_u2_ema_teacher/training_watchdog_step_00050000.json` | 2,027 | `897496b5ad3654a646c101d17ff2e5f00647367f95e2d2866e4b2909df743049` |
| `reports/runtime_compute_fairness/waiter_status.json` | 1,980 | `23c97dcd9919182e289012379631f1bd9a92796b4c7d918938508ab0aa4f8c9b` |
| `reports/controller_identity_guard_v1/guard_status.json` | 3,954 | `44cf7b1b5ae4d2ffba3d9a1cc441de43369486c7101b51ed441490a14f436ecf` |

These are mutable live reports. The identities above bind only this observation
and must not be substituted for terminal report identities.

## Compute-claim boundary

The persistent pair-monitor contention evidence has a maximum observation gap
of `16,377.985275` seconds. It also retained the bridge's own CoFiTok 50K
milestone sampler as an observed non-training GPU identity. Therefore raw
process wall-clock and throughput cannot be used for a direct architectural
speed comparison. The terminal pair monitor must preserve
`training_wall_clock.direct_comparison_allowed=false`; this is an evidence
limitation, not a training failure.

The independent runtime/compute waiter remains valid for the narrower facts it
actually proves: exact shared data, optimizer, micro-batch, accumulation,
effective batch, steps, images seen, parameter gap, resolved configs, and
physical recovery compute accounting. Its measured elapsed time and throughput
remain descriptive outcomes only. This boundary does not invalidate matched
FID, precision, recall, class-fidelity, checkpoint, sampling, or mechanism
comparisons.

## Downstream supervisors

The conditioning-ranking and 250M capacity chains were alive with fresh
heartbeats. All inspected supervisors were `waiting` with `child_pid=null`.
No downstream GPU work was active, and no full 300K launch was authorized by
the observed artifacts.

The next required transition remains:

1. dense reaches the exact 50K checkpoint;
2. checkpoint integrity and 2,048-sample DDIM-50 milestone evaluation finish;
3. the paired 50K milestone is built;
4. CoFiTok and dense exact-resume serially to 100K;
5. the matched terminal 10K DDIM-100 evaluation produces the quality-bridge
   result and source-selected follow-up decision.

