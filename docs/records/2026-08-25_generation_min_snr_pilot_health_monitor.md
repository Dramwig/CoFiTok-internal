# Matched Min-SNR 50K pilot health monitor (2026-08-25)

## Purpose

The generic generation-pair monitor cannot represent this pilot correctly:
the training recipe keeps a 100K cosine scheduler horizon while the physical
execution boundary is step 50,000. This observer therefore treats 100K only as
the immutable scheduler horizon and treats 50K as the hard per-method stop.

The observer is independent of the active training checkout. It is stdlib-only,
CUDA-hidden, CPU-only, read-only except for its own atomic status file and flock,
and permanently non-authorizing. It contains no process-signal or trainer launch
path.

## Bound execution

- training checkout:
  `/root/autodl-tmp/CoFiTok/checkouts/min-snr-matched-pilot-842a341/CoFiTok-internal`
- training revision: `842a34130e82f241330707118a05bf6ed01e263e`
- training tree: `3fd4c4538d15b85233b1f8b582dce0f185dedba2`
- training branch: `scale/generation-min-snr-matched-pilot-v1-20260825`
- output root:
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/min_snr_gamma5_matched_50k_pilot_v1`
- preparation SHA256:
  `8fa3b768015c08c6f34627dc767c4b2fddd465a43913495120a496ef411c6830`
- execution-gate SHA256:
  `19cb1560cca1feba38bf7a553652abf922e59d635d5574994996e37fcc72c014`
- controller launch-receipt SHA256:
  `6ce5a4a3f4d3225f68ece879cd07ceb01361d7a1315dc88d4beffab7b057f261`
- controller process-tree-audit SHA256:
  `9be864709967606a9c0da7eb90b73d84a765d505bf73b8350eb3891e6439fcca`
- controller runbook SHA256:
  `55cbf51e89c27da6c5c0fbf1e9a9e0c07b4b261b2342392c0e17c3bd846d6656`

## Checks

Every poll revalidates both Git identities and all source hashes; the exact
controller PID/start ticks/cmdline/cwd/PPID; controller ancestry; one direct
trainer with DataLoader workers allowed only below that trainer; sole-chain GPU
ownership; the execution flock through the controller FD symlink and fdinfo;
strict finite metrics with `samples_seen == step * 64`; fresh step 1; observed
Min-SNR downweighting; checkpoint cadence and metadata; completed training
reports and physical audits; disk runway; four evaluation-arm outputs; and the
exact final non-authorizing result boundary.

Startup grace is measured from the controller phase transition, not controller
lifetime. Active training metrics older than 1,800 seconds are reported as
`stalled`. Controller absence and flock release are legal only after a completed,
fully replayed pilot result. A completed result must retain `terminal_status=hold`,
`generation_advantage_proven=false`, and
`new_gate_required_for_any_followup=true`.

The detached deployment additionally requires PPID 1,
`CUDA_VISIBLE_DEVICES=""`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, nice 10,
and idle-class ionice.

## Verification before deployment

- `git diff --check`: pass
- Python compilation: pass
- focused pilot plus observer suite: `17 passed`
- Linux real-state one-shot probe: `running/training_cofitok`, `issues=[]`
- one-shot source SHA256:
  `21d0c7c83ef662c3e791a4780d6a33ac08bdc2afcce885c45d3fc621272bd5ad`
- one-shot observed CoFiTok state: step `1,050`, `67,200` samples, strict and
  finite; exact execution flock held; one controller; one direct trainer; sole
  GPU owner was that trainer; free bytes `264,439,267,328`

This observer cannot authorize training, sampling, evaluation, continuation
beyond 50K, full training, 300K, promotion, inference export, release, or process
signals. A monitor `pass` means only that the bounded pilot completed and its
non-authorizing result replayed exactly.

## Deployment

- implementation revision:
  `4bbc880ad3928b607ee61bc1be41d10a60853e4d`
- implementation tree: `6404585661c297cdb57447e81cdfd3f84331d569`
- remote checkout:
  `/root/autodl-tmp/CoFiTok/checkouts/min-snr-pilot-health-monitor-4bbc880/CoFiTok-internal`
- bundle: `17,102` bytes, SHA256
  `1343f548c36b500a87e3e7a0c11f6b1fa9db908f460539acf560bc7ff7b3e30e`
- launcher: `4,491` LF bytes, SHA256
  `990075d206c3993574d6fed24107b80eb4d6087e4bb4592bc952da04ee276d2e`
- live observer: PID `412598`, PPID `1`, start ticks `1729920946`
- canonical status:
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/min_snr_gamma5_matched_50k_pilot_v1/reports/health_monitor_v1/status.json`
- canonical deployment receipt: same directory, `deployment_receipt.json`,
  `4,869` bytes, SHA256
  `c085855775b75794ba715a3330069115e6cb6b632b78ffe4b59b17810b4b25a0`
- immutable initial status snapshot: `47,894` bytes, SHA256
  `42965d6398773e77b2ef807fff64d4a7bea546c8a031ccf60225b391407c63c9`

The prerequisite-aware bundle advertises only the observer branch and requires
the exact training revision `842a341`. Local and remote bundle verification,
Linux compilation, the 17-test focused suite, launcher syntax, clean checkout,
source hash, process identity, runtime limits, lock ownership, and three
successive monitor polls all passed. The sole GPU process remained trainer PID
`407563`; the observer never appeared in the GPU process list.

An earlier direct Windows-to-SSH launch command failed during local quoting of
the wrapper sleep argument, before any observer process, status, or lock was
created. The successful deployment used the independently hashed LF launcher;
the receipt preserves both outcomes without treating the failed command as a
deployment.
