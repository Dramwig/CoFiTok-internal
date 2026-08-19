# Conditioning pipeline lineage observer v1

Date: 2026-08-19

## Outcome

The five-stage conditioning-repair pipeline now has a persistent, CPU-only,
non-signaling lineage observer with an immutable deployment receipt. The
observer binds every waiting supervisor to its exact Git revision, source
files, process identity, standing authorization, and at-most-once launch
boundary. It does not authorize or start any experiment.

The deployment receipt is stored remotely at:

`/root/autodl-tmp/CoFiTok/checkpoints/generation/conditioning_pipeline_lineage_observer_v1/deployment_evidence_v1/deployment_receipt.json`

- bytes: `12,869`
- mode: `0444`
- SHA256:
  `c3b00ded4727b113d3c9b781876c247e6923aa2d646360caffab56c9da67b556`

The tracked machine-readable mirror is:

`artifacts/reports/generation/conditioning_pipeline_lineage_observer_v1_2026-08-19.json`

It is byte-identical to the remote receipt.

## Exact code identity

- deployed revision:
  `a58f2e28abdd671550d7c70b0303184bbdb029e4`
- deployed tree: `0c41f230d572f4524406b5679335775bbe2b23ac`
- branch: `analysis/generation-conditioning-pipeline-lineage-observer-v1`
- isolated checkout:
  `/root/autodl-tmp/CoFiTok/checkouts/conditioning-lineage-observer-a58f2e2/CoFiTok-internal`
- tracked state: clean
- plan SHA256:
  `9cbae4a436fe23ce4f1022c7a20654f9d4192b03a4ecf05af0538da95617c7af`

The incremental bundle is `20,217` bytes with SHA256
`014ffd4510b978293d678a988f86d72c20d992af11b224545528eabcfe046c89`.
It advertises only the deployed branch at `a58f2e2` and requires
`1ff6bb3db932ef9e43ddfafc067db779dab797d9`.

The newly recloned formal repository does not contain that prerequisite
object, so bundle verification against the formal repository correctly fails.
Verification against the isolated deployed checkout passes. No object was
fetched into the formal checkout solely to make this receipt pass.

## Persistent observer

- PID: `427175`
- process start ticks: `1676465875`
- nice level: `19`
- poll interval: `10` seconds
- stale threshold: `300` seconds
- timeout: `31,536,000` seconds
- CUDA allocation: forbidden and absent
- process signaling: forbidden and absent
- output:
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/conditioning_pipeline_lineage_observer_v1/lineage_report.json`
- log:
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/conditioning_pipeline_lineage_observer_v1/lineage_observer.log`
- lock:
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/conditioning_pipeline_lineage_observer_v1/lineage_observer.lock`

The process PID, start ticks, cwd, executable, and NUL-delimited command-line
SHA256 are bound in the receipt. At receipt validation the process remained
alive with command-line SHA256
`a1d017081d186effea7a3daa3daf970c541505d8ad40071ac880c4b70ef24ad5`.

## Immutable source snapshots

The deployment evidence directory contains read-only snapshots of the live
lineage report, observer log, and quality-bridge pair monitor:

- lineage report: `24,448` bytes, SHA256
  `b084ab9fa7d2759f33743b76d53f9a96aad41bf8f29f90d73bac7396c673d9f2`;
- observer log: `15,210` bytes and `90` lines, SHA256
  `2964518e2ba7c92d70e69a1aec447f09a0da53583c2daec6c4f5ae1be978f319`;
- pair monitor: `40,419` bytes, SHA256
  `b39cdf761d25aaa057951fd6aa11a1b8a36f67ab2f986b45801f1a9577e97fb6`.

The lineage snapshot reported `waiting`, no issues, valid standing
authorization, five healthy supervisors, no child process, and no launch. All
five stage observation counts were exactly `90`.

## Multi-poll evidence

Six independent read-only captures were taken across more than one minute.
The minimum stage observation count increased monotonically from `57` to
`65`; every report SHA was distinct; all six reports had zero issues, zero
children, and zero launches.

During the same interval the dense trainer advanced from step `27,150` to
`27,200` and its metrics file grew. During the later 26-check replay the
observer count had reached `121` and dense had reached step `27,350`. The only
GPU compute PID remained dense trainer `79894`.

This proves that the observer kept polling while the active training cadence
continued. It does not claim that the observer caused or improved training.

## Validation

- exact Linux conditioning-lineage tests: `13 passed`;
- Linux Python compile: passed;
- observer runbook `bash -n`: passed;
- `git diff --check`: passed;
- deployed checkout tracked state: clean;
- post-publication independent receipt replay: `26/26` checks passed.

Ruff is not installed in the pinned remote environment. The receipt records
this as `not_available_in_pinned_environment`; it does not claim a Ruff result
for this replay.

## Preserved state

The formal checkout remained unchanged at:

- revision: `1ebcc15210e63a776a2ba448481cbd8bb94a4066`
- tree: `659fa94726c4aec0afef49904f82b828bb62872b`
- branch: `scale/generative-system`
- tracked changes: `0`
- normal porcelain count: `89`
- porcelain SHA256:
  `18e5981f22a2ac255c7f60343429daa6ceea86412b1d7bd09bc474f2faf74004`
- `-uall` count: `301`

The five waiting supervisor PIDs remained `210203`, `245918`, `280369`,
`305233`, and `333312`. None was signaled or modified. The standing
authorization remained byte-identical with SHA256
`5fe64a0941acb55a21cb6479a726987c6259e93a772c47290f2d6883752de4df`.

## Scientific boundary

This receipt improves reproducibility and auditability. It is not evidence of
a generation-quality advantage. At the immutable snapshot CoFiTok was at
50K/100K and dense at 27,250/100K. Matched milestone sampling, terminal 100K
sampling, class fidelity, visual support, statistical uncertainty, and runtime
fairness must still complete before a broad CoFiTok generation advantage can
be claimed.
