# GPU contention provenance for matched generation cost (2026-08-02)

## Problem

The formal matched comparison reports exact steps, images seen, elapsed training
time, throughput, and peak VRAM. Steps, images, configs, and quality metrics are
source-bound, but raw wall time is also affected by other GPU jobs. During the
active stability dense 50K recovery, a FieldScope extraction process used about
15.4 GiB concurrently and changed recent dense 50-step intervals from roughly
130 seconds to roughly 278 seconds. Treating that slowdown as an architectural
cost difference would violate the matched-baseline claim.

## Evidence contract

`cofitok.gpu_contention` adds a persistent schema-v1 summary to every new pair
monitor refresh. It records:

- the monitor name and clean training Git identity;
- observation count, first/last timestamps, maximum poll gap, and whether
  coverage began before training and continued through pair completion;
- exact current training and GPU compute identities;
- every unrelated GPU PID/start-ticks identity with argv, cwd, process name,
  observation count, first/last observation, and maximum observed memory; and
- an explicit wall-clock decision with a machine-readable reason.

The monitor carries the prior summary through atomic rewrites and rejects a
changed monitor/Git binding, non-increasing timestamp, or reused PID/start
identity with different argv/cwd. A failed/incomplete NVIDIA or `/proc` query
and identity-list overflow preserve training health but make wall-clock coverage
incomplete. DataLoader children may
share trainer argv, but only PIDs present in the NVIDIA compute-process set are
treated as GPU processes.

Direct wall-clock and img/s comparison is allowed only when coverage starts
before any training step, observes active training, remains continuous within
2.5 polling intervals, reaches a clean pair completion, and observes no
unrelated GPU compute. Otherwise the same raw values remain available for
operations but are labelled observational-only. This does not weaken the direct
comparison of data, architecture, optimizer, steps, images, sampling protocol,
or quality metrics.

## Completion boundary

The large-scale comparison advances to schema v6 and binds the terminal pair
monitor as an additional source report. JSON rows, CSV, and Markdown carry the
wall-clock eligibility and reason; Markdown labels training time and throughput
as raw values. The terminal completion audit independently validates the bound
monitor evidence and rejects a promoted or inconsistent policy.

Both legacy-full and stability-full post-evaluation runbooks pass the exact
terminal monitor into the comparison builder. This change is local code and is
not deployed into the currently active immutable stability 50K training
checkout. It does not signal any process, restart a trainer, authorize full
300K, or reinterpret the active 50K quality gate.

## Verification

- Local project suite: `912 passed, 6 skipped` in 248.27 seconds.
- The two modified post-evaluation runbooks were streamed without deployment
  to `pro6000` and both passed the server's native `bash -n` parser (`2/2`).
- `git diff --check` passed. The local all-runbook checker was not used as
  syntax evidence because the available Windows/WSL environment has no
  `/bin/bash`; no remote project file was written for the native checks.
