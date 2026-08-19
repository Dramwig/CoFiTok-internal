# Full-data 40K matched trajectory waiter

Date: 2026-08-19

## Outcome

A source-bound, CPU-only waiter now guards publication of the full-data matched
training trajectory through step 40,000. It waits for the exact dense
EMA-teacher full-warmup audit, freezes both canonical JSONL prefixes, invokes
the existing resume-aware schema-3 builder, and requires two byte-identical
replays before publishing the report.

At the first post-launch check the waiter was healthy and waiting:

```text
pid: 691651
status: waiting
detail: dense_schedule_report_missing
dense step: 36,700 / 50,000
```

The authoritative paths are:

```text
status:
/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_base128_quality_bridge_v1/reports/matched_training_trajectory_step_00040000_waiter_status.json

future output:
/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_base128_quality_bridge_v1/reports/matched_training_trajectory_step_00040000_v1
```

The future output did not exist at deployment time. Its absence is expected
until the independently deployed dense schedule waiter publishes a passing
step-40K audit.

## Provenance

Control checkout:

```text
revision: fbda904ac97b6049f0efa1e96b66ea16526fdc8f
tree: 94f40df8c451af9496cd464aa73a42f3f2fbdbba
branch: analysis/generation-full-data-40k-trajectory-waiter-v1
source SHA256: 826663a67c8de2ad20e304b36f2622858bf8ac539df75562abb4a3d2c2795e66
```

Resume-aware builder checkout:

```text
revision: 40ee55202545ea01e48fb6642d1b5a51b5264663
tree: 2ae2599a7dddb000a6042132b3d1065046ed3944
branch: analysis/generation-resume-aware-matched-trajectory-v1
source SHA256: 2477345f81a99895f73b5c9a93844c8005ca959a2ed74ab3fa5a077c4753c21a
```

The incremental deployment bundle advertises only the control revision, has
prerequisite `40ee552`, is 10,148 bytes, and has SHA256
`776861751a4be7c6a52eb06ae862432755d514ceab7b7803044be53356930b86`.
It passed local and remote `git bundle verify`. Fetching the bundle did not
move the builder checkout HEAD.

The byte-identical local and remote deployment receipt is 3,564 bytes with
SHA256 `9c36d63e067c4d72447119b1bb3073688b4b34894ff27a12acaea77f801b2674`.

## Validation

- Local focused builder/waiter suite: `32 passed`.
- Linux waiter/builder/pair-contract/reporting selection: `42 passed` with
  `CUDA_VISIBLE_DEVICES=-1`.
- Python compilation and CLI help passed.
- A real-path `--once` rehearsal wrote only a waiting status under `/tmp`,
  created no trajectory output, and left no process behind.
- The authoritative waiter produced a second heartbeat after 60 seconds.
- A competing invocation failed with `matched trajectory waiter lock is held`.
- GPU state remained one process only: the active dense trainer PID `79894`.

## Safety and claim boundary

The waiter has no process-signal path and never imports or invokes a GPU
operation. Builder children receive `CUDA_VISIBLE_DEVICES=-1`. The status,
receipt, and eventual trajectory report all keep promotion, release, full
training, and full-300K authorization false.

The eventual 40K trajectory is a training-space diagnostic only. It cannot
establish sample quality, replace the matched 50K EMA evaluation, authorize a
later stage, or support a broad generation-advantage claim.
