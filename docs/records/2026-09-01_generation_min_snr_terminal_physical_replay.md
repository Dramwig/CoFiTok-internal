# Matched Min-SNR terminal physical replay (2026-09-01)

## Scope

This record documents the non-authorizing CPU-only physical replay of the
completed matched Min-SNR gamma=5 50K pilot. It did not launch training,
sampling, evaluation, continuation, promotion, export, release, or any GPU
process. The pilot remains screening-only and cannot replace the 100K
terminal hold.

## Fail-closed diagnosis and repair

The first replay from guard revision `eef0fc7e9ff8136fa71b4803da998867ad818407`
failed closed on `legacy_gamma0_cofitok sampling progress differs`. The
production `sampling_progress.json` schema is `1`, while
`SAMPLING_REPORT_SCHEMA_VERSION` is `6`; the guard had compared the former to
the latter. No canonical guard output was created by that attempt.

The guard-only fix uses an explicit `SAMPLING_PROGRESS_SCHEMA_VERSION = 1`
and updates the focused fixture to the production schema. The focused suite
reported `11 passed`; compilation and `git diff --check` also passed. The
remote isolated guard checkout is revision
`831caff20d6c7052d3012037ebc7ff622507b101`, tree
`8a4a5bf43c96789e418e3299c16b7eb23a7fc267`, branch
`analysis/generation-min-snr-terminal-physical-replay-v2-20260901`. Its
training replay source remains pinned to revision
`842a34130e82f241330707118a05bf6ed01e263e`, tree
`3fd4c4538d15b85233b1f8b582dce0f185dedba2`.

## Replay result

The canonical report is:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/min_snr_gamma5_matched_50k_pilot_v1/reports/terminal_physical_guard_v1/terminal_physical_guard.json
```

Its SHA256 is
`3ff6ad8314e0ea9e10c88aee785564d6713d2e93ddfd9f83f33e1ed33f0b99ed`.
The builder completed two identical physical replays and the verifier
completed a third replay with the same guard SHA. Source-code replay passed
three times; all four checkpoint payloads, sidecars, sample sets, classifier
weights, reports, and training-control bindings were physically rehashed.

The report preserves:

```text
status=pass
scientific_status=physical_evidence_replayed_non_authorizing
selection_status=no_shared_min_snr_candidate_at_50k
terminal_status=hold
generation_advantage_proven=false
```

The authorization boundary remains all false, including GPU execution,
sampling, continuation beyond 50K, full 300K, promotion, export, release, and
process signals.
