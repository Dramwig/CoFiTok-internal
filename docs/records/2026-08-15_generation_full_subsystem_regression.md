# Full generation-subsystem regression

Date: 2026-08-15

## Outcome

The complete set of generation-named test modules was executed as one coherent
regression on Windows and CUDA-hidden Linux. Windows collected 1,089 tests from
116 cross-platform modules and completed with 1,081 passed, zero failures or
errors, and eight explicit platform skips. Linux collected 1,092 tests from all
117 modules and completed with 1,090 passed, zero failures or errors, and only
the two expected real-CUDA skips. No implementation change was required.

This closes the prior evidence gap between individually passing capability
suites and the combined generation subsystem. It exercises interactions among
training, exact resume, checkpoints, sampling, inference, quality gates,
capacity progression, deployment, control continuity, strong-baseline
comparison, and terminal completion. It remains CPU-fixture regression evidence:
it does not prove that the pending full-data bridge or later physical training
ran, that samples are high quality, or that a promotion/export/release gate
passed.

## Coverage boundary

The exact selection rule at the clean source revision was every path returned by
`rg --files tests` whose relative path matched either `test_.*generation` or
`generation.*test`. This produced 117 files. Windows excluded only
`tests/test_generation_requested_class_visual_audit_waiter.py`, whose production
entrypoint imports POSIX `fcntl`; Linux included it and its three tests.

The combined modules cover:

- model/training configuration, pair contract, runtime selection, metrics,
  watchdogs, signals, exact resume, checkpoint retention/integrity, and
  completion accounting;
- sampling protocol, batch selection, sample archives, checkpoint evaluation,
  deployable inference artifacts, resumable sessions, and real-forward
  preflight;
- FID/IS/precision/recall support, class fidelity, requested-class panels,
  stability diagnostics, quality-bridge decisions, formal gates, and source
  provenance;
- 250M probe, 50K scaling, 100K completion, full-300K readiness/training,
  post-evaluation, lineage observation, and control-process continuity/relaunch;
- storage and GPU contention, workspace paths, runbook entrypoints/syntax,
  deployment receipts/conflicts/deduplication, strong-baseline comparison,
  final completion audit, and release-bound inference export.

## Windows evidence

The Windows run used:

- checkout: `C:\qbfd2`;
- branch: `scale/generation-capacity-control-continuity-v1`;
- revision: `de59b1d1d3eeb7cde180b2604cbbf7e9c7fd1820`;
- Python: `3.10.20`;
- CUDA policy: `CUDA_VISIBLE_DEVICES=-1`;
- tracked checkout state before and after testing: clean;
- result: 1,089 tests, 1,081 passed, zero failures/errors, eight skipped;
- JUnit time: `644.847` seconds.

The eight Windows skips were explicit and expected:

- two real-CUDA RNG/sampler restore regressions;
- two POSIX process-group watchdog/relaunch regressions;
- two synthetic `/proc` deployment-process scans;
- one symlink-evidence-parent rejection requiring Windows elevation;
- one POSIX partial-failure owned-process-group rollback regression.

The persistent Windows JUnit copy is:

- path:
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/control_plane_continuity/full_generation_regression_2026-08-15/pytest_windows_cuda_hidden_de59b1d.xml`;
- bytes: `169,037`;
- SHA256:
  `fcbaf310c6c13e51ba447d3070e7da0ec2b89d7fe4c1c1ccb437d156471d7e28`;
- file mode: `0644`.

## Linux evidence

The authoritative Linux rerun used:

- checkout:
  `/root/autodl-tmp/CoFiTok/checkouts/capacity-control-process-rehearsal-47e59a0/CoFiTok-internal`;
- branch: `scale/generation-capacity-control-process-relaunch-rehearsal-v1`;
- revision: `47e59a0ce926c3ac1fdaa4860de84d2f27e39f76`;
- tree: `3a99993b66db675f262ed2e04298e1d4d64411cf`;
- Python: `3.10.20`;
- production-equivalent import path: `$PROJECT:$PROJECT/src`;
- CUDA policy: `CUDA_VISIBLE_DEVICES=-1`;
- tracked checkout state: clean;
- result: 1,092 tests, 1,090 passed, zero failures/errors, two skipped;
- JUnit time: `2,144.302` seconds.

The only Linux skips were
`test_rng_restore_accepts_states_remapped_to_cuda` and
`test_stateful_sampler_restores_rng_state_loaded_on_cuda`. Both explicitly
require real CUDA and had already been preserved as the two expected skips in
the dedicated checkpoint-reproducibility revalidation. All six Windows-only
POSIX or `/proc` skips and all three `fcntl` waiter tests executed on Linux.

The authoritative persistent JUnit report is:

- path:
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/control_plane_continuity/full_generation_regression_2026-08-15/pytest_linux_cuda_hidden_47e59a0_rerun.xml`;
- bytes: `168,625`;
- SHA256:
  `da7e7fa267ebdf311ed9e90d5ddfa3cce5ce278104988cf14dface89c87d6460`;
- file mode: `0644`.

Only documentation changed between the Linux source revision and the Windows
revision, so source and tests were identical.

## Preserved initial Linux environment failure

The first Linux full-suite invocation intentionally remains preserved instead
of being overwritten. It omitted the production `PYTHONPATH`, so the subprocess
inside
`test_selector_entrypoint_sets_training_allocator_default` could import the
repository's `scripts` namespace from its working directory but could not import
the `src/cofitok` package. The result was 1,092 tests, one failure, zero errors,
and two CUDA skips. Production runbooks explicitly export the project root and
`src` before invoking the runtime selector, and the exact failed test passed
after reproducing that environment. The complete suite was then rerun under the
same production-equivalent import path.

The preserved non-passing report is:

- path:
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/control_plane_continuity/full_generation_regression_2026-08-15/pytest_linux_cuda_hidden_47e59a0.xml`;
- bytes: `174,356`;
- SHA256:
  `60ab993eddf5b1b09cf001f29574a9110ca016ad70278966cc65d31a2940787c`;
- JUnit time: `2,131.278` seconds.

That report is diagnostic history only and is not accepted as the regression
pass. The separate `_rerun.xml` identity above is the authoritative Linux
result.

## Preserved production state

Both complete runs used only temporary CPU fixtures. They did not deserialize
or rewrite production checkpoints, create production samples, modify decisions
or gates, launch a training controller, or signal an unrelated process. The
formal remote checkout was not used as the test checkout. During the final
runtime check, the GPU still contained only the unrelated FieldScope process;
the bounded CoFiTok recovery supervisor remained at attempt zero with no child
process and the capacity lineage observer reported no issues.
