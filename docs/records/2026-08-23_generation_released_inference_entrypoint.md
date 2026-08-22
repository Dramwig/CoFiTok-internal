# Completion-authorized released-inference entrypoint

Date: 2026-08-23 05:10 CST

## Purpose

The reusable `GenerationSession` and generic `infer_generation.py` already
supported terminal release receipts, but routine deployment still depended on
an operator remembering both authorization switches. That was correct for
research and pre-completion smoke workflows, but it left an avoidable
production usability and misconfiguration risk.

This change adds a distinct fail-closed released-inference entrypoint. It does
not create a release artifact or receipt, and it cannot turn the active 100K
quality bridge into a released system.

## Live-state precheck

Before development, the active remote chain was reread:

- CoFiTok: 100,000/100,000 steps and 6,400,000 images;
- dense: 83,900/100,000 steps and 5,369,600 images;
- pair monitor: `running`, stage `dense_identity_training`, `issues=[]`;
- sole GPU compute owner: dense trainer PID `219593`, 79,132 MiB;
- dense 90K/95K/100K physical waiters: healthy `waiting`;
- training checkout revision/tree:
  `cf0e5faa94bf4ab38d947b921935b3b765b5537a` /
  `6cef27723196fd363379bca2e7b85b1678ebd777`;
- training checkout branch:
  `scale/generation-stability-quality-bridge-100k`;
- full Git porcelain: empty;
- free `/root/autodl-tmp`: 301,481,607,168 bytes.

## Implementation

New entrypoint:

```text
scripts/infer_released_generation.py
```

Its CLI contract is deliberately narrower than the research entrypoint:

- `--completion-receipt` is required by argparse;
- only `--weights ema` is accepted;
- `require_release_authorization=True` is fixed internally;
- `require_completion_authorization=True` is fixed internally;
- neither authorization switch is exposed as a user-disableable CLI option;
- programmatic calls are re-normalized and reject a missing receipt or non-EMA
  request before delegating to the existing atomic/resumable inference core.

Consequently, released inference reuses the established pre-deserialization
chain:

1. inference-artifact sidecar and physical SHA/bytes;
2. embedded full-gate release authorization;
3. terminal release receipt;
4. bound terminal completion audit;
5. exact artifact/export-manifest/source provenance named by that receipt.

The generic `scripts/infer_generation.py` remains intentionally available for
training-checkpoint inspection, formal export smoke tests, and other research
workflows that must occur before a terminal receipt exists. Documentation now
identifies it as the research entrypoint rather than the routine production
launcher.

Code commit:

```text
revision: 89601f15a04bb27195db67edf58af575dc7bc118
tree:     89228f35f4bc345ee8c409d7ff4d583e009e4e23
subject:  Add completion-authorized inference entrypoint
```

## Verification

The related CUDA-hidden local suites collected 201 tests:

```text
tests/test_generation_inference_artifact.py: 31
tests/test_generation_quality_bridge_terminal_completion_audit.py: 34
tests/test_generation_released_inference_entrypoint.py: 7
tests/test_generation_runbook_entrypoints.py: 1
tests/test_generation_sampling_preflight.py: 6
tests/test_generation_session.py: 15
tests/test_generation_stability_inference_export_runbook.py: 5
tests/test_generation_system.py: 20
tests/test_large_scale_generation_completion_audit.py: 82
result: 199 passed, 2 CUDA-only skipped
```

Python compilation and `git diff --check` passed.

The exact commit was exported and rehearsed in an isolated Linux tree:

```text
archive: D:/cofitok-bundles/released-inference-entrypoint-89601f1.tar.gz
remote:  /tmp/released-inference-entrypoint-89601f1.tar.gz
bytes:   139,687,896
sha256:  68a6955ded7f8c419131ba312750ff2ce7585cfac055469fab445557fcc2844b
embedded commit: 89601f15a04bb27195db67edf58af575dc7bc118
```

The Linux rehearsal used `CUDA_VISIBLE_DEVICES=''`, `OMP_NUM_THREADS=1`, and
`MKL_NUM_THREADS=1`. It verified standalone `--help`, required receipt presence,
absence of user-disableable authorization flags, Python compilation, and the
same 201-test suite (`199 passed, 2 skipped`). The isolated rehearsal repository
was clean after testing. No GPU process was launched.

## Scope and authorization

This commit is not deployed into the active quality-bridge checkout and does
not alter any controller, trainer, waiter, checkpoint, sample set, gate, or
scientific decision. It is implementation hardening for a future genuinely
released artifact. Current state remains:

```text
generation_advantage_proven=false
full_training_launch_allowed=false
full_300k_launch_allowed=false
promotion_or_release_allowed=false
```

