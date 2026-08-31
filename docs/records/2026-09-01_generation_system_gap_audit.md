# Generation System Gap Audit (2026-09-01)

## Scope

This is a CPU-only current-state audit. It does not launch training or sampling,
modify the formal checkout, promote checkpoints, export an inference artifact,
or replace any locked scientific result.

## Authoritative state

The full-data bridge remains bound to:

- checkout: /tmp/cofitok-quality-bridge-execution-cf0e5fa/CoFiTok-internal
- revision: cf0e5faa94bf4ab38d947b921935b3b765b5537a
- tree: 6cef27723196fd363379bca2e7b85b1678ebd777
- branch: scale/generation-stability-quality-bridge-100k
- pair status: pass, issues: []

Both matched runs are complete at step 100000 and 6400000 images seen.
The latest checkpoint and report identities were re-read from the live server:

| method | latest step | latest SHA256 | training report SHA256 | training complete |
| --- | ---: | --- | --- | --- |
| CoFiTok K8 | 100000 | 8b5620f78b89c51a32e863f563a6311448a0545ad1bc6c06a1b532cc9eba079c | 9056b47fe7733b092608e29b9b5d14c95f5af6a35c170be4ec0dcd553f6b4a86 | true |
| dense identity | 100000 | 6da767b813dacc206a8a7e7de1aae05b6e8eacb61d742c6de816d4689de97efd | 5b38303da71f324acb668fca8c52cf7558fbee7ab8a38a5bc7768836387f08ec | true |

The standing authorization still hashes to
5fe64a0941acb55a21cb6479a726987c6259e93a772c47290f2d6883752de4df.

## Five-pillar completion audit

| pillar | current evidence | verdict | missing evidence |
| --- | --- | --- | --- |
| sample quality | Terminal 10K DDIM-100: CoFiTok FID 115.2622, precision 0.7556, recall 0.00832; dense FID 123.0210, precision 0.6653, recall 0.01000 | hold | absolute FID, recall floor, and class-fidelity qualification |
| training scale | Matched full-data pair completed at 100K | partial | authorized larger-scale qualification and eventual 300K protected milestones |
| strong-baseline fairness | v4 comparison operationally pass and keeps terminal scientific hold; CoFiTok/dense are the only matched-training rows | partial | a passing usable-quality gate and final release comparison |
| stable inference | Integrity-bound GenerationSession and inference contracts pass CPU tests | implementation ready | release-authorized EMA artifact and passing scientific gate |
| reproducible checkpoints | CoFiTok/dense 90K, 95K, 100K physical audits pass; dense replay for all three passes | partial | protected full-scale milestones and release completion replay |

The approved non-authorizing 1000-sample sampling-recovery diagnostic completed
with 16 observations and selected no_shared_sampling_recovery_candidate. It does
not authorize an independent confirmation or a larger training stage.

## Execution boundary

The two source-bound candidate gates are still:

- exposure_continuation: exact 100K resume to a bounded 110K target;
- capacity_qualification: fresh matched 256-channel 10K qualification with a 1K screen.

Both are status prepared, execution_ready false, and stage_authorization
not_authorized. Their output roots and stage-authorization sentinel are absent.
The live GPU reports 0 MiB used with no compute application; project free space is
approximately 250.8 GB. Training, sampling, full-300K, promotion, export,
release, and process-signal permissions remain false.

## CPU reproducibility verification

The isolated evidence branch ran the full code suite while excluding only the
four paper-layout tests that require an outer D:/paper tree:

- 1084 passed, 7 skipped in 356.99 seconds;
- compileall for src and scripts passed;
- git diff --check passed;
- verification record commit: 511fcc2.

The excluded paper-layout tests fail in a code-only checkout only because the
sibling paper tree is absent; no generation-system test failed.

## Completion verdict

The requested usable large-model generation system is not complete. Current
evidence supports the scoped ordered restricted dense-noise factorization and
prefix-controllable denoising claims, but not a broad usable-generation claim.
The next GPU action requires a separately created exact-stage authorization;
until it appears, continue CPU-only auditing and preserve
generation_advantage_proven=false and terminal_status=hold.
