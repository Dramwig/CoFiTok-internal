# Generation quality-bridge 50K milestone transition audit

Date: 2026-08-18

## Outcome

The active full-data quality bridge can follow its existing CoFiTok-50K to
milestone-evaluation to dense-50K transition without replacing the controller,
trainer, monitor, or physical checkpoint waiters. The transition is foreground,
serial, fail-closed, and recoverable from the evidence already published by a
partially completed milestone evaluation.

No correctness gap requiring an active-run hotfix was found. A CPU-only static
regression test was added for future revisions so the critical stage ordering
cannot silently become asynchronous or reorder dense training ahead of the
CoFiTok milestone evidence.

Machine-readable evidence:

`artifacts/reports/generation/quality_bridge_50k_milestone_transition_audit_2026-08-18/audit.json`

## Bound active identity

- training revision: `cf0e5faa94bf4ab38d947b921935b3b765b5537a`
- training tree: `6cef27723196fd363379bca2e7b85b1678ebd777`
- branch: `scale/generation-stability-quality-bridge-100k`
- output root:
  `/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_base128_quality_bridge_v1`

The exact active milestone runner, quality-bridge runbook, sampling preflight,
checkpoint evaluator, sampler, metrics evaluator, milestone builder/validator,
and training watchdog are bound by bytes and SHA256 in the machine-readable
report.

## Serial GPU transition

The training watchdog starts the trainer in its own process group and does not
return until the child has exited. On a watchdog failure it terminates the full
process group and waits for it. On a normal 50K stop it waits for exit code zero,
writes the terminal watchdog status, and only then returns to the shell runbook.

The exact runbook then executes, in order:

1. CoFiTok training to 50K;
2. a synchronous monitor snapshot;
3. CoFiTok milestone evaluation;
4. dense training to 50K;
5. a synchronous monitor snapshot;
6. dense milestone evaluation;
7. the paired 50K report.

None of the training, evaluation, or paired-report commands is backgrounded.
The milestone runner itself is also synchronous: sampling preflight, checkpoint
mechanism evaluation, 2,048-image DDIM-50 sampling, and torch-fidelity metrics.
Process exit releases CUDA resources before the next foreground stage can start.

The runbook has an idle-GPU assertion at controller launch. It has no additional
`nvidia-smi` assertion between these foreground stages. This is not a current
correctness blocker because the foreground process boundary is exact and the
pair monitor remains active. If a future revision changes any stage to
asynchronous execution or supports a shared GPU, it must add a stage-scoped GPU
ownership check; the new ordering regression test will already reject the
asynchronous/reordered form.

## Milestone recovery and provenance

The active milestone path is restart-safe under the same exact revision and
request:

- checkpoint evaluation uses an exclusive output lock, immutable request
  manifest, physical checkpoint verification, and validates a completed report
  before reuse;
- sampling uses an exclusive output lock, immutable checkpoint/protocol/runtime
  manifest, per-global-index batch-invariant RNG, atomic PNG publication,
  atomic progress, full PNG validation, and sample-set SHA256;
- metrics use an exclusive output lock, revalidate the sampling manifest,
  progress, sample-set digest, runtime, real-set tree, evaluator identity, and
  validate a completed report before reuse;
- the paired milestone report binds bytes/SHA256 identities for both generation
  and checkpoint-evaluation reports, requires the same checkpoint bytes and EMA
  weights within each method, enforces the matched protocol, and rehashes all
  four source reports on validation;
- an existing invalid paired report is never overwritten automatically.

Therefore a failure before the paired report causes the controller to fail
closed. A receipted exact-revision restart can reuse only matching completed
evidence and otherwise continues the incomplete stage. A valid paired report is
the only condition that skips a completed milestone.

## Regression protection and validation

Future safeguard commit:

- revision: `7c7bc42e26c80df809180339fa69279d238eb24f`
- tree: `6a3884177c64b8f066a1816535d9dcc31ccf1e76`
- branch: `analysis/generation-quality-bridge-50k-transition-readiness-v1`
- test: `tests/test_generation_quality_bridge_transition_runbook.py`

The incremental bundle was `114,546` bytes with SHA256
`bc3fa44077fe82ca99733ab3ebbf7192f1a889add2666d2fbde75bf05f9e325a`.
It required the exact active revision `cf0e5faa...`, verified successfully on
the server, and was checked out detached at:

`/tmp/qbtransition-rehearsal-Z1SZP0/CoFiTok-internal`

With CUDA hidden and one OMP/MKL thread:

- 71 checkpoint-evaluation, milestone, sampling, preflight, metrics-resume,
  watchdog, and transition-ordering tests passed;
- both relevant runbooks passed `bash -n`;
- the audited Python files passed `py_compile`;
- `git diff --check` and tracked-clean verification passed.

During validation, the authorized trainer advanced from step 43,350 to 43,550.
After validation the only GPU compute process was still trainer PID `619775`,
using `85,286 MiB`; no evaluator or additional trainer was launched.

## Boundary

This audit and regression test do not authorize a new experiment, larger
training horizon, promotion, release, or full 300K run. They do not modify the
active checkout or active output. The remaining empirical requirement is to
observe the real 50K checkpoint audit, CoFiTok milestone evaluation, and dense
transition when the existing controller reaches that boundary.
