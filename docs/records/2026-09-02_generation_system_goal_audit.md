# Generation-system goal audit (2026-09-02)

This change adds `scripts/audit_generation_system_goal.py`, a CPU-only,
read-only aggregate audit for the long-term generation-system objective.  It
is intentionally separate from the historical `validate_goal_completion.py`,
whose scope is the MVP/scoped paper evidence.

The new audit binds six requirements:

1. exact matched training scale and `samples_seen = step * effective_batch`;
2. sample-quality gate status and its source-report identities;
3. matched-training direct baseline rows with contextual baselines kept out of
   cross-tier numeric ranking;
4. completed, protocol-bound stable inference evidence;
5. physical checkpoint SHA256 rehash, sidecar and `latest.json` binding; and
6. the non-authorizing boundary, including `generation_advantage_proven=false`
   until the required quality evidence passes.

The default target is the formal 300K / 50K-sample / DDIM-250 system.  A
milestone audit can pass smaller explicit targets, but it cannot turn a
milestone into a release claim.  Any held, missing, or failed source remains
held, missing, or failed in the aggregate; the auditor never launches a
trainer or sampler, sends a process signal, changes an authorization, or
creates a candidate output root.

The current 100K quality bridge therefore remains incomplete: checkpoint and
inference provenance are useful evidence, while sample quality and the final
formal scale/protocol remain held.  This file records audit design only; it
does not authorize a GPU stage.

## Read-only revalidation (2026-09-02)

The auditor was hardened against the live report schema before this
revalidation.  Matched rows are now required to carry
`comparison_tier=matched_training_direct` and
`directly_comparable_to_cofitok=true`; contextual rows must carry
`comparison_tier=official_pretrained_contextual` and
`directly_comparable_to_cofitok=false`.  Sampling progress must also bind the
expected completed/total sample count and the prefix sample-set digest to the
sampling report.  Generation metrics bind checkpoint and sampling-runtime
identities through their nested `sample_provenance`; the evaluator runtime is
recorded separately.

Local focused tests passed (`9 passed`) and the script compiled successfully.
The remote audit used the explicit 100K / 10K / DDIM-100 milestone parameters
and returned:

| check | status |
| --- | --- |
| training scale and sample accounting | pass |
| sample quality | hold |
| matched direct/contextual baseline separation | pass |
| stable inference provenance | pass |
| physical checkpoint SHA256 and sidecar binding | pass |
| authorization boundary | pass |

The aggregate remains `status=hold`, `complete=false`, and
`generation_advantage_proven=false`.  The held quality checks are unchanged:
CoFiTok absolute FID, CoFiTok recall floor, and class fidelity.  No training,
sampling, process signal, promotion, release, or candidate output root was
created by this revalidation.

The same auditor was then run with the final target parameters (`300000`
training steps, `50000` samples, DDIM-250).  It returned
`status=hold`, `complete=false`, with `training_scale` and
`checkpoint_reproducibility` incomplete and `stable_inference` held for the
10K/DDIM-100 protocol mismatch.  This is an explicit audit result, not a
promotion or execution decision.

## Authoritative downstream recheck (2026-09-02)

The current versioned downstream evidence is also physically consistent:

* `quality_bridge_comparison_v4_authoritative/waiter_status.json` is
  operationally `pass` and binds the v3 terminal claim guard, while its
  comparison remains `terminal_status=hold` with decision
  `terminal_system_evidence_complete_without_qualified_matched_advantage`.
* `terminal_completion_audit_v4_authoritative/waiter_status.json` and
  `terminal_completion_audit.json` are `pass`/`hold` respectively, with
  `generation_advantage_proven=false`.
* The authoritative runtime-cost and statistical claim guards are `pass` and
  observational/non-authorizing.  The exposure-aware follow-up waiter has no
  decision or exposure report and remains `waiting`; the source-bound
  exposure/capacity preparation is not `execution_ready`.
* The six 90K/95K/100K physical-integrity waiter records remain `pass` for
  both matched methods.  The server GPU is idle and no training or sampling
  process is present.

These downstream statuses do not promote the terminal hold or authorize a
new stage.  Any future GPU action still requires a fresh exact
candidate-specific authorization; this audit performs no such action.

## Regression sweep (2026-09-02)

The CPU-only gate and audit regression set passed `60` tests with `2` expected
Windows skips (POSIX `flock` and symlink probing).  Compiling all tracked
`scripts/` and `src/` modules also passed.  A final-target audit rerun returned
`status=hold`, `complete=false`, with the existing 100K/10K evidence classified
as incomplete for the 300K/50K/DDIM-250 target rather than as a new execution
failure.

## Source-identity fail-closed hardening (2026-09-02)

The aggregate auditor now requires `source_reports` to be a non-empty mapping
for both the quality result and the strong-baseline comparison.  Every
descriptor is rehashed and must contain exactly `path`, `bytes`, and `sha256`;
malformed or silently omitted entries now fail the corresponding check instead
of being ignored.  The focused regression suite was rerun after this change
and passed with exit code `0` (the existing platform-dependent skips remain).

The authoritative remote reports were independently checked without changing
them: the quality result has 18 source descriptors and the comparison has 13;
all declared files matched their byte counts and SHA256 values, and none was a
symlink.  This strengthens provenance auditing only; it does not authorize a
new GPU stage.

The authorization-boundary check was also made fail-closed: the quality
bridge must explicitly provide `full_training_launch_allowed`,
`full_300k_launch_allowed`, `quality_bridge_execution_allowed`, and
`release_authorization_allowed`, all set to `false`; when a terminal audit is
present, its core training/sampling/GPU/promotion/release/process-signal
permissions are likewise required and must be `false`.  Missing boundaries or
missing fields are now failures rather than implicit permission.

## Checkpoint binding fail-closed hardening (2026-09-02)

The checkpoint reproducibility check now requires both `latest.json` and
`training_report.latest_checkpoint` to match the physically rehashed payload
and adjacent integrity sidecar for checkpoint filename, bytes, SHA256, format,
step, and manifest name.  A correct payload can no longer make the aggregate
audit pass if either reporting pointer names stale or substituted metadata.

The authoritative CoFiTok and dense 100K reports were rechecked read-only:
both bindings match exactly at step `100000`.  The focused checkpoint/audit
regression suite and module compilation passed after this change; no remote
artifact or authorization was modified.

The stable-inference check now also requires the paired sample-set digest to
be a well-formed 64-character hexadecimal SHA256 in both the sampling report
and progress record.  This prevents two internally consistent but malformed
digest strings from being treated as reproducible sample evidence.

## Stable-inference formal protocol hardening (2026-09-02)

The stable-inference gate is now fail-closed on the complete matched sampling
contract, rather than only on sampler name and step count.  For each method it
requires `clip_x0=true`, `guidance_scale=1.5`, `guidance_rescale=0.0`, batched
CFG, `bf16`, the balanced-modulo class schedule, seed `0`, and the method's
fixed terminal prefix budget (`[8]` for CoFiTok and `[1]` for dense identity).
The deterministic DDIM invariants remain required: `eta=0`, the exact
100-step timestep schedule, per-global-index random streams, and matching
sampling/progress manifest and sample-set SHA256 bindings.

The local focused audit test file passed (`17 passed`) and `compileall` over
`scripts/` and `src/` passed.  A read-only recheck of the authoritative remote
reports confirmed both terminal runs are completed at 10,000 samples with the
formal DDIM-100 contract, finite positive sampling time, complete progress,
and valid 64-character sample-set digests.  The GPU was idle and no trainer or
sampler process was present during the recheck.  This is provenance
hardening only; it did not launch or authorize any experiment and does not
change the scientific hold.

## Matched 1000-sample diagnostic route (2026-09-02)

The previously approved non-authorizing matched 1000-sample
sampling-recovery diagnostic is now closed as evidence: its versioned
recovery status is `completed`, the result report is `pass`, and it covers 8
matched cases / 16 observations.  It found
`selection_status=no_shared_sampling_recovery_candidate`; the diagnostic
therefore does not justify a sampler-only repair or a new confirmation run.

The downstream decision record only prepares a hypothesis for a fresh matched
Min-SNR training pilot.  It explicitly remains `execution_ready=false`, with
training, sampling, GPU, process-signal, promotion, release, and full-300K
permissions all false.  The exposure-aware and distribution-support waiters
remain CPU-only observational waiters with no decision/output, and the
authoritative terminal comparison remains `hold`.  No new GPU action was
taken, and the existing terminal hold and
`generation_advantage_proven=false` are unchanged.

## Sampling-manifest and protocol-contract binding hardening (2026-09-02)

Stable-inference auditing now delegates the shared report to
`sampling_protocol_contract` for the formal scaling protocol.  This checks the
protocol schema, inference API, exact DDIM timestep list, 1,000 training
timesteps, 10,000 samples, DDIM-100, and the fixed CFG/precision/seed/class
schedule fields.  The auditor additionally requires the per-global-index
random-stream scope and seed formula, the exact sample-digest framing, and
the method-specific terminal prefix.

The physical `sampling_manifest.json` is now rehashed and must match both the
sampling report and progress record.  Its sampling/checkpoint/runtime/step/
EMA fields must match the report, and the generation-metrics report must bind
the same manifest identity and a valid protocol-contract result.  The focused
audit plus shared sampling-protocol tests passed (`27 passed`), and all
`scripts/` and `src/` modules compiled.  Read-only remote hashing confirmed
the CoFiTok and dense terminal manifests match their report/progress SHA256
bindings.  No remote artifact, process, or authorization was modified.

## Quality-screen consistency hardening (2026-09-02)

The sample-quality audit now validates the quality screen itself before
accepting its hold/pass status.  It requires a non-empty check list, unique
check names, boolean `passed` flags, explicit observed/threshold values,
finite numeric content, a declared thresholds mapping, and an explicit
`non_authorizing=true` marker.  The reported `failed_checks` list and screen
status must exactly agree with the individual check outcomes.  This prevents
an incomplete or manually edited summary from hiding a failed quality gate.

The quality-screen regression set now passes (`20` focused tests; `29` when
combined with the shared sampling-protocol tests), with module compilation
still clean.  The authoritative remote quality result has 12 checks, whose
three failed outcomes exactly match its `failed_checks` list; its screen is
`hold` and explicitly non-authorizing.  This strengthens audit provenance
without changing the result, terminal hold, or execution permissions.

## Final local regression confirmation (2026-09-02)

After the quality-screen hardening, the complete local test suite was rerun
with `.venv\\Scripts\\python.exe -m pytest -q` and completed with
`PYTEST_EXIT=0`.  A subsequent `compileall` sweep over `scripts/` and `src/`
also completed with exit code `0`.  This confirms no local regression from the
latest audit changes; it does not alter any remote artifact, process, claim,
or authorization.

## Live guard fail-closed recheck (2026-09-02)

The remote guard chain was re-read after the regression sweep. The terminal
comparison and completion audit remain operationally `pass` but scientifically
`terminal_status=hold`; their training, sampling, GPU, process-signal,
promotion, export, and release permissions are all explicitly `false`. The
distribution-support and exposure-aware waiters remain `waiting` for a
follow-up decision, and the factorization supervisor remains `waiting` for
that same decision. The random-token wrapper is present, but its runbook
requires a completed factorization supervisor before it can reach its visual
diagnostic; no child diagnostic process or GPU process was present in the
snapshot (`nvidia-smi`: `0 MiB`, no running processes).

The authoritative training checkout is still clean at revision
`cf0e5faa94bf4ab38d947b921935b3b765b5537a`, tree
`6cef27723196fd363379bca2e7b85b1678ebd777`, and the standing-authorization
SHA256 still matches the expected value. No waiter was restarted, signalled,
or promoted by this recheck.

## Checkpoint waiter revalidation (2026-09-02)

All six historical 90K/95K/100K waiter records remain `pass`, with matching
physical-integrity audits. Each audit reports `physical_sha256_verified=true`,
the expected adjacent sidecar, the exact `scale/generation-stability-quality-
bridge-100k` revision and dataset identity, strictly increasing metrics, and
`samples_seen == step * 64`. The six payloads and sidecars are still present
with the recorded byte sizes. Their historical PIDs are absent because the
pass waiters have completed; no pass record was overwritten or restarted.

The runtime-fairness record remains `pass` and retains both orphan events
(`event_count=2`) with the required adjusted-compute lower bound. This is
read-only evidence maintenance and does not authorize training, sampling, or
promotion.

## Nested-scope authorization hardening (2026-09-02)

The aggregate authorization audit now also inspects a report's top-level
`scope` when that object exposes any known launch, GPU, process-signal,
promotion, or release permission. Such a scope must contain the complete
fail-closed permission set with every value explicitly `false`; a descriptive
scope with no permission fields is accepted without being treated as an
authorization surface. Regression tests cover both a nested `true` permission
and the descriptive-scope case. The focused goal-audit tests and the complete
local suite both exit `0`, and `compileall` remains clean. This is a guard-only
change and grants no execution authority.
