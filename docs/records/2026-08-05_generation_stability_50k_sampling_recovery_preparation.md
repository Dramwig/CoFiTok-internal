# Stability 50K frozen sampling-recovery preparation

Date: 2026-08-05

## Outcome

The only failed scaling gate remains `absolute_fid_quality`:

- CoFiTok formal EMA FID: `138.29702495267782`
- dense-identity formal EMA FID: `151.4476773464495`
- required absolute CoFiTok FID: `<= 100.0`
- promotion decision: `hold`
- `full_training_launch_allowed=false`

The matched 50K training pair and formal post-evaluation are complete and
passing. No training, checkpoint, matched-protocol, or provenance gate failed.
The next bounded discriminator is therefore a frozen-checkpoint inference
protocol diagnostic, not a 50K rerun or a full 300K launch.

Authoritative source root:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_scaling_50k_ema_teacher
```

## Predeclared diagnostic

The source-bound plan is:

```text
configs/generation/diagnostics/stability_50k_sampling_recovery_v1.json
```

It fixes an equal CoFiTok/dense sweep with:

- frozen step-50K EMA checkpoints;
- 512 images per method and case;
- DDIM-100;
- seed `0`, start index `0`, and balanced-modulo class labels;
- per-global-sample-index, batch-size-invariant random streams;
- bf16 and batched CFG;
- CFG/rescale cases `1.0/0.0`, `1.25/0.0`, `1.5/0.0`, `1.5/0.5`, and
  `1.5/1.0`;
- FID and Inception Score only, with precision/recall explicitly disabled for
  this small diagnostic.

The plan binds the frozen promotion gate, both formal 10K metrics reports, both
checkpoint SHA256 values, both integrity-sidecar SHA256 values, the formal real
set, the frozen training/evaluation Git identities, and the sampling/evaluator
runtime identities.

The diagnostic is permanently non-authorizing:

```text
diagnostic_non_authorizing=true
small_sample_fid_is_formal=false
replaces_frozen_promotion_gate=false
formal_protocol_change_allowed=false
matched_10000_confirmation_required=true
full_training_launch_allowed=false
full_300k_launch_allowed=false
```

## Fail-closed builder

The validator and summarizer is:

```text
scripts/build_generation_stability_sampling_recovery.py
```

It now:

- accepts report-only real-set metadata such as `root` while requiring exact
  equality of the bound digest schema, image count, and SHA256;
- rehashes the frozen gate and formal metrics reports;
- requires the exact sole failed gate to remain `absolute_fid_quality`;
- verifies formal checkpoint, sample-set, Git, runtime, evaluator, real-set,
  count, CFG, and random-stream provenance;
- verifies each diagnostic sampling and schema-v3 metrics report, including
  exact matched protocol, PRC-disabled policy, selected sample-set binding,
  positive elapsed time, and finite FID/IS values;
- requires all ten matched rows before ranking cases;
- emits no protocol change, promotion, full-training receipt, or launch
  authorization.

## GPU-deferred runbook

The prepared runbook is:

```text
artifacts/runbooks/generation_stability_frozen_50k_sampling_recovery_diagnostic.sh
```

It requires a separately clean checkout at an explicitly supplied exact
revision on `scale/generation-large-capacity`. Before any GPU work it acquires a
dedicated lock, verifies the immutable source reports, resolves `latest.json`,
rehashes both physical checkpoints through their required integrity sidecars,
and runs the builder preflight.

Every GPU stage first checks live compute processes and exits `75` without
launching if the GPU is busy. The only generation/evaluation entrypoints are
`scripts/generate_samples.py` and
`scripts/evaluate_generation_metrics.py --skip-prc`; there is no training,
checkpoint mutation, waiter replacement, or full-300K entrypoint. Outputs use
the independent root:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_scaling_50k_sampling_recovery_v1
```

## Verification

The initial recovery preparation verification completed:

```text
sampling-recovery and runbook targeted tests: 14 passed
remote streamed bash syntax check: pass
full repository suite: 1074 passed, 6 skipped
git diff --check: pass
```

The later execution-approval hardening is verified separately at exact code
revision `1940788522c787bcb12f01dc9487b9490a34326a`: `32` targeted tests,
`1,098 passed / 6 skipped` in the layout-independent full suite, and `4/4`
candidate paper-layout replays. See
`2026-08-05_generation_stability_sampling_execution_approval_hardening.md`.
That revision is now historical and must not be executed because its 10K
confirmation random stream overlapped the selection and frozen formal streams.
The replacement candidate is documented in
`2026-08-05_generation_stability_sampling_confirmation_stream_independence.md`.

The live remote state was re-read before preparing the runbook. FieldScope PID
`433140` was still running from
`/root/autodl-tmp/FieldScope/FieldScope-internal` with the exact
`fieldscope.cli extract-dataset` argv, approximately `15412 MiB` of GPU memory,
and 100% GPU utilization. Available `/root/autodl-tmp` storage was
`305852088320` bytes. No CoFiTok GPU diagnostic was launched, and no FieldScope
or other-project process was modified.

## Execution and decision boundary

Execution still requires an idle GPU, an exact clean deployed diagnostic
revision, `SAMPLING_RECOVERY_EXECUTION_ALLOWED=true`, and a user-created
execution-approval sentinel whose bytes/SHA256 bind the exact plan, Git
identity, output root, approval time/text, and permanently non-authorizing
scope. The runbook validates that sentinel before creating its output root or
acquiring the GPU-stage lock. A completed 512-sample sweep is
only a protocol-selection diagnostic. Any selected change must be rerun as a
fresh matched 10K evaluation in a separate output root before it can support a
new gate. That confirmation must use the predeclared disjoint global-index and
derived-seed interval `[10000, 20000)`, not the diagnostic/frozen interval
beginning at zero; the existing frozen gate must remain unchanged.

The recovery sentinel is an untracked, user-created JSON object with exactly
these top-level fields (placeholders are descriptive and are not an approval):

```json
{
  "schema_version": 1,
  "role": "generation_stability_sampling_execution_approval",
  "status": "approved",
  "scope": "stability_50k_sampling_recovery_v1_execution_only",
  "evidence": {
    "path": "<absolute path to stability_50k_sampling_recovery_v1.json>",
    "bytes": "<exact positive integer>",
    "sha256": "<exact lowercase SHA256>"
  },
  "git": {
    "revision": "<exact candidate revision>",
    "branch": "scale/generation-large-capacity",
    "tracked_dirty": false
  },
  "output_root": "/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_scaling_50k_sampling_recovery_v1",
  "approval_record": {
    "approved_by": "user",
    "approved_at": "<ISO-8601 timestamp with timezone>",
    "approval_text": "Approve the non-authorizing matched 512-sample sampling-recovery diagnostic only."
  },
  "authorization_boundary": {
    "sampling_recovery_execution_allowed": true,
    "sampling_confirmation_execution_allowed": false,
    "training_launch_allowed": false,
    "full_training_launch_allowed": false,
    "full_300k_launch_allowed": false,
    "release_authorization_allowed": false,
    "replaces_frozen_promotion_gate": false,
    "formal_protocol_change_allowed": false
  }
}
```

Extra or contradictory top-level fields, a non-user approver, a timezone-free
timestamp, a changed evidence/Git/output identity, or any weakened boundary
are rejected. No real approval sentinel was created during preparation or
verification.

If the bounded inference sweep does not show a plausible route toward the
absolute FID threshold, the next experiment must address training quality with
a separately named matched capacity/data/training-scale bridge. That bridge
must preserve the same CoFiTok/dense fairness contract and checkpoint audit
policy and still cannot silently authorize or start full matched 300K training.
