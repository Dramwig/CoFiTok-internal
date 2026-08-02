# Stability post-training completion supervisor (2026-07-31)

## Gap

The stability generation branch already had strict standalone runbooks for
full 300K training, formal 50K post-evaluation, final quality gating, EMA
inference export, and terminal completion audit. The post-training stages were
not joined by a source-bound, resumable supervisor. A completed multi-week
training pair could therefore wait for manual command reconstruction, and the
completion runbook still contained an obsolete hard-coded 50K evaluation
revision that contradicted the current v4 post-evaluation provenance.

## Provenance correction

`generation_stability_ema_teacher_completion_audit.sh` now requires the scaling
training and evaluation revisions and branches as explicit environment inputs.
It contains no historical `caab513` constant. The terminal audit therefore
uses the exact identities bound by the passing stability scaling gate.

## Supervisor

The new chain adds:

- `scripts/run_generation_stability_posttraining_supervisor.py`;
- `artifacts/runbooks/generation_stability_ema_teacher_posttraining_supervisor.sh`;
- `stability_full_300k_ema_teacher/reports/posttraining_supervisor.json`.

The supervisor cannot authorize or launch training. It requires an externally
created immutable full launch receipt and exact SHA256 before startup. It
rehashes the receipt and all nine bound sources, verifies the readiness and
scaling-gate bindings, checks exact full-training run paths, and waits for the
source-bound 300K pair monitor to report both runs at exactly 300,000 steps.

As of the 2026-08-03 schema-v3 launch receipt, the active source set contains
eleven reports. The supervisor rehashes all eleven and independently replays
the frozen supplemental verifier, including its raw post-evaluation source and
nested distribution/EMA-rollout sources. The historical nine-source statement
above describes the first supervisor revision only.

After training is complete and the GPU is idle, it executes:

1. formal matched 50K DDIM-250 post-evaluation;
2. strict full quality-gate validation;
3. release-authorized CoFiTok and dense EMA artifact export and smoke inference;
4. the read-only stability completion audit.

The full quality gate is a nonretryable scientific boundary. A failed or
source-mismatched gate stops the supervisor. Execution failures in sampling or
export receive bounded retries. If the completion audit writes a `failed` or
`incomplete` report, that report is also nonretryable rather than being treated
as a transient process exit.

## Recovery and completion

Existing final gates and completion reports are never trusted by filename.
They are reopened, their source reports are rehashed, and their complete
revision/SHA expectation maps must equal the current supervisor invocation.
Only a passing completion report with no failed or missing checks produces the
supervisor terminal `pass` state and
`formal_generation_completion_claimed=true`.

The supervisor is intentionally dormant until a human separately authorizes
and launches full 300K training, because the immutable full launch receipt does
not exist before that decision.
