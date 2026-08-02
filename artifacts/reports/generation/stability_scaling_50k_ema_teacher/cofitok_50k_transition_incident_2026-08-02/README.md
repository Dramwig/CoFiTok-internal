# CoFiTok 50K completion and dense-transition incident evidence

This evidence pack records a control-plane transition failure after the
CoFiTok member completed 50,000 optimizer steps. It intentionally contains no
checkpoint payload, generated sample tree, feature cache, or long training
log.

The final CoFiTok checkpoint was independently hashed on `pro6000` as
`ec7b9a0981f1d45420a9a86cdb80339d6d87b87fa77891c234db3d1b84376c2a`
over `1,006,325,418` bytes. The local sidecar, `latest.json`, training report,
and watchdog snapshots agree with step 50,000 and training revision
`2c2c1f5166b73d4f28df93b276901671ac1a7836`.

The matched runbook then stopped before creating the dense run because the
completion helper hard-coded branch `scale/generative-system`, while the
runbook and report correctly used
`scale/generation-stability-50k-preflight`. The 40K and 45K observers had an
independent false failure: their progress auditor treated the deliberately
retained stop-requested pause report from step 36,545 as if it had to match
newer resumed metrics.

At capture time, no matched-queue process remained and the dense run directory
did not exist. GPU PID `362355` belonged to FieldScope and was left untouched.
Full ImageNet-256 300K training remains unauthorized.

Recovery-control commit `e5c9ed7bd4590f5dd6dd0d78308c2ff8e868b58a` was
deployed into a new isolated checkout, then fast-forwarded only in that
checkout to controller revision
`5f57757a2162507c6166dbfe976246df6ae1af91`. The latter adds a non-blocking
single-controller lock without changing the immutable training checkout. Its
non-executing preflight completed as `prepared`, revalidating the completed
CoFiTok trust boundary, frozen runtime, matched configs, and
`68,809,579,484` bytes of storage headroom. A simultaneous second preflight
exited 15 before it could overwrite status. No queue or GPU process was
launched.

The recovery runbook intentionally leaves `pair_summary.json` absent. The
exact post-eval waiter must create it from the original locked config report,
preventing an equivalent-but-different recovery report path from weakening
source provenance.

`manifest.json` binds every copied file by byte count and SHA256. The raw
checkpoint and transition log remain only at their authoritative remote paths.
