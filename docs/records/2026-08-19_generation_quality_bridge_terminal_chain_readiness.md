# Quality-bridge terminal-chain readiness audit

Date: 2026-08-19

## Outcome

The active full-data matched quality bridge has a live, source-bound terminal
controller and a complete fail-closed follow-up control path. The audit found
no missing process responsible for producing `quality_bridge_result.json` or
`followup_experiment_decision.json` after the physical 100K evidence exists.

This is control-plane readiness, not terminal scientific evidence. The bridge
is still running, the terminal result and follow-up decision are absent, and no
broad generation-quality, full-300K, promotion, or release claim is allowed.

## Active execution identity

```text
output root:
/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_base128_quality_bridge_v1

controller PID: 618821
pair-monitor PID: 619509
dense watchdog PID: 79830
dense trainer PID: 79894

revision: cf0e5faa94bf4ab38d947b921935b3b765b5537a
tree: 6cef27723196fd363379bca2e7b85b1678ebd777
branch: scale/generation-stability-quality-bridge-100k
runbook SHA256: f0531763b4b888964a75923a8d59cb8cf8d1c79bd7782f845006c39e1a57fe73
```

The execution checkout was tracked-clean. The controller-identity guard bound
the exact PID, start ticks, executable, cwd, argv SHA256, runbook process, Git
identity, and monitor name. Its live state was `observing` with detail
`exact_bound_controller_identity_is_active`, zero identity-loss polls, and no
identity mismatch.

At `2026-08-19T05:44:40+08:00`:

```text
pair monitor: running / dense_identity_training
issues: []
CoFiTok first segment: 50,000 complete
dense first-segment tail: 17,050 / 50,000
dense samples_seen: 1,091,200
quality_bridge_result.json: absent
followup_experiment_decision.json: absent
GPU compute processes: PID 79894 only
/root/autodl-tmp available bytes: 317,615,255,552
```

## Verified terminal execution path

The exact execution runbook performs the following source-bound sequence:

1. Finish the paired 50K milestone, evaluate both methods with the matched
   2,048-sample DDIM-50 early-warning protocol, and build the paired milestone
   report.
2. Resume CoFiTok from 50K to exactly 100K, evaluate its 100K milestone, then
   resume dense identity from 50K to exactly 100K and evaluate the matched
   milestone.
3. Require both exact 100K training reports, protected 50K/100K checkpoints,
   integrity sidecars, pair validation, schedule/metrics audits, and a passing
   pair monitor.
4. Run terminal EMA preflight, checkpoint evaluation, 10,000-sample DDIM-100
   generation, FID/IS/precision/recall, and requested-class fidelity for both
   methods.
5. Build `quality_bridge_result.json`, immediately replay and verify it against
   all physical sources, and only then mark the execution controller complete.

The runbook permanently declares the bridge non-promotional:

```text
quality_bridge_only=true
report_is_promotion_gate=false
full_training_launch_allowed=false
full_300k_launch_allowed=false
```

## Follow-up decision path

The exact CPU-only decision waiter is live as PID `11777`:

```text
revision: 9b02fa83d20b1459a2706d6d82371caf5c023f54
tree: 474520aa848f3da143379a7b7f72d59a085fbe38
branch: scale/generation-quality-bridge-followup-decision-v1
runbook SHA256: 7d03191f0312ef90e08e25609be93c1ec4ad63badb59dc26bb049c6f971df89b
status: waiting_for_quality_bridge_result
```

After the result appears, it content-addresses and fully replays the bridge
result and both 50K/100K milestone reports before writing the immutable
follow-up decision. The routing policy is fail-closed:

- all terminal checks pass: prepare a new source-compatible formal gate;
- CoFiTok mechanism failure: bounded matched mechanism-recovery probe;
- matched CoFiTok quality regression: bounded factorization-quality probe;
- class-fidelity failure: matched class-conditioning diagnostic;
- only absolute-quality checks fail with shared 50K-to-100K FID improvement:
  prepare a bounded matched 250M capacity-qualification probe;
- only absolute-quality checks fail without shared improvement: reuse terminal
  samples for distribution-support diagnosis and a bounded recipe probe;
- contradictory or unclassified evidence: stop for source-bound reconciliation
  or policy extension.

Every route has `execution_ready=false`, `gpu_execution_allowed=false`,
`full_300k_launch_allowed=false`, and `release_authorization_allowed=false`.
Existing downstream supervisors may act only through their separately bound
standing authorization and exact route contracts.

## Guard and status audit

The controller guard checkout was exact and tracked-clean:

```text
revision: 57b2897a4cfbcde09bb24f3082d7fe3c67b83960
tree: bcc79d4a6c2cba1c8481a0f49956e35739922e29
branch: analysis/generation-quality-bridge-controller-identity-guard-v1
```

The terminal-system guard was also exact and tracked-clean:

```text
revision: 8ec9a09ddcd981c1ffbd06b6bda81386b33321de
tree: 344f6365d2e962e350b73f5ce4bc18c005f6dff9
branch: analysis/generation-terminal-system-claim-guard-v1
status: waiting_for_exact_terminal_system_sources
```

A recursive scan over the active bridge, capacity, terminal-uncertainty,
claim-guard, and conditioning supervisor status files found current counts of
`32 waiting`, `2 running`, `1 observing`, `6 pass`, and `1 completed`. The only
two `failed` files were immutable snapshots under
`reports/recovery_incident_2026-08-17_pin_memory/`; neither is a current
controller status. No current status carried a non-empty issue list.

## CPU-only rehearsal

All checks used `CUDA_VISIBLE_DEVICES=-1`, one numerical-library thread,
`nice -n 19`, and idle IO priority. Exact-checkout targeted tests passed:

```text
quality-bridge execution and result contract: 15 passed
follow-up decision and route replay: 13 passed
controller identity guard: 15 passed
terminal-system claim guard: 7 passed
total: 50 passed
```

Both terminal runbooks passed `bash -n`. All four exact checkouts remained
tracked-clean. The active dense trainer stayed the sole GPU compute process and
continued advancing during the rehearsal.

## Remaining evidence

Control-plane readiness does not close the system objective. Still required:

- physical paired 100K completion and protected checkpoint audits;
- matched 100K milestone and terminal 10K sample metrics;
- absolute FID/precision/recall and requested-class fidelity qualification;
- a source-replayed follow-up decision and any separately controlled bounded
  repair/capacity experiment it selects;
- only after a new passing gate, formal larger-scale training, 50K formal
  sampling, stable inference artifacts, completion audit, and release receipt.
