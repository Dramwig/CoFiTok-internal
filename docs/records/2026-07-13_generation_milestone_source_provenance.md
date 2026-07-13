# Generation milestone source provenance

Date: 2026-07-13

Branch: `scale/generative-system`

The full ImageNet-256 queue evaluates matched CoFiTok and dense checkpoints at
50K, 100K, 200K, and 300K. The previous aggregate milestone report validated
its inputs while being built, but it did not retain content-addressed identities
for those inputs. Resume logic could therefore skip a stale aggregate after an
underlying metrics or checkpoint-evaluation report changed, and the final
completion audit could not independently reconstruct that binding.

Milestone schema v2 now records the authoritative path, byte count, and SHA256
for all four source reports:

- CoFiTok generation metrics.
- Dense generation metrics.
- CoFiTok checkpoint mechanism evaluation.
- Dense checkpoint mechanism evaluation.

The accepted paths are method- and step-specific under the two full 300K run
directories. `cofitok_ddim_sampling_v1` now has an explicit non-claim
`milestone` contract: 2,048 samples, EMA DDIM-50, CFG 1.5, zero guidance
rescale, batched CFG, `eta=0`, clipped `x0`, bf16, seed zero, balanced-modulo
classes, batch 32, 1,000 training timesteps, and invariant per-index streams.

`scripts/validate_generation_milestone_report.py` rehashes all source reports
and recomputes protocol, metrics, matched deltas, and quality-alert state. The
full training runbook invokes it both before skipping an existing milestone and
immediately after building one. The large-scale completion audit receives a
fresh source verification for every milestone and rejects source drift,
non-authoritative paths, malformed checkpoint/sample hashes, non-finite metrics,
or matched protocol weakening.

Verification on 2026-07-13:

```text
targeted milestone/protocol/transition/completion tests: 60 passed
full local suite after initial implementation: 481 passed
```

Milestone alerts remain nonblocking early warnings. They are preserved in the
completion report but never substitute for the final 50K DDIM-250 quality gate.
