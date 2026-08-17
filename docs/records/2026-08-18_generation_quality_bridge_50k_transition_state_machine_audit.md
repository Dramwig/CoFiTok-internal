# 2026-08-18 quality-bridge 50K milestone transition audit

## Question

The active runbook pauses training at CoFiTok step `50,000`, performs a
2,048-sample DDIM-50 milestone evaluation, and only then starts dense training.
Because this evaluation can last longer than the monitor's `600`-second idle
grace, the transition must not be mistaken for a dead training queue.

This audit also checks how the legacy GPU-contention schema will classify the
runbook-owned milestone sampler during that transition.

## Source binding

The inspected monitor, watchdog, GPU provenance module, runbook, and focused
test are byte-identical to active training revision
`cf0e5faa94bf4ab38d947b921935b3b765b5537a`. The local comparison against that
revision was empty for all inspected paths.

Primary source SHA256 values:

```text
src/cofitok/monitoring.py
  516d7d374702b62e0c8476379f3ed8f6ea35dd7abfc52dd9eddbc6aca6cac845
src/cofitok/gpu_contention.py
  dc19e81331467ec29f7e3eedb83c8006a83b7804688f1ea9883cf08c504f349c
src/cofitok/training/watchdog.py
  9ca7ec0439e978439ced61afa7644e28c6e11a2eccf0a7c74c244fdf979dc51e
artifacts/runbooks/generation_stability_full_data_quality_bridge_100k_execute.sh
  c4f05b36a98ff7b0bcc10229dedf9131bdda377bf8e86be0127a9829fbf4c924
```

## Queue-state result

The pair monitor has an explicit runbook-only transition branch:

```text
training processes absent
runbook process present
incomplete pair
=> status=waiting, stage=<current_method>_transition, issues=[]
```

The background monitor exits only for `pass`, `failed`, or `stalled`; it keeps
polling in `waiting`. Therefore a long milestone evaluation does not terminate
the pair monitor and does not consume the idle-failure grace.

The exact Linux test passed with CUDA hidden, bytecode disabled, and pytest
cache disabled:

```text
tests/test_monitor_generation_10pct_pair.py::test_monitor_waits_during_runbook_only_milestone_transition
1 passed
```

When dense training starts, its watchdog allows `600` seconds for a fresh pair
monitor report while the monitor polls every `300` seconds. A prior `waiting`
report is not a terminal failure, and the next healthy poll arrives within the
startup grace. The train-to-eval-to-dense state machine therefore does not need
an active-checkout patch or monitor restart.

## GPU provenance limitation

GPU-contention schema v1 defines expected GPU work solely by membership in the
`train_generation.py` PID set. A runbook-owned milestone sampler is not a
training PID. A source-exact synthetic replay therefore produced:

```text
pair status/stage: waiting / cofitok_transition
legal sampler classified as unrelated: true
unrelated observed: true
training wall-clock direct comparison allowed: false
```

This label is a provenance limitation, not a queue failure and not external
contention by itself. Any sampler identity observed during the 50K transition
must be interpreted from its exact argv, cwd, and runbook ancestry before being
described as another project's GPU process.

For the current bridge, raw wall-clock comparison is already disallowed because
the historical monitoring gap is `16,377.985275` seconds. Consequently this
legacy classification cannot remove an efficiency claim that was still
available. It also does not affect matched data, parameters, steps, checkpoint
integrity, or sample-quality comparison.

The correct current reporting boundary is:

- keep matched quality comparison;
- report training time and throughput as observational only;
- report milestone sampling time separately;
- do not describe a runbook-owned evaluator as external contention;
- require stage-aware GPU provenance from before training for any future
  exclusive efficiency claim.

## Live state and decision

At the authoritative snapshot, CoFiTok was at step `30,600`, dense had not
started, pair status was `running / cofitok_training`, and issues were empty.
No source, process, monitor, runbook, or training checkout was modified by this
audit.

Decision:

```text
allow the existing 50K milestone transition: yes
restart or replace the active monitor: no
quality comparison remains valid: yes
raw training wall-clock direct comparison: no
```

Compact machine evidence:

```text
artifacts/reports/generation/quality_bridge_50k_transition_state_machine_audit_2026-08-18.json
canonical LF bytes: 4,965
canonical LF SHA256: 40ccce33849ff39424f4bdda9cf48cbe7aad39f7ac471a75e13e46537df015aa
Git blob OID: c8ca111da333760dd46aee3d66a0f679fbf1af77
```

This audit does not evaluate sample quality and cannot authorize promotion,
larger training, broad superiority, or release.
