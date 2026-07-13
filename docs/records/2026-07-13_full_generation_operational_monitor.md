# Full-generation operational monitor (2026-07-13)

The full ImageNet-256 matched queue runs for multiple weeks. Pipeline and
supervisor status identify the current shell stage but do not expose optimizer
step, metric freshness, checkpoint cadence, GPU state, or disk state.

Monitoring is now split into a pure `cofitok.monitoring` module and a generic
`scripts/monitor_generation_pair.py` CLI. The existing 10% entry point remains
a thin compatibility wrapper. The monitor is read-only: it parses JSONL/JSON,
stats checkpoint files, queries processes, filesystem usage, and `nvidia-smi`,
and atomically replaces one small status report. It never loads or hashes model
weights, uses the GPU, signals a process, or edits training assets.

For each method it validates:

- strictly increasing metric steps and finite, non-negative loss/runtime fields;
- target-step bounds and exact final completion fields;
- non-empty checkpoint files and 5,000-step checkpoint cadence with a 250-step
  publication grace window;
- metric freshness while a training process is active.

The full runbook starts or reuses a PID-recorded background monitor polling every
300 seconds. It also runs a synchronous one-shot health snapshot after every
CoFiTok and dense milestone training segment, before milestone sampling. A live
training process with no metric update for 1,800 seconds is `stalled`. During
milestone evaluation the runbook remains alive but no training process exists,
so the state is intentionally `waiting` rather than a false stall.

After both exact 300K training reports exist, a final one-shot snapshot must
produce `pass`. The large-scale completion audit requires that report, binds it
to the clean deployed revision and exact run directories, and verifies nonempty
metrics plus retained 50K, 100K, 200K, and 300K checkpoint stat evidence for both
methods.
