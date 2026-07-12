# Generation runbook CLI contracts

Date: 2026-07-12

Branch: `scale/generative-system`

The post-training pipeline spans migration, training, runtime selection,
preflight, sampling, metrics, visual audit, gates, comparison, inference export,
and final completion audit. Shell syntax checks cannot detect a renamed or
removed Python CLI option, and isolated module tests do not prove that runbooks
still call the current parser contract.

`tests/test_generation_runbook_entrypoints.py` reads the six authoritative
post-training runbooks and extracts every multiline `python scripts/*.py`
command. It currently covers 18 unique entry points. For each command it:

- requires the discovered script set to match the locked expected entrypoint set;
- executes the real script with `--help` in-process, proving imports and parser
  startup succeed;
- extracts every `--option` used by the runbook and requires it to appear in the
  corresponding CLI help.

The option set is derived from the runbook source rather than copied into a
second hand-maintained map. Adding an unreviewed entrypoint, removing a parser
option, or leaving a stale shell argument therefore fails before remote
deployment.
