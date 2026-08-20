# Generation completion receipt required-check contract

Date: 2026-08-20

## Purpose

Harden the final inference release trust boundary. Previously,
`cofitok.generation.release` accepted a completion audit when its top-level
summary claimed completion and the single inference-export check passed. A
manually constructed audit could therefore omit the remaining terminal checks
while keeping `complete=true`, `failed_checks=[]`, and `missing_checks=[]`.

This did not bypass the separately bound full quality gate inside an inference
artifact, but it made the terminal completion receipt weaker than the auditors
that produced it.

## Implementation

Implementation commit:

```text
0ec45d2ec9f49c7b755b361c2e52809a7c47a424
```

The receipt layer now owns exact ordered contracts for all supported audit
profiles:

- `large_scale_generation_v1`: 18 required checks.
- `stability_generation_system_v1`: 18 required checks.
- `capacity_full_generation_system_v1`: 12 required checks.

Before any inference artifact verification, receipt construction now rejects:

- a missing or malformed check list;
- missing required checks;
- duplicate check names;
- unknown extra checks;
- reordered checks;
- any status other than `pass`;
- missing non-mapping check evidence;
- incomplete or duplicated inference-export evidence.

Receipt consumption reconstructs the same payload from the bound completion
audit, so the contract is revalidated whenever an artifact is authorized.

## Verification

- Legal receipt construction and consumption were exercised for all three
  profiles.
- Negative tests cover forged minimal completion, missing checks, duplicate
  checks, hidden non-pass status, missing evidence, unknown checks, and order
  drift.
- Monkeypatch sentinels prove an invalid audit is rejected before artifact
  verification during receipt construction and before `torch.load` during
  production inference consumption.
- AST extraction independently compared each auditor's `_check(...)` sequence
  with the release contract: `18/18`, `18/18`, and `12/12`, all exact matches.
- Full CPU pytest run collected 1,728 tests and exited successfully.

## Operational boundary

This change was developed only in the clean local worktree
`C:/qbfinalreleaseaudit`. It was not deployed to the active remote training
checkout, did not access the GPU, did not signal any process, and does not
alter or authorize the running 100K quality bridge or any later training,
sampling, promotion, or release stage.

This is release-provenance hardening, not scientific evidence that CoFiTok has
generation-quality superiority over the matched dense baseline.
