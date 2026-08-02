# Terminal generation release receipt (2026-08-03)

## Gap

The EMA inference artifact already carried two pre-deserialization
authorizations: the scaling gate that allowed full training and the final
quality gate that allowed export. Export, preflight, and smoke inference are
inputs to the terminal generation-system audit, however, so none of those
artifacts could also prove that the terminal audit itself had passed. The
documentation required completion before routine deployment, but
`--require-release-authorization` could enforce only the earlier final gate.

## Receipt

Both terminal completion runbooks now publish a deterministic
`release_receipt.json` after, and only after, their completion audit succeeds.
Schema 1 binds:

- the physical completion-audit path, bytes, and SHA256;
- the completion profile and its frozen expected revisions/receipts;
- the unique passing inference-artifact check;
- both distinct CoFiTok and dense artifact paths, bytes, SHA256 values, source
  checkpoint sizes, source environment/Git identity, training and release
  authorizations, export/inference environment identity, and smoke PNG hashes.

Receipt creation independently reopens both artifact sidecars and rehashes the
physical artifacts. Reuse is accepted only when the existing receipt is exactly
equal to the deterministic reconstruction. A changed audit, changed artifact,
missing inference check, failed/missing completion check, duplicate artifact
path, or unsupported completion profile fails closed.

## Consumer enforcement

`GenerationSession`, `load_generation_model`, the routine inference CLI, and
the real-forward preflight accept a completion receipt. Production consumers
can require it with:

```text
--completion-receipt <release_receipt.json>
--require-completion-authorization
```

The loader verifies the artifact sidecar, receipt, bound completion audit, and
selected artifact identity before `torch.load`. Completion authorization also
implies the existing final-release requirement. The verified receipt/audit
identity and method are propagated through session, preflight, inference
manifest, progress, and report metadata, so exact resume cannot switch to a
different terminal authorization.

This receipt does not authorize training and does not replace the exact-resume
300K checkpoint. It is the consumer-side proof that a smaller EMA artifact has
passed the entire training, formal quality, matched comparison, reproducibility,
smoke, and completion chain—not merely that its final quality gate passed.

## Verification

Tests cover a real tiny training checkpoint, two EMA exports, both terminal
completion profiles, deterministic receipt publication, production session
generation, real-forward preflight, missing receipt, audit mutation, incomplete
audit, and an artifact not named by the receipt. Formal runbook entrypoint tests
require the receipt builder and verify its live argparse options. Focused
inference/runbook/completion regression passed, and the complete local suite was
`966 passed / 6 skipped in 270.40s`. Both changed runbooks passed native Linux
`bash -n` on pro6000 using their Git-clean-filtered working-tree blobs.
