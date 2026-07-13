# Formal generation dataset provenance (2026-07-13)

## Finding

Generation configs and completed training reports previously bound only the
dataset alias and root path. Because the matched methods run serially, those
fields could remain equal even if the authoritative ImageNet manifest or split
contents changed between CoFiTok and dense training. Runtime environment and Git
provenance do not close that data-identity gap.

## Contract

`cofitok.data.provenance` defines the two formal generation identities from the
existing experiment-condition records:

| alias | manifest bytes | manifest SHA256 | train | val |
|---|---:|---|---:|---:|
| `imagenet_256_10pct` | 54,885,982 | `dcdd622564941ad418fffa960051f14f0ad41852c681ed9a8329c5e6f561cca7` | 128,161 | 50,000 |
| `imagenet_256` | 405,484,553 | `9a2eec642f0d56162bffaafed84a41267f22abfc9feff4cf41fed9f6881173f0` | 1,281,167 | 50,000 |

Before constructing the model, production training hashes
`metadata/image_manifest.jsonl`, checks bytes and loader split sizes, rejects a
symlinked dataset root or manifest, and computes a canonical identity SHA over
alias, resolved root, manifest identity, and split sizes.

## Propagation

The verified record is written to the run manifest, checkpoint-free runtime
benchmark, checkpoint payload, integrity sidecar, `latest.json`, and final
training report. Exact resume compares the expected identity with the integrity
sidecar before `torch.load`, then revalidates the payload copy before restoring
model, EMA, optimizer, scheduler, scaler, or RNG state.

Completed-pair validation schema v2 requires both reports and checkpoint
pointers to carry the same formal identity. The full 300K completion audit is
strict. The active `781a014` 10% pair predates this field, so only its two
reports may jointly use `--allow-legacy-missing-dataset-provenance`; a mixed
legacy/bound pair is rejected and the validator retains a visible warning.
Training runtime selection is likewise schema v2: stale benchmark caches without
valid data provenance are archived, every completed method/candidate must share
one identity, and the selected identity must match both final training reports.

This identity binds the authoritative sample manifest and recorded split, not a
new claim that every derived JPEG was individually rehashed. Source parquet
checksums and export lineage remain documented in
`docs/experiment_conditions/imagenet_256_full_2026-07-09.md` and
`docs/experiment_conditions/imagenet_256_10pct_2026-07-09.md`.
