# Deterministic generation visual-quality audit (2026-07-12)

Formal scalar metrics are necessary but not sufficient for reviewing generated
images. The promotion and final post-evaluation runbooks now produce deterministic
visual evidence with `scripts/build_generation_visual_audit.py`.

## Panels

The audit writes three independent PNGs:

- fixed-index CoFiTok endpoint samples;
- the same fixed-index dense-identity endpoint samples;
- CoFiTok prefix paths with columns `1,2,4,8` and fixed seeds by row.

Endpoint indices span the beginning, middle, and tail of the 10K/50K formal
sample set. Prefix diagnostics contain only 64 samples, so they use a separate
fixed `0..15` index list. These index domains must not be conflated.

## Provenance and checks

The builder requires completed sampling reports and verifies output directories,
checkpoint SHA/step, sample-set SHA/count, prefix checkpoint identity, exact PNG
existence, and consistent image shapes. Every selected source file and generated
panel receives a SHA256. It also records pixel range/mean/std and rejects exact
duplicate endpoint PNGs among the fixed samples.

The report explicitly declares that the panels are not quantitative metrics.
They support human inspection and cannot replace FID, IS, precision, or recall.
The final completion audit binds the panel sources to the exact formal generation
reports and requires all three panel hashes, nonzero pixel variation, zero exact
endpoint duplicates, and prefix budgets `1,2,4,8`.

Panels remain separate files so downstream paper layouts can arrange them without
baking labels or unrelated formatting into the source evidence.
