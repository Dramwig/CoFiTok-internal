# Official 50K NPZ completion guard (2026-07-11)

D-AR and ReTok write 50,048 PNG files because their official distributed
samplers pad the last batch. Their ADM-evaluator NPZ files contain exactly
50,000 samples. The extra raw PNGs are retained as provenance and are not part
of metric computation.

Paper-facing completion now requires all of the following:

- at least 50,000 raw PNG files;
- NPZ key `arr_0` with header shape `[50000, 256, 256, 3]`, dtype `uint8`, and
  C-order storage;
- finite FID, sFID, Inception Score, Precision, and Recall values.

MAR additionally requires the sampling report to verify the pinned LTH14 repo
commit, official PTH `model_ema`, KL-16 VAE state, asset SHA256 values, and the
full README sampling protocol. The header check reads only the embedded NPY
metadata and does not load the roughly 10 GB sample array.

Implementation:

```text
scripts/baselines/build_official_related_methods_table.py
```
