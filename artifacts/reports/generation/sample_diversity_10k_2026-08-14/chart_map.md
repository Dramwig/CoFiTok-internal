# Chart map

## Global nearest-neighbor dHash candidate rates

- Question: Do the generated cohorts contain more low-level near-duplicate candidates than a matched real reference?
- Family/type: grouped bar.
- Fields: cohort, Hamming threshold (`<=4`, `<=8`), candidate fraction and count.
- Claim supported: exact-copy collapse is absent and dHash-close candidates are not elevated in either generated cohort.
- Palette: hard two-root cap for the two thresholds, plus neutral scaffolding; cohort identity remains on the axis.

## Per-class nearest-neighbor dHash distribution

- Question: Is within-class low-level diversity systematically narrower in generated samples?
- Family/type: box plot.
- Fields: cohort, class index, median nearest-neighbor Hamming distance among 10 images in that class.
- Grain: deterministic matched 500-class subset for the MCP snapshot; the durable JSON retains all 1,000 classes per cohort.
- Claim supported: the distribution of within-class dHash proximity is not shifted toward substantially closer neighbors in the generated cohorts.
- Palette: single-root preferred; cohort labels and box positions carry identity without requiring color alone.

The first visual compares cumulative candidate rates; the second shows a class-level distribution. Repeated bar-family views were avoided because the analytical questions are comparison versus distribution.
