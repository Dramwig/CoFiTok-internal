# Generation-system upgrade start (2026-07-12)

The locked mechanism-validation state was committed locally as `88c3aab` on
branch `paper-evidence-locked`. New generation work was branched to
`scale/generative-system` so no locked experiment, report, or paper table is
silently replaced.

Initial implementation adds a scalable class-conditional U-Net predictor,
restricted CoFiTok and matched dense configurations, production checkpoint and
EMA infrastructure, exact model RNG/data-order resume, and EMA CFG/DDIM sample
export. Local verification includes the full test suite, exact interrupted vs.
uninterrupted CPU training metrics, and generated PNGs from an EMA checkpoint.

This record marks engineering progress only. Large-scale generation is not
complete until the 10% and full-data promotion gates in `docs/GENERATION_SYSTEM.md`
are executed on `pro6000` and their reports/checkpoints are archived.
