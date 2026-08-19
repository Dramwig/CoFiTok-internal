# Exposure-aware quality follow-up waiter deployment

This evidence records the CPU-only waiter deployed at commit
`85e3ece1196fd318cd6823439824e19fca4275a3`. The waiter does not interpret an
improving but low-quality terminal result without first replaying the exact
matched 100K training exposure.

The deployed process is PID `587876` in an independent clean checkout. At the
recorded observation it was correctly waiting for the terminal quality result;
neither that result nor the terminal exposure report existed yet. The active
matched run remained healthy at CoFiTok step `51,650` and dense step `50,000`,
with no unrelated GPU process.

If both terminal sources arrive, the waiter may write only
`followup_experiment_decision_exposure_aware_v2.json`. It cannot launch GPU
work, training, sampling, evaluation, promotion, release, or full 300K, and it
does not overwrite or automatically supersede the existing v1 decision.

See `deployment_receipt.json` for exact identities and
`docs/records/2026-08-20_generation_quality_bridge_exposure_aware_followup_waiter.md`
for the rationale and verification boundary.
