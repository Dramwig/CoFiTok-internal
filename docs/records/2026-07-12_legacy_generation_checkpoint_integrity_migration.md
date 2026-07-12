# Legacy generation checkpoint integrity migration (2026-07-12)

## Scope

The active ImageNet-256 10% matched 50K queue was intentionally kept on commit
`781a014` so CoFiTok and `dense_identity` use one training revision. That commit
atomically renames checkpoint files but predates the later integrity-sidecar
protocol. Restarting the pair merely to gain metadata would discard GPU work.

## Migration

After both 50K trainings complete and before sampling, the post-evaluation
runbook runs `scripts/migrate_generation_checkpoint_integrity.py` for each final
checkpoint. The migration:

- computes SHA256 and byte count without modifying checkpoint bytes;
- loads the payload and requires model, EMA, optimizer, scheduler, RNG, config,
  and stateful sampler fields needed by the exact-resume path;
- writes the standard atomic `.pt.integrity.json` sidecar;
- updates `latest.json` and `training_report.json` to the same hash and step;
- records a separate migration report beside the run.

This proves payload readability, field presence, byte identity, and provenance.
It does not claim that a post-migration optimizer step has already been run; the
full branch's CPU resume smoke remains the executable recovery test.
