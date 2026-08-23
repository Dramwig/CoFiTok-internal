# Terminal route supersession interlock

The full-data 100K quality bridge has one authoritative terminal evidence chain,
but three previously deployed GPU-capable consumers still bind
`reports/terminal_system_claim_guard_v1/terminal_system_claim_guard.json`.
That legacy terminal guard consumes a runtime guard whose terminal pair-monitor
SHA256 is `501e24dd...`; the completed authoritative pair monitor is
`78306a02...`, and the corrected strict runtime replay is
`f39e3e42...`.

This branch adds a deployment-time, CPU-only, non-authorizing interlock. Before
the terminal route exists, it verifies the exact live legacy consumers, source
hashes, checkouts, waiting statuses, sole dense terminal sampler, and the stale
versus strict runtime bindings. It then publishes three static markers:

- a pre-existing lock directory for the legacy factorization output;
- a pre-existing lock directory for the legacy conditioning output;
- a regular-file marker at the legacy random-token output path.

The exact legacy supervisors check the first two paths before `Popen`; their
runbooks also refuse an existing lock directory. The random-token runbook
refuses an output path that exists but is not a directory, in addition to
requiring the exact completed factorization status. Thus the old consumers can
remain untouched while being made fail-closed. No process is signaled and no
GPU work, training, promotion, export, release, or 300K launch is authorized.

Any subsequently selected machine route must use the corrected runtime-strict
terminal guard and a new versioned output root. This interlock does not select
that route and does not change `generation_advantage_proven=false`.
