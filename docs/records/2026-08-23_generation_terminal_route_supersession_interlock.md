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

## Deployment

The deployed code checkout is:

```text
/root/autodl-tmp/CoFiTok/checkouts/terminal-route-supersession-interlock-bdcab4b/CoFiTok-internal
revision: bdcab4b15bbbb18a78deaba85ff96eac6c129964
tree: f94f85b227f2c6496f6d9c2ed845a03639df857b
branch: analysis/generation-terminal-route-supersession-interlock-v1-20260823
source SHA256: 9c795fa660a73eb5c48496667d1fecbd29211c5d367e14bde4a90e3b6deb889c
```

The incremental bundle is
`D:/cofitok-bundles/terminal-route-supersession-interlock-bdcab4b.bundle`,
14,231 bytes, SHA256
`2b5279a8c9914fd210d37c0ae17fae342b94cf3affc140b1a39268602566573b`.
It requires `b9c03a70204fe7e89af39c041894539f31097086` and advertises only the
deployment branch at `bdcab4b`.

Remote Linux verification completed with `10 passed`. The live preflight then
verified the sole GPU owner as dense DDIM-100 terminal sampler PID `128678`,
the stale runtime binding, and exact child-free factorization/conditioning
supervisors plus the random-token wrapper. At deployment time dense sampling
was `1376/10000`.

The canonical remote receipt is:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_base128_quality_bridge_v1/reports/terminal_route_supersession_interlock_v1/supersession_receipt.json
bytes: 24968
SHA256: ce7bd71d7ef4892a7c2eafb2d3c46c69d89f3e9c14c8a2ed7ede81bf40e9b860
```

The three active markers are:

| legacy route | marker SHA256 | bytes | mode |
|---|---:|---:|---:|
| factorization quality regression | `79469ddaeb0a11448018b9021260cfcfbb88d296918ec558b540038952ded6ea` | 2,483 | `0444`, parent lock directory `0555` |
| conditioning ranking | `a6388095e6a54cb350e375ac4efc319136035bce9aab38ab22de20bf8b4f8262` | 2,431 | `0444`, parent lock directory `0555` |
| random-token semantic visual | `e31e72391d95468121629ca437fc9497970c86109d57f5dfbd826c569a33c22c` | 2,464 | `0444` regular-file output marker |

An independent post-write replay returned `pass`, found all three interlocks,
found no legacy GPU child, and observed the same sole dense sampler. The random
wrapper's transient child was verified as only `sleep 300`, not a GPU process.

An earlier non-deploying rehearsal at `b92248b` stopped during preflight because
one expected random-token shell snippet used the expanded rather than literal
`${QUALITY_ROOT}` form. It created no marker, receipt, process, or experiment.
The corrected commit above passed both tests and live preflight before the
single deployment.
