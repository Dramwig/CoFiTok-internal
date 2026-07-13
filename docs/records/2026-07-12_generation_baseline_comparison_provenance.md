# Strong-baseline comparison provenance

Date: 2026-07-12

Branch: `scale/generative-system`

The large-scale result has two intentionally separate comparison tiers.

`CoFiTok K=8` and `Dense identity` are the only direct matched-training rows.
Their parameter count, effective batch, images seen, training time/throughput,
peak VRAM, sampling batch, sampling time/throughput, checkpoint SHA, and sample
set SHA are derived from the same 300K training and 50K sampling evidence used
by the final gate. Each row also carries the matched real-set tree SHA and
evaluator runtime-environment SHA.

`D-AR`, `MAR`, and `ReTok` are official-pretrained ImageNet-256 50K contextual
rows. They are not compute-matched: their training budgets differ and their
reported metrics use the pinned ADM TensorFlow baseline protocol rather than the
matched torch-fidelity evaluator. The comparison policy therefore sets
`cross_tier_numeric_ranking_allowed=false`.

Comparison schema version 5 binds the tracked
`official_related_methods_table.json` bytes by SHA256 and preserves each row's
alias, exact method identity, eval-only status, secondary-only table role,
protocol, metric source path, and distribution metrics. The final completion
audit rereads that table, recomputes its SHA, checks all three identities and
metrics, and independently recomputes every direct compute field from training
and sampling reports. Empty contextual placeholders, mislabeled rows, edited
metrics, stale source tables, unmatched real data/evaluator environments, and
cross-tier ranking policies all fail closed.

Schema v5 also binds the two full training reports, two formal 50K metrics
reports, and final generation gate by authoritative path, byte count, and
SHA256. Completion rereads all five files and independently verifies matched
FID, IS, precision, recall, evaluator identity, sampling invocation count,
derived FID summary, training cost, sampling cost, and protocol provenance.
