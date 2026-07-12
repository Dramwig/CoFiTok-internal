# Tiny Dense-Epsilon Generated Quality

Date: 2026-07-09

Scope: close the remaining Tiny formal64 paper-table partial row by evaluating
the existing 5k same-backbone dense-epsilon checkpoint.

Runbook:

```text
artifacts/runbooks/p0_tiny_dense_generated_quality_2026-07-09.sh
```

Checkpoint:

```text
/root/autodl-tmp/CoFiTok/checkpoints/train_tiny_imagenet_k8_epsilononly_p150eval_5k_2026-07-08/checkpoint_final.pt
```

Output:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generated_quality_stream_tiny_epsilononly_5k_1024_ddim50_2026-07-09/generated_quality_report.json
artifacts/reports/summary_2026-07-09_p0_formal64_tiny_dense_generated/
artifacts/reports/formal64_p0_horizontal_table_2026-07-09_tiny_external_dense_complete/
artifacts/reports/paper_comparison_matrix_2026-07-09_p0_formal64_tiny_external_dense_complete/
```

Metric:

| Method | Dataset | Steps | NFE | Lowres Frechet | Inception Frechet |
|---|---|---:|---:|---:|---:|
| Dense epsilon | `tiny_imagenet_200` | 5000 | 50 | 8.2676 | 385.5078 |

Result: the Tiny Dense-epsilon row is now `completed` in the formal64 table.
The matrix status counts remain `completed=45, missing=26, needs_adapter=32,
partial=1, protocol_blocked=8` because the matrix's remaining `partial` cell
belongs to `cifar10`.
