# EDM Adapter

This adapter keeps NVLabs EDM as an external repository and stores only
CoFiTok-side glue.

External repo:

```text
/root/autodl-tmp/CoFiTok/baselines/repos/edm
```

CoFiTok uses EDM directly through its public scripts:

- `dataset_tool.py` is not required for the current formal64 path because EDM
  accepts image directories directly.
- `train.py` trains an unconditional EDM/DDPM++ model on the same dataset
  folders used by CoFiTok.
- `generate.py` exports PNG samples from `network-snapshot-*.pkl`.

The formal runbook writes `baseline_train_report.json` and
`baseline_eval_report.json` so `build_paper_comparison_matrix.py` can count EDM
as external baseline evidence without copying the repo into `CoFiTok-internal`.
