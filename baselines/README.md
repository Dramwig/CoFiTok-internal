# Baseline Code Assets

This directory tracks baseline metadata and lightweight CoFiTok-side glue only.
Do not place full external repositories here.

External baseline repositories live on `pro6000` under:

```text
/root/autodl-tmp/CoFiTok/baselines/repos/<alias>
```

The first setup pass clones only the baselines marked `clone_on_setup: true` in
`baselines/registry.json`. That set is intentionally limited to methods needed
to answer the main MVP threats:

- `edm`
- `improved_diffusion`
- `d_ar`
- `ml_flextok`
- `mar`
- `titok_1d_tokenizer`
- `retok`

The parameter-matched direct dense predictor and endpoint-only factorized
control are internal and are implemented through CoFiTok configs/runners, not
by cloning another repo.

Run from `CoFiTok-internal` on `pro6000`:

```bash
python scripts/baselines/clone_baseline_repos.py \
  --registry baselines/registry.json \
  --repo-root /root/autodl-tmp/CoFiTok/baselines/repos \
  --manifest-output docs/experiment_conditions/baselines_repo_manifest_2026-07-09.json
```

The manifest records URL, commit hash, license file candidates, clone status,
and target path. Fair-training adapters, patches, and configs should be added
after this code-asset step is stable.
