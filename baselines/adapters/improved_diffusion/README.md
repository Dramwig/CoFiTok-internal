# improved-diffusion Adapter

This adapter keeps OpenAI `improved-diffusion` unmodified while making the repo
usable inside the current `pf-vlm` environment for single-process baseline
smoke and fair-training runs.

The remote upstream clone is:

```text
/root/autodl-tmp/CoFiTok/baselines/repos/improved_diffusion
```

The adapter supplies local-only shims for:

- `mpi4py`: one-rank `COMM_WORLD`.
- `blobfile`: local filesystem calls used by the upstream scripts.

Use it by prepending this directory and the upstream repo to `PYTHONPATH`:

```bash
export PYTHONPATH=/root/autodl-tmp/CoFiTok/CoFiTok-internal/baselines/adapters/improved_diffusion:/root/autodl-tmp/CoFiTok/baselines/repos/improved_diffusion
```

No third-party repo files are edited. Any future behavioral patches must be
saved under `baselines/patches/improved_diffusion/`.
