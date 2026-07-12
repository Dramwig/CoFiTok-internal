# ADM evaluator asset (2026-07-11)

The D-AR/ADM evaluator resolves its Inception graph relative to the current
working directory. To keep generated assets out of pinned third-party repos,
the canonical graph is stored at:

```text
/root/autodl-tmp/CoFiTok/checkpoints/baselines/official_refs/classify_image_graph_def.pb
```

Source:

```text
https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/classify_image_graph_def.pb
```

Verified properties:

```text
size: 95,673,916 bytes
sha256: 009d6814d1bc560d4e7b236e170e9b2d5ca6f4b57bd8037f6db05776204415c6
```

Runbooks execute the evaluator with this directory as `cwd`. Existing runtime
copies inside D-AR/ReTok may be deleted after the active MAR evaluation finishes;
they are caches, not source modifications or repository patches.
