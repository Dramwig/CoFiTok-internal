# Stable generation session and inference CLI (2026-07-12)

The generation path is now a reusable package boundary rather than logic owned
by one formal-evaluation script.

- `GenerationSession` owns one verified loaded model and diffusion schedule.
- immutable `GenerationRequest` owns seeds, classes, prefix, CFG, DDIM, eta,
  precision, and clipping policy;
- `GenerationResult` returns finite CPU images plus checkpoint and protocol
  provenance;
- `infer_generation.py` provides class/seed/prefix-controlled atomic PNG output
  with per-image SHA256;
- `generate_samples.py` uses the same session for formal 10K/50K evaluation.

Formal sampling manifests identify
`cofitok.generation.GenerationSession@1`. The final completion audit requires
that API marker, so a separate or stale inference path cannot silently provide
the paper-scale generation evidence.

CPU tests prove repeat requests are bit-identical, model-specific class/token
validation is fail-closed, the CLI writes all prefix outputs and hashes, and the
formal resume/progress tests continue to pass after the refactor.
