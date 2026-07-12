# Formal sampling CLI integration

Date: 2026-07-12

Branch: `scale/generative-system`

The formal 10K/50K runbooks call `scripts/generate_samples.py`, while most
sampler tests previously exercised `ddim_sample`, `predict_epsilon`, or
`GenerationSession` directly. After the inference-session refactor, the CLI
manifest still referenced a removed local `schedule` variable. A formal run
would have failed before generating its first image even though lower-level
tests passed.

The CLI now derives `actual_timesteps` from `session.schedule`. A subprocess
integration test creates a real tiny class-conditional checkpoint with an
integrity sidecar, invokes the same CLI entry point, performs DDIM sampling,
publishes a PNG atomically, and verifies the completed sampling report including
actual timesteps and sample-set SHA256. The command also creates the immutable
sampling manifest and atomic sampling-progress record used by resumable formal
runs.

This complements the exact batched-versus-sequential CFG test, per-index RNG
batch/resume invariance test, corrupt-PNG recovery test, and target GPU/checkpoint
preflight.
