# Training runtime selection environment binding (2026-07-13)

## Finding

The full ImageNet-256 runtime selector already bound benchmark configs, Git
revision, optimizer-step timing, throughput, and CUDA memory. It did not carry
the canonical runtime environment used by the real benchmark. A benchmark cache
from another Python/PyTorch/CUDA/driver/GPU/project-lock environment could
therefore influence microbatch selection, while the final 300K reports were
validated only against each other.

## Contract

`train_generation.py` benchmark reports now include the same canonical runtime
environment and SHA256 used by exact-resume checkpoints. A cached benchmark is
accepted only when its config, clean Git revision, benchmark horizon, checkpoint-
free role, and recomputed environment SHA all match.

`select_generation_training_runtime.py` rejects malformed fingerprints,
CoFiTok/dense environment differences, and environment changes across any pair
of successfully completed candidates. The selected report carries the shared
SHA256 in addition to each source benchmark's complete environment.

The terminal completion audit independently recomputes every completed
candidate fingerprint, verifies clean source Git provenance, and requires the
selected environment SHA to equal both full 300K training reports. Thus runtime
packing remains an execution optimization inside one reproducible environment,
not an untracked source of protocol variation.

## Verification

Focused tests cover valid selection, malformed hashes, within-candidate drift,
cross-candidate drift, dirty benchmark caches, benchmark report emission, final
training drift, and direct benchmark-evidence tampering.

- focused runtime-selection/exact-resume/completion tests: 53 passed;
- complete local repository suite: 475 passed.
