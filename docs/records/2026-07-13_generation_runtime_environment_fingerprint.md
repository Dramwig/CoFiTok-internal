# Generation runtime-environment fingerprint (2026-07-13)

Exact model, optimizer, scheduler, scaler, RNG, sampler, config, and Git state do
not by themselves prove exact resume when the numerical software or GPU runtime
changes. Full ImageNet-256 training now captures a stable runtime-environment
object before constructing data iterators or loading checkpoint state.

The fingerprint contains:

- Python implementation, version, and executable;
- operating system release and machine architecture;
- NumPy, Pillow, PyTorch, torchvision, and tqdm package versions;
- PyTorch, CUDA, cuDNN, NVIDIA driver, matmul/TF32/determinism, and thread state;
- CUDA device name, compute capability, total memory, and processor count;
- relevant CUDA/PyTorch/Python environment variables;
- byte size and SHA256 for `pyproject.toml` and `uv.lock`.

The canonical JSON SHA256 is stored in the run manifest, training report,
checkpoint extra state, checkpoint integrity sidecar, and `latest.json`. On
resume, the loader verifies payload/sidecar consistency and recursively compares
the current environment before loading model, EMA, optimizer, scheduler, scaler,
or RNG state. A mismatch fails with exact field paths and cannot advance metrics
or write a new checkpoint.

The final completion audit independently recomputes both full-training report
hashes, requires the latest checkpoint pointers to bind them, verifies required
language/framework/package/project-file fields, and requires CoFiTok and dense to
have the same environment SHA.

This rule applies to upgrade-branch full 300K checkpoints. The pinned 10% legacy
checkpoints are migrated only for byte integrity and are not rewritten. Exported
EMA inference artifacts retain their own provenance and may be loaded in a
compatible deployment environment; exact optimizer-trajectory resume is the
stricter operation protected here.
