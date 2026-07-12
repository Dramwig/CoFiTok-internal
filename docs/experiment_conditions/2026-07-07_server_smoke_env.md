# 2026-07-07 Server Smoke Environment

Server:

```text
pro6000
/root/autodl-tmp/CoFiTok/CoFiTok-internal
```

Execution environment:

```text
/root/miniconda3/bin/conda run -n pf-vlm python
Python 3.10.20
```

GPU observed before scaffold verification:

```text
NVIDIA RTX PRO 6000 Blackwell Server Edition
memory.total: 97887 MiB
memory.used: 0 MiB
utilization.gpu: 0 %
```

Validation commands:

```bash
cd /root/autodl-tmp/CoFiTok/CoFiTok-internal
PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python -m pytest -q
PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/smoke_forward.py \
  --config configs/smoke_random_cpu.json \
  --output /root/autodl-tmp/CoFiTok/checkpoints/smoke/random_forward_report.json
PYTHONPATH=src /root/miniconda3/bin/conda run -n pf-vlm python scripts/smoke_forward.py \
  --config configs/smoke_random_cuda.json \
  --output /root/autodl-tmp/CoFiTok/checkpoints/smoke/random_forward_cuda_report.json
```

Results:

```text
pytest: 5 passed
CPU smoke: wrote /root/autodl-tmp/CoFiTok/checkpoints/smoke/random_forward_report.json
CUDA smoke: wrote /root/autodl-tmp/CoFiTok/checkpoints/smoke/random_forward_cuda_report.json
```

Local archived report copies:

```text
C:/Users/zixi-/Desktop/PaperField/CoFiTok/CoFiTok-internal/artifacts/reports/smoke/random_forward_report.json
C:/Users/zixi-/Desktop/PaperField/CoFiTok/CoFiTok-internal/artifacts/reports/smoke/random_forward_cuda_report.json
```

Smoke diagnostics:

```text
CPU smoke actual_device: cpu
CPU epsilon shape: 4 x 3 x 32 x 32
CPU zero-token component energy: 0.0

CUDA smoke actual_device: cuda
CUDA epsilon shape: 8 x 3 x 32 x 32
CUDA zero-token component energy: 0.0
```
