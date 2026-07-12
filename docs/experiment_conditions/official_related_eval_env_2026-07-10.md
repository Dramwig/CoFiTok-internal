# Official Related Eval Environment - 2026-07-10

Host: `pro6000`

Environment:

```text
conda env: pf-vlm
python: /root/autodl-tmp/conda/envs/pf-vlm/bin/python
torch: 2.7.1+cu128
torchvision: 0.22.1+cu128
```

Packages added for official related-method eval smoke:

```text
webdataset==1.0.2
torch-fidelity==0.4.0
braceexpand==0.1.7
tensorflow-cpu==2.21.0
opencv-python-headless==5.0.0.93
tensorboard==2.21.0
```

Install command:

```bash
python -m pip install -i https://pypi.org/simple webdataset torch-fidelity
python -m pip install -i https://pypi.org/simple tensorflow-cpu
python -m pip install -i https://pypi.org/simple opencv-python-headless tensorboard
```

Reason:

- D-AR imports require `webdataset`.
- MAR official evaluator imports require `torch_fidelity`.
- MAR official `main_mar.py` imports `cv2` and `SummaryWriter`.
- D-AR ADM evaluator imports `tensorflow.compat.v1`.
- ReTok basic imports were already available in `pf-vlm`.

Validation:

```text
D-AR imports ok: autoregressive.models.gpt, dar_tool, webdataset
MAR imports ok: models.mar, torch_fidelity
MAR runtime imports ok: cv2 5.0.0, tensorboard 2.21.0
ReTok basic imports ok: torch, omegaconf, timm
TensorFlow evaluator ok: tensorflow 2.21.0
```
