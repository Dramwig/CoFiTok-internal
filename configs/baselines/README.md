# Baseline Configs

Baseline configs belong here by alias:

```text
configs/baselines/<alias>/
```

Each config must state the dataset alias, split, resolution, preprocessing,
conditioning, training steps, batch size, optimizer, seed, parameter count
policy, NFE/sampling steps, and evaluation scripts.

Keep `imagenet_1k_64x64_hf` and strict-source `downsampled_imagenet_64`
separate. A row trained on one alias cannot be reported under the other.
