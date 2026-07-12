from __future__ import annotations

import os
from pathlib import Path

import torch
from torchvision.utils import save_image


def save_tensor_png(
    image: torch.Tensor,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing image {output}")
    temporary = output.with_name(f".{output.name}.part")
    try:
        save_image(
            (image.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5,
            temporary,
            format="png",
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output
