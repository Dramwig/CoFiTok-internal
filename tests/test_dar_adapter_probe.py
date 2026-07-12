from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "baselines" / "probe_dar_adapter.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("probe_dar_adapter", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_extract_image_size_choices() -> None:
    module = load_script_module()
    source = 'parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=256)'

    assert module._extract_image_size_choices(source) == [256, 384, 512]


def test_build_report_marks_formal64_incompatible(tmp_path: Path) -> None:
    module = load_script_module()
    write(tmp_path / "README.md", "D-AR models trained on ImageNet 256x256")
    write(tmp_path / "GETTING_STARTED.md", "python tokenizer/tokenizer_image/utils_repa.py # Download REPA DINO")
    write(tmp_path / "requirements.txt", "torch>=2.1.0\n")
    write(tmp_path / "configs/tokenizer_v1.yaml", "model:\n  num_all_queries: 256\n")
    write(
        tmp_path / "tokenizer/tokenizer_image/vq_train_accelerate.py",
        'parser.add_argument("--image-size", type=int, choices=[128, 256, 384, 512], default=256)\n'
        "dino_weight = 0.5\n",
    )
    write(
        tmp_path / "autoregressive/train/train_c2i_accelerate.py",
        'parser.add_argument("--image-size", type=int, choices=[256,384,448,512], default=256)\n'
        "from dataset.imagenet import build_imagenet\n",
    )
    write(
        tmp_path / "autoregressive/sample/sample_c2i.py",
        'parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=256)\n',
    )
    write(tmp_path / "dataset/imagenet.py", "from torchvision.datasets import ImageFolder\n")

    report = module.build_report(tmp_path)

    assert report["status"] == "feasibility_only"
    assert report["formal64_status"] == "not_protocol_compatible_without_code_and_protocol_changes"
    assert report["image_size_choices"]["ar_train"] == [256, 384, 448, 512]
    assert any("exclude 64" in blocker for blocker in report["blockers"])
    assert any("class-conditional" in blocker for blocker in report["blockers"])
