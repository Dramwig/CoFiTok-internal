from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cofitok.generation import build_epsilon_stability_sampling_design
from cofitok.generation_gate_sources import gate_source_report_identity
from cofitok.image_integrity import image_tree_sha256
from cofitok.reporting import write_json_report
from scripts import prepare_generation_epsilon_stability_real_artifact_subset as subset
from scripts import run_generation_epsilon_stability_sampling_recovery as controller


def _execution() -> dict:
    return {
        "random_stream": {
            "seed": 884_241,
            "start_index": 0,
        },
        "methods": {
            "cofitok": {
                "checkpoint": {"path": "/checkpoints/cofitok.pt"},
                "prefix_budget": 8,
            },
            "dense_identity": {
                "checkpoint": {"path": "/checkpoints/dense.pt"},
                "prefix_budget": 1,
            },
        },
    }


def test_gpu_process_parser_is_strict() -> None:
    assert controller.parse_gpu_process_pids("") == []
    assert controller.parse_gpu_process_pids("19\n7\n19\n") == [7, 19]
    with pytest.raises(ValueError, match="unparseable"):
        controller.parse_gpu_process_pids("python")


def test_duplicate_controller_scan_binds_output_root(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    (proc / "101").mkdir(parents=True)
    (proc / "102").mkdir(parents=True)
    output = tmp_path / "diagnostic_v1"
    (proc / "101/cmdline").write_bytes(
        b"python\0scripts/run_generation_epsilon_stability_sampling_recovery.py\0"
        + output.resolve().as_posix().encode()
    )
    (proc / "102/cmdline").write_bytes(
        b"python\0scripts/run_generation_epsilon_stability_sampling_recovery.py\0"
        b"/another/output"
    )
    assert controller.duplicate_controller_pids(
        output_root=output,
        own_pid=999,
        proc_root=proc,
    ) == [101]


@pytest.mark.parametrize(
    ("case_id", "required", "forbidden"),
    [
        (
            "legacy_terminal_hard_clip",
            ["--x0-constraint", "clip"],
            [
                "--start-timestep",
                "--scale-initial-noise-by-sigma",
                "--recompute-epsilon-after-x0-constraint",
            ],
        ),
        (
            "start975_sigma_dynamic_threshold_recompute",
            [
                "--start-timestep",
                "975",
                "--scale-initial-noise-by-sigma",
                "--x0-constraint",
                "dynamic_threshold",
                "--recompute-epsilon-after-x0-constraint",
            ],
            [],
        ),
    ],
)
def test_sampling_command_materializes_only_the_bound_case(
    case_id: str,
    required: list[str],
    forbidden: list[str],
) -> None:
    command = controller.sampling_command(
        python=Path("/python"),
        project=Path("/project"),
        execution=_execution(),
        design=build_epsilon_stability_sampling_design(),
        case_id=case_id,
        method="cofitok",
        sample_root=Path("/output/samples"),
    )
    assert command.count("--num-samples") == 1
    assert command[command.index("--num-samples") + 1] == "1000"
    assert command[command.index("--prefix-budgets") + 1] == "8"
    assert command[command.index("--seed") + 1] == "884241"
    assert command[command.index("--start-index") + 1] == "0"
    for value in required:
        assert value in command
    for value in forbidden:
        assert value not in command


def test_runbook_is_single_controller_and_non_authorizing() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "artifacts/runbooks/generation_epsilon_stability_sampling_recovery_v1.sh"
    ).read_text(encoding="utf-8")
    assert source.count("run_generation_epsilon_stability_sampling_recovery.py") == 1
    assert "--required-idle-polls 3" in source
    assert "nvidia-smi --query-compute-apps=pid" in source
    assert "train_generation.py" not in source
    assert "kill " not in source
    assert "pkill" not in source
    assert "10000" not in source
    assert "300k" not in source.lower()


def test_real_reference_subset_is_decoded_pixel_exact_and_replayable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    sources = []
    for index, (value, suffix) in enumerate(((17, ".jpg"), (91, ".png"))):
        source = real / f"source_{index}{suffix}"
        Image.fromarray(
            np.full((256, 256, 3), value, dtype=np.uint8),
            mode="RGB",
        ).save(source)
        sources.append(source)
    images = sorted(sources)
    contract = tmp_path / "real_set.json"
    write_json_report(
        contract,
        {
            "real_set": {
                "digest_schema": "cofitok_image_tree_sha256_v1",
                "sha256": image_tree_sha256(images, root=real),
                "root": real.resolve().as_posix(),
                "image_count": 2,
            }
        },
    )
    monkeypatch.setattr(subset, "SAMPLE_COUNT", 2)
    output = tmp_path / "subset"
    first = subset.build_subset(
        real_dir=real,
        real_set_contract_path=contract,
        expected_real_set_contract_sha256=gate_source_report_identity(contract)[
            "sha256"
        ],
        output_dir=output,
        resume=False,
    )
    replay = subset.build_subset(
        real_dir=real,
        real_set_contract_path=contract,
        expected_real_set_contract_sha256=gate_source_report_identity(contract)[
            "sha256"
        ],
        output_dir=output,
        resume=True,
    )
    assert replay == first
    assert sorted(path.name for path in output.iterdir()) == [
        "000000.png",
        "000001.png",
    ]
    assert first["materialization"] == (
        "decoded_rgb_pixels_to_lossless_numbered_png_v1"
    )
    assert [row["source_format"] for row in first["source_images"]] == [
        "JPEG",
        "PNG",
    ]
    for source, destination in zip(images, sorted(output.glob("*.png"))):
        with (
            Image.open(source) as source_image,
            Image.open(destination) as output_image,
        ):
            assert output_image.format == "PNG"
            assert output_image.mode == "RGB"
            assert output_image.size == (256, 256)
            assert output_image.tobytes() == source_image.tobytes()
