import json
from pathlib import Path

import pytest

from cofitok.inference_replay import file_identity
from scripts.verify_generation_exposure_semantic_trajectory_replay import (
    _render_json_bytes,
    _verify_bound_file,
)


def test_render_json_bytes_normalizes_types_and_uses_lf() -> None:
    payload = {
        "steps": (1_250, 2_500, 5_000),
        "by_step": {1_250: {"timesteps": (500, 700, 900)}},
    }
    rendered = _render_json_bytes(payload)

    assert b"\r\n" not in rendered
    assert rendered.endswith(b"\n")
    assert json.loads(rendered) == {
        "steps": [1_250, 2_500, 5_000],
        "by_step": {"1250": {"timesteps": [500, 700, 900]}},
    }


def test_bound_file_replay_rejects_changed_payload(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text('{"status":"pass"}\n', encoding="utf-8")
    expected = file_identity(source)

    assert _verify_bound_file(expected, label="source") == expected

    source.write_text('{"status":"changed"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="physical identity differs"):
        _verify_bound_file(expected, label="source")
