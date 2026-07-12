from __future__ import annotations

import json

import pytest

from cofitok import reporting


def test_write_json_report_is_atomic_on_failure(tmp_path, monkeypatch) -> None:
    target = tmp_path / "report.json"
    target.write_text('{"status": "previous"}\n', encoding="utf-8")

    def failing_dump(payload, handle, **kwargs):
        del payload, kwargs
        handle.write('{"status": "partial"')
        raise RuntimeError("serialization interrupted")

    monkeypatch.setattr(reporting.json, "dump", failing_dump)
    with pytest.raises(RuntimeError, match="serialization interrupted"):
        reporting.write_json_report(target, {"status": "new"})

    assert json.loads(target.read_text(encoding="utf-8")) == {"status": "previous"}
    assert list(tmp_path.glob(".report.json.*.tmp")) == []


def test_write_json_report_replaces_complete_payload(tmp_path) -> None:
    target = tmp_path / "report.json"
    target.write_text('{"status": "previous"}\n', encoding="utf-8")

    reporting.write_json_report(target, {"status": "completed", "step": 5})

    assert json.loads(target.read_text(encoding="utf-8")) == {
        "status": "completed",
        "step": 5,
    }
    assert list(tmp_path.glob(".report.json.*.tmp")) == []


def test_write_text_report_replaces_complete_payload(tmp_path) -> None:
    target = tmp_path / "report.md"
    target.write_text("previous\n", encoding="utf-8")

    reporting.write_text_report(target, "completed\n")

    assert target.read_text(encoding="utf-8") == "completed\n"
    assert list(tmp_path.glob(".report.md.*.tmp")) == []
