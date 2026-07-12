from __future__ import annotations

from scripts.train_generation import StopController


def test_stop_controller_records_signal() -> None:
    controller = StopController()

    controller.request(15, None)

    assert controller.requested is True
    assert controller.signal_number == 15
