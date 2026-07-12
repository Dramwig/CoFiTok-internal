import pytest

from cofitok.configs import LossConfig
from scripts.train_short import _scheduled_loss_config


def test_scheduled_loss_config_preserves_static_decor_weight() -> None:
    config = LossConfig(component_decorrelation_weight=0.005)

    scheduled = _scheduled_loss_config(config, step=1)

    assert scheduled.component_decorrelation_weight == pytest.approx(0.005)


def test_scheduled_loss_config_delays_and_warms_decor_weight() -> None:
    config = LossConfig(
        component_decorrelation_weight=0.01,
        component_decorrelation_start_step=100,
        component_decorrelation_warmup_steps=200,
    )

    assert _scheduled_loss_config(config, step=100).component_decorrelation_weight == 0.0
    assert _scheduled_loss_config(config, step=200).component_decorrelation_weight == pytest.approx(0.005)
    assert _scheduled_loss_config(config, step=300).component_decorrelation_weight == pytest.approx(0.01)
    assert _scheduled_loss_config(config, step=500).component_decorrelation_weight == pytest.approx(0.01)


def test_scheduled_loss_config_rejects_negative_schedule() -> None:
    with pytest.raises(ValueError):
        _scheduled_loss_config(
            LossConfig(
                component_decorrelation_weight=0.01,
                component_decorrelation_start_step=-1,
            ),
            step=1,
        )
