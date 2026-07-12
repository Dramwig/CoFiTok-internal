import pytest

from cofitok.models.blocks import group_norm_groups


@pytest.mark.parametrize(
    ("channels", "expected"),
    [(24, 8), (32, 8), (30, 6), (28, 7), (27, 3), (5, 5)],
)
def test_group_norm_groups_uses_largest_valid_group_count(
    channels: int, expected: int
) -> None:
    assert group_norm_groups(channels) == expected
