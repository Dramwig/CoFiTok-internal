import pytest

from scripts.build_frozen_loss_gradient_source_catalog import select_rows


def sample_rows():
    return [{"path": f"extracted/val/n{i:08d}/{j}.jpg", "wnid": f"n{i:08d}",
             "label": i, "height": 256, "width": 256}
            for i in range(40) for j in range(3)]


def test_selection_is_order_invariant_unique_and_excludes_previous_classes():
    rows = sample_rows()
    excluded = {"n00000000", "n00000001"}
    chosen = select_rows(rows, excluded)
    assert chosen == select_rows(list(reversed(rows)), excluded)
    assert len(chosen) == 32
    assert len({row["wnid"] for row in chosen}) == 32
    assert not {row["wnid"] for row in chosen} & excluded


def test_selection_requires_enough_distinct_classes():
    with pytest.raises(ValueError, match="insufficient"):
        select_rows(sample_rows()[:12], set())


@pytest.mark.parametrize("field,value", [("path", "../escape.jpg"), ("label", True), ("width", 224)])
def test_selection_rejects_invalid_source_rows(field, value):
    rows = sample_rows()
    rows[0][field] = value
    with pytest.raises(ValueError, match="contract differs"):
        select_rows(rows, set())


def test_selection_rejects_duplicate_path():
    rows = sample_rows()
    with pytest.raises(ValueError, match="duplicate"):
        select_rows(rows + rows[:1], set())
