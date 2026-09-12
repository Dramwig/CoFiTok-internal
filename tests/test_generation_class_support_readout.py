import pytest

from scripts.audit_generation_class_support_readout import close, independent_summary


def rows_for(predictions):
    return [dict(sample_index=i, filename=f"{i:06d}.png", requested_class=i % 5,
                 predicted_top1_class=p, predicted_top5_classes=[p] + [k for k in range(5) if k != p],
                 requested_probability=0.2, image_sha256=f"{i:064x}")
            for i, p in enumerate(predictions)]


def test_independent_ami_perfect_labels():
    summary = independent_summary(rows_for(list(range(5)) * 4), 5)
    assert summary["adjusted_mutual_information"] == pytest.approx(1)
    assert summary["top1_correct"] == 20
    assert summary["effective_class_count"] == pytest.approx(5)


def test_independent_ami_constant_prediction():
    summary = independent_summary(rows_for([0] * 20), 5)
    assert summary["adjusted_mutual_information"] == pytest.approx(0, abs=1e-12)
    assert summary["top1_accuracy"] == 0.2
    assert summary["predicted_class_count"] == 1


def test_independent_cyclic_alignment():
    summary = independent_summary(rows_for([(i + 2) % 5 for i in range(20)]), 5)
    assert summary["best_cyclic_offset"] == 2
    assert summary["top1_correct"] == 0
    assert summary["best_cyclic_accuracy"] == 1


def test_rejects_prediction_index_drift():
    rows = rows_for([0] * 20)
    rows[3]["sample_index"] = 5
    with pytest.raises(ValueError, match="contract differs"):
        independent_summary(rows, 5)


def test_independent_comparison_fails_closed():
    with pytest.raises(ValueError, match="differs"):
        close(0.001, 0.01, "AMI")
