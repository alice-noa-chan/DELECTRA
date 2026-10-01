import pytest
import torch

from deletcra.metrics import binary_ranking_metrics


@pytest.mark.parametrize(
    ("scores", "labels", "auc", "ap"),
    [
        ([0, 1, 2, 3], [0, 0, 1, 1], 1, 1),
        ([0, 1, 2, 3], [1, 1, 0, 0], 0, 5 / 12),
        ([1, 1, 1, 1], [0, 1, 0, 1], 0.5, 0.5),
        ([3, 2, 2, 1], [1, 0, 1, 0], 0.875, 5 / 6),
    ],
)
def test_ranking_metrics_handle_perfect_reversed_and_tied_scores(
    scores, labels, auc, ap
):
    result = binary_ranking_metrics(torch.tensor(scores), torch.tensor(labels))
    assert result["rtd_auroc"] == pytest.approx(auc)
    assert result["rtd_average_precision"] == pytest.approx(ap)
    # Calibration offsets change the 0.5 classifier but not ranking metrics.
    assert result == binary_ranking_metrics(
        torch.tensor(scores) - 100, torch.tensor(labels)
    )


def test_single_class_metrics_are_explicitly_undefined():
    assert binary_ranking_metrics(torch.ones(3), torch.zeros(3)) == {
        "rtd_auroc": None,
        "rtd_average_precision": None,
    }


@pytest.mark.parametrize("scores,labels", [([], []), ([1], [2]), ([float("nan")], [0])])
def test_invalid_ranking_inputs_are_rejected(scores, labels):
    with pytest.raises(ValueError):
        binary_ranking_metrics(torch.tensor(scores), torch.tensor(labels))
