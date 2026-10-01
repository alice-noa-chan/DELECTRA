"""Binary ranking metrics with exact score ties and explicit undefined cases."""

import torch
from torch import Tensor


def binary_ranking_metrics(scores: Tensor, labels: Tensor) -> dict:
    scores = scores.detach().cpu().double().flatten()
    labels = labels.detach().cpu().flatten()
    if scores.numel() == 0 or scores.shape != labels.shape:
        raise ValueError("scores and labels must have matching nonempty shapes")
    if not torch.isfinite(scores).all() or not ((labels == 0) | (labels == 1)).all():
        raise ValueError("scores must be finite and labels binary")
    labels = labels.bool()
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if not positives or not negatives:
        return {"rtd_auroc": None, "rtd_average_precision": None}
    order = scores.argsort(descending=True, stable=True)
    sorted_scores, sorted_labels = scores[order], labels[order]
    ends = torch.cat(
        (
            torch.where(sorted_scores[:-1] != sorted_scores[1:])[0],
            torch.tensor([len(scores) - 1]),
        )
    )
    tp = sorted_labels.double().cumsum(0)[ends]
    fp = ends.double() + 1 - tp
    recall = tp / positives
    precision = tp / (tp + fp)
    previous_recall = torch.cat((torch.zeros(1), recall[:-1]))
    ap = ((recall - previous_recall) * precision).sum()
    tpr = torch.cat((torch.zeros(1), recall))
    fpr = torch.cat((torch.zeros(1), fp / negatives))
    auc = torch.trapezoid(tpr, fpr)
    return {"rtd_auroc": auc.item(), "rtd_average_precision": ap.item()}
