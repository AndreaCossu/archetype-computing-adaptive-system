"""
Performance metrics for model evaluation.
"""

import numpy as np


def nrmse(preds: np.ndarray, target: np.ndarray) -> float:
    """Compute Normalized Root Mean Squared Error.

    Args:
        preds: Predictions array.
        target: Ground truth array (must match preds shape).

    Returns:
        NRMSE value (lower is better).
    """
    preds = preds.flatten()
    target = target.flatten()
    assert preds.shape == target.shape, "Predictions and target must have the same shape"
    mse = np.mean(np.square(preds - target))
    norm = np.sqrt(np.mean(np.square(target)))
    return np.sqrt(mse) / (norm + 1e-9)

def accuracy(preds: np.ndarray, target: np.ndarray) -> float:
    """Compute accuracy for classification tasks.

    Args:
        preds: Predictions array (class probabilities or logits).
        target: Ground truth array (class labels).

    Returns:
        Accuracy value (between 0 and 1).
    """
    assert preds.shape[0] == target.shape[0], "Number of predictions and targets must match"
    pred_labels = np.argmax(preds, axis=1)
    correct = np.sum(pred_labels == target)
    return correct / target.shape[0]


__all__ = ["nrmse", "accuracy"]
