"""Evaluation metric helpers shared across pipeline steps."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def top1_accuracy(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """Compute top-1 accuracy from probability matrix."""
    return float(accuracy_score(y_true, np.argmax(y_pred_proba, axis=1)))


def per_class_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None = None,
) -> dict:
    """Return sklearn classification report as a dictionary."""
    return classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )


def confusion_mat(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Thin wrapper around sklearn confusion_matrix."""
    return confusion_matrix(y_true, y_pred)
