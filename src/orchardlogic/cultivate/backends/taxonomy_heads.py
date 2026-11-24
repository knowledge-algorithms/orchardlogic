"""ModernBERT+logistic taxonomy heads loaded from portable ``npz`` files.

No heads are packaged. ``load_head`` / ``save_head`` read and write
``coef_``, ``intercept_`` and ``classes_``; load remaps ``predict_proba`` columns
onto the taxonomy's ``label_order``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from orchardlogic.cultivate.backends.modernbert import (
    FEATURE_CONFIG,
    MODERNBERT_MODEL_ID,
    MODERNBERT_REVISION,
)
from orchardlogic.cultivate.exceptions import InvalidIdentityError

HEAD_SCHEMA_VERSION = "orchard_taxonomy_head_v1"
HEAD_HYPERPARAMS = {
    "C": 0.1,
    "class_weight": "balanced",
    "max_iter": 1000,
    "random_state": 20260725,
}
SOURCE_STUDENT_SHA256 = {
    "domain": "b927c99282099f322396b5671e58d2796d2f6c23ff8e1241f2bd6fe609b4021b",
    "function": "13fa57af1f0e18d568c36091a465499baa6f6e661dce877fad714c4b741d0114",
}
def remap_head_to_label_order(
    coef: np.ndarray,
    intercept: np.ndarray,
    classes: Sequence[str],
    label_order: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Permute student-order arrays onto Orchard ``label_order`` (same set)."""
    stored = [str(label) for label in classes]
    ordered = tuple(label_order)
    if set(stored) != set(ordered):
        raise InvalidIdentityError(
            "head class set must match packaged label_order exactly"
        )
    if len(stored) != len(ordered):
        raise InvalidIdentityError("head classes must be unique and match label_order")
    index = [stored.index(label) for label in ordered]
    return (
        np.asarray(coef, dtype=np.float64)[index],
        np.asarray(intercept, dtype=np.float64)[index],
        np.asarray(ordered),
    )


def rebuild_logistic_regression(
    coef: np.ndarray,
    intercept: np.ndarray,
    classes: Sequence[str],
) -> LogisticRegression:
    """Rebuild the Phase 2B student estimator from arrays (do not re-train)."""
    estimator = LogisticRegression(
        C=float(HEAD_HYPERPARAMS["C"]),
        class_weight=str(HEAD_HYPERPARAMS["class_weight"]),
        max_iter=int(HEAD_HYPERPARAMS["max_iter"]),
        random_state=int(HEAD_HYPERPARAMS["random_state"]),
    )
    estimator.classes_ = np.asarray(classes)
    estimator.coef_ = np.asarray(coef, dtype=np.float64)
    estimator.intercept_ = np.asarray(intercept, dtype=np.float64)
    estimator.n_features_in_ = int(estimator.coef_.shape[1])
    return estimator


def load_head(
    path: str | Path,
    label_order: Sequence[str],
) -> LogisticRegression:
    """Load a replacement head from a portable ``npz`` and remap to ``label_order``."""
    target = Path(path)
    with np.load(target, allow_pickle=False) as archive:
        coef = np.asarray(archive["coef_"], dtype=np.float64)
        intercept = np.asarray(archive["intercept_"], dtype=np.float64)
        classes = np.asarray(archive["classes_"]).astype(str).tolist()
    remapped_coef, remapped_intercept, remapped_classes = remap_head_to_label_order(
        coef, intercept, classes, label_order
    )
    return rebuild_logistic_regression(
        remapped_coef, remapped_intercept, remapped_classes.astype(str).tolist()
    )


def save_head(
    path: str | Path,
    coef: np.ndarray,
    intercept: np.ndarray,
    classes: Sequence[str],
) -> Path:
    """Write a portable replacement head (``coef_``, ``intercept_``, ``classes_``)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        coef_=np.asarray(coef, dtype=np.float64),
        intercept_=np.asarray(intercept, dtype=np.float64),
        classes_=np.asarray([str(label) for label in classes]),
    )
    return target


def predict_proba_rows(
    classifier: LogisticRegression,
    features: np.ndarray,
    label_order: Sequence[str],
) -> np.ndarray:
    """Full probability rows over ``label_order`` (not argmax-only)."""
    probabilities = np.asarray(classifier.predict_proba(features), dtype=np.float64)
    class_index = {
        str(label): index for index, label in enumerate(classifier.classes_.tolist())
    }
    if set(class_index) != set(label_order):
        raise InvalidIdentityError("classifier class set must match label_order")
    columns = [class_index[label] for label in label_order]
    return probabilities[:, columns]


def feature_model_provenance() -> dict[str, Any]:
    return {
        "id": MODERNBERT_MODEL_ID,
        "revision": MODERNBERT_REVISION,
        "pooling": FEATURE_CONFIG["pooling"],
        "max_length": FEATURE_CONFIG["max_length"],
        "dimensions": FEATURE_CONFIG["dimensions"],
        "batch_size": FEATURE_CONFIG["batch_size"],
    }
