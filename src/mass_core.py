#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MASS core utilities for TRIAGE-MIL.

Quality-aware filtering:
- magnitude percentile filtering: P2/P98
- statistical outlier filtering: max absolute z-score <= 3.5
- feature entropy filtering: entropy >= P3
- composite quality score Q(x_j) >= 0.30

MASS axes:
- MH: morphologic heterogeneity via feature variance
- TR: tissue regularity via feature magnitude / variance
- MI: microenvironmental interface via local variation in MH among spatial neighbors
- TD: tissue diversity via farthest-point sampling in feature space
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors


@dataclass
class MASSConfig:
    """Manuscript-aligned MASS configuration for CLAM + UNI features."""

    # Selection budget fractions. These defaults are stable release defaults.
    # They can be overridden by config or Optuna-selected fractions.
    mh_fraction: float = 0.35
    tr_fraction: float = 0.35
    mi_fraction: float = 0.15
    td_fraction: float = 0.15

    # Quality-aware filtering thresholds from the manuscript.
    magnitude_percentile_low: float = 2.0
    magnitude_percentile_high: float = 98.0
    outlier_z_threshold: float = 3.5
    entropy_percentile_low: float = 3.0
    min_quality_threshold: float = 0.30

    # MASS spatial/local neighborhood for MI.
    mi_neighbors: int = 17

    # Safety.
    eps: float = 1e-8

    def __post_init__(self):
        total = self.mh_fraction + self.tr_fraction + self.mi_fraction + self.td_fraction
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(
                "MASS fractions must sum to 1.0. "
                f"Got MH+TR+MI+TD={total:.6f}."
            )


def _normalize01(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Min-max normalize a vector to [0, 1]."""
    x = np.asarray(x, dtype=np.float32)
    return (x - x.min()) / (x.max() - x.min() + eps)


def compute_feature_entropy(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Compute entropy over absolute feature activations per tile."""
    X_abs = np.abs(X).astype(np.float32)
    probs = X_abs / (X_abs.sum(axis=1, keepdims=True) + eps)
    entropy = -np.sum(probs * np.log(probs + eps), axis=1)
    return entropy.astype(np.float32)


def quality_filter_embeddings(
    X: np.ndarray,
    config: MASSConfig,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Manuscript-aligned feature-space quality filtering.

    Returns
    -------
    keep_mask : np.ndarray
        Boolean mask of retained tiles.
    quality_scores : np.ndarray
        Composite quality scores in [0, 1].
    metadata : dict
        Filtering statistics.
    """
    if X.ndim != 2:
        raise ValueError(f"Expected X with shape [N, D], got {X.shape}")

    N = X.shape[0]
    eps = config.eps

    magnitude = np.linalg.norm(X, axis=1)
    mag_low = np.percentile(magnitude, config.magnitude_percentile_low)
    mag_high = np.percentile(magnitude, config.magnitude_percentile_high)
    mag_keep = (magnitude >= mag_low) & (magnitude <= mag_high)

    mu = X.mean(axis=0)
    sigma = X.std(axis=0) + eps
    max_abs_z = np.max(np.abs((X - mu) / sigma), axis=1)
    outlier_keep = max_abs_z <= config.outlier_z_threshold

    entropy = compute_feature_entropy(X, eps=eps)
    entropy_thr = np.percentile(entropy, config.entropy_percentile_low)
    entropy_keep = entropy >= entropy_thr

    # Composite Q(x_j): high score = non-extreme magnitude, non-outlier, high entropy.
    mag_centered = 1.0 - np.abs(_normalize01(magnitude, eps) - 0.5) * 2.0
    z_score = 1.0 - np.clip(max_abs_z / (config.outlier_z_threshold + 1.0), 0.0, 1.0)
    entropy_score = _normalize01(entropy, eps)

    quality_scores = (
        0.34 * np.clip(mag_centered, 0.0, 1.0)
        + 0.33 * np.clip(z_score, 0.0, 1.0)
        + 0.33 * np.clip(entropy_score, 0.0, 1.0)
    ).astype(np.float32)

    quality_keep = quality_scores >= config.min_quality_threshold
    keep_mask = mag_keep & outlier_keep & entropy_keep & quality_keep

    # Hard safety: avoid returning an empty slide; keep top-quality tiles if needed.
    if keep_mask.sum() == 0:
        n_keep = min(max(1, int(0.05 * N)), N)
        top_idx = np.argsort(-quality_scores)[:n_keep]
        keep_mask[top_idx] = True

    metadata = {
        "total_tiles": int(N),
        "kept_tiles": int(keep_mask.sum()),
        "removed_tiles": int((~keep_mask).sum()),
        "magnitude_low": float(mag_low),
        "magnitude_high": float(mag_high),
        "entropy_threshold": float(entropy_thr),
        "outlier_z_threshold": float(config.outlier_z_threshold),
        "quality_threshold": float(config.min_quality_threshold),
    }

    if verbose:
        print(
            f"[MASS Quality] Kept {metadata['kept_tiles']}/{metadata['total_tiles']} "
            f"tiles ({metadata['kept_tiles'] / max(1, N) * 100:.1f}%)."
        )

    return keep_mask, quality_scores, metadata


# Backward-compatible name used by older selector scripts.
def filter_artifacts_comprehensive(
    X: np.ndarray,
    coords: Optional[np.ndarray],
    config: MASSConfig,
    verbose: bool = True,
    is_preprocessed: bool = False,
):
    """
    Backward-compatible wrapper around manuscript-aligned quality filtering.

    The coords and is_preprocessed arguments are intentionally ignored because
    the manuscript filtering is feature-space based.
    """
    keep_mask, quality_scores, _ = quality_filter_embeddings(X, config, verbose=verbose)
    return keep_mask, quality_scores


def compute_mass_axis_scores(
    X: np.ndarray,
    coords: Optional[np.ndarray],
    config: MASSConfig,
) -> dict:
    """
    Compute manuscript-defined MASS axis scores.

    MH = Var(x_j)
    TR = ||x_j||_2 / (Var(x_j) + eps)
    MI = local std of MH among spatial neighbors
    TD is selected by farthest-point sampling, so no scalar TD score is needed.
    """
    eps = config.eps

    mh = np.var(X, axis=1).astype(np.float32)
    magnitude = np.linalg.norm(X, axis=1).astype(np.float32)
    tr = (magnitude / (mh + eps)).astype(np.float32)

    if coords is not None and len(coords) == len(X):
        k = min(config.mi_neighbors, len(X))
        nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto")
        nbrs.fit(coords.astype(np.float32))
        _, idx = nbrs.kneighbors(coords.astype(np.float32))
        mi = np.std(mh[idx], axis=1).astype(np.float32)
    else:
        # If coordinates are unavailable, fall back to feature-space neighbors.
        k = min(config.mi_neighbors, len(X))
        nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto")
        nbrs.fit(X.astype(np.float32))
        _, idx = nbrs.kneighbors(X.astype(np.float32))
        mi = np.std(mh[idx], axis=1).astype(np.float32)

    return {"MH": mh, "TR": tr, "MI": mi}


def farthest_point_sampling(
    X: np.ndarray,
    candidate_indices: np.ndarray,
    n_select: int,
    start_scores: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Feature-space farthest-point sampling for TD axis.

    Parameters
    ----------
    X : np.ndarray
        Full feature matrix [N, D].
    candidate_indices : np.ndarray
        Candidate tile indices.
    n_select : int
        Number of tiles to select.
    start_scores : Optional[np.ndarray]
        Optional score vector used to choose the first point.
    """
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    if n_select <= 0 or len(candidate_indices) == 0:
        return np.array([], dtype=np.int64)

    n_select = min(n_select, len(candidate_indices))
    Xc = X[candidate_indices].astype(np.float32)

    if start_scores is not None:
        first_local = int(np.argmax(start_scores[candidate_indices]))
    else:
        first_local = 0

    selected_local = [first_local]
    min_dist = np.linalg.norm(Xc - Xc[first_local][None, :], axis=1)

    for _ in range(1, n_select):
        next_local = int(np.argmax(min_dist))
        if next_local in selected_local:
            break
        selected_local.append(next_local)
        dist = np.linalg.norm(Xc - Xc[next_local][None, :], axis=1)
        min_dist = np.minimum(min_dist, dist)

    return candidate_indices[np.asarray(selected_local, dtype=np.int64)]
