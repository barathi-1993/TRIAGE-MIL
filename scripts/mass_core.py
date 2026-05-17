#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MASS v2 Core Implementation - Optimized for Speed (CPU, device-aware API)

Encoder-agnostic multi-axis stratified sampling for survival analysis.

Optimizations:
1. Farthest Point Sampling uses vectorized running-minimum updates (O(NK)).
2. Interaction Zone interpolation uses cKDTree batch querying.
3. NearestNeighbors forced to single-thread to allow process-level parallelism.

NOTE:
- The API is now "device-aware": functions accept a `device` argument
  (e.g., "cpu", "cuda:0") so they are compatible with the multi-GPU
  mass_selector_optuna_fast.py.
- Internally, computations are still CPU-based (NumPy + SciPy + sklearn),
  because NearestNeighbors and cKDTree are CPU-only. The `device` argument
  is accepted but not used to move computations to GPU yet.
"""

import numpy as np
from scipy.spatial import KDTree, cKDTree
from sklearn.neighbors import NearestNeighbors
from typing import Tuple, Optional, List
from dataclasses import dataclass


# ============================================================================
# MASSConfig and Factory Methods
# ============================================================================

@dataclass
class MASSConfig:
    """
    MASS hyperparameters with preprocessing-aware defaults.
    
    Default selection fractions (35-35-15-15) are biologically motivated:
    - 35% high-risk (tumor cores with dense cellularity)
    - 35% low-risk (stromal regions, baseline for comparison)
    - 15% interaction (tumor-stroma boundaries, invasion zones)
    - 15% diversity (morphologically unique features)
    
    These values were validated via Optuna optimization, which converged to
    similar fractions (40.6-29.0-10.1-20.3) with no significant performance
    difference (p=0.82), confirming the defaults lie within a flat optimum.
    """
    
    # ========== SELECTION FRACTIONS (DEFAULTS - USE FOR ALL DATASETS) ==========
    high_risk_fraction: float = 0.35      # 35% tumor cores
    low_risk_fraction: float = 0.35       # 35% stromal regions
    interaction_fraction: float = 0.15    # 15% tumor-stroma boundaries
    diversity_fraction: float = 0.15      # 15% morphologically diverse
    
    # ========== QUALITY FILTERING (PREPROCESSING-AWARE) ==========
    # CRITICAL: These should be set based on preprocessing method!
    # See create_ae_config() and create_clam_config() factory methods below.
    
    min_quality_threshold: float = 0.05           # Default: lenient (for AE)
    variance_percentile_low: float = 0.5          # Default: lenient
    variance_percentile_high: float = 99.5        # Default: lenient
    magnitude_percentile_low: float = 0.5         # Default: lenient
    magnitude_percentile_high: float = 99.5       # Default: lenient
    outlier_z_threshold: float = 5.5              # Default: lenient (for AE)
    entropy_percentile_low: float = 1.0           # Default: lenient
    density_sigma_threshold: float = 4.0          # Default: lenient
    spatial_radius: float = 1024.0
    min_neighbors: int = 1                        # Default: lenient (for AE)
    
    # ========== RISK SCORING WEIGHTS (OPTUNA-OPTIMIZED - USE THESE!) ==========
    # These were optimized via Optuna and show consistent improvement
    heterogeneity_weight: float = 0.235           # Variance within tile
    extremeness_weight: float = 0.349             # Distance from cohort mean
    dissimilarity_weight: float = 0.324           # Distance to neighbors
    density_weight: float = 0.092                 # Feature magnitude
    
    def __post_init__(self):
        """Validate configuration."""
        # Check selection fractions sum to 1.0
        total = (self.high_risk_fraction + 
                 self.low_risk_fraction + 
                 self.interaction_fraction + 
                 self.diversity_fraction)
        
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(
                f"Selection fractions must sum to 1.0, got {total:.6f}\n"
                f"  high_risk: {self.high_risk_fraction:.3f}\n"
                f"  low_risk: {self.low_risk_fraction:.3f}\n"
                f"  interaction: {self.interaction_fraction:.3f}\n"
                f"  diversity: {self.diversity_fraction:.3f}"
            )
        
        # Check risk scoring weights sum to ~1.0
        risk_total = (self.heterogeneity_weight + 
                      self.extremeness_weight + 
                      self.dissimilarity_weight + 
                      self.density_weight)
        
        if not np.isclose(risk_total, 1.0, atol=1e-3):
            print(f"⚠️  WARNING: Risk weights sum to {risk_total:.3f}, normalizing...")
            norm = risk_total
            self.heterogeneity_weight /= norm
            self.extremeness_weight  /= norm
            self.dissimilarity_weight /= norm
            self.density_weight      /= norm


def create_ae_config() -> MASSConfig:
    """
    Create MASS config optimized for AE-preprocessed data.
    
    AE preprocessing already removes ~80% of tiles (keeping only "informative" ones),
    so we use VERY LENIENT quality filtering to avoid over-filtering.
    
    Expected removal rate: 5-15%
    """
    return MASSConfig(
        high_risk_fraction=0.35,
        low_risk_fraction=0.35,
        interaction_fraction=0.15,
        diversity_fraction=0.15,
        
        # LENIENT quality filtering for AE
        min_quality_threshold=0.05,           # Very low (AE already filtered)
        variance_percentile_low=0.5,          # Only extreme outliers
        variance_percentile_high=99.5,
        magnitude_percentile_low=0.5,
        magnitude_percentile_high=99.5,
        outlier_z_threshold=5.5,              # Very lenient (5.5σ)
        entropy_percentile_low=1.0,           # Almost no entropy filtering
        density_sigma_threshold=4.0,          # Lenient density check
        spatial_radius=1024.0,
        min_neighbors=3,                      # Allow isolated tiles
        
        heterogeneity_weight=0.235,
        extremeness_weight=0.349,
        dissimilarity_weight=0.324,
        density_weight=0.092,
    )


def create_clam_config() -> MASSConfig:
    """
    Create MASS config optimized for CLAM-extracted data.
    
    CLAM extracts ALL tissue tiles without quality filtering,
    so we use STRICT filtering to remove artifacts, blurry tiles, etc.
    
    Expected removal rate: 20-35%
    """
    return MASSConfig(
        high_risk_fraction=0.35,
        low_risk_fraction=0.35,
        interaction_fraction=0.15,
        diversity_fraction=0.15,
        
        # STRICT quality filtering for CLAM
        min_quality_threshold=0.30,
        variance_percentile_low=2.0,
        variance_percentile_high=98.0,
        magnitude_percentile_low=2.0,
        magnitude_percentile_high=98.0,
        outlier_z_threshold=3.5,
        entropy_percentile_low=3.0,
        density_sigma_threshold=3.0,
        spatial_radius=1024.0,
        min_neighbors=3,
        
        heterogeneity_weight=0.235,
        extremeness_weight=0.349,
        dissimilarity_weight=0.324,
        density_weight=0.092,
    )


def create_optuna_config() -> MASSConfig:
    """
    Create MASS config with Optuna-optimized selection fractions.
    
    These fractions (40.6-29.0-10.1-20.3) were found via Optuna but show
    no significant improvement over defaults (C-index: 0.735 vs 0.734, p=0.82).
    
    Use this only for ablation studies to prove robustness.
    """
    return MASSConfig(
        high_risk_fraction=0.406,
        low_risk_fraction=0.290,
        interaction_fraction=0.101,
        diversity_fraction=0.203,
        
        # Lenient filtering (assuming AE preprocessing)
        min_quality_threshold=0.05,
        variance_percentile_low=0.5,
        variance_percentile_high=99.5,
        magnitude_percentile_low=0.5,
        magnitude_percentile_high=99.5,
        outlier_z_threshold=5.5,
        entropy_percentile_low=1.0,
        density_sigma_threshold=4.0,
        spatial_radius=1024.0,
        min_neighbors=1,
        
        heterogeneity_weight=0.235,
        extremeness_weight=0.349,
        dissimilarity_weight=0.324,
        density_weight=0.092,
    )


# ============================================================================
# Quality Filtering
# ============================================================================

def filter_artifacts_comprehensive(
    X: np.ndarray,
    coords: Optional[np.ndarray],
    config: MASSConfig,
    verbose: bool = True,
    is_preprocessed: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Dual-mode artifact filtering.
    
    NOTE: The is_preprocessed flag is now LESS important because config
    should be set correctly via create_ae_config() or create_clam_config().
    This flag is kept for backward compatibility and as a safety check.
    """
    N = X.shape[0]
    
    if verbose:
        print(f"[Quality Filter] Analyzing {N} tiles...")
        mode_str = "LENIENT (AE/preprocessed)" if is_preprocessed else "STANDARD (from config)"
        print(f"  Mode: {mode_str}")
        print(f"  Quality threshold: {config.min_quality_threshold:.3f}")
        print(f"  Outlier z-threshold: {config.outlier_z_threshold:.1f}")
    
    keep = np.ones(N, dtype=bool)
    quality_scores = np.ones(N, dtype=np.float32)
    
    # Use config parameters directly
    var_low_pct  = config.variance_percentile_low
    var_high_pct = config.variance_percentile_high
    mag_low_pct  = config.magnitude_percentile_low
    mag_high_pct = config.magnitude_percentile_high
    z_threshold  = config.outlier_z_threshold
    entropy_low_pct = config.entropy_percentile_low
    min_neighbors   = config.min_neighbors
    
    # Adaptive skipping
    skip_spatial = (min_neighbors <= 1) or coords is None
    skip_density = (config.density_sigma_threshold > 3.5)
    
    # ========== FILTER 1: Variance ==========
    variance = np.var(X, axis=1)
    var_low  = np.percentile(variance, var_low_pct)
    var_high = np.percentile(variance, var_high_pct)
    
    blurry_mask   = variance < var_low
    high_var_mask = variance > var_high
    keep &= ~blurry_mask & ~high_var_mask
    
    var_norm = (variance - variance.min()) / (variance.max() - variance.min() + 1e-6)
    quality_scores *= np.clip(var_norm, 0.1, 1.0)
    
    if verbose:
        print(f"  Filter 1 (Variance): Removed {blurry_mask.sum()} blurry + "
              f"{high_var_mask.sum()} high-var")
    
    # ========== FILTER 2: Magnitude ==========
    magnitude = np.linalg.norm(X, axis=1)
    mag_low   = np.percentile(magnitude, mag_low_pct)
    mag_high  = np.percentile(magnitude, mag_high_pct)
    
    background_mask = magnitude < mag_low
    penmark_mask    = magnitude > mag_high
    keep &= ~background_mask & ~penmark_mask
    
    mag_norm    = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-6)
    mag_centered = 1.0 - 2.0 * np.abs(mag_norm - 0.5)
    quality_scores *= np.clip(mag_centered, 0.1, 1.0)
    
    if verbose:
        print(f"  Filter 2 (Magnitude): Removed {background_mask.sum()} background + "
              f"{penmark_mask.sum()} pen marks")
    
    # ========== FILTER 3: Statistical Outliers ==========
    X_mean = X.mean(axis=0)
    X_std  = X.std(axis=0) + 1e-8
    Z      = (X - X_mean) / X_std
    max_z  = np.max(np.abs(Z), axis=1)
    
    outlier_mask = max_z > z_threshold
    keep &= ~outlier_mask
    
    outlier_score = np.clip(1.0 - (max_z / (z_threshold + 2.0)), 0.0, 1.0)
    quality_scores *= outlier_score
    
    if verbose:
        print(f"  Filter 3 (Outliers): Removed {outlier_mask.sum()} outliers (z>{z_threshold:.1f})")
    
    # ========== FILTER 4: Entropy ==========
    X_abs = np.abs(X)
    X_prob = X_abs / (X_abs.sum(axis=1, keepdims=True) + 1e-8)
    entropy = -np.sum(X_prob * np.log(X_prob + 1e-8), axis=1)
    
    entropy_threshold = np.percentile(entropy, entropy_low_pct)
    low_info_mask = entropy < entropy_threshold
    keep &= ~low_info_mask
    
    entropy_norm = (entropy - entropy.min()) / (entropy.max() - entropy.min() + 1e-6)
    quality_scores *= np.clip(entropy_norm, 0.2, 1.0)
    
    if verbose:
        print(f"  Filter 4 (Entropy): Removed {low_info_mask.sum()} low-info tiles")
    
    # ========== FILTER 5: Spatial Isolation ==========
    if not skip_spatial:
        tree = cKDTree(coords)
        dists, _ = tree.query(coords, k=min_neighbors+1, workers=1)
        has_neighbors = dists[:, min_neighbors] <= config.spatial_radius
        isolated_mask = ~has_neighbors
        
        nearest_dist = dists[:, 1]
        spatial_quality = np.clip(1.0 - (nearest_dist / (config.spatial_radius * 2)), 0.0, 1.0)
        
        keep &= ~isolated_mask
        quality_scores *= spatial_quality
        
        if verbose:
            print(f"  Filter 5 (Spatial): Removed {isolated_mask.sum()} isolated tiles")
    elif verbose:
        print(f"  Filter 5 (Spatial): SKIPPED")
    
    # ========== FILTER 6: Density ==========
    if not skip_density:
        density_score = np.linalg.norm(X, axis=1)
        density_mean  = density_score.mean()
        density_std   = density_score.std()
        density_mask  = np.abs(density_score - density_mean) > \
                        config.density_sigma_threshold * density_std
        keep &= ~density_mask
        
        if verbose:
            print(f"  Filter 6 (Density): Removed {density_mask.sum()} extreme density tiles")
    elif verbose:
        print(f"  Filter 6 (Density): SKIPPED")
    
    # ========== SUMMARY ==========
    removed = (~keep).sum()
    removal_rate = removed / N * 100.0
    
    if verbose:
        print(f"\n[Quality Filter] Kept {keep.sum()}/{N} tiles "
              f"(removed {removed}, {removal_rate:.1f}%)")
        
        if is_preprocessed and removal_rate > 25:
            print(f"  ⚠️  WARNING: High removal for preprocessed data ({removal_rate:.1f}%)")
            print(f"      Consider using create_ae_config() for more lenient filtering")
        elif not is_preprocessed and removal_rate < 15:
            print(f"  ⚠️  WARNING: Low removal for CLAM data ({removal_rate:.1f}%)")
            print(f"      Consider using create_clam_config() for stricter filtering")
    
    return keep, quality_scores


# ============================================================================
# Risk Scoring (device-aware API, CPU backend)
# ============================================================================

def compute_surrogate_risk_score(
    X: np.ndarray,
    quality_scores: np.ndarray,
    config: MASSConfig,
    device: str = "cpu"
) -> np.ndarray:
    """
    Compute surrogate risk scores (quality-adjusted).

    Parameters
    ----------
    X : (N, D) np.ndarray
    quality_scores : (N,) np.ndarray
    config : MASSConfig
    device : str
        Included for API compatibility with multi-GPU caller.
        Currently computations remain on CPU (NumPy + sklearn).
    """
    N = X.shape[0]
    
    # 1. Heterogeneity
    tile_variance = np.var(X, axis=1)
    heterogeneity_score = (tile_variance - tile_variance.min()) / \
                          (tile_variance.max() - tile_variance.min() + 1e-6)
    
    # 2. Extremeness
    X_mean = X.mean(axis=0)
    extremeness = np.linalg.norm(X - X_mean, axis=1)
    extremeness_score = (extremeness - extremeness.min()) / \
                        (extremeness.max() - extremeness.min() + 1e-6)
    
    # 3. Dissimilarity
    nn = NearestNeighbors(n_neighbors=min(10, N), n_jobs=1)
    nn.fit(X)
    dists, _ = nn.kneighbors(X)
    dissimilarity = dists[:, 1:].mean(axis=1)
    dissimilarity_score = (dissimilarity - dissimilarity.min()) / \
                          (dissimilarity.max() - dissimilarity.min() + 1e-6)
    
    # 4. Density
    density = np.linalg.norm(X, axis=1)
    density_score = (density - density.min()) / \
                    (density.max() - density.min() + 1e-6)
    
    # Combine with Optuna-optimized weights
    raw_risk = (
        config.heterogeneity_weight * heterogeneity_score +
        config.extremeness_weight  * extremeness_score +
        config.dissimilarity_weight * dissimilarity_score +
        config.density_weight      * density_score
    )
    
    # Quality adjustment
    quality_adjusted_risk = raw_risk * quality_scores
    quality_adjusted_risk = (quality_adjusted_risk - quality_adjusted_risk.min()) / \
                            (quality_adjusted_risk.max() - quality_adjusted_risk.min() + 1e-6)
    
    return quality_adjusted_risk.astype(np.float32)


# ============================================================================
# Interaction Zones
# ============================================================================

def identify_interaction_zones(
    X: np.ndarray,
    coords: np.ndarray,
    surrogate_risk: np.ndarray,
    spatial_radius: float = 768.0,
    max_tiles_to_process: int = 5000,
    verbose: bool = True
) -> np.ndarray:
    """
    Detect tumor-stroma interaction zones.
    Uses cKDTree for batched interpolation.
    """
    N = len(coords)
    
    if verbose:
        print(f"  [Interaction] Processing {N} tiles (radius={spatial_radius:.0f})...")
    
    tree = cKDTree(coords)
    interaction_score = np.zeros(N, dtype=np.float32)
    
    # Sample tiles if too many (for speed)
    if N > max_tiles_to_process:
        if verbose:
            print(f"  [Interaction] Sampling {max_tiles_to_process} tiles for efficiency...")
        sample_idx = np.random.choice(N, max_tiles_to_process, replace=False)
        process_idx = sample_idx
    else:
        sample_idx = None
        process_idx = np.arange(N)
    
    # Score calculation
    report_interval = max(1, len(process_idx) // 5)
    
    for count, i in enumerate(process_idx):
        if verbose and count % report_interval == 0:
            print(f"    Score Calc: {count}/{len(process_idx)}", end='\r')
        
        neighbor_idx = tree.query_ball_point(coords[i], r=spatial_radius)
        if len(neighbor_idx) < 3:
            continue
        
        neighbor_risks = surrogate_risk[neighbor_idx]
        interaction_score[i] = np.std(neighbor_risks)
    
    # Interpolate remaining if sampled
    if sample_idx is not None:
        if verbose:
            print(f"\n  [Interaction] Interpolating remaining {N - len(sample_idx)} tiles...")
        
        sampled_scores = interaction_score[sample_idx]
        
        mask = np.ones(N, dtype=bool)
        mask[sample_idx] = False
        remaining_idx = np.where(mask)[0]
        
        if len(remaining_idx) > 0:
            sampled_coords = coords[sample_idx]
            sample_tree = cKDTree(sampled_coords)
            
            dists, nn_idx = sample_tree.query(coords[remaining_idx], k=3, workers=1)
            
            weights = 1.0 / (dists + 1e-6)
            weights /= weights.sum(axis=1, keepdims=True)
            
            interpolated = (weights * sampled_scores[nn_idx]).sum(axis=1)
            interaction_score[remaining_idx] = interpolated
    
    # Normalize
    if interaction_score.max() > 0:
        interaction_score = (interaction_score - interaction_score.min()) / \
                            (interaction_score.max() - interaction_score.min() + 1e-6)
    
    if verbose:
        print(f"\n  [Interaction] Done! Score range: "
              f"[{interaction_score.min():.3f}, {interaction_score.max():.3f}]")
    
    return interaction_score


# ============================================================================
# Diversity Selection (Farthest Point Sampling)
# ============================================================================

def greedy_feature_diversity(
    X: np.ndarray,
    K: int,
    verbose: bool = False
) -> np.ndarray:
    """
    Optimized Farthest Point Sampling (FPS).
    Maintains a running 'min_dist' array to reduce complexity to O(NK).
    """
    N = X.shape[0]
    if K >= N:
        return np.arange(N, dtype=np.int32)
    
    if verbose:
        print(f"  [Diversity] Selecting {K} diverse tiles from {N} candidates...")
    
    selected_indices = np.zeros(K, dtype=np.int32)
    
    # Start with point farthest from centroid
    center = X.mean(axis=0)
    first_idx = np.argmax(np.linalg.norm(X - center, axis=1))
    selected_indices[0] = first_idx
    
    # Initialize min distances to first point
    min_dists = np.linalg.norm(X - X[first_idx], axis=1)
    
    report_interval = max(1, K // 10)
    
    for k in range(1, K):
        if verbose and k % report_interval == 0:
            print(f"    Progress: {k}/{K} ({k / K * 100:.0f}%)", end='\r')
        
        new_idx = np.argmax(min_dists)
        selected_indices[k] = new_idx
        
        new_point = X[new_idx]
        dist_new = np.linalg.norm(X - new_point, axis=1)
        min_dists = np.minimum(min_dists, dist_new)
    
    if verbose:
        print(f"\n  [Diversity] Selected {K} diverse tiles")
    
    return selected_indices


# ============================================================================
# Main MASS v2 Pipeline
# ============================================================================

def mass_tile_selection_v2(
    X: np.ndarray,
    coords: Optional[np.ndarray],
    tile_ids: List[str],
    K: int,
    config: MASSConfig,
    verbose: bool = True,
    is_preprocessed: bool = False,
    device: str = "cpu"
) -> Tuple[np.ndarray, List[str], dict]:
    """
    Complete MASS v2 pipeline.
    """
    N = X.shape[0]
    
    if K >= N:
        selected = np.arange(N, dtype=np.int32)
        selected_ids = [tile_ids[i] for i in selected]
        metadata = {'note': 'K >= N, selected all tiles'}
        return selected, selected_ids, metadata
    
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"MASS TILE SELECTION v2 (K={K})")
        print(f"{'=' * 70}")
    
    # Stage 0: Quality filtering
    quality_mask, quality_scores = filter_artifacts_comprehensive(
        X, coords, config, verbose=verbose,
        is_preprocessed=is_preprocessed
    )
    
    # Adaptive quality threshold
    if is_preprocessed:
        base_threshold = 0.005 # Very lenient for AE
    else:
        base_threshold = config.min_quality_threshold  # Stricter for CLAM
    
    quality_mask &= (quality_scores >= base_threshold)
    
    # Progressive threshold lowering if needed
    attempts = 0
    while quality_mask.sum() < K and attempts < 3:
        base_threshold *= 0.6
        quality_mask = quality_scores >= base_threshold
        attempts += 1
        
        if verbose:
            print(f"  -> Lowered threshold to {base_threshold:.3f} "
                  f"({quality_mask.sum()} tiles)")
    
    if quality_mask.sum() < K:
        if verbose:
            print(f"\n⚠️  Only {quality_mask.sum()} quality tiles (requested {K})")
            print(f"    Using all available tiles")
        quality_mask = np.ones(N, dtype=bool)
    
    clean_idx = np.where(quality_mask)[0]
    X_clean = X[clean_idx]
    coords_clean = coords[clean_idx] if coords is not None else None
    quality_clean = quality_scores[clean_idx]
    
    if verbose:
        print(f"\n[Clean Dataset] {len(clean_idx)} high-quality tiles")
    
    # Stage 1: Risk scoring
    if verbose:
        print(f"\n[Stage 1] Computing surrogate risk scores...")
    
    surrogate_risk = compute_surrogate_risk_score(
        X_clean, quality_clean, config, device=device
    )
    
    # Stage 2: Selection
    if verbose:
        print(f"\n[Stage 2] Selecting diverse tiles...")
    
    K_high        = int(K * config.high_risk_fraction)
    K_low         = int(K * config.low_risk_fraction)
    K_interaction = int(K * config.interaction_fraction)
    
    # High-risk tiles
    high_risk_rank  = np.argsort(-surrogate_risk)
    high_risk_local = high_risk_rank[:min(K_high, len(clean_idx))]
    
    # Low-risk tiles
    low_risk_rank  = np.argsort(surrogate_risk)
    low_risk_local = low_risk_rank[:min(K_low, len(clean_idx))]
    
    if verbose:
        print(f"  High-risk: {len(high_risk_local)} "
              f"(mean={surrogate_risk[high_risk_local].mean():.3f})")
        print(f"  Low-risk: {len(low_risk_local)} "
              f"(mean={surrogate_risk[low_risk_local].mean():.3f})")
    
    # Interaction zones
    if coords_clean is not None and len(clean_idx) >= K_interaction and K_interaction > 0:
        interaction_scores = identify_interaction_zones(
            X_clean, coords_clean, surrogate_risk,
            spatial_radius=config.spatial_radius,
            verbose=verbose
        )
        
        used = set(high_risk_local) | set(low_risk_local)
        remaining_local = np.array([i for i in range(len(clean_idx)) if i not in used])
        
        if len(remaining_local) > 0:
            interaction_rank = np.argsort(-interaction_scores[remaining_local])
            interaction_local = remaining_local[
                interaction_rank[:min(K_interaction, len(remaining_local))]
            ]
        else:
            interaction_local = np.array([], dtype=np.int32)
        
        if verbose:
            print(f"  Interaction: {len(interaction_local)}")
    else:
        interaction_local = np.array([], dtype=np.int32)
    
    # Diversity tiles
    K_diversity = K - len(high_risk_local) - len(low_risk_local) - len(interaction_local)
    
    used = set(high_risk_local) | set(low_risk_local) | set(interaction_local)
    remaining_local = np.array([i for i in range(len(clean_idx)) if i not in used])
    
    if len(remaining_local) > 0 and K_diversity > 0:
        # Select from mid-risk range
        mid_mask = (surrogate_risk[remaining_local] > 0.3) & \
                   (surrogate_risk[remaining_local] < 0.7)
        mid_candidates = remaining_local[mid_mask]
        
        MAX_DIVERSITY_CANDIDATES = 6000
        if len(mid_candidates) > MAX_DIVERSITY_CANDIDATES:
            if verbose:
                print(f"  [Diversity] Pre-sampling {len(mid_candidates)} → "
                      f"{MAX_DIVERSITY_CANDIDATES} for speed")
            mid_candidates = np.random.choice(
                mid_candidates, MAX_DIVERSITY_CANDIDATES, replace=False
            )
        
        if len(mid_candidates) > 0:
            diversity_subset = greedy_feature_diversity(
                X_clean[mid_candidates],
                min(K_diversity, len(mid_candidates)),
                verbose=verbose
            )
            diversity_local = mid_candidates[diversity_subset]
        else:
            diversity_local = remaining_local[:K_diversity]
        
        if verbose:
            print(f"  Diversity: {len(diversity_local)}")
    else:
        diversity_local = np.array([], dtype=np.int32)
    
    # Combine all categories
    all_selected_local = np.concatenate([
        high_risk_local,
        low_risk_local,
        interaction_local,
        diversity_local
    ])[:K]
    
    # Map back to original indices
    selected_original = clean_idx[all_selected_local]
    selected_ids = [tile_ids[i] for i in selected_original]
    
    category_labels = {}
    
    for local_idx in high_risk_local:
        orig_idx = clean_idx[local_idx]
        category_labels[int(orig_idx)] = 'high_risk'
    
    for local_idx in low_risk_local:
        orig_idx = clean_idx[local_idx]
        category_labels[int(orig_idx)] = 'low_risk'
    
    for local_idx in interaction_local:
        orig_idx = clean_idx[local_idx]
        category_labels[int(orig_idx)] = 'interaction'
    
    for local_idx in diversity_local:
        orig_idx = clean_idx[local_idx]
        category_labels[int(orig_idx)] = 'diversity'
    
    metadata = {
        'total_tiles': N,
        'artifacts_removed': N - len(clean_idx),
        'selected': len(selected_original),
        'high_risk': len(high_risk_local),
        'low_risk': len(low_risk_local),
        'interaction': len(interaction_local),
        'diversity': len(diversity_local),
        'mean_quality': float(quality_scores[selected_original].mean()),
        'mean_risk': float(surrogate_risk[all_selected_local].mean()),
        'category_labels': category_labels
    }
    
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"[FINAL] Selected {len(selected_original)}/{N} tiles")
        print(f"  Artifacts removed: {metadata['artifacts_removed']}")
        print(f"  High-risk: {metadata['high_risk']} "
              f"({metadata['high_risk'] / len(selected_original) * 100:.1f}%)")
        print(f"  Low-risk: {metadata['low_risk']} "
              f"({metadata['low_risk'] / len(selected_original) * 100:.1f}%)")
        print(f"  Interaction: {metadata['interaction']} "
              f"({metadata['interaction'] / len(selected_original) * 100:.1f}%)")
        print(f"  Diversity: {metadata['diversity']} "
              f"({metadata['diversity'] / len(selected_original) * 100:.1f}%)")
        print(f"  Quality score: {metadata['mean_quality']:.3f}")
        print(f"  Risk score: {metadata['mean_risk']:.3f}")
        print(f"{'=' * 70}\n")
    
    return selected_original, selected_ids, metadata
