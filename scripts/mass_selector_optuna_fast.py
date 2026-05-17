#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MASS Selector (MASS Edition) - GPU Accelerated + Optuna [FIXED VERSION]
======================================================
1. Uses Multi-Axis Stratified Sampling (MASS) for dataset-agnostic selection.
2. Fully GPU-accelerated (10-50x faster).
3. Optuna tunes the FRACTIONS (Complexity vs Structure vs Interaction) instead of weights.
4. Auto-CPU Fallback for OOM cases.
5. **FIXED: locals() bug in min_axis_frac logic**
6. **FIXED: Better error reporting for debugging**
7. **FIXED: Strict Artifact Filtering (No fallback to artifacts if K < TopK)**

- MASS v2:
    * Axis 1: MH    (variance-based)
    * Axis 2: TR     (magnitude / variance)
    * Axis 3: MI   (neighborhood variance)
    * Axis 4: TD     (k-center / farthest-point sampling in feature space)

- Optuna tunes axis fractions (MH / TR / MI); Diversity gets remainder.
- min-axis-frac enforces a minimum coverage per major axis (0–2), stealing from Diversity.

CLAM-style
--------------
python mass_selector_optuna_fast.py \
  --feature-root /path_to/Extracted_features/UNI/pt_files \
  --encoder uni \
  --out-cache-best /path_to/Extracted_features/MASS \
  --format-type clam \
  --gpus "0,1,2,3,4,5,6,7" \
  --n-workers 7 --topk 4096 --min-axis-frac 0.05 --diversity-mode kcenter --save-artifacts --tile-dir /data_64T_3/Shared/Barathi/Extracted_tiles --skip-optuna --only-pid TCGA-CM-4744-01Z-00-DX1.527ead53-bd55-4321-adea-079bf5e2e8a5

python mass_selector_optuna_fast.py \
  --feature-root /path_to/Extracted_features/UNI/pt_files \
  --encoder uni \
  --out-cache-best /path_to/Extracted_features/MASS \
  --format-type clam \
  --gpus "2,3,4,5,6,7" \
  --n-workers 5 \
  --topk 4096 \
  --min-axis-frac 0.05 \
  --diversity-mode kcenter \
  --save-artifacts \
  --tile-dir /path_to/Extracted_tiles \
  --skip-optuna \
  --only-pid TCGA-CM-4744-01Z-00-DX1.527ead53-bd55-4321-adea-079bf5e2e8a5
"""

import os
# CRITICAL: Prevent deadlock between multiprocessing and numpy/sklearn
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import multiprocessing
# CRITICAL: Set spawn method BEFORE importing torch to avoid CUDA fork issues
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

import numpy as np
import torch
import json
import argparse
import time
import tqdm  # module import
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import traceback
import shutil
import optuna
from optuna.samplers import TPESampler

# Import loaders
from mass_io_utils import (
    load_patient_features,
    discover_patients
)
from mass_core import MASSConfig, filter_artifacts_comprehensive

# ============================================================================
# OPTIONAL: GPU K-MEANS (kept for potential use, not default for diversity)
# ============================================================================

def gpu_kmeans(X, K, n_iters=10):
    """Simple GPU K-Means implementation (Robust to empty clusters)."""
    N, D = X.shape
    device = X.device

    # Initialize centroids randomly
    indices = torch.randperm(N, device=device)[:K]
    centroids = X[indices]

    for _ in range(n_iters):
        # Assign clusters: ||x - c||^2
        dists = torch.cdist(X, centroids)
        labels = torch.argmin(dists, dim=1)

        # Update centroids
        new_centroids = []
        for k in range(K):
            mask = labels == k
            if mask.sum() > 0:
                new_centroids.append(X[mask].mean(dim=0))
            else:
                # Re-init empty cluster: Fix dimensions for stack
                random_idx = torch.randint(0, N, (1,), device=device)
                new_centroids.append(X[random_idx].squeeze(0))
        centroids = torch.stack(new_centroids)

    # Find closest samples to final centroids
    dists = torch.cdist(X, centroids)
    closest_indices = torch.argmin(dists, dim=0)  # [K] indices
    return closest_indices.unique()

# ============================================================================
# MASS v2: GPU-ACCELERATED SELECTION + SEMANTIC LABELS + COVERAGE
# ============================================================================

def universal_mass_selection_gpu_with_labels(
    X_tensor: torch.Tensor,
    coords_tensor: Optional[torch.Tensor],
    K: int,
    config: MASSConfig,
    device: torch.device,
    min_axis_frac: float = 0.0,
    diversity_mode: str = "kcenter",
    frac_complex: float = None,
    frac_struct: float = None,
    frac_inter: float = None,
):
    """
    MASS v2: GPU-Optimized Selection + Semantic Labels.
    """
    N = X_tensor.shape[0]
    if N == 0 or K <= 0:
        return (
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.uint8),
            {
                "total_tiles": int(N),
                "selected": 0,
                "complexity": 0,
                "structure": 0,
                "interaction": 0,
                "diversity": 0,
                "mean_risk": 0.0,
                "K_1": 0,
                "K_2": 0,
                "K_3": 0,
                "K_4": 0,
                "frac_complex": 0.0,
                "frac_struct": 0.0,
                "frac_inter": 0.0,
                "frac_div": 0.0,
                "min_axis_frac": float(min_axis_frac),
                "diversity_mode": diversity_mode,
            },
        )

    # move to device
    X = X_tensor.to(device)
    coords = coords_tensor.to(device) if coords_tensor is not None else None

    # ----------------------------------------------------------------------
    # 1) Determine axis fractions (Optuna overrides or MASSConfig defaults)
    # ----------------------------------------------------------------------
    if frac_complex is None:
        frac_complex = getattr(config, "high_risk_fraction", 0.32)
    if frac_struct is None:
        frac_struct = getattr(config, "low_risk_fraction", 0.18)
    if frac_inter is None:
        frac_inter = getattr(config, "interaction_fraction", 0.27)

    # Diversity gets remainder
    frac_div = 1.0 - (frac_complex + frac_struct + frac_inter)
    if hasattr(config, "diversity_fraction"):
        if (
            frac_complex == getattr(config, "high_risk_fraction", 0.32)
            and frac_struct == getattr(config, "low_risk_fraction", 0.18)
            and frac_inter == getattr(config, "interaction_fraction", 0.27)
        ):
            frac_div = getattr(config, "diversity_fraction")

    # Renormalize if fractions not valid
    total_frac = frac_complex + frac_struct + frac_inter + max(frac_div, 0.0)
    if total_frac <= 0:
        frac_complex = frac_struct = frac_inter = frac_div = 0.25
        total_frac = 1.0

    frac_complex /= total_frac
    frac_struct /= total_frac
    frac_inter /= total_frac
    frac_div = max(1.0 - (frac_complex + frac_struct + frac_inter), 0.0)

    # Raw budgets
    K_1 = int(round(K * frac_complex))  # Complexity
    K_2 = int(round(K * frac_struct))   # Structure
    K_3 = int(round(K * frac_inter))    # Interaction

    # Ensure at least 1 tile if fraction > 0
    if frac_complex > 0 and K_1 == 0:
        K_1 = 1
    if frac_struct > 0 and K_2 == 0:
        K_2 = 1
    if frac_inter > 0 and K_3 == 0:
        K_3 = 1

    K_4 = max(K - (K_1 + K_2 + K_3), 0)  # Diversity

    # ----------------------------------------------------------------------
    # 2) Semantic coverage minimum per major axis (0–2) via min_axis_frac
    # ----------------------------------------------------------------------
    min_axis_tiles = 0
    if min_axis_frac > 0.0:
        min_axis_tiles = max(1, int(round(K * min_axis_frac)))

    if min_axis_tiles > 0:
        # Ensure minimum coverage for each major axis
        if K_1 < min_axis_tiles:
            deficit = min_axis_tiles - K_1
            steal = min(deficit, K_4)
            K_1 += steal
            K_4 -= steal
        
        if K_2 < min_axis_tiles:
            deficit = min_axis_tiles - K_2
            steal = min(deficit, K_4)
            K_2 += steal
            K_4 -= steal
        
        if K_3 < min_axis_tiles:
            deficit = min_axis_tiles - K_3
            steal = min(deficit, K_4)
            K_3 += steal
            K_4 -= steal

    # Safety in case we overshoot K
    sum_major = K_1 + K_2 + K_3
    if sum_major > K:
        scale = K / float(sum_major)
        K_1 = int(round(K_1 * scale))
        K_2 = int(round(K_2 * scale))
        K_3 = int(round(K_3 * scale))
        K_4 = max(K - (K_1 + K_2 + K_3), 0)

    # ----------------------------------------------------------------------
    # 3) Base scores for axes 0–2
    # ----------------------------------------------------------------------
    # Complexity: variance across feature dims
    var_scores = torch.var(X, dim=1)  # [N]

    # Structure: magnitude / variance
    mag_scores = torch.norm(X, dim=1)  # [N]
    struct_scores = mag_scores / (var_scores + 1e-6)

    # AXIS 1: COMPLEXITY
    K_1_eff = min(K_1, N)
    if K_1_eff > 0:
        _, idx_complex = torch.topk(var_scores, k=K_1_eff)
    else:
        idx_complex = torch.tensor([], device=device, dtype=torch.long)

    mask_remain = torch.ones(N, dtype=torch.bool, device=device)
    if idx_complex.numel() > 0:
        mask_remain[idx_complex] = False

    # AXIS 2: STRUCTURE
    available = int(mask_remain.sum().item())
    K_2_eff = min(K_2, available)
    if K_2_eff > 0 and available > 0:
        cands_struct = torch.nonzero(mask_remain).squeeze()
        if cands_struct.ndim == 0 and cands_struct.numel() > 0:
            cands_struct = cands_struct.unsqueeze(0)
        struct_vals = struct_scores[cands_struct]
        _, top_struct = torch.topk(struct_vals, k=K_2_eff)
        idx_struct = cands_struct[top_struct]
        mask_remain[idx_struct] = False
    else:
        idx_struct = torch.tensor([], device=device, dtype=torch.long)

    # AXIS 3: INTERACTION
    available = int(mask_remain.sum().item())
    K_3_eff = min(K_3, available)
    if K_3_eff > 0 and available > 0 and coords is not None:
        cands_inter = torch.nonzero(mask_remain).squeeze()
        if cands_inter.ndim == 0 and cands_inter.numel() > 0:
            cands_inter = cands_inter.unsqueeze(0)

        signal = var_scores  # [N]
        cand_coords = coords[cands_inter]  # [M, 2]
        dists = torch.cdist(cand_coords, coords)  # [M, N]
        k_neighbors = min(17, dists.shape[1])
        _, neighbors = torch.topk(dists, k=k_neighbors, largest=False)
        neighbor_signals = signal[neighbors]         # [M, k_neighbors]
        inter_scores = torch.std(neighbor_signals, dim=1)  # [M]

        _, top_inter = torch.topk(inter_scores, k=K_3_eff)
        idx_inter = cands_inter[top_inter]
        mask_remain[idx_inter] = False
    else:
        idx_inter = torch.tensor([], device=device, dtype=torch.long)

    # ----------------------------------------------------------------------
    # 4) AXIS 4: DIVERSITY via k-center / random
    # ----------------------------------------------------------------------
    cands_div = torch.nonzero(mask_remain).squeeze()
    if cands_div.ndim == 0 and cands_div.numel() > 0:
        cands_div = cands_div.unsqueeze(0)

    available = int(cands_div.numel())
    K_4_eff = min(K_4, available)

    if K_4_eff <= 0 or available == 0:
        idx_div = torch.tensor([], device=device, dtype=torch.long)
    else:
        diversity_mode = (diversity_mode or "kcenter").lower()
        if diversity_mode == "kcenter":
            Xc = X[cands_div]  # [M, D]

            var_cands = var_scores[cands_div]
            start_idx = torch.argmax(var_cands)  # index in [0..M-1]

            selected_local = [start_idx]
            M = Xc.shape[0]

            min_dist = torch.full((M,), float("inf"), device=device)

            def update_min_dist(center_idx: int, min_dist_vec: torch.Tensor):
                center = Xc[center_idx].unsqueeze(0)       # [1, D]
                d = torch.norm(Xc - center, dim=1)         # [M]
                return torch.minimum(min_dist_vec, d)

            min_dist = update_min_dist(start_idx, min_dist)

            for _ in range(1, K_4_eff):
                next_idx = int(torch.argmax(min_dist).item())
                if next_idx in selected_local:
                    break
                selected_local.append(next_idx)
                min_dist = update_min_dist(next_idx, min_dist)

            selected_local = torch.tensor(selected_local, device=device, dtype=torch.long)
            idx_div = cands_div[selected_local]
        else:
            # Fallback to random diversity
            perm = torch.randperm(available, device=device)[:K_4_eff]
            idx_div = cands_div[perm]

    # ----------------------------------------------------------------------
    # 5) Assemble final indices + labels
    # ----------------------------------------------------------------------
    
    # Build label map BEFORE concatenation
    label_map = torch.full((N,), 255, device=device, dtype=torch.uint8)
    label_map[idx_complex] = 0
    label_map[idx_struct]  = 1
    label_map[idx_inter]   = 2
    label_map[idx_div]     = 3

    # Concatenate indices in semantic order (complexity → structure → interaction → diversity)
    final_indices = torch.cat([idx_complex, idx_struct, idx_inter, idx_div])
    
    # Remove duplicates WITHOUT sorting (preserve semantic order)
    if final_indices.numel() > 0:
        # Convert to list for Python-based deduplication (preserves order)
        seen = set()
        unique_indices = []
        for idx_item in final_indices.tolist():
            if idx_item not in seen:
                seen.add(idx_item)
                unique_indices.append(idx_item)
        
        # Convert back to tensor
        final_indices = torch.tensor(unique_indices, device=device, dtype=torch.long)
    
    # NOW assign labels from the label map (this will be correct!)
    final_labels = label_map[final_indices]
    
    selected_count = int(final_indices.numel())
    
    # Handle edge case: no tiles selected
    if selected_count == 0:
        _, idx_one = torch.topk(var_scores, k=1)
        final_indices = idx_one
        final_labels = torch.tensor([0], device=device, dtype=torch.uint8)
        selected_count = 1
    
    # If overshoot K (rare), keep top-K by variance
    if final_indices.numel() > K:
        scores_final = var_scores[final_indices]
        _, keep_idx = torch.topk(scores_final, k=K)
        # Sort keep_idx to preserve semantic order
        keep_idx = torch.sort(keep_idx)[0]
        final_indices = final_indices[keep_idx]
        final_labels = final_labels[keep_idx]
        selected_count = K
    
    final_risk = var_scores[final_indices]  # variance proxy

    # ----------------------------------------------------------------------
    # 6) Metadata
    # ----------------------------------------------------------------------
    metadata = {
        "total_tiles": int(N),
        "selected": selected_count,
        "complexity": int(idx_complex.numel()),
        "structure": int(idx_struct.numel()),
        "interaction": int(idx_inter.numel()),
        "diversity": int(idx_div.numel()),
        "mean_risk": float(final_risk.mean().item()),
        "K_1": int(K_1),
        "K_2": int(K_2),
        "K_3": int(K_3),
        "K_4": int(K_4),
        "frac_complex": float(frac_complex),
        "frac_struct": float(frac_struct),
        "frac_inter": float(frac_inter),
        "frac_div": float(frac_div),
        "min_axis_frac": float(min_axis_frac),
        "diversity_mode": diversity_mode,
    }

    return (
        final_indices.detach().cpu().numpy().astype(np.int32),
        final_risk.detach().cpu().numpy().astype(np.float32),
        final_labels.detach().cpu().numpy(),
        metadata,
    )

# ============================================================================
# WORKER FUNCTION (FIXED)
# ============================================================================

def process_patient_wrapper(
    pid: str,
    args: argparse.Namespace,
    config: MASSConfig,
    output_dir: str,
    gpu_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Worker function to process a single patient with CPU fallback."""
    try:
        # Choose device
        device_str = "cpu"
        if gpu_ids:
            gid = gpu_ids[hash(pid) % len(gpu_ids)]
            device_str = f"cuda:{gid}"

        # Load data
        try:
            X, tile_ids, coords, is_preprocessed = load_patient_features(
                args.feature_root,
                args.encoder,
                pid,
                tile_dir=args.tile_dir,
                format_type=args.format_type,
                verbose=False,
            )
        except FileNotFoundError as e:
            return {
                "status": "skipped",
                "patient_id": pid,
                "reason": f"Missing files: {str(e)}"
            }

        # ---------------------------------------------------------------------
        # 1. INITIAL CHECK
        # ---------------------------------------------------------------------
        if len(X) == 0:
            return {
                "status": "skipped",
                "patient_id": pid,
                "reason": "No tiles available after loading"
            }

        # ---------------------------------------------------------------------
        # 2. QUALITY FILTER (Strict)
        # ---------------------------------------------------------------------
        quality_mask, _ = filter_artifacts_comprehensive(
            X, coords, config, verbose=False, is_preprocessed=is_preprocessed
        )
        
        # ---------------------------------------------------------------------
        # 3. HANDLE ARTIFACT SAVING (Optional)
        # ---------------------------------------------------------------------
        if args.save_artifacts and output_dir:
            artifact_indices = np.where(~quality_mask)[0]
            
            if len(artifact_indices) > 0:
                artifact_ids = [tile_ids[i] for i in artifact_indices]
                
                # Setup directories
                discard_base = Path(output_dir) / "discarded_artifacts"
                discard_base.mkdir(parents=True, exist_ok=True)
                
                # 1. Save list of IDs
                with open(discard_base / f"{pid}_discarded_ids.txt", "w") as f:
                    f.write("\n".join(artifact_ids))

                # 2. Copy images (Recursive Search)
                if args.tile_dir:
                    patient_artifact_dir = discard_base / pid
                    patient_artifact_dir.mkdir(exist_ok=True)
                    
                    patient_root = Path(args.tile_dir) / pid
                    
                    # BUILD FILE MAP: Scan all subfolders
                    file_map = {}
                    if patient_root.exists():
                        for ext in ["*.png", "*.jpg", "*.jpeg"]:
                            for p in patient_root.rglob(ext):
                                file_map[p.name] = p
                    
                    copied_count = 0
                    for tid in artifact_ids:
                        # Handle potential missing extensions in IDs
                        src_path = file_map.get(tid)
                        if not src_path:
                            src_path = file_map.get(f"{tid}.png")
                        if not src_path:
                            src_path = file_map.get(f"{tid}.jpg")
                            
                        if src_path:
                            try:
                                shutil.copy2(src_path, patient_artifact_dir / src_path.name)
                                copied_count += 1
                            except Exception:
                                pass 
                    
                    if copied_count == 0 and len(artifact_ids) > 0:
                        print(f"[WARN] {pid}: Found artifacts but could not locate source images in {patient_root}")
                
                elif len(artifact_ids) > 0:
                    if not hasattr(process_patient_wrapper, '_warned_tiledir'):
                        print("[WARN] --save-artifacts is ON but --tile-dir is missing.")
                        process_patient_wrapper._warned_tiledir = True

        # ---------------------------------------------------------------------
        # 4. PREPARE CLEAN DATA (FIXED: STRICT CLAMPING)
        # ---------------------------------------------------------------------
        clean_idx = np.where(quality_mask)[0]
        
        # If filtering removed everything, we must skip
        if len(clean_idx) == 0:
             return {
                "status": "skipped",
                "patient_id": pid,
                "reason": "All tiles were filtered out as artifacts"
            }

        X_clean = X[clean_idx]
        coords_clean = coords[clean_idx] if coords is not None else None
        
        # >>> FIX: Clamping Logic <<<
        # If clean tiles (e.g. 5000) > topk (4096) --> effective_K = 4096 (Select Best)
        # If clean tiles (e.g. 3000) <= topk (4096) --> effective_K = 3000 (Select All)
        effective_K = min(len(X_clean), args.topk)

        X_t = torch.from_numpy(X_clean).float()
        C_t = torch.from_numpy(coords_clean).float() if coords_clean is not None else None

        # ---------------------------------------------------------------------
        # 5. RUN MASS SELECTION
        # ---------------------------------------------------------------------
        try:
            device = torch.device(device_str)
            
            selected_local, risks, labels, metadata = universal_mass_selection_gpu_with_labels(
                X_t, C_t, effective_K, config, device,
                min_axis_frac=args.min_axis_frac, diversity_mode=args.diversity_mode,
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and device_str.startswith("cuda"):
                torch.cuda.empty_cache()
                device_cpu = torch.device("cpu")
                selected_local, risks, labels, metadata = universal_mass_selection_gpu_with_labels(
                    X_t, C_t, effective_K, config, device_cpu,
                    min_axis_frac=args.min_axis_frac, diversity_mode=args.diversity_mode,
                )
            else:
                raise e

        # ---------------------------------------------------------------------
        # 6. MAP BACK INDICES
        # ---------------------------------------------------------------------
        # selected_local are indices into X_clean
        # We need indices into original X (for tile_ids)
        selected_original = clean_idx[selected_local]
        selected_ids = [tile_ids[i] for i in selected_original]
        
        if coords is not None:
             selected_coords = coords[selected_original]
        else:
             selected_coords = np.zeros((len(selected_original), 2), dtype=np.int32)

        # Update metadata
        metadata["artifacts_removed"] = int(len(X) - len(clean_idx))
        metadata["requested_K"] = int(args.topk)
        metadata["adaptive_K"] = int(effective_K)
        metadata["original_tiles"] = int(len(X))
        metadata["used_full_K"] = bool(effective_K == args.topk)
        
        # Save results
        if output_dir:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)

            np.save(out_path / f"{pid}_topk_idx.npy", selected_original.astype(np.int32))
            np.save(out_path / f"{pid}_topk_risks.npy", risks.astype(np.float32))
            np.save(out_path / f"{pid}_topk_labels.npy", labels.astype(np.uint8))
            np.save(out_path / f"{pid}_topk_coords.npy", selected_coords.astype(np.int32))
            
            with open(out_path / f"{pid}_topk_ids.txt", "w") as f:
                f.write("\n".join(selected_ids))

            with open(out_path / f"{pid}_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

        return {
            "status": "success",
            "patient_id": pid,
            "mean_risk": metadata["mean_risk"],
            "artifacts_removed": metadata["artifacts_removed"],
            "effective_K": effective_K,
        }

    except Exception as e:
        error_details = {
            "status": "error",
            "patient_id": pid,
            "error_msg": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc()
        }
        if not hasattr(process_patient_wrapper, '_error_count'):
            process_patient_wrapper._error_count = 0
        if process_patient_wrapper._error_count < 3:
            print(f"\n[ERROR] Patient {pid} failed:")
            print(f"  Type: {error_details['error_type']}")
            print(f"  Message: {error_details['error_msg']}")
            process_patient_wrapper._error_count += 1
        return error_details

# ============================================================================
# MAIN CACHE BUILDER
# ============================================================================

def parse_gpu_ids(gpu_str: str) -> List[str]:
    if not gpu_str:
        return []
    return [g.strip() for g in gpu_str.split(",") if g.strip() != ""]

def build_cache_parallel(args, patient_ids, config, output_dir, gpu_ids):
    os.makedirs(output_dir, exist_ok=True)

    # Save MASS config (for reproducibility)
    with open(Path(output_dir) / "mass_config.json", "w") as f:
        json.dump(
            {
                "complexity_fraction": float(config.high_risk_fraction),
                "structure_fraction": float(config.low_risk_fraction),
                "interaction_fraction": float(config.interaction_fraction),
                "diversity_fraction": float(config.diversity_fraction),
                "min_axis_frac": float(args.min_axis_frac),
                "diversity_mode": args.diversity_mode,
            },
            f,
            indent=2,
        )

    print(f"\n[Cache] Processing {len(patient_ids)} patients...")

    worker = partial(
        process_patient_wrapper,
        args=args,
        config=config,
        output_dir=output_dir,
        gpu_ids=gpu_ids,
    )

    start_time = time.time()

    # Use explicit spawn context to avoid CUDA fork issues
    mp_context = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=args.n_workers, mp_context=mp_context) as executor:
        results = list(
            tqdm.tqdm(executor.map(worker, patient_ids), total=len(patient_ids))
        )

    success = sum(1 for r in results if r["status"] == "success")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    errors = [r for r in results if r["status"] == "error"]
    skipped_list = [r for r in results if r["status"] == "skipped"]
    
    print(
        f"\n[Cache] Done in {time.time() - start_time:.1f}s"
    )
    print(f"  ✓ Success: {success}/{len(patient_ids)}")
    print(f"  ⊘ Skipped: {skipped}/{len(patient_ids)}")
    print(f"  ✗ Errors: {len(errors)}/{len(patient_ids)}")
    
    # Report skipped patients
    if skipped_list:
        print(f"\n[SKIPPED SUMMARY] {len(skipped_list)} patients skipped:")
        skip_reasons = {}
        for skip in skipped_list:
            reason = skip.get('reason', 'Unknown')
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
        
        for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count} patients")
        
        # Save skipped log
        skipped_log_path = Path(output_dir) / "skipped_log.json"
        with open(skipped_log_path, "w") as f:
            json.dump(skipped_list, f, indent=2)
        print(f"\n[SKIPPED LOG] Saved to {skipped_log_path}")
    
    # Report errors
    if errors:
        print(f"\n[ERROR SUMMARY] {len(errors)} patients failed:")
        error_types = {}
        for err in errors:
            err_type = err.get('error_type', 'Unknown')
            error_types[err_type] = error_types.get(err_type, 0) + 1
        
        for err_type, count in sorted(error_types.items(), key=lambda x: -x[1]):
            print(f"  {err_type}: {count} patients")
        
        # Save error log
        error_log_path = Path(output_dir) / "error_log.json"
        with open(error_log_path, "w") as f:
            json.dump(errors, f, indent=2)
        print(f"\n[ERROR LOG] Saved to {error_log_path}")

# ============================================================================
# OPTUNA OBJECTIVE (Mass Version)
# ============================================================================

class MassMASSObjective:
    def __init__(self, args, sample_pids: List[str], gpu_ids: Optional[List[str]] = None):
        self.args = args
        self.sample_pids = sample_pids
        self.gpu_ids = gpu_ids

    def __call__(self, trial: optuna.Trial) -> float:
        # Fractions for Complexity / Structure / Interaction
        f_complex = trial.suggest_float("frac_complexity", 0.15, 0.40)
        f_struct  = trial.suggest_float("frac_structure",  0.15, 0.40)
        f_inter   = trial.suggest_float("frac_interaction", 0.10, 0.30)
        f_div = 1.0 - (f_complex + f_struct + f_inter)

        # Basic feasibility
        if f_div < 0.05 or f_div > 0.40:
            raise optuna.TrialPruned()

        config = MASSConfig(
            high_risk_fraction=f_complex,
            low_risk_fraction=f_struct,
            interaction_fraction=f_inter,
            diversity_fraction=f_div,
            min_quality_threshold=0.3,
            heterogeneity_weight=0.33,
            extremeness_weight=0.33,
            dissimilarity_weight=0.33,
            density_weight=0.01,  # Dummy
        )

        rng = np.random.default_rng(self.args.seed + trial.number)
        trial_pids = rng.choice(
            self.sample_pids, size=min(12, len(self.sample_pids)), replace=False
        )

        # Proxy objective: mean variance-based risk of selected tiles
        div_scores = []
        for pid in trial_pids:
            res = process_patient_wrapper(
                pid,
                self.args,
                config,
                output_dir="",  # no saving during Optuna
                gpu_ids=self.gpu_ids,
            )
            if res["status"] == "success":
                div_scores.append(res["mean_risk"])
            # Skip patients with errors or missing files

        if len(div_scores) == 0:
            # All patients failed, prune this trial
            raise optuna.TrialPruned()

        return float(np.mean(div_scores))

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--tile-dir", default=None)
    parser.add_argument("--format-type", default="auto")
    parser.add_argument("--out-cache-best", required=True)
    parser.add_argument("--study-dir", default=None)
    parser.add_argument("--topk", type=int, default=4096)
    parser.add_argument("--skip-optuna", action="store_true")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--gpus", default="")
    parser.add_argument("--n-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save-artifacts", 
        action="store_true", 
        help="If set, saves a list of discarded artifact tile IDs to a 'discarded' folder."
    )
    parser.add_argument(
        "--min-axis-frac",
        type=float,
        default=0.0,
        help="Minimum fraction of K reserved for each major axis (0–2). "
             "Tiles are 'stolen' from Diversity to satisfy coverage.",
    )
    parser.add_argument(
        "--diversity-mode",
        type=str,
        default="kcenter",
        choices=["kcenter", "random"],
        help="Strategy for Diversity axis selection.",
    )
    parser.add_argument(
        "--only-pid",
        nargs="+",
        default=None,
        help="Optional: one or more patient IDs to process (e.g., P-0001 TCGA-XX-YYYY).",
    )
    args = parser.parse_args()

    gpu_ids = parse_gpu_ids(args.gpus)

    # Discover patients
    pids = discover_patients(args.feature_root, args.encoder, args.format_type, verbose=True)

    if args.only_pid is not None:
        requested = set([str(p) for p in args.only_pid])
        pids = [p for p in pids if str(p) in requested]
        print(f"[Filter] Restricting to {len(pids)} patients: {pids}")

    if len(pids) == 0:
        print("[Error] No patients to process after applying --only-pid filter.")
        return

    # MASS configuration (with or without Optuna)
    if args.skip_optuna:
        print("[Config] Using Standard Balanced MASS Config (32/18/27/23)")
        config = MASSConfig(
            high_risk_fraction=0.19585112746335845,
            low_risk_fraction=0.22606056073988443,
            interaction_fraction=0.20495128632644755,
            diversity_fraction=0.3731370254703096,
            min_quality_threshold=0.1,
            heterogeneity_weight=1.0,
            extremeness_weight=1.0,
            dissimilarity_weight=1.0,
            density_weight=0.0,
        )
    else:
        if args.study_dir is None:
            args.study_dir = str(Path(args.out_cache_best).parent / "mass_study")
        os.makedirs(args.study_dir, exist_ok=True)

        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=args.seed),
        )
        objective = MassMASSObjective(args, pids, gpu_ids)

        print(f"[Optuna] Tuning MASS Fractions ({args.n_trials} trials)...")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=args.n_trials, n_jobs=1)

        best = study.best_params
        f_div = 1.0 - (
            best["frac_complexity"] + best["frac_structure"] + best["frac_interaction"]
        )
        print(
            f"[Optuna] Best Fractions: "
            f"Comp={best['frac_complexity']:.2f}, "
            f"Struct={best['frac_structure']:.2f}, "
            f"Inter={best['frac_interaction']:.2f}, "
            f"Div={f_div:.2f}"
        )

        config = MASSConfig(
            high_risk_fraction=best["frac_complexity"],
            low_risk_fraction=best["frac_structure"],
            interaction_fraction=best["frac_interaction"],
            diversity_fraction=f_div,
            min_quality_threshold=0.1,
            heterogeneity_weight=1.0,
            extremeness_weight=1.0,
            dissimilarity_weight=1.0,
            density_weight=0.0,
        )

    # Build cache with MASS v2 + semantic labels
    build_cache_parallel(args, pids, config, args.out_cache_best, gpu_ids)

if __name__ == "__main__":
    # Ensure spawn context is set (redundant safety check)
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    
    main()
