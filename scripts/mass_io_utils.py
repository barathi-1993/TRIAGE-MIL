#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MASS I/O Utilities - Universal loader for CLAM and AE formats
Updated with 'verbose' flags for clean parallel processing.
"""

import numpy as np
import h5py
import glob
from pathlib import Path
from typing import Tuple, List, Optional
import re


def _tile_id_from_path(path: str) -> str:
    """Extract tile ID from filepath"""
    return Path(path).stem


def _extract_coords_from_filename(filename: str) -> Optional[Tuple[int, int]]:
    """
    Extract coordinates from filename.
    Supports formats:
    - P-0001_512_w2048_4_194_h55296_108_171.png  -> (2048, 55296)
    - TCGA-XX_512_w4608_9_264_h45056_88_206.png  -> (4608, 45056)
    """
    try:
        # Pattern: w{X}_...h{Y}_
        w_match = re.search(r'_w(\d+)_', filename)
        h_match = re.search(r'_h(\d+)_', filename)
        
        if w_match and h_match:
            x = int(w_match.group(1))
            y = int(h_match.group(1))
            return (x, y)
    except:
        pass
    
    return None


# ============================================================================
# AE Format Loaders
# ============================================================================

def load_ae_format(
    feature_root: str,
    encoder: str,
    patient_id: str,
    tile_dir: Optional[str] = None,
    verbose: bool = True  # <--- NEW
) -> Tuple[np.ndarray, List[str], Optional[np.ndarray], bool]:
    """
    Load AE-based preprocessing format.
    """
    patient_dir = Path(feature_root) / encoder / patient_id
    
    # Find the combined features file
    feature_files = list(patient_dir.glob(f"{patient_id}_Informative_set*_combined_features.npy"))
    
    if not feature_files:
        raise FileNotFoundError(f"No AE features found in {patient_dir}")
    
    if verbose:
        print(f"[AE Loader] Found {len(feature_files)} feature files")
    
    all_features = []
    all_tile_ids = []
    all_coords = []
    
    for feature_file in sorted(feature_files):
        # Extract set name (e.g., "set1" from "Informative_set1")
        set_name = feature_file.stem.split('_Informative_')[-1].replace('_combined_features', '')
        
        # Load features
        try:
            features = np.load(feature_file, allow_pickle=True)
        except ValueError:
            features = np.load(feature_file, allow_pickle=False)
        
        # Handle object arrays
        if features.dtype == object:
            if features.shape == () and isinstance(features.item(), dict):
                data = features.item()
                if 'features' in data:
                    features = data['features']
                elif 'feat' in data:
                    features = data['feat']
        
        features = features.astype(np.float32)
        
        # Ensure 2D
        if features.ndim == 1:
            features = features[None, :]
        
        all_features.append(features)
        n_tiles = features.shape[0]
        
        # ========== NEW: Load coordinates from *_coords.npy ==========
        coords_file = patient_dir / f"{patient_id}_Informative_{set_name}_combined_features_coords.npy"
        coords_loaded = False
        
        if coords_file.exists():
            try:
                coords = np.load(coords_file, allow_pickle=False)
                
                # Validate shape
                if coords.shape[0] == n_tiles and coords.shape[1] == 2:
                    all_coords.extend(coords.tolist())
                    coords_loaded = True
                    if verbose:
                        print(f"  Loaded coordinates from {coords_file.name}")
                else:
                    if verbose:
                        print(f"  Warning: Coord shape mismatch in {coords_file.name}: "
                              f"{coords.shape} (expected {n_tiles}x2)")
            except Exception as e:
                if verbose:
                    print(f"  Warning: Could not load coords from {coords_file.name}: {e}")
        
        # ========== Load tile IDs from *_filenames.npy ==========
        filenames_file = patient_dir / f"{patient_id}_Informative_{set_name}_combined_features_filenames.npy"
        
        if filenames_file.exists():
            try:
                filenames = np.load(filenames_file, allow_pickle=True)
                
                # Handle different formats
                if isinstance(filenames, np.ndarray):
                    if filenames.dtype == object:
                        # Array of strings
                        filenames = [str(f) for f in filenames]
                    else:
                        filenames = filenames.tolist()
                
                # Extract tile IDs from filenames
                if len(filenames) == n_tiles:
                    for fname in filenames:
                        # Extract stem (filename without extension)
                        if isinstance(fname, (str, Path)):
                            tile_id = Path(fname).stem
                        else:
                            tile_id = str(fname)
                        all_tile_ids.append(tile_id)
                        
                        # If coords weren't loaded from file, try extracting from filename
                        if not coords_loaded:
                            coord = _extract_coords_from_filename(str(fname))
                            if coord:
                                all_coords.append(coord)
                    
                    if verbose:
                        print(f"  Loaded {len(filenames)} tile IDs from {filenames_file.name}")
                else:
                    if verbose:
                        print(f"  Warning: Filename count mismatch: {len(filenames)} vs {n_tiles} features")
                    # Fall back to generic IDs
                    for i in range(n_tiles):
                        all_tile_ids.append(f"{patient_id}_{set_name}_{i}")
                        
            except Exception as e:
                if verbose:
                    print(f"  Warning: Could not load filenames from {filenames_file.name}: {e}")
                # Fall back to generic IDs
                for i in range(n_tiles):
                    all_tile_ids.append(f"{patient_id}_{set_name}_{i}")
        else:
            # No filenames file - try tile_dir approach (fallback)
            if tile_dir:
                tile_set_dir = Path(tile_dir) / patient_id / f"Informative_{set_name}"
                
                if tile_set_dir.exists():
                    tile_files = sorted(tile_set_dir.glob("*.png"))
                    
                    if len(tile_files) == n_tiles:
                        for tile_file in tile_files:
                            tile_id = tile_file.stem
                            all_tile_ids.append(tile_id)
                            
                            # Extract coordinates from filename if not loaded from coords file
                            if not coords_loaded:
                                coord = _extract_coords_from_filename(tile_file.name)
                                if coord:
                                    all_coords.append(coord)
                    else:
                        for i in range(n_tiles):
                            all_tile_ids.append(f"{patient_id}_{set_name}_{i}")
                else:
                    for i in range(n_tiles):
                        all_tile_ids.append(f"{patient_id}_{set_name}_{i}")
            else:
                # No tile_dir - use generic IDs
                for i in range(n_tiles):
                    all_tile_ids.append(f"{patient_id}_{set_name}_{i}")
    
    # Concatenate all sets
    X = np.concatenate(all_features, axis=0)
    
    # Handle coordinates
    coords_array = None
    if all_coords and len(all_coords) == len(all_tile_ids):
        coords_array = np.array(all_coords, dtype=np.float32)
        if verbose:
            print(f"  Total: Loaded coordinates for {len(coords_array)} tiles")
    else:
        if all_coords and len(all_coords) != len(all_tile_ids):
            if verbose:
                print(f"  Warning: Coord count mismatch: {len(all_coords)} coords vs {len(all_tile_ids)} tiles")
        if verbose:
            print(f"  No coordinates available")
    
    return X, all_tile_ids, coords_array, True


# ============================================================================
# CLAM Format Loaders
# ============================================================================

def load_clam_format(
    feature_root: str,
    encoder: str,
    patient_id: str,
    tile_dir: Optional[str] = None,
    verbose: bool = True
) -> Tuple[np.ndarray, List[str], Optional[np.ndarray], bool]:
    """
    Load CLAM-based preprocessing format.
    """
    # Try multiple possible locations for .pt file
    pt_candidates = [
        Path(feature_root) / encoder / "pt_files" / f"{patient_id}.pt",
        Path(feature_root) / encoder / f"{patient_id}.pt",
        Path(feature_root) / "pt_files" / f"{patient_id}.pt",
        Path(feature_root) / f"{patient_id}.pt"
    ]
    
    pt_file = None
    for candidate in pt_candidates:
        if candidate.exists():
            pt_file = candidate
            break
    
    if pt_file is None:
        raise FileNotFoundError(
            f"No CLAM .pt file found for {patient_id}. Tried:\n" + 
            "\n".join(f"  - {c}" for c in pt_candidates)
        )
    
    if verbose:
        print(f"[CLAM Loader] Loading {pt_file}")
    
    # Load PyTorch features
    try:
        import torch
        data = torch.load(pt_file, map_location='cpu')  # ← Load as 'data', not 'features'
        
        # ✅ CRITICAL FIX: Extract coords from dict BEFORE extracting features
        coords_from_pt = None
        
        if isinstance(data, dict):
            # Extract coordinates first (if present)
            if 'coords' in data:
                coords_from_pt = data['coords']
                
                # Convert Tensor → numpy immediately
                if isinstance(coords_from_pt, torch.Tensor):
                    coords_from_pt = coords_from_pt.cpu().numpy()
                
                coords_from_pt = coords_from_pt.astype(np.int32)  # ← Force int32
                
                if verbose:
                    print(f"  ✅ Loaded coords from .pt: shape={coords_from_pt.shape}, "
                          f"range X=[{coords_from_pt[:, 0].min()}, {coords_from_pt[:, 0].max()}], "
                          f"Y=[{coords_from_pt[:, 1].min()}, {coords_from_pt[:, 1].max()}]")
            
            # Now extract features
            if 'features' in data:
                features = data['features']
            elif 'feat' in data:
                features = data['feat']
            else:
                # Take first tensor value
                features = next(v for v in data.values() if isinstance(v, torch.Tensor))
        else:
            # Raw tensor (no dict)
            features = data
        
        if isinstance(features, torch.Tensor):
            features = features.numpy()
        
        features = features.astype(np.float32)
        
    except ImportError:
        raise ImportError("PyTorch required for CLAM .pt files")
    
    # Ensure 2D
    if features.ndim == 1:
        features = features[None, :]
    
    N = features.shape[0]
    if verbose:
        print(f"  Loaded {N} features (dim={features.shape[1]})")
    
    # ✅ NEW: Use coords from .pt file if available
    coords_array = coords_from_pt  # Start with coords from .pt
    tile_ids = []
    
    # Only try loading from .h5 if coords weren't in .pt file
    if coords_array is None and tile_dir:
        # Try multiple possible locations for .h5 file
        h5_candidates = [
            Path(tile_dir) / "patches" / f"{patient_id}.h5",
            Path(tile_dir) / f"{patient_id}.h5",
            Path(tile_dir) / "patches" / patient_id / f"{patient_id}.h5",
        ]
        
        h5_file = None
        for candidate in h5_candidates:
            if candidate.exists():
                h5_file = candidate
                break
        
        if h5_file and h5_file.exists():
            try:
                with h5py.File(h5_file, 'r') as f:
                    # CLAM h5 structure: coords dataset [N, 2]
                    if 'coords' in f:
                        coords = f['coords'][:]
                        
                        # Validate
                        if coords.shape[0] == N:
                            coords_array = coords.astype(np.int32)  # ← int32!
                            if verbose:
                                print(f"  Loaded coordinates for {N} tiles from {h5_file.name}")
                        else:
                            if verbose:
                                print(f"  Warning: Coord count mismatch: {coords.shape[0]} coords vs {N} features")
                    else:
                        if verbose:
                            print(f"  Warning: No 'coords' dataset in {h5_file.name}")
                    
                    # Try to load tile IDs if available
                    if 'tile_ids' in f:
                        tile_ids = [tid.decode('utf-8') if isinstance(tid, bytes) else str(tid) 
                                   for tid in f['tile_ids'][:]]
            except Exception as e:
                if verbose:
                    print(f"  Warning: Could not load h5 file: {e}")
        else:
            if verbose:
                print(f"  Note: No .h5 file found (coords already loaded from .pt)")
    
    # ✅ VALIDATION: Check if coords were loaded
    if coords_array is not None:
        # Verify not all zeros
        if coords_array.max() == 0 and coords_array.min() == 0:
            if verbose:
                print(f"  ⚠️  WARNING: All coordinates are zero!")
        
        # Verify shape matches
        if coords_array.shape[0] != N:
            if verbose:
                print(f"  ⚠️  WARNING: Coord count mismatch: {coords_array.shape[0]} != {N}")
            # Truncate to match
            min_len = min(coords_array.shape[0], N)
            coords_array = coords_array[:min_len]
            features = features[:min_len]
            N = min_len
    else:
        if verbose:
            print(f"  ⚠️  WARNING: No coordinates loaded (will use zeros)")
    
    # Generate tile IDs if not loaded
    if not tile_ids or len(tile_ids) != N:
        tile_ids = [f"{patient_id}_tile_{i}" for i in range(N)]
    
    # Return False to trigger STRICT artifact filtering
    return features, tile_ids, coords_array, False


# ============================================================================
# Universal Loader (Auto-detect format)
# ============================================================================

def load_patient_features(
    feature_root: str,
    encoder: str,
    patient_id: str,
    tile_dir: Optional[str] = None,
    format_type: str = "auto",
    verbose: bool = True  # <--- NEW
) -> Tuple[np.ndarray, List[str], Optional[np.ndarray], bool]:
    """
    Universal loader that auto-detects CLAM vs AE format.
    """
    
    if format_type == "auto":
        # Auto-detect format
        
        # Check for CLAM format (.pt files)
        clam_pt = Path(feature_root) / encoder / "pt_files" / f"{patient_id}.pt"
        clam_pt_alt = Path(feature_root) / encoder / f"{patient_id}.pt"
        
        if clam_pt.exists() or clam_pt_alt.exists():
            if verbose:
                print(f"[Auto-detect] CLAM format detected for {patient_id}")
            format_type = "clam"
        else:
            # Check for AE format (patient_id directory)
            ae_dir = Path(feature_root) / encoder / patient_id
            if ae_dir.exists() and ae_dir.is_dir():
                if verbose:
                    print(f"[Auto-detect] AE format detected for {patient_id}")
                format_type = "ae"
            else:
                raise FileNotFoundError(
                    f"Could not detect format for {patient_id}. "
                    f"Checked:\n  CLAM: {clam_pt}\n  AE: {ae_dir}"
                )
    
    # Load based on detected format, passing VERBOSE down
    if format_type == "clam":
        return load_clam_format(feature_root, encoder, patient_id, tile_dir, verbose=verbose)
    elif format_type == "ae":
        return load_ae_format(feature_root, encoder, patient_id, tile_dir, verbose=verbose)
    else:
        raise ValueError(f"Unknown format_type: {format_type}")


# ============================================================================
# Patient List Discovery
# ============================================================================

def discover_patients(
    feature_root: str,
    encoder: str,
    format_type: str = "auto",
    verbose: bool = True # <--- NEW
) -> List[str]:
    """
    Discover all available patients in the feature directory.
    """
    base = Path(feature_root) / encoder
    
    # If base doesn't exist, try without encoder subdirectory
    if not base.exists():
        base = Path(feature_root)
    
    if not base.exists():
        raise FileNotFoundError(f"Feature directory not found: {base}")
    
    patients = []
    
    if format_type == "auto":
        # Try CLAM first (pt_files subdirectory)
        pt_dir = base / "pt_files"
        if pt_dir.exists():
            pt_files = list(pt_dir.glob("*.pt"))
            if pt_files:
                patients = [f.stem for f in pt_files]
                if verbose:
                    print(f"[Discovery] Found {len(patients)} patients (CLAM format with pt_files/)")
                return sorted(patients)
        
        # Alternative CLAM location (no pt_files)
        pt_files = list(base.glob("*.pt"))
        if pt_files:
            patients = [f.stem for f in pt_files]
            if verbose:
                print(f"[Discovery] Found {len(patients)} patients (CLAM format, flat structure)")
            return sorted(patients)
        
        # Try AE format
        patient_dirs = [d for d in base.iterdir() if d.is_dir() and d.name != "pt_files"]
        if patient_dirs:
            # Verify it's AE format (has combined_features.npy files)
            sample_dir = patient_dirs[0]
            ae_files = list(sample_dir.glob("*_combined_features.npy"))
            if ae_files:
                patients = [d.name for d in patient_dirs]
                if verbose:
                    print(f"[Discovery] Found {len(patients)} patients (AE format)")
                return sorted(patients)
    
    elif format_type == "clam":
        # CLAM format
        pt_dir = base / "pt_files"
        if pt_dir.exists():
            pt_files = list(pt_dir.glob("*.pt"))
        else:
            pt_files = list(base.glob("*.pt"))
        patients = [f.stem for f in pt_files]
    
    elif format_type == "ae":
        patient_dirs = [d for d in base.iterdir() if d.is_dir()]
        patients = [d.name for d in patient_dirs]
    
    if not patients:
        raise ValueError(f"No patients found in {base}")
    
    if verbose:
        print(f"[Discovery] Found {len(patients)} patients")
    return sorted(patients)