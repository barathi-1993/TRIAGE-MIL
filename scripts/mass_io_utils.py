#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MASS I/O Utilities for CLAM-Style WSI Feature Files
===================================================

This module provides clean, GitHub-ready utilities for loading WSI tile
features extracted using a CLAM-style preprocessing pipeline.

Supported input format:
-----------------------
1. CLAM-style PyTorch feature files:
   
   feature_root/
   └── encoder/
       └── pt_files/
           ├── patient_001.pt
           ├── patient_002.pt
           └── ...

   or:

   feature_root/
   └── encoder/
       ├── patient_001.pt
       ├── patient_002.pt
       └── ...

2. Optional CLAM patch coordinate files:

   tile_dir/
   └── patches/
       ├── patient_001.h5
       ├── patient_002.h5
       └── ...

Expected .pt structure:
-----------------------
The .pt file can be either:

1. A tensor directly:
   Tensor[N, D]

2. A dictionary containing features:
   {
       "features": Tensor[N, D],
       "coords": Tensor[N, 2]   # optional
   }

Accepted feature keys:
----------------------
- "features"
- "feat"
- "feats"

Returned values:
----------------
features : np.ndarray
    Tile-level feature matrix with shape [N, D].

tile_ids : list[str]
    Tile identifiers.

coords : Optional[np.ndarray]
    Tile coordinates with shape [N, 2], if available.

is_preprocessed : bool
    Always False for CLAM-style features, so downstream MASS filtering can
    apply strict quality filtering.
"""

from pathlib import Path
from typing import List, Optional, Tuple
import re

import h5py
import numpy as np


# -------------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------------

def _extract_coords_from_filename(filename: str) -> Optional[Tuple[int, int]]:
    """
    Extract tile coordinates from a filename if the filename follows a
    coordinate-containing pattern.

    Supported examples:
    -------------------
    P-0001_512_w2048_4_194_h55296_108_171.png -> (2048, 55296)
    TCGA-XX_512_w4608_9_264_h45056_88_206.png -> (4608, 45056)

    Parameters
    ----------
    filename : str
        Tile filename.

    Returns
    -------
    Optional[Tuple[int, int]]
        Extracted (x, y) coordinates if available, otherwise None.
    """
    try:
        w_match = re.search(r"_w(\d+)_", filename)
        h_match = re.search(r"_h(\d+)_", filename)

        if w_match and h_match:
            x = int(w_match.group(1))
            y = int(h_match.group(1))
            return x, y

    except Exception:
        return None

    return None


def _find_clam_pt_file(feature_root: str, encoder: str, patient_id: str) -> Path:
    """
    Find a CLAM-style .pt feature file for a patient.

    Parameters
    ----------
    feature_root : str
        Root directory containing feature files.

    encoder : str
        Encoder folder name, for example "UNI" or "UNI/pt_files".

    patient_id : str
        Patient or slide identifier.

    Returns
    -------
    Path
        Path to the located .pt file.

    Raises
    ------
    FileNotFoundError
        If no .pt file is found.
    """
    feature_root = Path(feature_root)
    encoder_path = Path(encoder)

    candidates = [
        feature_root / encoder_path / f"{patient_id}.pt",
        feature_root / encoder_path / "pt_files" / f"{patient_id}.pt",
        feature_root / "pt_files" / f"{patient_id}.pt",
        feature_root / f"{patient_id}.pt",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    tried_paths = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(
        f"No CLAM .pt feature file found for patient_id='{patient_id}'.\n"
        f"Tried:\n{tried_paths}"
    )


def _find_clam_h5_file(tile_dir: str, patient_id: str) -> Optional[Path]:
    """
    Find an optional CLAM .h5 patch coordinate file.

    Parameters
    ----------
    tile_dir : str
        Directory containing CLAM patch files.

    patient_id : str
        Patient or slide identifier.

    Returns
    -------
    Optional[Path]
        Path to .h5 file if found, otherwise None.
    """
    if tile_dir is None:
        return None

    tile_dir = Path(tile_dir)

    candidates = [
        tile_dir / "patches" / f"{patient_id}.h5",
        tile_dir / f"{patient_id}.h5",
        tile_dir / "patches" / patient_id / f"{patient_id}.h5",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


# -------------------------------------------------------------------------
# CLAM feature loader
# -------------------------------------------------------------------------

def load_clam_format(
    feature_root: str,
    encoder: str,
    patient_id: str,
    tile_dir: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, List[str], Optional[np.ndarray], bool]:
    """
    Load CLAM-style WSI tile features for one patient/slide.

    The loader first searches for a .pt file. Coordinates are loaded in the
    following priority order:

    1. From the .pt file if the dictionary contains a "coords" key.
    2. From the corresponding CLAM .h5 patch file if tile_dir is provided.
    3. Otherwise coords is returned as None.

    Parameters
    ----------
    feature_root : str
        Root directory containing feature files.

    encoder : str
        Encoder subdirectory. Examples:
        - "UNI"
        - "UNI/pt_files"
        - "pt_files"

    patient_id : str
        Patient or slide identifier.

    tile_dir : Optional[str], default=None
        Optional CLAM patch directory containing .h5 files with coordinates.

    verbose : bool, default=True
        Whether to print loading details.

    Returns
    -------
    features : np.ndarray
        Feature matrix with shape [N, D].

    tile_ids : List[str]
        Generated tile identifiers.

    coords_array : Optional[np.ndarray]
        Coordinates with shape [N, 2], if available.

    is_preprocessed : bool
        Always False for CLAM-style features.
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required to load CLAM .pt feature files. "
            "Install it with: pip install torch"
        ) from exc

    pt_file = _find_clam_pt_file(feature_root, encoder, patient_id)

    if verbose:
        print(f"[CLAM Loader] Loading features from: {pt_file}")

    data = torch.load(pt_file, map_location="cpu")

    features = None
    coords_array = None

    # Case 1: .pt file stores a dictionary
    if isinstance(data, dict):
        # Load coordinates if present
        if "coords" in data and data["coords"] is not None:
            coords_array = data["coords"]

            if isinstance(coords_array, torch.Tensor):
                coords_array = coords_array.cpu().numpy()
            else:
                coords_array = np.asarray(coords_array)

            coords_array = coords_array.astype(np.int32)

        # Load features
        for key in ["features", "feat", "feats"]:
            if key in data:
                features = data[key]
                break

        # Fallback: use first tensor-like value that is not coords
        if features is None:
            for key, value in data.items():
                if key == "coords":
                    continue
                if isinstance(value, torch.Tensor):
                    features = value
                    break

        if features is None:
            raise ValueError(
                f"No feature tensor found in {pt_file}. "
                f"Expected one of the keys: 'features', 'feat', or 'feats'."
            )

    # Case 2: .pt file directly stores a tensor
    else:
        features = data

    if isinstance(features, torch.Tensor):
        features = features.cpu().numpy()
    else:
        features = np.asarray(features)

    features = features.astype(np.float32)

    # Ensure features are 2D: [N, D]
    if features.ndim == 1:
        features = features[None, :]

    if features.ndim != 2:
        raise ValueError(
            f"Expected a 2D feature matrix [N, D], but got shape {features.shape} "
            f"from {pt_file}."
        )

    n_tiles = features.shape[0]

    if verbose:
        print(f"  Loaded features: shape={features.shape}")

    # If coordinates were not stored inside .pt, try CLAM .h5 file
    if coords_array is None and tile_dir is not None:
        h5_file = _find_clam_h5_file(tile_dir, patient_id)

        if h5_file is not None:
            if verbose:
                print(f"[CLAM Loader] Loading coordinates from: {h5_file}")

            try:
                with h5py.File(h5_file, "r") as f:
                    if "coords" in f:
                        coords_array = np.asarray(f["coords"][:]).astype(np.int32)

                        if verbose:
                            print(f"  Loaded coordinates: shape={coords_array.shape}")
                    else:
                        if verbose:
                            print(f"  Warning: No 'coords' dataset found in {h5_file}")

            except Exception as exc:
                if verbose:
                    print(f"  Warning: Failed to load coordinates from {h5_file}: {exc}")

        else:
            if verbose:
                print("  Note: No matching CLAM .h5 coordinate file found.")

    # Validate coordinate shape
    if coords_array is not None:
        if coords_array.ndim != 2 or coords_array.shape[1] != 2:
            if verbose:
                print(
                    f"  Warning: Invalid coordinate shape {coords_array.shape}. "
                    "Coordinates will be ignored."
                )
            coords_array = None

        elif coords_array.shape[0] != n_tiles:
            min_len = min(coords_array.shape[0], n_tiles)

            if verbose:
                print(
                    f"  Warning: Coordinate count ({coords_array.shape[0]}) does not "
                    f"match feature count ({n_tiles}). Truncating both to {min_len}."
                )

            features = features[:min_len]
            coords_array = coords_array[:min_len]
            n_tiles = min_len

        elif coords_array.max() == 0 and coords_array.min() == 0:
            if verbose:
                print("  Warning: All loaded coordinates are zero.")

    else:
        if verbose:
            print("  Warning: No coordinates loaded. Downstream spatial operations may be limited.")

    # Generate generic tile IDs
    tile_ids = [f"{patient_id}_tile_{i}" for i in range(n_tiles)]

    # False means CLAM features are not pre-filtered by AE-style preprocessing.
    # Downstream MASS should apply strict quality filtering.
    is_preprocessed = False

    return features, tile_ids, coords_array, is_preprocessed


# -------------------------------------------------------------------------
# Public loader API
# -------------------------------------------------------------------------

def load_patient_features(
    feature_root: str,
    encoder: str,
    patient_id: str,
    tile_dir: Optional[str] = None,
    format_type: str = "clam",
    verbose: bool = True,
) -> Tuple[np.ndarray, List[str], Optional[np.ndarray], bool]:
    """
    Load patient/slide features.

    This GitHub release supports CLAM-style feature loading only.

    Parameters
    ----------
    feature_root : str
        Root directory containing feature files.

    encoder : str
        Encoder subdirectory, e.g., "UNI/pt_files" or "UNI".

    patient_id : str
        Patient or slide identifier.

    tile_dir : Optional[str], default=None
        Optional directory containing CLAM .h5 patch coordinate files.

    format_type : str, default="clam"
        Kept for compatibility. Accepted values:
        - "clam"
        - "auto"

        Both load CLAM-style features only.

    verbose : bool, default=True
        Whether to print loading details.

    Returns
    -------
    Tuple[np.ndarray, List[str], Optional[np.ndarray], bool]
        features, tile_ids, coordinates, is_preprocessed
    """
    if format_type not in {"clam", "auto"}:
        raise ValueError(
            f"Unsupported format_type='{format_type}'. "
            "This GitHub release supports only CLAM-style features. "
            "Use format_type='clam' or format_type='auto'."
        )

    return load_clam_format(
        feature_root=feature_root,
        encoder=encoder,
        patient_id=patient_id,
        tile_dir=tile_dir,
        verbose=verbose,
    )


# -------------------------------------------------------------------------
# Patient discovery
# -------------------------------------------------------------------------

def discover_patients(
    feature_root: str,
    encoder: str,
    format_type: str = "clam",
    verbose: bool = True,
) -> List[str]:
    """
    Discover available patient/slide IDs from CLAM-style .pt feature files.

    Supported layouts:
    ------------------
    1. feature_root/encoder/pt_files/*.pt
    2. feature_root/encoder/*.pt
    3. feature_root/pt_files/*.pt
    4. feature_root/*.pt

    Parameters
    ----------
    feature_root : str
        Root directory containing feature files.

    encoder : str
        Encoder subdirectory.

    format_type : str, default="clam"
        Kept for compatibility. Accepted values:
        - "clam"
        - "auto"

    verbose : bool, default=True
        Whether to print discovery details.

    Returns
    -------
    List[str]
        Sorted patient/slide IDs.
    """
    if format_type not in {"clam", "auto"}:
        raise ValueError(
            f"Unsupported format_type='{format_type}'. "
            "This GitHub release supports only CLAM-style feature discovery."
        )

    feature_root = Path(feature_root)
    encoder_path = Path(encoder)

    candidate_dirs = [
        feature_root / encoder_path,
        feature_root / encoder_path / "pt_files",
        feature_root / "pt_files",
        feature_root,
    ]

    pt_files = []

    for directory in candidate_dirs:
        if directory.exists() and directory.is_dir():
            found = sorted(directory.glob("*.pt"))
            if found:
                pt_files = found
                break

    if not pt_files:
        tried_dirs = "\n".join(f"  - {p}" for p in candidate_dirs)
        raise FileNotFoundError(
            f"No CLAM .pt feature files found.\n"
            f"Tried directories:\n{tried_dirs}"
        )

    patients = sorted([pt_file.stem for pt_file in pt_files])

    if verbose:
        print(f"[Discovery] Found {len(patients)} CLAM-style patients/slides.")
        print(f"[Discovery] Feature directory: {pt_files[0].parent}")

    return patients


# -------------------------------------------------------------------------
# Optional quick test
# -------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test CLAM-style feature loading for TRIAGE-MIL."
    )

    parser.add_argument(
        "--feature-root",
        type=str,
        required=True,
        help="Root directory containing CLAM-style feature files.",
    )

    parser.add_argument(
        "--encoder",
        type=str,
        default="UNI/pt_files",
        help="Encoder subdirectory, e.g., 'UNI/pt_files' or 'UNI'.",
    )

    parser.add_argument(
        "--patient-id",
        type=str,
        default=None,
        help="Patient/slide ID to test. If omitted, the first discovered patient is used.",
    )

    parser.add_argument(
        "--tile-dir",
        type=str,
        default=None,
        help="Optional CLAM patch directory containing .h5 coordinate files.",
    )

    args = parser.parse_args()

    patient_ids = discover_patients(
        feature_root=args.feature_root,
        encoder=args.encoder,
        format_type="clam",
        verbose=True,
    )

    if args.patient_id is None:
        patient_id = patient_ids[0]
    else:
        patient_id = args.patient_id

    features, tile_ids, coords, is_preprocessed = load_patient_features(
        feature_root=args.feature_root,
        encoder=args.encoder,
        patient_id=patient_id,
        tile_dir=args.tile_dir,
        format_type="clam",
        verbose=True,
    )

    print("\n[Test Summary]")
    print(f"  Patient ID      : {patient_id}")
    print(f"  Features shape  : {features.shape}")
    print(f"  Number tile IDs : {len(tile_ids)}")
    print(f"  Coords shape    : {None if coords is None else coords.shape}")
    print(f"  Is preprocessed : {is_preprocessed}")
