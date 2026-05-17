#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLAM-style feature I/O utilities for TRIAGE-MIL.

This public release supports the manuscript setting:
CLAM tissue masking / patch extraction + UNI feature embeddings saved as .pt files.

Expected layouts
----------------
feature_root/
└── UNI/
    └── pt_files/
        ├── patient_001.pt
        └── patient_002.pt

or:

feature_root/
└── UNI/
    ├── patient_001.pt
    └── patient_002.pt

Each .pt file may be:
1) a tensor of shape [N, D], or
2) a dict with keys such as "features", "feat", or "feats", and optionally "coords".
"""

from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np


def _find_pt_file(feature_root: str, encoder: str, patient_id: str) -> Path:
    """Find a CLAM-style .pt feature file."""
    feature_root = Path(feature_root)
    encoder_path = Path(encoder)

    candidates = [
        feature_root / encoder_path / f"{patient_id}.pt",
        feature_root / encoder_path / "pt_files" / f"{patient_id}.pt",
        feature_root / "pt_files" / f"{patient_id}.pt",
        feature_root / f"{patient_id}.pt",
    ]

    for path in candidates:
        if path.exists():
            return path

    tried = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(
        f"No CLAM .pt feature file found for patient_id='{patient_id}'.\nTried:\n{tried}"
    )


def _find_h5_file(tile_dir: Optional[str], patient_id: str) -> Optional[Path]:
    """Find an optional CLAM .h5 coordinate file."""
    if tile_dir is None:
        return None

    tile_dir = Path(tile_dir)
    candidates = [
        tile_dir / "patches" / f"{patient_id}.h5",
        tile_dir / f"{patient_id}.h5",
        tile_dir / "patches" / patient_id / f"{patient_id}.h5",
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


def load_patient_features(
    feature_root: str,
    encoder: str,
    patient_id: str,
    tile_dir: Optional[str] = None,
    verbose: bool = True,
):
    """
    Load CLAM-style feature matrix and coordinates.

    Parameters
    ----------
    feature_root : str
        Root feature directory.
    encoder : str
        Encoder subdirectory, for example "UNI/pt_files" or "UNI".
    patient_id : str
        Patient or slide ID.
    tile_dir : Optional[str]
        Optional CLAM patch directory containing .h5 coordinate files.
    verbose : bool
        Print diagnostic messages.

    Returns
    -------
    features : np.ndarray
        Shape [N, D], float32.
    tile_ids : list[str]
        Generated tile identifiers.
    coords : Optional[np.ndarray]
        Shape [N, 2], int32 if available.
    is_preprocessed : bool
        Always False because CLAM features require MASS quality filtering.
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required to load .pt feature files.") from exc

    pt_file = _find_pt_file(feature_root, encoder, patient_id)
    if verbose:
        print(f"[I/O] Loading {pt_file}")

    data = torch.load(pt_file, map_location="cpu")

    features = None
    coords = None

    if isinstance(data, dict):
        for key in ("features", "feat", "feats"):
            if key in data:
                features = data[key]
                break

        if features is None:
            for key, value in data.items():
                if key == "coords":
                    continue
                if isinstance(value, torch.Tensor):
                    features = value
                    break

        if "coords" in data and data["coords"] is not None:
            coords = data["coords"]
    else:
        features = data

    if features is None:
        raise ValueError(
            f"No features found in {pt_file}. Expected key 'features', 'feat', or 'feats'."
        )

    if isinstance(features, torch.Tensor):
        features = features.cpu().numpy()
    else:
        features = np.asarray(features)

    features = features.astype(np.float32)

    if features.ndim == 1:
        features = features[None, :]
    if features.ndim != 2:
        raise ValueError(f"Expected features with shape [N, D], got {features.shape}")

    n_tiles = features.shape[0]

    if coords is not None:
        if isinstance(coords, torch.Tensor):
            coords = coords.cpu().numpy()
        else:
            coords = np.asarray(coords)
        coords = coords.astype(np.int32)

    if coords is None and tile_dir is not None:
        h5_file = _find_h5_file(tile_dir, patient_id)
        if h5_file is not None:
            with h5py.File(h5_file, "r") as f:
                if "coords" in f:
                    coords = np.asarray(f["coords"][:]).astype(np.int32)

    if coords is not None:
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(f"Coordinates must have shape [N, 2], got {coords.shape}")
        if coords.shape[0] != n_tiles:
            min_len = min(coords.shape[0], n_tiles)
            if verbose:
                print(
                    f"[I/O] Warning: feature/coord mismatch for {patient_id}: "
                    f"{n_tiles} features vs {coords.shape[0]} coords. Truncating to {min_len}."
                )
            features = features[:min_len]
            coords = coords[:min_len]
            n_tiles = min_len

    tile_ids = [f"{patient_id}_tile_{i}" for i in range(n_tiles)]
    is_preprocessed = False

    if verbose:
        coord_msg = None if coords is None else coords.shape
        print(f"[I/O] Loaded features={features.shape}, coords={coord_msg}")

    return features, tile_ids, coords, is_preprocessed


def discover_patients(
    feature_root: str,
    encoder: str,
    verbose: bool = True,
) -> List[str]:
    """
    Discover patient/slide IDs from CLAM-style .pt files.
    """
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
        tried = "\n".join(f"  - {p}" for p in candidate_dirs)
        raise FileNotFoundError(f"No .pt feature files found. Tried:\n{tried}")

    patient_ids = sorted([p.stem for p in pt_files])

    if verbose:
        print(f"[I/O] Found {len(patient_ids)} patient/slide feature files in {pt_files[0].parent}")

    return patient_ids
