#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Precompute semantic hierarchical hypergraphs for TRIAGE-MIL.

Manuscript-aligned behavior:
- MASS labels define semantic groups: MH, TR, MI, TD.
- For each semantic group, tiles are clustered in embedding space using k-means
  to form super-nodes.
- Hyperedges represent:
  1) tile-to-super-node containment,
  2) intra-semantic super-node similarity,
  3) inter-semantic interactions across MASS semantic groups.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from mass_io_utils import discover_patients, load_patient_features


def _load_config(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _safe_kmeans(X: np.ndarray, n_clusters: int, random_state: int = 35) -> np.ndarray:
    """Run k-means with safe handling of small groups."""
    n = X.shape[0]
    if n == 0:
        return np.array([], dtype=np.int64)
    if n_clusters <= 1 or n <= 1:
        return np.zeros(n, dtype=np.int64)

    n_clusters = min(n_clusters, n)
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    return km.fit_predict(X).astype(np.int64)


def build_semantic_hierarchical_hypergraph(
    X: np.ndarray,
    coords: np.ndarray,
    mass_labels: np.ndarray,
    k_intra: int = 10,
    k_inter: int = 5,
    tiles_per_super: int = 12,
    random_state: int = 35,
) -> Tuple[torch.Tensor, Dict]:
    """
    Build manuscript-aligned semantic hierarchical hypergraph.

    Parameters
    ----------
    X : np.ndarray
        Selected tile embeddings [K, D].
    coords : np.ndarray
        Selected tile coordinates [K, 2]. Coordinates are stored for metadata
        but super-node clustering is performed in embedding space.
    mass_labels : np.ndarray
        MASS semantic labels [K], where 0=MH, 1=TR, 2=MI, 3=TD.
    k_intra : int
        Number of intra-semantic super-node neighbors.
    k_inter : int
        Number of inter-semantic super-node neighbors.
    tiles_per_super : int
        Approximate number of tiles per semantic super-node.
    """
    X = np.asarray(X, dtype=np.float32)
    mass_labels = np.asarray(mass_labels, dtype=np.int64)

    N, D = X.shape
    if N == 0:
        raise ValueError("Cannot build hypergraph for zero selected tiles.")

    tile_to_super = np.zeros(N, dtype=np.int64)
    super_features = []
    super_labels = []
    super_members = []
    current_super = 0

    for sem_label in range(4):
        tile_idx = np.where(mass_labels == sem_label)[0]
        n_type = len(tile_idx)
        if n_type == 0:
            continue

        n_super = max(1, int(np.ceil(n_type / float(tiles_per_super))))
        local_cluster = _safe_kmeans(X[tile_idx], n_super, random_state=random_state)

        for c in range(local_cluster.max() + 1):
            members_local = np.where(local_cluster == c)[0]
            members_global = tile_idx[members_local]
            if len(members_global) == 0:
                continue

            tile_to_super[members_global] = current_super
            super_features.append(X[members_global].mean(axis=0))
            super_labels.append(sem_label)
            super_members.append(members_global.astype(np.int64))
            current_super += 1

    S = len(super_features)
    if S == 0:
        H = torch.eye(N).to_sparse_coo()
        return H, {"n_tiles": N, "n_super_nodes": 0}

    super_features = np.stack(super_features).astype(np.float32)
    super_labels = np.asarray(super_labels, dtype=np.int64)

    hyperedge_columns = []

    # 1) Containment hyperedges: one per super-node, connecting member tiles.
    for members in super_members:
        col = np.zeros(N, dtype=np.float32)
        col[members] = 1.0
        hyperedge_columns.append(col)

    # 2) Intra-semantic hyperedges over similar super-nodes in same semantic group.
    for sem_label in range(4):
        s_idx = np.where(super_labels == sem_label)[0]
        if len(s_idx) <= 1:
            continue

        k = min(k_intra + 1, len(s_idx))
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(super_features[s_idx])
        _, neigh = nn.kneighbors(super_features[s_idx])

        for row_i, neigh_local in enumerate(neigh):
            center_super = s_idx[row_i]
            neighbor_supers = s_idx[neigh_local[1:]]
            involved_supers = np.concatenate([[center_super], neighbor_supers])

            col = np.zeros(N, dtype=np.float32)
            for s in involved_supers:
                col[super_members[s]] = 1.0
            if col.sum() > 0:
                hyperedge_columns.append(col)

    # 3) Inter-semantic hyperedges across MASS groups.
    # Constrained cross-type rules follow the biological interaction motivation:
    # MH/TR interact with MI; MI interacts with all; TD interacts mainly with MI.
    cross_rules = {
        0: [1, 2],     # MH -> TR, MI
        1: [0, 2],     # TR -> MH, MI
        2: [0, 1, 3],  # MI -> MH, TR, TD
        3: [2],        # TD -> MI
    }

    for s in range(S):
        src_label = int(super_labels[s])
        target_labels = cross_rules.get(src_label, [])
        target_supers = np.where(np.isin(super_labels, target_labels))[0]

        if len(target_supers) == 0:
            continue

        k = min(k_inter, len(target_supers))
        d = np.linalg.norm(super_features[target_supers] - super_features[s][None, :], axis=1)
        nearest = target_supers[np.argsort(d)[:k]]
        involved_supers = np.concatenate([[s], nearest])

        col = np.zeros(N, dtype=np.float32)
        for ss in involved_supers:
            col[super_members[ss]] = 1.0
        if col.sum() > 0:
            hyperedge_columns.append(col)

    H_dense = np.stack(hyperedge_columns, axis=1).astype(np.float32)
    H = torch.from_numpy(H_dense).to_sparse_coo().coalesce()

    metadata = {
        "n_tiles": int(N),
        "n_super_nodes": int(S),
        "n_hyperedges": int(H_dense.shape[1]),
        "n_containment_edges": int(S),
        "k_intra": int(k_intra),
        "k_inter": int(k_inter),
        "tiles_per_super": int(tiles_per_super),
        "tile_to_super": tile_to_super.tolist(),
        "super_labels": super_labels.tolist(),
    }

    return H, metadata


def process_patient(pid: str, cfg: dict, args):
    feature_root = cfg["feature_root"]
    encoder = cfg.get("selected_encoder", "UNI/pt_files")
    cache_dir = Path(cfg["precompute_cache"]["cache_dir"])

    idx_path = cache_dir / f"{pid}_topk_idx.npy"
    label_path = cache_dir / f"{pid}_topk_labels.npy"
    risk_path = cache_dir / f"{pid}_topk_risks.npy"

    if not idx_path.exists() or not label_path.exists():
        raise FileNotFoundError(f"Missing MASS cache for {pid} in {cache_dir}")

    idx = np.load(idx_path)
    labels = np.load(label_path)
    risks = np.load(risk_path) if risk_path.exists() else np.zeros(len(idx), dtype=np.float32)

    X_all, _, coords_all, _ = load_patient_features(
        feature_root=feature_root,
        encoder=encoder,
        patient_id=pid,
        tile_dir=cfg.get("tile_dir", None),
        verbose=False,
    )

    idx = idx[idx < len(X_all)]
    labels = labels[: len(idx)]
    risks = risks[: len(idx)]

    X = X_all[idx]
    if coords_all is None:
        coords = np.zeros((len(idx), 2), dtype=np.int32)
    else:
        coords = coords_all[idx]

    hcfg = cfg.get("semantic_hierarchy_config", {})
    k_intra = int(hcfg.get("k_intra", 10))
    k_inter = int(hcfg.get("k_inter", 5))
    tiles_per_super = int(hcfg.get("tiles_per_super", 12))

    H, meta = build_semantic_hierarchical_hypergraph(
        X=X,
        coords=coords,
        mass_labels=labels,
        k_intra=k_intra,
        k_inter=k_inter,
        tiles_per_super=tiles_per_super,
        random_state=int(cfg.get("random_seed", 35)),
    )

    out_dir = cache_dir / "hypergraphs"
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = f"k{k_intra}_i{k_inter}_t{tiles_per_super}"
    out_path = out_dir / f"{pid}_H_{tag}.pt"

    torch.save(
        {
            "H": H,
            "risk_scores": torch.from_numpy(risks.astype(np.float32)),
            "mass_labels": torch.from_numpy(labels.astype(np.int64)),
            "meta": meta,
        },
        out_path,
    )

    return out_path


def main():
    parser = argparse.ArgumentParser(description="Precompute TRIAGE-MIL semantic hypergraphs.")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--only-pid", default=None, type=str)
    args = parser.parse_args()

    cfg = _load_config(args.config)

    if args.only_pid:
        patients = [args.only_pid]
    else:
        patients = discover_patients(cfg["feature_root"], cfg.get("selected_encoder", "UNI/pt_files"))

    print(f"[Hypergraph] Processing {len(patients)} patients/slides.")
    for i, pid in enumerate(patients, start=1):
        print(f"[{i}/{len(patients)}] {pid}", end=" ... ", flush=True)
        try:
            out_path = process_patient(pid, cfg, args)
            print(f"saved {out_path.name}")
        except Exception as exc:
            print(f"failed: {exc}")


if __name__ == "__main__":
    main()
