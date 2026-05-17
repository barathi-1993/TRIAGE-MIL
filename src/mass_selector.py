#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MASS tile selector for TRIAGE-MIL.

This script is aligned with the manuscript:
1) CLAM-style .pt features are loaded.
2) Feature-space quality filtering is applied.
3) A fixed budget K is selected through four phenotype-inspired axes:
   MH, TR, MI, TD.
4) The selected indices, semantic labels, risk/proxy scores, and metadata are saved.

Outputs per patient/slide
-------------------------
{pid}_topk_idx.npy
{pid}_topk_labels.npy
{pid}_topk_risks.npy
{pid}_mass_metadata.json

Semantic labels
---------------
0 = MH: Morphologic Heterogeneity
1 = TR: Tissue Regularity
2 = MI: Microenvironmental Interface
3 = TD: Tissue Diversity
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from mass_io_utils import discover_patients, load_patient_features
from mass_core import (
    MASSConfig,
    compute_mass_axis_scores,
    farthest_point_sampling,
    quality_filter_embeddings,
)


def _budget_from_fractions(K: int, cfg: MASSConfig) -> Dict[str, int]:
    """Convert fractions into exact integer budgets that sum to K."""
    budgets = {
        "MH": int(round(K * cfg.mh_fraction)),
        "TR": int(round(K * cfg.tr_fraction)),
        "MI": int(round(K * cfg.mi_fraction)),
    }
    budgets["TD"] = max(K - budgets["MH"] - budgets["TR"] - budgets["MI"], 0)

    # Correct rounding drift.
    total = sum(budgets.values())
    if total != K:
        budgets["TD"] += K - total

    return budgets


def select_mass_tiles(
    X: np.ndarray,
    coords: Optional[np.ndarray],
    K: int,
    cfg: MASSConfig,
    verbose: bool = True,
):
    """
    Select a fixed-size subset using manuscript-aligned MASS.

    Selection is sequential and mutually exclusive:
    MH -> TR -> MI -> TD.
    """
    N = X.shape[0]
    if N == 0:
        raise ValueError("Cannot run MASS on an empty feature matrix.")

    keep_mask, quality_scores, filter_meta = quality_filter_embeddings(X, cfg, verbose=verbose)
    clean_indices = np.where(keep_mask)[0]

    if len(clean_indices) == 0:
        raise RuntimeError("Quality filtering removed all tiles.")

    K_eff = min(K, len(clean_indices))
    X_clean = X[clean_indices]
    coords_clean = coords[clean_indices] if coords is not None else None

    scores = compute_mass_axis_scores(X_clean, coords_clean, cfg)
    budgets = _budget_from_fractions(K_eff, cfg)

    remaining = np.arange(len(clean_indices), dtype=np.int64)
    selected_local = []
    labels = []

    def take_top(score_name: str, budget: int, label: int):
        nonlocal remaining, selected_local, labels
        if budget <= 0 or len(remaining) == 0:
            return
        n = min(budget, len(remaining))
        vals = scores[score_name][remaining]
        chosen_local_positions = np.argsort(-vals)[:n]
        chosen = remaining[chosen_local_positions]

        selected_local.extend(chosen.tolist())
        labels.extend([label] * len(chosen))

        selected_set = set(chosen.tolist())
        remaining = np.asarray([idx for idx in remaining if idx not in selected_set], dtype=np.int64)

    take_top("MH", budgets["MH"], 0)
    take_top("TR", budgets["TR"], 1)
    take_top("MI", budgets["MI"], 2)

    # TD via feature-space farthest-point sampling from remaining candidates.
    if budgets["TD"] > 0 and len(remaining) > 0:
        td_chosen = farthest_point_sampling(
            X_clean,
            candidate_indices=remaining,
            n_select=min(budgets["TD"], len(remaining)),
            start_scores=scores["MH"],
        )
        selected_local.extend(td_chosen.tolist())
        labels.extend([3] * len(td_chosen))

    selected_local = np.asarray(selected_local, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.uint8)

    # If K was not filled due to edge cases, fill by quality score.
    if len(selected_local) < K_eff:
        already = set(selected_local.tolist())
        fill_candidates = np.asarray(
            [i for i in range(len(clean_indices)) if i not in already],
            dtype=np.int64,
        )
        n_fill = min(K_eff - len(selected_local), len(fill_candidates))
        if n_fill > 0:
            fill_vals = quality_scores[clean_indices[fill_candidates]]
            fill_local = fill_candidates[np.argsort(-fill_vals)[:n_fill]]
            selected_local = np.concatenate([selected_local, fill_local])
            labels = np.concatenate([labels, np.full(n_fill, 3, dtype=np.uint8)])

    selected_local = selected_local[:K_eff]
    labels = labels[:K_eff]

    selected_global = clean_indices[selected_local].astype(np.int32)

    # Risk/proxy score used only as a selected-tile proxy, not supervised survival risk.
    mh_norm = scores["MH"]
    mh_norm = (mh_norm - mh_norm.min()) / (mh_norm.max() - mh_norm.min() + cfg.eps)
    proxy_risk = mh_norm[selected_local].astype(np.float32)

    metadata = {
        "total_tiles": int(N),
        "clean_tiles": int(len(clean_indices)),
        "selected_tiles": int(len(selected_global)),
        "K_requested": int(K),
        "K_effective": int(K_eff),
        "axis_counts": {
            "MH": int((labels == 0).sum()),
            "TR": int((labels == 1).sum()),
            "MI": int((labels == 2).sum()),
            "TD": int((labels == 3).sum()),
        },
        "budgets": budgets,
        "filtering": filter_meta,
    }

    return selected_global, labels, proxy_risk, metadata


def process_patient(pid: str, args, cfg: MASSConfig):
    """Run MASS for one patient/slide and save outputs."""
    try:
        X, _, coords, _ = load_patient_features(
            feature_root=args.feature_root,
            encoder=args.encoder,
            patient_id=pid,
            tile_dir=args.tile_dir,
            verbose=False,
        )

        idx, labels, risks, meta = select_mass_tiles(
            X=X,
            coords=coords,
            K=args.topk,
            cfg=cfg,
            verbose=False,
        )

        out_dir = Path(args.out_cache)
        out_dir.mkdir(parents=True, exist_ok=True)

        np.save(out_dir / f"{pid}_topk_idx.npy", idx)
        np.save(out_dir / f"{pid}_topk_labels.npy", labels)
        np.save(out_dir / f"{pid}_topk_risks.npy", risks)

        with open(out_dir / f"{pid}_mass_metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        return {"patient_id": pid, "status": "ok", "selected": int(len(idx))}

    except Exception as exc:
        return {"patient_id": pid, "status": "failed", "error": str(exc)}


def main():
    parser = argparse.ArgumentParser(description="Run MASS tile selection for TRIAGE-MIL.")
    parser.add_argument("--feature-root", required=True, type=str)
    parser.add_argument("--encoder", default="UNI/pt_files", type=str)
    parser.add_argument("--tile-dir", default=None, type=str)
    parser.add_argument("--out-cache", required=True, type=str)
    parser.add_argument("--topk", default=4096, type=int)
    parser.add_argument("--only-pid", default=None, type=str)
    parser.add_argument("--mh-fraction", default=0.35, type=float)
    parser.add_argument("--tr-fraction", default=0.35, type=float)
    parser.add_argument("--mi-fraction", default=0.15, type=float)
    parser.add_argument("--td-fraction", default=0.15, type=float)
    args = parser.parse_args()

    cfg = MASSConfig(
        mh_fraction=args.mh_fraction,
        tr_fraction=args.tr_fraction,
        mi_fraction=args.mi_fraction,
        td_fraction=args.td_fraction,
    )

    if args.only_pid:
        patients = [args.only_pid]
    else:
        patients = discover_patients(args.feature_root, args.encoder, verbose=True)

    print(f"[MASS] Processing {len(patients)} patient/slide files.")
    print(f"[MASS] Output cache: {args.out_cache}")

    results = []
    for i, pid in enumerate(patients, start=1):
        print(f"[{i}/{len(patients)}] {pid}", end=" ... ", flush=True)
        res = process_patient(pid, args, cfg)
        results.append(res)
        if res["status"] == "ok":
            print(f"ok ({res['selected']} tiles)")
        else:
            print(f"failed: {res['error']}")

    out_dir = Path(args.out_cache)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "mass_selection_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    n_ok = sum(r["status"] == "ok" for r in results)
    print(f"[MASS] Completed {n_ok}/{len(results)} successfully.")


if __name__ == "__main__":
    main()
