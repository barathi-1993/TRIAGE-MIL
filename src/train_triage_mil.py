#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TRIAGE-MIL training script.

Manuscript-aligned components:
- CLAM + UNI .pt features
- MASS-selected fixed-budget tile subset
- semantic hierarchical hypergraph message passing
- multi-head gated attention pooling
- Cox partial likelihood survival training

Camera-ready evaluation protocol:
- Patient-level 5-fold cross-validation
- 60% train / 20% validation / 20% held-out test per fold
- Validation split is used only for checkpoint selection / early stopping
- Held-out test split is used only for final C-index and prediction export
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lifelines.utils import concordance_index
from torch.utils.data import DataLoader, Dataset

from mass_io_utils import load_patient_features


# -------------------------------------------------------------------------
# Reproducibility and survival utilities
# -------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def c_index(times, events, risks):
    return concordance_index(times, -risks, events)


def cox_ph_loss(risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(time, descending=True)
    risk = risk[order]
    event = event[order]

    log_risk_sum = torch.logcumsumexp(risk, dim=0)
    log_lik = risk - log_risk_sum

    n_events = (event == 1).sum()
    if n_events == 0:
        return risk.sum() * 0.0

    return -(log_lik[event == 1].mean())


# -------------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------------

class CaseBagDataset(Dataset):
    """Dataset for MASS-selected CLAM/UNI feature bags."""

    def __init__(
        self,
        cfg: dict,
        labels_df: pd.DataFrame,
        case_ids: List[str],
    ):
        self.cfg = cfg
        self.df = labels_df.copy()
        self.case_ids = [str(x) for x in case_ids]

        self.id_col = cfg.get("id_col", "patient_id")
        self.time_col = cfg.get("time_col", "survival_time")
        self.event_col = cfg.get("event_col", "survival_event")

        self.feature_root = cfg["feature_root"]
        self.encoder = cfg.get("selected_encoder", "UNI/pt_files")
        self.tile_dir = cfg.get("tile_dir", None)

        self.cache_dir = Path(cfg["precompute_cache"]["cache_dir"])
        hcfg = cfg.get("semantic_hierarchy_config", {})
        self.hypergraph_tag = (
            f"k{int(hcfg.get('k_intra', 10))}"
            f"_i{int(hcfg.get('k_inter', 5))}"
            f"_t{int(hcfg.get('tiles_per_super', 12))}"
        )
        self.hypergraph_dir = self.cache_dir / "hypergraphs"
        self.max_tiles = cfg.get("max_tiles", 4096)

        self.df = self.df[self.df[self.id_col].astype(str).isin(self.case_ids)].copy()
        self.df[self.id_col] = self.df[self.id_col].astype(str)
        self.row_map = {str(r[self.id_col]): r for _, r in self.df.iterrows()}

        missing = sorted(set(self.case_ids) - set(self.row_map.keys()))
        if missing:
            raise ValueError(f"Some case_ids are missing from labels_df: {missing[:10]}")

    def __len__(self):
        return len(self.case_ids)

    def _load_hypergraph(self, pid: str):
        path = self.hypergraph_dir / f"{pid}_H_{self.hypergraph_tag}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing precomputed hypergraph: {path}")

        data = torch.load(path, map_location="cpu", weights_only=False)
        H = data["H"]

        if H.is_sparse:
            H = H.coalesce()

        return H

    def __getitem__(self, idx: int):
        pid = self.case_ids[idx]
        row = self.row_map[pid]

        X_all, _, coords_all, _ = load_patient_features(
            self.feature_root,
            self.encoder,
            pid,
            tile_dir=self.tile_dir,
            verbose=False,
        )

        idx_path = self.cache_dir / f"{pid}_topk_idx.npy"
        label_path = self.cache_dir / f"{pid}_topk_labels.npy"
        risk_path = self.cache_dir / f"{pid}_topk_risks.npy"

        if not idx_path.exists():
            raise FileNotFoundError(f"Missing MASS index file: {idx_path}")

        sel = np.load(idx_path)
        sel = sel[sel < len(X_all)]

        if self.max_tiles is not None:
            sel = sel[: int(self.max_tiles)]

        X = X_all[sel].astype(np.float32)

        if coords_all is None:
            coords = np.zeros((len(sel), 2), dtype=np.float32)
        else:
            coords = coords_all[sel].astype(np.float32)

        mass_labels = (
            np.load(label_path).astype(np.int64)[: len(sel)]
            if label_path.exists()
            else np.zeros(len(sel), dtype=np.int64)
        )

        risk_scores = (
            np.load(risk_path).astype(np.float32)[: len(sel)]
            if risk_path.exists()
            else np.zeros(len(sel), dtype=np.float32)
        )

        H = self._load_hypergraph(pid)

        if H.shape[0] != X.shape[0]:
            raise RuntimeError(
                f"H/X mismatch for {pid}: H has {H.shape[0]} nodes, "
                f"but X has {X.shape[0]} selected tiles."
            )

        return {
            "id": pid,
            "features": torch.from_numpy(X).float(),
            "coords": torch.from_numpy(coords).float(),
            "mass_labels": torch.from_numpy(mass_labels).long(),
            "risk_scores": torch.from_numpy(risk_scores).float(),
            "H": H,
            "time": torch.tensor(float(row[self.time_col]), dtype=torch.float32),
            "event": torch.tensor(int(row[self.event_col]), dtype=torch.long),
        }


def collate_bags(batch):
    return batch


# -------------------------------------------------------------------------
# Model
# -------------------------------------------------------------------------

class HypergraphGatedMILSurvival(nn.Module):
    """Semantic hypergraph MIL with multi-head gated attention."""

    def __init__(
        self,
        feat_dim: int,
        hidden: int = 512,
        heads: int = 6,
        num_hyper_layers: int = 4,
        attn_dropout: float = 0.22,
        learnable_temp: bool = True,
    ):
        super().__init__()

        self.feat_dim = feat_dim
        self.hidden = hidden
        self.heads = heads
        self.num_hyper_layers = num_hyper_layers
        self.learnable_temp = learnable_temp

        self.theta_layers = nn.ModuleList(
            [nn.Linear(feat_dim, hidden) for _ in range(num_hyper_layers)]
        )
        self.phi_layers = nn.ModuleList(
            [nn.Linear(hidden, feat_dim) for _ in range(num_hyper_layers)]
        )
        self.norm_layers = nn.ModuleList(
            [nn.LayerNorm(feat_dim) for _ in range(num_hyper_layers)]
        )

        self.V = nn.ModuleList([nn.Linear(feat_dim, hidden) for _ in range(heads)])
        self.U = nn.ModuleList([nn.Linear(feat_dim, hidden) for _ in range(heads)])
        self.w = nn.ModuleList([nn.Linear(hidden, 1, bias=False) for _ in range(heads)])

        self.dropout = nn.Dropout(attn_dropout)
        self.fuse = nn.Linear(heads * feat_dim, feat_dim)
        self.norm_bag = nn.LayerNorm(feat_dim)
        self.cox_head = nn.Linear(feat_dim, 1)

        if learnable_temp:
            self.attention_temp = nn.Parameter(torch.ones(heads))

    def hypergraph_mp(self, X: torch.Tensor, H: Optional[torch.Tensor]):
        if H is None or H.numel() == 0:
            return X

        X_out = X
        H = H.float()

        if not H.is_sparse:
            H = H.to_sparse_coo().coalesce()

        for i in range(self.num_hyper_layers):
            He = H.transpose(0, 1).coalesce()

            deg_e = torch.sparse.sum(He, dim=1).to_dense().clamp(min=1).unsqueeze(1)
            deg_v = torch.sparse.sum(H, dim=1).to_dense().clamp(min=1).unsqueeze(1)

            E = torch.sparse.mm(He, torch.tanh(self.theta_layers[i](X_out))) / deg_e
            X_up = torch.sparse.mm(H, torch.tanh(self.phi_layers[i](E))) / deg_v
            X_out = self.norm_layers[i](X_out + X_up)

        return X_out

    def forward(self, X: torch.Tensor, H: Optional[torch.Tensor] = None, return_attn: bool = False):
        X = self.hypergraph_mp(X, H)

        head_embeds = []
        head_weights = []

        for i, (V, U, w) in enumerate(zip(self.V, self.U, self.w)):
            z = torch.tanh(V(X)) * torch.sigmoid(U(X))
            scores = w(self.dropout(z)).squeeze(-1)

            if self.learnable_temp:
                temp = torch.clamp(self.attention_temp[i], min=0.1, max=5.0)
                attn = torch.softmax(scores / temp, dim=0)
            else:
                attn = torch.softmax(scores, dim=0)

            bag = torch.matmul(attn.unsqueeze(0), X).squeeze(0)
            head_embeds.append(bag)
            head_weights.append(attn)

        Z = self.norm_bag(self.fuse(torch.cat(head_embeds, dim=-1)))
        risk = self.cox_head(Z).squeeze(-1)

        if return_attn:
            return risk, head_weights

        return risk


# -------------------------------------------------------------------------
# Split handling: manuscript-aligned train/val/test protocol
# -------------------------------------------------------------------------

def _make_stratification_labels(df: pd.DataFrame, cfg: dict) -> np.ndarray:
    """Create event + time-quartile labels for stratified splitting."""
    time_col = cfg.get("time_col", "survival_time")
    event_col = cfg.get("event_col", "survival_event")

    events = df[event_col].astype(int).values
    times = df[time_col].astype(float)

    try:
        time_bins = pd.qcut(times, q=4, labels=False, duplicates="drop")
        time_bins = pd.Series(time_bins).fillna(0).astype(int).values
    except Exception:
        time_bins = np.zeros(len(df), dtype=int)

    return np.array([f"{e}_{b}" for e, b in zip(events, time_bins)])


def _safe_stratified_outer_split(ids: List[str], strat_labels: np.ndarray, cfg: dict, fold: int):
    """Return train_val IDs and test IDs using StratifiedKFold when possible."""
    from sklearn.model_selection import KFold, StratifiedKFold

    k_folds = int(cfg.get("k_folds", 5))
    seed = int(cfg.get("random_seed", 35))

    ids_arr = np.asarray(ids)

    try:
        splitter = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
        splits = list(splitter.split(ids_arr, strat_labels))
    except Exception:
        splitter = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
        splits = list(splitter.split(ids_arr))

    train_val_idx, test_idx = splits[fold]
    train_val_ids = ids_arr[train_val_idx].tolist()
    test_ids = ids_arr[test_idx].tolist()
    train_val_labels = strat_labels[train_val_idx]

    return train_val_ids, test_ids, train_val_labels


def _safe_train_val_split(train_val_ids: List[str], train_val_labels: np.ndarray, cfg: dict, fold: int):
    """
    Split the 80% non-test patients into 60% train and 20% val overall.
    Since val is 20/80 = 0.25 of the train_val pool, test_size=0.25.
    """
    from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

    ids_arr = np.asarray(train_val_ids)
    seed = int(cfg.get("random_seed", 35)) + int(fold)

    try:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
        train_idx, val_idx = next(sss.split(ids_arr, train_val_labels))
    except Exception:
        train_idx, val_idx = train_test_split(
            np.arange(len(ids_arr)),
            test_size=0.25,
            random_state=seed,
            shuffle=True,
        )

    train_ids = ids_arr[train_idx].tolist()
    val_ids = ids_arr[val_idx].tolist()

    return train_ids, val_ids


def _validate_disjoint_splits(train_ids: List[str], val_ids: List[str], test_ids: List[str], fold: int):
    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)

    overlap_train_val = train_set & val_set
    overlap_train_test = train_set & test_set
    overlap_val_test = val_set & test_set

    if overlap_train_val or overlap_train_test or overlap_val_test:
        raise ValueError(
            f"Patient leakage detected in fold {fold}. "
            f"train∩val={len(overlap_train_val)}, "
            f"train∩test={len(overlap_train_test)}, "
            f"val∩test={len(overlap_val_test)}."
        )

    if len(train_ids) == 0 or len(val_ids) == 0 or len(test_ids) == 0:
        raise ValueError(
            f"Fold {fold} has an empty split: "
            f"train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}."
        )


def make_splits(df: pd.DataFrame, cfg: dict, fold: int) -> Tuple[List[str], List[str], List[str]]:
    """
    Load or create patient-disjoint 60/20/20 train/val/test splits.

    Preferred split-file format:
        patient_id,split
        P001,train
        P002,val
        P003,test

    If split files are not provided, a deterministic stratified 5-fold outer split is created,
    followed by a stratified train/val split inside the non-test patients.
    """
    id_col = cfg.get("id_col", "patient_id")
    split_col = cfg.get("split_col", "split")
    splits_dir = cfg.get("splits_dir", None)

    df = df.copy()
    df[id_col] = df[id_col].astype(str)

    # ------------------------------------------------------------------
    # Option 1: Load explicit fold split file
    # ------------------------------------------------------------------
    if splits_dir is not None:
        split_file = Path(splits_dir) / f"fold_{fold}.csv"

        if split_file.exists():
            s = pd.read_csv(split_file)
            s[id_col] = s[id_col].astype(str)

            if split_col not in s.columns:
                raise ValueError(
                    f"Split file {split_file} must contain a '{split_col}' column "
                    "with train/val/test labels."
                )

            s[split_col] = s[split_col].astype(str).str.lower()

            train_label = str(cfg.get("train_label", "train")).lower()
            val_label = str(cfg.get("val_label", "val")).lower()
            test_label = str(cfg.get("test_label", "test")).lower()

            train_ids = s.loc[s[split_col] == train_label, id_col].tolist()
            val_ids = s.loc[s[split_col].isin([val_label, "valid", "validation"]), id_col].tolist()
            test_ids = s.loc[s[split_col] == test_label, id_col].tolist()

            _validate_disjoint_splits(train_ids, val_ids, test_ids, fold)
            return train_ids, val_ids, test_ids

    # ------------------------------------------------------------------
    # Option 2: Deterministically create manuscript-aligned 60/20/20 split
    # ------------------------------------------------------------------
    ids = sorted(df[id_col].astype(str).tolist())
    df_sorted = df.set_index(id_col).loc[ids].reset_index()

    strat_labels = _make_stratification_labels(df_sorted, cfg)

    train_val_ids, test_ids, train_val_labels = _safe_stratified_outer_split(
        ids=ids,
        strat_labels=strat_labels,
        cfg=cfg,
        fold=fold,
    )

    train_ids, val_ids = _safe_train_val_split(
        train_val_ids=train_val_ids,
        train_val_labels=train_val_labels,
        cfg=cfg,
        fold=fold,
    )

    _validate_disjoint_splits(train_ids, val_ids, test_ids, fold)
    return train_ids, val_ids, test_ids


# -------------------------------------------------------------------------
# Training / evaluation
# -------------------------------------------------------------------------

def infer_feat_dim(cfg: dict, case_id: str):
    X, _, _, _ = load_patient_features(
        cfg["feature_root"],
        cfg.get("selected_encoder", "UNI/pt_files"),
        case_id,
        tile_dir=cfg.get("tile_dir", None),
        verbose=False,
    )
    return X.shape[1]


def train_one_epoch(model, loader, optimizer, device, cfg):
    model.train()

    use_amp = bool(cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    accum_steps = int(cfg.get("gradient_accumulation_steps", 1))

    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    n_steps = 0

    pending_risk = []
    pending_time = []
    pending_event = []

    for batch_idx, items in enumerate(loader):
        for sample in items:
            X = sample["features"].to(device)
            H = sample["H"].to(device)
            t = sample["time"].to(device).view(1)
            e = sample["event"].to(device).view(1)

            with torch.cuda.amp.autocast(enabled=use_amp):
                r = model(X, H).view(1)

            pending_risk.append(r)
            pending_time.append(t)
            pending_event.append(e)

        do_step = ((batch_idx + 1) % accum_steps == 0) or (batch_idx + 1 == len(loader))

        if do_step:
            risks = torch.cat(pending_risk)
            times = torch.cat(pending_time)
            events = torch.cat(pending_event)

            if (events == 1).sum() > 0:
                with torch.cuda.amp.autocast(enabled=False):
                    loss = cox_ph_loss(risks.float(), times.float(), events.long())

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                total_loss += float(loss.detach().cpu())
                n_steps += 1

            pending_risk = []
            pending_time = []
            pending_event = []

    return total_loss / max(1, n_steps)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    rows = []

    for items in loader:
        for sample in items:
            X = sample["features"].to(device)
            H = sample["H"].to(device)

            risk = model(X, H).detach().cpu().item()

            rows.append(
                {
                    "patient_id": sample["id"],
                    "time": float(sample["time"].item()),
                    "event": int(sample["event"].item()),
                    "risk": float(risk),
                }
            )

    pred_df = pd.DataFrame(rows)

    if len(pred_df) > 0 and pred_df["event"].sum() > 0 and pred_df["risk"].nunique() > 1:
        ci = c_index(pred_df["time"].values, pred_df["event"].values, pred_df["risk"].values)
    else:
        ci = np.nan

    return ci, pred_df


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_bags,
    )


def main():
    parser = argparse.ArgumentParser(description="Train TRIAGE-MIL.")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--fold", default=0, type=int)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = json.load(f)

    set_seed(int(cfg.get("random_seed", 35)) + int(args.fold))

    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")

    labels = pd.read_csv(cfg["csv_labels"])
    id_col = cfg.get("id_col", "patient_id")
    labels[id_col] = labels[id_col].astype(str)

    train_ids, val_ids, test_ids = make_splits(labels, cfg, args.fold)

    feat_dim = infer_feat_dim(cfg, train_ids[0])

    print(
        f"[Train] Fold {args.fold}: "
        f"train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}, "
        f"feat_dim={feat_dim}"
    )

    # Save the exact split used for traceability
    save_dir = Path(cfg.get("save_dir", "./results/TRIAGE_MIL_CLAM_UNI")) / f"fold_{args.fold}"
    save_dir.mkdir(parents=True, exist_ok=True)

    split_df = pd.DataFrame(
        [{"patient_id": x, "split": "train"} for x in train_ids]
        + [{"patient_id": x, "split": "val"} for x in val_ids]
        + [{"patient_id": x, "split": "test"} for x in test_ids]
    )
    split_df.to_csv(save_dir / "split_used.csv", index=False)

    # Datasets
    train_ds = CaseBagDataset(cfg, labels, train_ids)
    val_ds = CaseBagDataset(cfg, labels, val_ids)
    test_ds = CaseBagDataset(cfg, labels, test_ids)

    # Loaders
    train_loader = make_loader(
        train_ds,
        batch_size=int(cfg.get("batch_size", 4)),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 0)),
    )

    train_eval_loader = make_loader(
        train_ds,
        batch_size=1,
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 0)),
    )

    val_loader = make_loader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 0)),
    )

    test_loader = make_loader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 0)),
    )

    # Model
    model = HypergraphGatedMILSurvival(
        feat_dim=feat_dim,
        hidden=int(cfg.get("hidden", 512)),
        heads=int(cfg.get("heads", 6)),
        num_hyper_layers=int(cfg.get("num_hyper_layers", 4)),
        attn_dropout=float(cfg.get("attn_dropout", 0.22)),
        learnable_temp=bool(cfg.get("learnable_temp", True)),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("lr", 2e-4)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )

    best_val_ci = -np.inf
    patience = int(cfg.get("early_patience", 40))
    min_epochs = int(cfg.get("early_min_epochs", 60))
    max_epochs = int(cfg.get("epochs", 200))

    bad_epochs = 0
    logs = []

    best_path = save_dir / "best_model.pt"

    # ------------------------------------------------------------------
    # Training: validation is used only for checkpoint selection
    # ------------------------------------------------------------------
    for epoch in range(1, max_epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, device, cfg)
        val_ci, val_pred_df = evaluate(model, val_loader, device)

        logs.append(
            {
                "epoch": epoch,
                "loss": loss,
                "val_cindex": val_ci,
                "best_val_cindex_so_far": best_val_ci if best_val_ci > -np.inf else np.nan,
            }
        )

        print(
            f"[Fold {args.fold}] "
            f"epoch={epoch:03d} loss={loss:.4f} val_cindex={val_ci:.4f}"
        )

        if not np.isnan(val_ci) and val_ci > best_val_ci:
            best_val_ci = val_ci
            bad_epochs = 0

            torch.save(
                {
                    "model": model.state_dict(),
                    "cfg": cfg,
                    "fold": args.fold,
                    "best_val_cindex": float(best_val_ci),
                    "epoch": epoch,
                },
                best_path,
            )

            # For traceability only; this is not final test performance.
            val_pred_df.to_csv(save_dir / "val_predictions_at_best_checkpoint.csv", index=False)

        else:
            bad_epochs += 1

        pd.DataFrame(logs).to_csv(save_dir / "training_log.csv", index=False)

        if bool(cfg.get("early_stopping", True)) and epoch >= min_epochs and bad_epochs >= patience:
            print(f"[Fold {args.fold}] Early stopping at epoch {epoch}.")
            break

    if not best_path.exists():
        raise RuntimeError(f"No best checkpoint was saved for fold {args.fold}.")

    # ------------------------------------------------------------------
    # Final evaluation: held-out test set only
    # ------------------------------------------------------------------
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])

    train_ci, train_pred_df = evaluate(model, train_eval_loader, device)
    val_ci_final, val_pred_df_final = evaluate(model, val_loader, device)
    test_ci, test_pred_df = evaluate(model, test_loader, device)

    train_pred_df.to_csv(save_dir / "train_predictions.csv", index=False)
    val_pred_df_final.to_csv(save_dir / "val_predictions_final_checkpoint.csv", index=False)
    test_pred_df.to_csv(save_dir / "test_predictions.csv", index=False)

    metrics = {
        "fold": int(args.fold),
        "n_train": int(len(train_ids)),
        "n_val": int(len(val_ids)),
        "n_test": int(len(test_ids)),
        "best_val_cindex": float(best_val_ci),
        "train_cindex_at_best_checkpoint": None if np.isnan(train_ci) else float(train_ci),
        "val_cindex_at_best_checkpoint": None if np.isnan(val_ci_final) else float(val_ci_final),
        "test_cindex_at_best_checkpoint": None if np.isnan(test_ci) else float(test_ci),
        "best_checkpoint_epoch": int(checkpoint.get("epoch", -1)),
    }

    with open(save_dir / "metrics_summary.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[Fold {args.fold}] Best validation C-index: {best_val_ci:.4f}")
    print(f"[Fold {args.fold}] Final held-out test C-index: {test_ci:.4f}")
    print(f"[Fold {args.fold}] Saved outputs to: {save_dir}")


if __name__ == "__main__":
    main()
