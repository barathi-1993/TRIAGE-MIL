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
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from lifelines.utils import concordance_index
from torch.utils.data import DataLoader, Dataset

from mass_io_utils import load_patient_features


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
        self.case_ids = list(case_ids)

        self.id_col = cfg.get("id_col", "patient_id")
        self.time_col = cfg.get("time_col", "survival_time")
        self.event_col = cfg.get("event_col", "survival_event")

        self.feature_root = cfg["feature_root"]
        self.encoder = cfg.get("selected_encoder", "UNI/pt_files")
        self.tile_dir = cfg.get("tile_dir", None)

        self.cache_dir = Path(cfg["precompute_cache"]["cache_dir"])
        hcfg = cfg.get("semantic_hierarchy_config", {})
        self.hypergraph_tag = f"k{int(hcfg.get('k_intra', 10))}_i{int(hcfg.get('k_inter', 5))}_t{int(hcfg.get('tiles_per_super', 12))}"
        self.hypergraph_dir = self.cache_dir / "hypergraphs"
        self.max_tiles = cfg.get("max_tiles", 4096)

        self.df = self.df[self.df[self.id_col].astype(str).isin(self.case_ids)].copy()
        self.df[self.id_col] = self.df[self.id_col].astype(str)
        self.row_map = {str(r[self.id_col]): r for _, r in self.df.iterrows()}

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
                f"H/X mismatch for {pid}: H has {H.shape[0]} nodes, X has {X.shape[0]} tiles."
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

        self.theta_layers = nn.ModuleList([nn.Linear(feat_dim, hidden) for _ in range(num_hyper_layers)])
        self.phi_layers = nn.ModuleList([nn.Linear(hidden, feat_dim) for _ in range(num_hyper_layers)])
        self.norm_layers = nn.ModuleList([nn.LayerNorm(feat_dim) for _ in range(num_hyper_layers)])

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


def make_splits(df: pd.DataFrame, cfg: dict, fold: int):
    """Use provided split file when available; otherwise make deterministic K-fold splits."""
    id_col = cfg.get("id_col", "patient_id")
    splits_dir = cfg.get("splits_dir", None)

    if splits_dir is not None:
        split_file = Path(splits_dir) / f"fold_{fold}.csv"
        if split_file.exists():
            s = pd.read_csv(split_file)
            s[id_col] = s[id_col].astype(str)
            if "split" in s.columns:
                train_ids = s.loc[s["split"].str.lower() == "train", id_col].tolist()
                val_ids = s.loc[s["split"].str.lower().isin(["val", "valid", "validation", "test"]), id_col].tolist()
                return train_ids, val_ids
            if "train" in s.columns:
                train_ids = s.loc[s["train"] == 1, id_col].tolist()
                val_ids = s.loc[s["train"] == 0, id_col].tolist()
                return train_ids, val_ids

    from sklearn.model_selection import KFold

    ids = sorted(df[id_col].astype(str).tolist())
    k_folds = int(cfg.get("k_folds", 5))
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=int(cfg.get("random_seed", 35)))
    splits = list(kf.split(ids))
    train_idx, val_idx = splits[fold]
    train_ids = [ids[i] for i in train_idx]
    val_ids = [ids[i] for i in val_idx]
    return train_ids, val_ids


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

    pending_risk, pending_time, pending_event = [], [], []

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

            pending_risk, pending_time, pending_event = [], [], []

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
    if pred_df["event"].sum() > 0 and pred_df["risk"].nunique() > 1:
        ci = c_index(pred_df["time"].values, pred_df["event"].values, pred_df["risk"].values)
    else:
        ci = np.nan

    return ci, pred_df


def main():
    parser = argparse.ArgumentParser(description="Train TRIAGE-MIL.")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--fold", default=0, type=int)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = json.load(f)

    set_seed(int(cfg.get("random_seed", 35)) + args.fold)

    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")

    labels = pd.read_csv(cfg["csv_labels"])
    id_col = cfg.get("id_col", "patient_id")
    labels[id_col] = labels[id_col].astype(str)

    train_ids, val_ids = make_splits(labels, cfg, args.fold)

    feat_dim = infer_feat_dim(cfg, train_ids[0])
    print(f"[Train] Fold {args.fold}: train={len(train_ids)}, val={len(val_ids)}, feat_dim={feat_dim}")

    train_ds = CaseBagDataset(cfg, labels, train_ids)
    val_ds = CaseBagDataset(cfg, labels, val_ids)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.get("batch_size", 4)),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 0)),
        collate_fn=collate_bags,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 0)),
        collate_fn=collate_bags,
    )

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

    save_dir = Path(cfg.get("save_dir", "./results/TRIAGE_MIL_CLAM_UNI")) / f"fold_{args.fold}"
    save_dir.mkdir(parents=True, exist_ok=True)

    best_ci = -np.inf
    patience = int(cfg.get("early_patience", 40))
    min_epochs = int(cfg.get("early_min_epochs", 60))
    bad_epochs = 0
    logs = []

    for epoch in range(1, int(cfg.get("epochs", 200)) + 1):
        loss = train_one_epoch(model, train_loader, optimizer, device, cfg)
        val_ci, pred_df = evaluate(model, val_loader, device)

        logs.append({"epoch": epoch, "loss": loss, "val_cindex": val_ci})
        print(f"[Fold {args.fold}] epoch={epoch:03d} loss={loss:.4f} val_cindex={val_ci:.4f}")

        if not np.isnan(val_ci) and val_ci > best_ci:
            best_ci = val_ci
            bad_epochs = 0
            torch.save({"model": model.state_dict(), "cfg": cfg, "best_cindex": best_ci}, save_dir / "best_model.pt")
            pred_df.to_csv(save_dir / "predictions.csv", index=False)
        else:
            bad_epochs += 1

        pd.DataFrame(logs).to_csv(save_dir / "training_log.csv", index=False)

        if bool(cfg.get("early_stopping", True)) and epoch >= min_epochs and bad_epochs >= patience:
            print(f"[Fold {args.fold}] Early stopping at epoch {epoch}.")
            break

    print(f"[Fold {args.fold}] Best validation C-index: {best_ci:.4f}")


if __name__ == "__main__":
    main()
