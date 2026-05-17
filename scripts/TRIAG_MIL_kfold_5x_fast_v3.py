#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""

TRIAG-MIL_kfold_5x_fast_v3.py

=======================================

Enhanced with MASS Semantically-Aware Hierarchical Hypergraph

NEW FEATURES:

- Loads MASS semantic labels from cache (_topk_labels.npy)

- Builds semantic hierarchical hypergraph with typed super-nodes

- Inter-semantic edges model biological interactions

with Auto-resume
----------------
CUDA_VISIBLE_DEVICES=3 AUTO_RESUME=true python TRIAG-MIL_kfold_5x_fast_v3.py --config config_TRIAG-MIL_kfold_5x_earlystop-AE.json --fold 0

Usage:

CUDA_VISIBLE_DEVICES=0 python TRIAG-MIL_kfold_5x_fast_v3.py --config config_TRIAG-MIL_kfold_5x_earlystop-AE.json --fold 0 &
CUDA_VISIBLE_DEVICES=1 python TRIAG-MIL_kfold_5x_fast_v3.py --config config_TRIAG-MIL_kfold_5x_earlystop-AE.json --fold 1 &
CUDA_VISIBLE_DEVICES=2 python TRIAG-MIL_kfold_5x_fast_v3.py --config config_TRIAG-MIL_kfold_5x_earlystop-AE.json --fold 2 &
CUDA_VISIBLE_DEVICES=3 python TRIAG-MIL_kfold_5x_fast_v3.py --config config_TRIAG-MIL_kfold_5x_earlystop-AE.json --fold 3 &
CUDA_VISIBLE_DEVICES=4 python TRIAG-MIL_kfold_5x_fast_v3.py --config config_TRIAG-MIL_kfold_5x_earlystop-AE.json --fold 4 &

CUDA_VISIBLE_DEVICES=3 python TRIAG-MIL_kfold_5x_fast_v3.py --config config_TRIAG-MIL_kfold_5x_earlystop-CLAM.json --fold 0 &
CUDA_VISIBLE_DEVICES=5 python TRIAG-MIL_kfold_5x_fast_v3.py --config config_TRIAG-MIL_kfold_5x_earlystop-CLAM.json --fold 1 &
CUDA_VISIBLE_DEVICES=4 AUTO_RESUME=true python TRIAG-MIL_kfold_5x_fast_v3.py --config config_TRIAG-MIL_kfold_5x_earlystop-CLAM.json --fold 2 &
CUDA_VISIBLE_DEVICES=5 AUTO_RESUME=true python TRIAG-MIL_kfold_5x_fast_v3.py --config config_TRIAG-MIL_kfold_5x_earlystop-CLAM.json --fold 3 &
CUDA_VISIBLE_DEVICES=4 python TRIAG-MIL_kfold_5x_fast_v3.py --config config_TRIAG-MIL_kfold_5x_earlystop-CLAM.json --fold 4 &

"""

from __future__ import annotations
import os, re, glob, json, warnings, random, time, contextlib
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import h5py
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from lifelines.utils import concordance_index

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kw): return x

warnings.filterwarnings("ignore")

# Speed toggles
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# DDP helpers
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_ddp():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        global_rank = int(os.environ.get('RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        torch.cuda.set_device(local_rank)
        return True, local_rank, global_rank, world_size
    return False, 0, 0, 1

# Basics
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def c_index(times, events, risks):
    return concordance_index(times, -risks, events)


def cox_ph_loss(risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(time, descending=True)
    risk = risk[order]; event = event[order]
    log_risk_sum = torch.logcumsumexp(risk, dim=0)
    loglik = risk - log_risk_sum
    n_events = (event == 1).sum()
    if n_events == 0:
        return torch.zeros((), dtype=risk.dtype, device=risk.device, requires_grad=True) * risk.sum()
    return -(loglik[event == 1].mean())

# ═══════════════════════════════════════════════════════════════════════════
# SEMANTIC HIERARCHICAL HYPERGRAPH (STEP 3)
# ═══════════════════════════════════════════════════════════════════════════

def build_semantic_hierarchical_hypergraph_fast(
    X, coords, mass_labels, risk_scores, 
    k_intra=5, k_inter=3, tiles_per_super=16
):
    """
    GPU-Optimized Semantically-Aware Hierarchical Hypergraph.
    
    NOVELTY: Creates typed super-nodes per MASS semantic axis.
    """
    N, D = X.shape
    device = X.device
    
    # Create typed super-nodes (GPU-only)
    super_feats_list = []
    super_labels_list = []
    tile_to_super = torch.zeros(N, dtype=torch.long, device=device)
    current_super_idx = 0
    
    for sem_type in range(4):  # 0=Complexity, 1=Structure, 2=Interaction, 3=Diversity
        type_mask = (mass_labels == sem_type)
        indices = torch.where(type_mask)[0]
        n_type = len(indices)
        
        if n_type == 0:
            continue
        
        n_super = max(1, n_type // tiles_per_super)
        sub_coords = coords[indices]
        
        # Fast GPU clustering: Random anchors + Voronoi
        if n_type > n_super:
            perm = torch.randperm(n_type, device=device)[:n_super]
            anchors = sub_coords[perm]
            dists = torch.cdist(sub_coords, anchors)
            local_clusters = torch.argmin(dists, dim=1)
        else:
            local_clusters = torch.zeros(n_type, dtype=torch.long, device=device)
            n_super = 1
        
        global_cluster_ids = local_clusters + current_super_idx
        tile_to_super[indices] = global_cluster_ids
        
        # Compute super-node features (scatter mean)
        type_super_feats = torch.zeros((n_super, D), device=device)
        counts = torch.zeros(n_super, device=device)
        
        sub_X = X[indices]
        type_super_feats.index_add_(0, local_clusters, sub_X)
        counts.index_add_(0, local_clusters, torch.ones(n_type, device=device))
        type_super_feats = type_super_feats / (counts.unsqueeze(1) + 1e-6)
        
        super_feats_list.append(type_super_feats)
        super_labels_list.append(torch.full((n_super,), sem_type, device=device, dtype=torch.long))
        
        current_super_idx += n_super
    
    if not super_feats_list:
        return torch.eye(N, device=device), X[:1], {'n_super': 0}
    
    super_nodes = torch.cat(super_feats_list, dim=0)
    super_labels = torch.cat(super_labels_list, dim=0)
    K = super_nodes.shape[0]
    
    # Build hyperedges
    H_cont = torch.zeros((N, K), device=device)
    H_cont[torch.arange(N, device=device), tile_to_super] = 1.0
    
    # Intra-semantic edges
    H_intra_list = []
    super_norm = F.normalize(super_nodes, p=2, dim=1)
    
    for i in range(N):
        my_type = mass_labels[i].item()
        my_super = tile_to_super[i].item()
        
        same_type_mask = (super_labels == my_type)
        same_type_indices = torch.where(same_type_mask)[0]
        
        if len(same_type_indices) > 1:
            my_super_feat = super_norm[my_super]
            same_type_feats = super_norm[same_type_indices]
            sims = torch.matmul(same_type_feats, my_super_feat)
            sims[same_type_indices == my_super] = -1.0
            
            k_actual = min(k_intra, len(same_type_indices) - 1)
            if k_actual > 0:
                _, top_idx = torch.topk(sims, k=k_actual)
                neighbors = same_type_indices[top_idx]
                edge_vec = torch.zeros(K, device=device)
                edge_vec[neighbors] = 1.0
                H_intra_list.append(edge_vec)
            else:
                H_intra_list.append(torch.zeros(K, device=device))
        else:
            H_intra_list.append(torch.zeros(K, device=device))
    
    H_intra = torch.stack(H_intra_list, dim=0)
    
    # Inter-semantic edges (THE NOVELTY!)
    H_inter_list = []
    cross_type_rules = {
        0: [1, 2],  # Complexity → Structure, Interaction
        1: [0, 2],  # Structure → Complexity, Interaction
        2: [0, 1, 3],  # Interaction → All
        3: [2]  # Diversity → Interaction
    }
    
    for i in range(N):
        my_type = mass_labels[i].item()
        my_super_feat = super_norm[tile_to_super[i]]
        target_types = cross_type_rules.get(my_type, [])
        
        if target_types:
            target_mask = torch.zeros(K, dtype=torch.bool, device=device)
            for tt in target_types:
                target_mask |= (super_labels == tt)
            
            target_indices = torch.where(target_mask)[0]
            
            if len(target_indices) > 0:
                target_feats = super_norm[target_indices]
                sims = torch.matmul(target_feats, my_super_feat)
                k_actual = min(k_inter, len(target_indices))
                _, top_idx = torch.topk(sims, k=k_actual)
                neighbors = target_indices[top_idx]
                edge_vec = torch.zeros(K, device=device)
                edge_vec[neighbors] = 1.0
                H_inter_list.append(edge_vec)
            else:
                H_inter_list.append(torch.zeros(K, device=device))
        else:
            H_inter_list.append(torch.zeros(K, device=device))
    
    H_inter = torch.stack(H_inter_list, dim=0)
    
    # Combine & filter
    H_intra_nz = H_intra[:, H_intra.sum(dim=0) > 0]
    H_inter_nz = H_inter[:, H_inter.sum(dim=0) > 0]
    H_combined = torch.cat([H_cont, H_intra_nz, H_inter_nz], dim=1)
    
    metadata = {
        'n_tiles': N,
        'n_super_nodes': K,
        'n_containment_edges': K,
        'n_intra_edges': H_intra_nz.shape[1],
        'n_inter_edges': H_inter_nz.shape[1]
    }
    metadata.update({
    "tile_to_super": tile_to_super.detach().cpu(),
    "super_labels": super_labels.detach().cpu()
})
    
    return H_combined, super_nodes, metadata

# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING (Same as v2)
# ═══════════════════════════════════════════════════════════════════════════

def load_features_and_coords(file_path):
    """Load features and coords from .pt or .h5 file."""
    file_path = Path(file_path)
    if file_path.suffix == '.pt':
        data = torch.load(file_path, map_location='cpu')
        if isinstance(data, dict):
            features = data.get('features', data.get('feat', data.get('feats', None)))
            coords = data.get('coords', None)
        else:
            features = data
            coords = None
        if features is None:
            raise ValueError(f"No features found in {file_path}")
        if not isinstance(features, torch.Tensor):
            features = torch.from_numpy(np.asarray(features)).float()
        while features.ndim > 2:
            features = features.squeeze(0)
        if features.ndim == 1:
            features = features.unsqueeze(0)
        if coords is None:
            coords = np.zeros((features.shape[0], 2), dtype=np.int32)
        elif isinstance(coords, torch.Tensor):
            coords = coords.cpu().numpy()
        else:
            coords = np.asarray(coords)
        if coords.shape[0] != features.shape[0]:
            m = min(coords.shape[0], features.shape[0])
            features = features[:m]
            coords = coords[:m]
        return features.numpy().astype(np.float32), coords.astype(np.int32)
    elif file_path.suffix == '.h5':
        with h5py.File(file_path, 'r') as f:
            feats_key = 'features' if 'features' in f else list(f.keys())[0]
            features = np.array(f[feats_key][:]).astype(np.float32)
            coords = np.array(f['coords'][:]).astype(np.int32) if 'coords' in f else np.zeros((features.shape[0], 2), dtype=np.int32)
            while features.ndim > 2:
                features = features.squeeze(0)
        return features, coords
    else:
        raise TypeError(f"Unknown feature file type: {file_path}")

def load_npy_combined(npypath: str):
    """Load combined .npy file with sidecar coords."""
    obj = np.load(npypath, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.shape == ():
        obj = obj.item()

    feats, coords = None, None

    if isinstance(obj, dict):
        for k in ('features', 'feats', 'feat', 'X'):
            if k in obj:
                feats = obj[k]; break
        for k in ('coords', 'tile_coords', 'xy', 'xy_coords'):
            if k in obj:
                coords = obj[k]; break
        if isinstance(feats, str) and os.path.isfile(feats):
            arr = np.load(feats, allow_pickle=True)
            if isinstance(arr, np.lib.npyio.NpzFile):
                arr = arr[list(arr.keys())[0]]
            feats = arr
    elif isinstance(obj, np.lib.npyio.NpzFile):
        keys = list(obj.keys())
        for k in keys:
            arr = obj[k]
            if isinstance(arr, np.ndarray) and arr.ndim == 2:
                feats = arr; break
        for k in ('coords', 'xy', 'tile_coords'):
            if k in obj:
                coords = obj[k]; break
    elif isinstance(obj, np.ndarray):
        feats = obj

    if feats is None:
        raise ValueError(f"{npypath}: could not locate 'features'")

    feats = np.asarray(feats)
    if feats.ndim == 1:
        feats = feats[None, :]
    feats = feats.astype(np.float32, copy=False)

    if coords is None:
        coords_path = npypath.replace('.npy', '_coords.npy')
        if os.path.isfile(coords_path):
            try:
                coords = np.load(coords_path)
                if coords.shape[0] != feats.shape[0]:
                    coords = None
            except Exception:
                coords = None

    if coords is None:
        coords = np.zeros((feats.shape[0], 2), dtype=np.int32)
    else:
        coords = np.asarray(coords)
        if coords.ndim == 1 and coords.size == 2:
            coords = np.tile(coords[None, :], (feats.shape[0], 1))
        if coords.shape[0] != feats.shape[0]:
            m = min(coords.shape[0], feats.shape[0])
            feats = feats[:m]
            coords = coords[:m]
        coords = coords.astype(np.int32, copy=False)

    return feats, coords

# [Include all TCGA scanning functions from original - keeping code concise]
_TCGA_CASE_RE = re.compile(r'^(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})', re.IGNORECASE)

def tcga_case_from_stem(s: str) -> str:
    m = _TCGA_CASE_RE.match(s)
    return m.group(1) if m else s

def scan_npy_combined(feature_root: str) -> dict:
    """Find per-patient *_combined_features.npy files."""
    out = {}
    for pid_dir in glob.glob(os.path.join(feature_root, "*")):
        if not os.path.isdir(pid_dir):
            continue
        dir_name = os.path.basename(pid_dir)
        matches = glob.glob(os.path.join(pid_dir, "*_combined_features.npy"))
        if matches:
            out[dir_name] = {'type': 'npy_layout', 'path': matches[0]}
            if dir_name.startswith('TCGA-'):
                case_match = re.match(r'^(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})', dir_name)
                if case_match:
                    short_case_id = case_match.group(1)
                    if short_case_id not in out:
                        out[short_case_id] = {'type': 'npy_layout', 'path': matches[0]}
    return out

# ═══════════════════════════════════════════════════════════════════════════
# DATASET WITH SEMANTIC LABELS (STEP 2)
# ═══════════════════════════════════════════════════════════════════════════
class CaseBagDataset(Dataset):
    def __init__(self, feature_root: str, encoder: str,
                 labels_df: pd.DataFrame, id_col: str, time_col: str, event_col: str,
                 file_pattern: str = "*.npy", max_tiles: Optional[int] = None,
                 tile_cache: Optional[Dict[str, np.ndarray]] = None,
                 cache_dir: Optional[str] = None,
                 hypergraph_dir: Optional[str] = None,
                 hypergraph_tag: Optional[str] = None): 
        
        self.enc_dir = Path(feature_root) / encoder
        self.df = labels_df.copy()
        self.id_col, self.t_col, self.e_col = id_col, time_col, event_col
        self.file_pattern = file_pattern
        self.max_tiles = max_tiles
        self.tile_cache = tile_cache or {}
        self.cache_dir = cache_dir
        self.hypergraph_dir = Path(hypergraph_dir) if hypergraph_dir else None
        self.hypergraph_tag = hypergraph_tag
        
        self.feature_info = {}
        label_pids = set(self.df[id_col].astype(str))
        found_pids = set()

        # Scan for feature files
        for pid in label_pids:
            pt_file = self.enc_dir / f"{pid}.pt"
            h5_file = self.enc_dir / f"{pid}.h5"
            if pt_file.exists():
                self.feature_info[pid] = {'type': 'single_file', 'path': pt_file}
                found_pids.add(pid)
            elif h5_file.exists():
                self.feature_info[pid] = {'type': 'single_file', 'path': h5_file}
                found_pids.add(pid)
            else:
                candidates = list(self.enc_dir.glob(f"{pid}*.pt"))
                if candidates:

                    self.feature_info[pid] = {'type': 'single_file', 'path': candidates[0]}
                    found_pids.add(pid)

        extra = scan_npy_combined(str(self.enc_dir))
        for pid, info in extra.items():
            if pid in label_pids and pid not in found_pids:
                self.feature_info[pid] = info
                found_pids.add(pid)
        
        self.case_ids = sorted(list(found_pids))
        self.df = self.df[self.df[id_col].isin(self.case_ids)].reset_index(drop=True)

    def __len__(self): 
        return len(self.case_ids)

    def _load_cached_hypergraph(self, pid: str) -> torch.Tensor:
        if self.hypergraph_dir is None or self.hypergraph_tag is None:
            return None 

        h_path = self.hypergraph_dir / f"{pid}_H_{self.hypergraph_tag}.pt"
        if not h_path.exists() and pid.startswith("TCGA-"):
            short_id = pid[:12]
            candidates = sorted(self.hypergraph_dir.glob(f"{short_id}*_H_{self.hypergraph_tag}.pt"))
            if candidates: h_path = candidates[0]

        if not h_path.exists():
            return None

        try:
            gd = torch.load(h_path, map_location="cpu", weights_only=False)
            H = gd["H"]
            if not H.is_sparse: return H.float()
            return H.coalesce()
        except Exception as e:
            return None

    def __getitem__(self, idx: int):
        pid = self.case_ids[idx]
        info = self.feature_info[pid]
        
        try:
            # 1. Load RAW Features
            if info['type'] == 'single_file':
                X_all, coords_all = load_features_and_coords(info['path'])
                xs_all, ys_all = coords_all[:, 0], coords_all[:, 1]
            elif info['type'] == 'npy_layout':
                X_all, coords_all = load_npy_combined(info['path'])
                xs_all, ys_all = coords_all[:, 0], coords_all[:, 1]
            else:
                raise ValueError(f"Unknown data type")
    
            N = X_all.shape[0]
            if N == 0: raise ValueError(f"No features loaded")

            # 2. Determine Indices (STRICT MODE)
            sel = None
            if pid in self.tile_cache:
                sel = self.tile_cache[pid]
            elif self.cache_dir:
                # Try Exact Match
                idx_path = os.path.join(self.cache_dir, f"{pid}_topk_idx.npy")
                if os.path.exists(idx_path):
                    sel = np.load(idx_path)
                else:
                    # Try Wildcard Match (for long filenames)
                    candidates = glob.glob(os.path.join(self.cache_dir, f"{pid}*_topk_idx.npy"))
                    if candidates:
                        sel = np.load(candidates[0])

            # CRITICAL CHECK 1: Missing Index File
            if sel is None:
                raise FileNotFoundError(f"STRICT MODE: Could not find index file for {pid} in {self.cache_dir}. Cannot verify which tiles were used for the graph.")

            if self.max_tiles is not None: sel = sel[:self.max_tiles]
            sel = sel[sel < N]
            
            X_tad = X_all[sel]
            xs_tad = xs_all[sel]
            ys_tad = ys_all[sel]
            coords_tad = np.stack([xs_tad, ys_tad], axis=1)
    
            # 3. Load Labels
            row = self.df[self.df[self.id_col] == pid].iloc[0]
            risk_tensor = torch.zeros(X_tad.shape[0]).float()
            mass_labels = torch.zeros(X_tad.shape[0], dtype=torch.long)
            
            # 4. Load Hypergraph
            H = self._load_cached_hypergraph(pid)

            # CRITICAL CHECK 2: Size Mismatch
            if H is not None:
                h_count = H.shape[0]
                x_count = X_tad.shape[0]
                
                if h_count != x_count:
                    # STRICT MODE: Crash immediately on mismatch
                    raise RuntimeError(f"FATAL MISMATCH for {pid}: Graph H has {h_count} nodes, but Features X (loaded from index file) has {x_count}. Data integrity compromised.")

            return {
                'id': pid,
                'features': torch.from_numpy(X_tad).float(),
                'coords': torch.from_numpy(coords_tad).float(),
                'risk_scores': risk_tensor,
                'mass_labels': mass_labels,
                'H': H,  
                'time': torch.tensor(float(row[self.t_col]), dtype=torch.float32),
                'event': torch.tensor(int(row[self.e_col]), dtype=torch.long)
            }
        
        except Exception as e:
            # Re-raise the error so the main training loop stops or catches it properly
            print(f"❌ [Dataset Error] {pid}: {e}")
            raise e

def collate_bags(batch: List[Dict]): return batch

# ═══════════════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════

class HypergraphGatedMILSurvival_v2(nn.Module):
    """Enhanced TRIAG-MIL v2 with semantic hierarchical hypergraph support."""
    def __init__(self, feat_dim, heads=8, hidden=384, bins=1, use_clinical=False, clin_dim=0, 
                 attn_dropout=0.2, num_hyper_layers=1, use_self_attn=False, 
                 self_attn_heads=4, learnable_temp=True):
        super().__init__()
        self.feat_dim = feat_dim
        self.heads = heads
        self.learnable_temp = learnable_temp
        
        # ... [Layers definition stays the same] ...
        self.num_hyper_layers = num_hyper_layers
        self.theta_layers = nn.ModuleList([nn.Linear(feat_dim, hidden) for _ in range(num_hyper_layers)])
        self.phi_layers   = nn.ModuleList([nn.Linear(hidden, feat_dim) for _ in range(num_hyper_layers)])
        self.norm_hyper_layers = nn.ModuleList([nn.LayerNorm(feat_dim) for _ in range(num_hyper_layers)])

        if use_self_attn:
            self.self_attn = nn.MultiheadAttention(feat_dim, self_attn_heads, dropout=attn_dropout, batch_first=True)
            self.norm_self_attn = nn.LayerNorm(feat_dim)
        self.use_self_attn = use_self_attn

        self.V = nn.ModuleList([nn.Linear(feat_dim, hidden) for _ in range(heads)])
        self.U = nn.ModuleList([nn.Linear(feat_dim, hidden) for _ in range(heads)])
        self.w = nn.ModuleList([nn.Linear(hidden, 1, bias=False) for _ in range(heads)])
        self.dropout = nn.Dropout(attn_dropout)

        if learnable_temp:
            # Initialize to 0.5 to encourage sharpness from the start
            self.attention_temp = nn.Parameter(torch.ones(heads) * 1.0)

        self.fuse = nn.Linear(heads * feat_dim, feat_dim)
        self.norm_bag = nn.LayerNorm(feat_dim)
        self.clin_gate = (nn.Sequential(nn.Linear(clin_dim, feat_dim), nn.Sigmoid()) if (use_clinical and clin_dim > 0) else None)
        self.cox_head = nn.Linear(feat_dim, 1)
        self.hazard_head = nn.Linear(feat_dim, bins)

    def hypergraph_mp(self, X, H):
        if H is None or H.numel() == 0: return X
        X_out = X
        with torch.cuda.amp.autocast(enabled=False):
            # Ensure inputs are Float32 (Dense)
            H = H.float()
            X_out = X_out.float()
            
            for i in range(self.num_hyper_layers):
                He = H.t()
                # .to_dense() prevents the sparse clamp error
                deg_e = He.sum(1, keepdim=True).to_dense().clamp(min=1)
                
                # Cast weights to Float32 for this calculation
                theta_w = self.theta_layers[i].weight.float()
                theta_b = self.theta_layers[i].bias.float()
                
                # Manual linear layer
                h_theta = F.linear(X_out, theta_w, theta_b)
                
                E = He @ torch.tanh(h_theta) / deg_e
                
                deg_v = H.sum(1, keepdim=True).to_dense().clamp(min=1)
                
                phi_w = self.phi_layers[i].weight.float()
                phi_b = self.phi_layers[i].bias.float()
                e_phi = F.linear(E, phi_w, phi_b)
                
                X_up = H @ torch.tanh(e_phi) / deg_v
                X_out = X_out + X_up

        # Norm layer handles the transition back to mixed precision
        X_out = self.norm_hyper_layers[0](X_out)
        return X_out     

    # def hypergraph_mp(self, X, H):
    #     if H is None or H.numel() == 0: return X
    #     X_out = X
    #     for i in range(self.num_hyper_layers):
    #         He = H.t()
    #         deg_e = He.sum(1, keepdim=True).clamp(min=1)
    #         E = He @ torch.tanh(self.theta_layers[i](X_out)) / deg_e
    #         deg_v = H.sum(1, keepdim=True).clamp(min=1)
    #         X_up = H @ torch.tanh(self.phi_layers[i](E)) / deg_v
    #         X_out = self.norm_hyper_layers[i](X_out + X_up)
    #     return X_out

    def forward(self, X, H=None, clinical=None, return_attn=False):
        if H is not None: X = self.hypergraph_mp(X, H)
        
        if self.use_self_attn:
            X_a, _ = self.self_attn(X.unsqueeze(0), X.unsqueeze(0), X.unsqueeze(0))
            X = self.norm_self_attn(X + X_a.squeeze(0))
        
        head_embeds, head_weights = [], []
        
        for i, (V, U, w) in enumerate(zip(self.V, self.U, self.w)):
            z = torch.tanh(V(X)) * torch.sigmoid(U(X))
            s = w(self.dropout(z)).squeeze(-1)
            
            # <--- NEW: Temperature Clamping --->
            if self.learnable_temp:
                temp = torch.clamp(self.attention_temp[i], min=0.1, max=5.0) 
                a = torch.softmax(s / temp, dim=0)
            else:
                a = torch.softmax(s, dim=0)
                

            
            bag = (a.unsqueeze(0) @ X).squeeze(0)
            head_embeds.append(bag)
            head_weights.append(a)
        
        Z = self.norm_bag(self.fuse(torch.cat(head_embeds, dim=-1)))
        if self.clin_gate is not None and clinical is not None:
            Z = Z * self.clin_gate(clinical)
        
        risk = self.cox_head(Z).squeeze(-1)
        hazard = torch.sigmoid(self.hazard_head(Z))

        if return_attn:
            return risk, hazard, head_weights
            
        return risk, hazard

# ═══════════════════════════════════════════════════════════════════════════
# TRAINING & EVALUATION
# ═══════════════════════════════════════════════════════════════════════════


def train_one_epoch(model, loader, optimizer, device, use_amp, cfg,
                    is_main=True):
    model.train()
    running, steps = 0.0, 0
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    accum_steps = int(cfg.get('gradient_accumulation_steps', 1))
    micro_step = 0

    accum_risks, accum_times, accum_events = [], [], []

    pbar = tqdm(loader, desc="Train", leave=False) if is_main else loader

    for batch_idx, items in enumerate(pbar):
        for sample in items:
            X = sample['features'].to(device, non_blocking=True)
            H = sample['H'].to(device, non_blocking=True)

            # 🔒 HARD SAFETY CHECK
            assert H.shape[0] == X.shape[0], \
                f"[FATAL] H/X mismatch for {sample['id']} → {H.shape} vs {X.shape}"

            t = sample['time'].to(device).view(1)
            e = sample['event'].to(device).view(1)

            with torch.cuda.amp.autocast(enabled=use_amp):
                r, _ = model(X, H, clinical=None)

            accum_risks.append(r.view(1))
            accum_times.append(t)
            accum_events.append(e)

        micro_step += 1
        is_last = (batch_idx == len(loader) - 1)

        if micro_step >= accum_steps or is_last:
            all_risks = torch.cat(accum_risks)
            all_times = torch.cat(accum_times)
            all_events = torch.cat(accum_events)

            if (all_events == 1).sum() == 0:
                accum_risks, accum_times, accum_events = [], [], []
                micro_step = 0
                continue

            with torch.cuda.amp.autocast(enabled=False):
                loss = cox_ph_loss(all_risks.float(),
                                   all_times.float(),
                                   all_events.float()) / accum_steps

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            running += float(loss.detach().cpu())
            steps += 1

            accum_risks, accum_times, accum_events = [], [], []
            micro_step = 0

            if is_main and isinstance(pbar, tqdm):
                pbar.set_postfix({'loss': f'{running / steps:.4f}'})

    return running / max(1, steps)


@torch.no_grad()
def evaluate(model, loader, device, use_amp: bool, cfg: dict,
             return_preds: bool=False, save_attention: bool=False,
             save_dir: Optional[str]=None,
             attention_aggregation: str = "average"):
    """
    Evaluation using ONLY cached hypergraphs.
    No graph construction allowed here.
    """
    model.eval()

    local_ids, local_times, local_events, local_risks = [], [], [], []
    all_attention_weights, all_coords = [], []

    it = tqdm(loader, desc="Eval", leave=False)

    for batch_idx, items in enumerate(it):
        for sample in items:
            pid = sample['id']

            X = sample['features'].to(device, non_blocking=True).float()
            coords = sample['coords'].cpu().numpy()
            H = sample['H'].to(device, non_blocking=True)

            # 🔒 HARD SAFETY CHECK
            assert H.shape[0] == X.shape[0], \
                f"[FATAL] Eval H/X mismatch for {pid}: {H.shape} vs {X.shape}"

            t = float(sample['time'].item())
            e = int(sample['event'].item())

            with torch.cuda.amp.autocast(enabled=use_amp):
                if save_attention:
                    r, _, head_weights = model(X, H, clinical=None, return_attn=True)
                else:
                    r, _ = model(X, H, clinical=None, return_attn=False)

            risk = float(r.detach().cpu().item())

            local_ids.append(pid)
            local_times.append(t)
            local_events.append(e)
            local_risks.append(risk)

            # ================= ATTENTION AGGREGATION =================
            if save_attention:
                N = X.shape[0]

                if head_weights is not None and len(head_weights) > 0:
                    if attention_aggregation == "average":
                        attn = torch.stack(head_weights).mean(0).cpu().numpy()

                    elif attention_aggregation == "variance_weighted":
                        variances = torch.tensor([h.var().item() for h in head_weights])
                        weights = torch.softmax(variances / 0.1, dim=0)
                        attn = (weights[:, None] * torch.stack(head_weights)).sum(0).cpu().numpy()

                    elif attention_aggregation == "best_head":
                        variances = [h.var().item() for h in head_weights]
                        attn = head_weights[int(np.argmax(variances))].cpu().numpy()

                    elif attention_aggregation == "all_heads":
                        attn = [h.cpu().numpy() for h in head_weights]

                    else:
                        raise ValueError(f"Unknown aggregation: {attention_aggregation}")
                else:
                    attn = np.full((N,), 1.0 / max(N, 1), dtype=np.float32)

                all_attention_weights.append(attn)
                all_coords.append(coords)
            # =========================================================

    ci = c_index(
        np.array(local_times),
        np.array(local_events),
        np.array(local_risks)
    )

    out = {'c_index': ci, 'n': len(local_ids)}

    if return_preds:
        out.update({
            'ids': local_ids,
            'times': local_times,
            'events': local_events,
            'risks': local_risks
        })

    if save_attention and save_dir:
        torch.save({
            'patient_ids': local_ids,
            'attention_weights': all_attention_weights,
            'tile_coords': all_coords,
            'risks': local_risks,
            'times': local_times,
            'events': local_events,
            'aggregation_method': attention_aggregation
        }, os.path.join(save_dir, 'attention_data.pt'))

    return out


# ═══════════════════════════════════════════════════════════════════════════
# UTILS (splits, early stopping, etc - same as v2)
# ═══════════════════════════════════════════════════════════════════════════

def build_strata(labels_df: pd.DataFrame, event_col: str, time_col: str, n_bins: int) -> np.ndarray:
    eps = 1e-6; df = labels_df[[event_col, time_col]].copy()
    unc = df[df[event_col].astype(int) == 1]
    try:
        _, bins = pd.qcut(unc[time_col], q=n_bins, retbins=True, duplicates='drop')
    except Exception:
        mn = df[time_col].min() - eps; mx = df[time_col].max() + eps
        bins = np.linspace(mn, mx, n_bins + 1)
    bins = np.asarray(bins, float); bins[0] = df[time_col].min() - eps; bins[-1] = df[time_col].max() + eps
    tb = pd.cut(df[time_col], bins=bins, labels=False, right=False, include_lowest=True).astype('Int64').fillna(n_bins // 2).astype(int)
    ev = df[event_col].astype(int).values
    return ev * (len(bins) - 1) + tb.values

def stratified_kfold_ids(labels_df: pd.DataFrame, id_col: str, event_col: str, time_col: str,
                         n_bins: int, n_splits: int, seed: int) -> List[Tuple[List[str], List[str]]]:
    from sklearn.model_selection import StratifiedKFold
    strata = build_strata(labels_df, event_col, time_col, n_bins)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    idx = np.arange(len(labels_df)); folds = []
    for tr, va in skf.split(idx, strata):
        folds.append((labels_df.iloc[tr][id_col].tolist(), labels_df.iloc[va][id_col].tolist()))
    return folds

def load_tile_cache(cache_dir: str, case_ids: List[str], max_tiles: Optional[int]) -> Dict[str, np.ndarray]:
    cache = {}
    if cache_dir and os.path.isdir(cache_dir):
        for pid in case_ids:
            fp = os.path.join(cache_dir, f"{pid}_topk_idx.npy")
            if os.path.isfile(fp):
                arr = np.load(fp)
                cache[pid] = arr if max_tiles is None else arr[:max_tiles]
    return cache

class EarlyStoppingCIndex:
    def __init__(self, warmup=0, patience=20, min_epochs=40, verbose=True):
        self.warmup = int(warmup); self.patience = int(patience); self.min_epochs = int(min_epochs)
        self.verbose = verbose; self.best = None; self.counter = 0
        self.early_stop = False; self.best_path = None
    
    def step(self, epoch: int, value: float, model: nn.Module, ckpt_path: str):
        if epoch <= self.warmup: return False
        improve = (self.best is None) or (value > self.best)
        if improve:
            self.best = value; self.counter = 0
            torch.save({'model': model.state_dict()}, ckpt_path)
            self.best_path = ckpt_path
            if self.verbose: print(f"[EarlyStop] new best C-index={value:.4f} @ epoch {epoch}")
        else:
            self.counter += 1
            if self.verbose: print(f"[EarlyStop] no improv ({self.counter}/{self.patience})")
            if (epoch >= self.min_epochs) and (self.counter >= self.patience):
                self.early_stop = True
                if self.verbose: print("[EarlyStop] triggered")
        return self.early_stop

# ═══════════════════════════════════════════════════════════════════════════
# CSV AGGREGATION FOR SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

CSV_COLUMNS = ['fold', 'seed', 'n_train', 'n_val', 'val_final_c', 'val_best_c']

def compute_global_oof_cindex(save_root):
    all_preds = []
    for fold_dir in glob.glob(os.path.join(save_root, "fold_*", "seed_*")):
        pred_path = os.path.join(fold_dir, "predictions.csv")
        if os.path.isfile(pred_path):
            df = pd.read_csv(pred_path)
            all_preds.append(df)

    if not all_preds:
        return None

    df_all = pd.concat(all_preds, ignore_index=True)

    ci = c_index(
        df_all['time'].values,
        df_all['event'].values,
        df_all['risk'].values
    )
    return ci


def _csv_lock_path(path: str) -> str:
    return path + ".lock"

@contextlib.contextmanager
def _acquire_lock(path: str, timeout: float = 300.0, poll: float = 0.2):
    lock = _csv_lock_path(path)
    start = time.time()
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            os.close(fd)
            break
        except FileExistsError:
            if time.time() - start > timeout:
                raise TimeoutError(f"Timeout waiting for lock {lock}")
            time.sleep(poll)
    try:
        yield
    finally:
        with contextlib.suppress(Exception):
            os.remove(lock)

def append_row_safely(csv_path: str, row: Dict):
    """Append a single row to CSV with file locking."""
    with _acquire_lock(csv_path):
        if os.path.isfile(csv_path):
            df = pd.read_csv(csv_path)
        else:
            df = pd.DataFrame(columns=CSV_COLUMNS)
        for c in CSV_COLUMNS:
            if c not in df.columns:
                df[c] = np.nan
        mask_num_fold = pd.to_numeric(df['fold'], errors='coerce').notnull()
        df_data = df[mask_num_fold].copy()
        df_data.loc[len(df_data)] = [row.get(c, np.nan) for c in CSV_COLUMNS]
        extras = df.loc[~mask_num_fold]
        df_final = pd.concat([df_data, extras], ignore_index=True)
        df_final.to_csv(csv_path, index=False)

def maybe_write_final_summary(csv_path: str, expected_rows: int):
    """Write final summary row with mean±std if all folds complete."""
    with _acquire_lock(csv_path):
        if not os.path.isfile(csv_path):
            return
        df = pd.read_csv(csv_path)
        mask_num = pd.to_numeric(df['fold'], errors='coerce').notnull()
        df_data = df[mask_num].copy()
        if len(df_data) < expected_rows:
            return
        mean_c = float(df_data['val_best_c'].mean())
        std_c  = float(df_data['val_best_c'].std(ddof=1) if len(df_data) > 1 else 0.0)
        df = df[mask_num].copy()
        if 'val_mean' not in df.columns: df['val_mean'] = np.nan
        if 'val_std' not in df.columns:  df['val_std']  = np.nan
        if 'note' not in df.columns:     df['note']     = ""
        agg_row = {
            'fold': 'ALL', 'seed': 'ALL',
            'n_train': int(df_data['n_train'].mean()) if 'n_train' in df_data.columns else '',
            'n_val':   int(df_data['n_val'].mean())   if 'n_val'   in df_data.columns else '',
            'val_final_c': np.nan, 'val_best_c': np.nan,
            'val_mean': mean_c, 'val_std': std_c, 'note': 'mean±std over all rows'
        }
        df_out = pd.concat([df, pd.DataFrame([agg_row])], ignore_index=True)
        df_out.to_csv(csv_path, index=False)
        print(f"\n[Summary] mean±std over {len(df_data)} row(s): {mean_c:.4f} ± {std_c:.4f}")
        print(f"[Summary CSV] {csv_path}")

# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main(cfg_path: str, fold_override: Optional[int]=None):
    cfg = json.load(open(cfg_path))
    feature_root = cfg['feature_root']
    encoder = cfg.get('selected_encoder', (cfg.get('encoders') or [None])[0])
    labels = pd.read_csv(cfg['csv_labels'])
    id_col, time_col, event_col = cfg['id_col'], cfg['time_col'], cfg['event_col']
    labels[id_col] = labels[id_col].astype(str)

    is_ddp, local_rank, global_rank, world_size = setup_ddp()
    is_main = (not is_ddp) or (global_rank == 0)
    device = torch.device(f'cuda:{local_rank}' if is_ddp else 'cuda' if torch.cuda.is_available() else 'cpu')

    labels = labels[labels[event_col].notna() & labels[time_col].notna()].reset_index(drop=True)
    
    k_folds = int(cfg.get('k_folds', 5))
    base_seed = int(cfg.get('random_seed', 42))
    save_root = cfg['save_dir']
    os.makedirs(save_root, exist_ok=True)

# -------------------------------------------------------------------------
    # FOLD GENERATION STRATEGY (Boolean Mask Loader)
    # -------------------------------------------------------------------------
    folds = None
    splits_dir = cfg.get('splits_dir', None)

    if splits_dir and os.path.isdir(splits_dir):
        if is_main: print(f"[Splits] Looking for split files in: {splits_dir}")
        loaded_folds = []
        all_splits_valid = True
        
        for k in range(k_folds):
            # EXPECTED FORMAT: splits_0.csv, splits_1.csv (CLAM standard)
            split_path = os.path.join(splits_dir, f"splits_{k}.csv")
            
            if os.path.exists(split_path):
                try:
                    df_split = pd.read_csv(split_path)
                    
                    # 1. Identify ID Column (usually 'case_id' or first column)
                    if 'case_id' in df_split.columns:
                        id_col_name = 'case_id'
                    elif 'patient_id' in df_split.columns:
                        id_col_name = 'patient_id'
                    else:
                        id_col_name = df_split.columns[0] # Fallback to first col
                    
                    # 2. Parse Boolean Masks (train=1, val=1)
                    # We cast to int just in case they are saved as floats/strings
                    train_mask = df_split['train'].astype(int) == 1
                    val_mask   = df_split['val'].astype(int) == 1
                    
                    train_ids = df_split.loc[train_mask, id_col_name].astype(str).tolist()
                    val_ids   = df_split.loc[val_mask, id_col_name].astype(str).tolist()
                    
                    # 3. Validation Check
                    if len(train_ids) == 0 or len(val_ids) == 0:
                        if is_main: print(f"[Splits] Warning: Fold {k} is empty? Tr={len(train_ids)}, Val={len(val_ids)}")
                        all_splits_valid = False
                        break
                        
                    loaded_folds.append((train_ids, val_ids))
                    
                except Exception as e:
                    if is_main: print(f"[Splits] Error parsing {split_path}: {e}")
                    all_splits_valid = False
                    break
            else:
                if is_main: print(f"[Splits] Missing file: {split_path}")
                all_splits_valid = False
                break
        
        if all_splits_valid and len(loaded_folds) == k_folds:
            if is_main: print(f"[Splits] ✅ Successfully loaded {k_folds} folds from disk (Boolean Format).")
            folds = loaded_folds

    # 2. Fallback: Generate On-the-Fly
    if folds is None:
        if is_main: print(f"[Splits] ⚠️  Files not found or invalid. Generating on-the-fly (Seed {base_seed})...")
        folds = stratified_kfold_ids(
            labels[[id_col, event_col, time_col]],
            id_col, event_col, time_col,
            n_bins=int(cfg.get('stratify_time_bins', 4)),
            n_splits=k_folds,
            seed=base_seed
        )
    # -------------------------------------------------------------------------

    batch_size = int(cfg.get('batch_size', 4))
    epochs = int(cfg.get('epochs', 200))
    use_amp = bool(cfg.get('amp', True))
    lr = float(cfg.get('lr', 3e-4))
    wd = float(cfg.get('weight_decay', 1e-4))
    max_tiles = cfg.get('max_tiles', None)
    cache_dir = cfg.get('precompute_cache', {}).get('cache_dir', '')
    hypergraph_dir = os.path.join(cache_dir, "hypergraphs")
    
    h_cfg = cfg.get('semantic_hierarchy_config', {})
    k_intra = int(h_cfg.get('k_intra', 5))
    k_inter = int(h_cfg.get('k_inter', 3))
    tiles_per_super = int(h_cfg.get('tiles_per_super', 16))
    
    hypergraph_tag = f"k{k_intra}_i{k_inter}_t{tiles_per_super}"
    
    # v2 parameters
    heads = int(cfg.get('heads', 8))
    hidden = int(cfg.get('hidden', 384))
    num_hyper_layers = int(cfg.get('num_hyper_layers', 2))
    use_self_attn = bool(cfg.get('use_self_attn', True))

    # Early stopping
    es_enabled = bool(cfg.get('early_stopping', True))
    es_warmup = int(cfg.get('early_warmup', 0))
    es_patience = int(cfg.get('early_patience', 20))
    es_min_epochs = int(cfg.get('early_min_epochs', 40))

    num_workers = int(cfg.get('num_workers', 8))
    
    # NEW: Enable/disable semantic hierarchy
    use_semantic_hierarchy = bool(cfg.get('use_semantic_hierarchy', True))
    
    # CSV summary file
    summary_csv = os.path.join(save_root, "triagmil_fast_summary.csv")
    expected_total_rows = k_folds
    
    if is_main:
        print(f"\n{'='*80}")
        print(f"🧬 TRIAG-MIL with Semantic Hierarchical Hypergraph")
        print(f"{'='*80}")
        print(f"Hypergraph Dir: {hypergraph_dir}")
        print(f"Hypergraph Tag: {hypergraph_tag}")
        print(f"Feature root: {feature_root}")
        print(f"Encoder: {encoder}")
        print(f"Cache dir: {cache_dir}")
        print(f"Semantic hierarchy: {'ENABLED' if use_semantic_hierarchy else 'DISABLED'}")
        print(f"Summary CSV: {summary_csv}")
        print(f"{'='*80}\n")

    run_folds = range(k_folds) if (fold_override is None) else [int(fold_override)]

    for k in run_folds:
        fold_seed = base_seed
        fold_root = os.path.join(save_root, f"fold_{k}", f"seed_{fold_seed}")
        results_path = os.path.join(fold_root, "results.pt")
        
        # 🟢 INTELLIGENT RESUME & SUMMARY RECOVERY
        if os.path.isfile(results_path):
            if is_main:
                print(f"\n⏭️  FOLD {k} already complete.")
                try:
                    # Load existing result (weights_only=False for numpy support)
                    res = torch.load(results_path, map_location='cpu', weights_only=False)
                    
                    # Extract Metrics
                    val_c = res.get('best_val', res.get('val_cindex', 0.0))
                    final_c = res.get('val_cindex', 0.0)
                    
                    # Get IDs to report correct counts (folds is already loaded/generated above)
                    tr_ids_Rec, va_ids_Rec = folds[k]
                    
                    # Create Row
                    row = {
                        'fold': k,
                        'seed': fold_seed,
                        'n_train': len(tr_ids_Rec),
                        'n_val': len(va_ids_Rec),
                        'val_final_c': final_c,
                        'val_best_c': val_c
                    }
                    
                    # Force append to CSV (function handles duplicates/updates)
                    append_row_safely(summary_csv, row)
                    print(f"   ✓ Added to Summary CSV: C-Index = {val_c:.4f}")
                    
                except Exception as e:
                    print(f"   ⚠️ Could not recover metrics for summary: {e}")
            
            # Skip training for this fold
            continue
        
        if is_main:
            print(f"\n{'='*80}")
            print(f"🔬 FOLD {k} | SEED {fold_seed}")
            print(f"{'='*80}")
        
        set_seed(fold_seed)
        
        tr_ids, va_ids = folds[k]
        tr_cache = load_tile_cache(cache_dir, tr_ids, max_tiles)
        va_cache = load_tile_cache(cache_dir, va_ids, max_tiles)

        ds_train = CaseBagDataset(
            feature_root=feature_root, 
            encoder=encoder,
            labels_df=labels[labels[id_col].isin(tr_ids)],
            id_col=id_col, 
            time_col=time_col, 
            event_col=event_col,
            file_pattern="*.npy", 
            max_tiles=max_tiles, 
            tile_cache=tr_cache, 
            cache_dir=cache_dir,
            # ▼▼▼ YOU MUST ADD THESE LINES ▼▼▼
            hypergraph_dir=hypergraph_dir,
            hypergraph_tag=hypergraph_tag
        )
        ds_val = CaseBagDataset(
            feature_root=feature_root, 
            encoder=encoder,
            labels_df=labels[labels[id_col].isin(va_ids)],
            id_col=id_col, 
            time_col=time_col, 
            event_col=event_col,
            file_pattern="*.npy", 
            max_tiles=max_tiles, 
            tile_cache=va_cache, 
            cache_dir=cache_dir,
            # ▼▼▼ YOU MUST ADD THESE LINES ▼▼▼
            hypergraph_dir=hypergraph_dir,
            hypergraph_tag=hypergraph_tag
        )

        try:
            feat_dim = int(ds_train[0]['features'].shape[1])
        except:
            print(f"ERROR: No training data for fold {k}")
            continue

        dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=True,
                             collate_fn=collate_bags, worker_init_fn=seed_worker)
        dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True,
                           collate_fn=collate_bags)

        os.makedirs(fold_root, exist_ok=True)

        model = HypergraphGatedMILSurvival_v2(
            feat_dim=feat_dim, heads=heads, hidden=hidden,
            bins=1, attn_dropout=0.2,
            num_hyper_layers=num_hyper_layers,
            use_self_attn=use_self_attn
        ).to(device)
            
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        es = EarlyStoppingCIndex(es_warmup, es_patience, es_min_epochs) if es_enabled else None
        best_val = -1.0
        best_path = None

        if is_main:
            print(f"Train: {len(tr_ids)} | Val: {len(va_ids)}")
        SMOOTH_WINDOW = 5  # 5 is reasonable for ~70 val samples
        
        start_epoch = 1
        val_ci_buffer = []  # Initialize buffer for smoothing
        resume_checkpoint = None
        
        # Look for existing checkpoints in this fold directory
        existing_checkpoints = sorted(glob.glob(os.path.join(fold_root, "ep*.pt")))
        
        if existing_checkpoints:
            # Find the checkpoint with highest epoch number
            # Extract epoch numbers from filenames like "ep82_val0.6683.pt"
            checkpoint_epochs = []
            for ckpt_path in existing_checkpoints:
                fname = os.path.basename(ckpt_path)
                match = re.search(r'ep(\d+)_', fname)
                if match:
                    checkpoint_epochs.append((int(match.group(1)), ckpt_path))
            
            if checkpoint_epochs:
                # Sort by epoch number and get the latest
                checkpoint_epochs.sort(key=lambda x: x[0])
                latest_epoch, resume_checkpoint = checkpoint_epochs[-1]
                
                print(f"\n{'='*80}")
                print(f"🔄 Found existing checkpoints for Fold {k}")
                print(f"   Latest: {os.path.basename(resume_checkpoint)} (Epoch {latest_epoch})")
                print(f"{'='*80}")
                
                # Auto-resume if running in non-interactive mode or environment variable set
                auto_resume = os.getenv('AUTO_RESUME', 'false').lower() == 'true'
                
                if auto_resume:
                    resume_decision = 'y'
                    print("   AUTO_RESUME=true detected, resuming automatically...")
                else:
                    resume_decision = input("   Resume from this checkpoint? (y/n): ").strip().lower()
                
                if resume_decision == 'y':
                    try:
                        print(f"\n📥 Loading checkpoint: {resume_checkpoint}")
                        ckpt_data = torch.load(resume_checkpoint, map_location=device, weights_only=False)
                        
                        # Restore model state
                        model.load_state_dict(ckpt_data['model'])
                        print("   ✓ Model state restored")
                        
                        # Restore optimizer state
                        if 'optimizer' in ckpt_data:
                            optimizer.load_state_dict(ckpt_data['optimizer'])
                            print("   ✓ Optimizer state restored")
                        else:
                            print("   ⚠ Optimizer state not found, using fresh optimizer")
                        
                        # Restore training progress
                        start_epoch = ckpt_data.get('epoch', latest_epoch) + 1
                        current_ckpt_val = ckpt_data.get('val_c', -1.0)
                        best_val_path = os.path.join(fold_root, "best_val.pt")
                        if os.path.exists(best_val_path):
                            try:
                                # Load strictly the metadata to get the score
                                best_data = torch.load(best_val_path, map_location='cpu',weights_only=False)
                                global_best = best_data.get('val_c', -1.0)
                                # The best value is the max of the current checkpoint or the stored best file
                                best_val = max(current_ckpt_val, global_best)
                                print(f"   ✓ Restored True Best C-Index: {best_val:.4f} (from best_val.pt)")
                            except Exception as e:
                                print(f"   ⚠ Could not read best_val.pt, using checkpoint val: {e}")
                                best_val = current_ckpt_val
                        else:
                            best_val = current_ckpt_val
                        
                        # Restore smoothing buffer
                        if 'val_ci_buffer' in ckpt_data:
                            val_ci_buffer = ckpt_data['val_ci_buffer']
                            print(f"   ✓ Smoothing buffer restored ({len(val_ci_buffer)} values)")
                        else:
                            # Reconstruct buffer from last SMOOTH_WINDOW checkpoints
                           
                            print(f"   ⚠ Buffer not found, reconstructing from last {SMOOTH_WINDOW} checkpoints...")
                            recent_checkpoints = checkpoint_epochs[-SMOOTH_WINDOW:]
                            for _, ckpt_path in recent_checkpoints:
                                try:
                                    temp_data = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                                    val_ci_buffer.append(temp_data.get('val_c', 0.0))
                                except:
                                    pass
                        
                        # Restore early stopping state if it exists
                        if es is not None and 'early_stopping' in ckpt_data:
                            es_state = ckpt_data['early_stopping']
                            es.best = es_state.get('best', None)
                            es.counter = es_state.get('counter', 0)
                            es.best_path = es_state.get('best_path', None)
                            print(f"   ✓ Early stopping state restored (counter: {es.counter})")
                        
                        print(f"\n✅ Successfully resumed from epoch {ckpt_data.get('epoch', latest_epoch)}")
                        print(f"   Starting at epoch: {start_epoch}")
                        print(f"   Best val C-index so far: {best_val:.4f}")
                        print(f"   Smoothed buffer size: {len(val_ci_buffer)}")
                        print(f"{'='*80}\n")
                        
                    except Exception as e:
                        print(f"\n❌ Error loading checkpoint: {e}")
                        print("   Starting from scratch...\n")
                        start_epoch = 1
                        val_ci_buffer = []
                        best_val = -1.0
                else:
                    print("\n   Skipping resume, starting from scratch...\n")
                    start_epoch = 1
                    val_ci_buffer = []
                    best_val = -1.0
        else:
            # No checkpoints found, fresh start
            if is_main:
                print(f"\n{'='*80}")
                print(f"🆕 No existing checkpoints found for Fold {k}")
                print(f"   Starting fresh training...")
                print(f"{'='*80}\n")
            start_epoch = 1
            val_ci_buffer = []
            best_val = -1.0
            
        for ep in range(start_epoch, epochs + 1):
            tr_loss = train_one_epoch(
                model, dl_train, optimizer, device, use_amp, cfg,
                is_main=is_main
            )
               
            val_metrics = evaluate(model, dl_val, device, use_amp, cfg)
            val_c = val_metrics['c_index']
            
            val_ci_buffer.append(val_c)
            if len(val_ci_buffer) > SMOOTH_WINDOW:
                val_ci_buffer.pop(0)
            
            val_ci_smoothed = float(np.mean(val_ci_buffer))

            if is_main:
                print(
                    f"Fold {k} | Ep {ep:02d}/{epochs} | "
                    f"tr={tr_loss:.4f} | "
                    f"valCI={val_c:.4f} | "
                    f"valCI_smooth={val_ci_smoothed:.4f}"
                )

            if is_main:
                # Save regular checkpoint
                ckpt = os.path.join(fold_root, f"ep{ep:02d}_val{val_c:.4f}.pt")
                es_state = None
                if es is not None:
                    es_state = {
                        'best': es.best,
                        'counter': es.counter,
                        'best_path': es.best_path
                    }
                torch.save({
                    'model': model.state_dict(), 
                    'feat_dim': feat_dim,
                    'optimizer': optimizer.state_dict(),
                    'epoch': ep,
                    'val_c': val_c,
                    'val_ci_smoothed': val_ci_smoothed,
                    'val_ci_buffer': val_ci_buffer,
                    'early_stopping': es_state
                }, ckpt)
                
                # Track best by RAW metric
                if val_c > best_val:
                    best_val = val_c
                    best_path = ckpt
                    # Save a copy as best_val_raw.pt
                    torch.save({
                        'model': model.state_dict(), 
                        'feat_dim': feat_dim,
                        'epoch': ep,
                        'val_c': val_c
                    }, os.path.join(fold_root, "best_val.pt"))
            
            # Early stopping uses smoothed
            if es is not None:
                ckpt_best = os.path.join(fold_root, "best_val_smoothed.pt")
                stop = es.step(ep, val_ci_smoothed, model, ckpt_path=ckpt_best)
                if stop:
                    if is_main:
                        print(f"[EarlyStop] stopping at epoch {ep}")
                    break
        if is_main and best_path is None:
            potential_best = os.path.join(fold_root, "best_val.pt")
            if os.path.isfile(potential_best):
                best_path = potential_best
                print(f"   ✓ No new best found during resume. Using existing best: {best_path}")
        # Final evaluation
        if is_main and best_path and os.path.isfile(best_path):
            print(f"\n[Final Eval] Loading best model from {best_path}")
            state = torch.load(best_path, map_location=device, weights_only=False)
            model.load_state_dict(state['model'], strict=False)
            
            final = evaluate(model, dl_val, device, use_amp, cfg,
                           return_preds=True, save_attention=True,
                           save_dir=fold_root)
            
            final_c = final['c_index']
            print(f"[Fold {k}] Final VAL C-index={final_c:.4f}")

            df_preds = pd.DataFrame({
                'patient_id': final['ids'],
                'risk': final['risks'],
                'time': final['times'],
                'event': final['events']
            })
            df_preds.to_csv(os.path.join(fold_root, 'predictions.csv'), index=False)
            
            torch.save({
                'val_cindex': final_c,
                'fold': k,
                'seed': fold_seed,
                'best_val': best_val
            }, os.path.join(fold_root, "results.pt"))
            
            # Write to summary CSV
            row = {
                'fold': k,
                'seed': fold_seed,
                'n_train': len(tr_ids),
                'n_val': len(va_ids),
                'val_final_c': final_c,
                'val_best_c': best_val
            }
            append_row_safely(summary_csv, row)
            
            print(f"\n{'='*80}")
            print(f"✅ FOLD {k} COMPLETE | Final C-index: {best_val:.4f}")
            print(f"{'='*80}\n")

    # Write final summary after all folds
    if is_main:
        maybe_write_final_summary(summary_csv, expected_rows=expected_total_rows)
        
        global_ci = compute_global_oof_cindex(save_root)
        if global_ci is not None:
            print(f"\n🌍 GLOBAL OOF C-index (ALL folds combined): {global_ci:.4f}\n")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--fold", type=int, default=None)
    args = ap.parse_args()
    main(args.config, fold_override=args.fold)