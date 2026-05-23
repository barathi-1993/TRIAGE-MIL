#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Summarize TRIAGE-MIL cross-validation test C-index results.

Reads:
  results_dir/fold_*/metrics_summary.json

Writes:
  results_dir/cv_test_cindex_by_fold.csv
  results_dir/cv_test_cindex_summary.csv
  results_dir/cv_test_cindex_summary.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Summarize TRIAGE-MIL CV test C-index.")
    parser.add_argument(
        "--results-dir",
        required=True,
        type=str,
        help="Directory containing fold_0, fold_1, ..., fold_4 result folders.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    rows = []
    for fold_dir in sorted(results_dir.glob("fold_*")):
        metrics_file = fold_dir / "metrics_summary.json"

        if not metrics_file.exists():
            print(f"[WARNING] Missing metrics file: {metrics_file}")
            continue

        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        test_ci = metrics.get("test_cindex_at_best_checkpoint", None)

        rows.append(
            {
                "fold": metrics.get("fold", fold_dir.name),
                "n_train": metrics.get("n_train", None),
                "n_val": metrics.get("n_val", None),
                "n_test": metrics.get("n_test", None),
                "best_val_cindex": metrics.get("best_val_cindex", None),
                "test_cindex": test_ci,
                "best_checkpoint_epoch": metrics.get("best_checkpoint_epoch", None),
            }
        )

    if not rows:
        raise RuntimeError(f"No fold metrics found under {results_dir}")

    df = pd.DataFrame(rows)
    df = df.sort_values("fold")

    valid_test = df["test_cindex"].dropna().astype(float).values

    if len(valid_test) == 0:
        raise RuntimeError("No valid test C-index values found.")

    test_mean = float(np.mean(valid_test))
    test_std = float(np.std(valid_test, ddof=1)) if len(valid_test) > 1 else 0.0

    summary = {
        "results_dir": str(results_dir),
        "n_folds_found": int(len(df)),
        "n_valid_test_folds": int(len(valid_test)),
        "test_cindex_mean": test_mean,
        "test_cindex_std": test_std,
        "test_cindex_values": [float(x) for x in valid_test],
    }

    df.to_csv(results_dir / "cv_test_cindex_by_fold.csv", index=False)

    with open(results_dir / "cv_test_cindex_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    pd.DataFrame([summary]).to_csv(results_dir / "cv_test_cindex_summary.csv", index=False)

    print("")
    print("===== TRIAGE-MIL CV Test C-index Summary =====")
    print(df[["fold", "best_val_cindex", "test_cindex", "best_checkpoint_epoch"]])
    print("")
    print(f"Test C-index: {test_mean:.4f} +/- {test_std:.4f} across {len(valid_test)} folds")
    print("")
    print(f"Saved: {results_dir / 'cv_test_cindex_by_fold.csv'}")
    print(f"Saved: {results_dir / 'cv_test_cindex_summary.json'}")
    print(f"Saved: {results_dir / 'cv_test_cindex_summary.csv'}")


if __name__ == "__main__":
    main()
