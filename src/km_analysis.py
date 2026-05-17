#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kaplan-Meier and log-rank analysis for TRIAGE-MIL predictions.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import logrank_test


def analyze_predictions(pred_file: Path, output_file: Path, method: str = "median"):
    df = pd.read_csv(pred_file)

    required = {"time", "event", "risk"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{pred_file} missing columns: {missing}")

    times = df["time"].values
    events = df["event"].values.astype(int)
    risks = df["risk"].values

    if method == "median":
        threshold = np.median(risks)
        groups = (risks >= threshold).astype(int)
        labels = ["Low Risk", "High Risk"]
    elif method == "tertiles":
        thresholds = np.percentile(risks, [33.33, 66.67])
        groups = np.digitize(risks, thresholds)
        labels = ["Low Risk", "Medium Risk", "High Risk"]
    else:
        raise ValueError("method must be 'median' or 'tertiles'")

    low_mask = groups == 0
    high_mask = groups == (len(labels) - 1)

    lr = logrank_test(
        times[high_mask],
        times[low_mask],
        events[high_mask],
        events[low_mask],
    )

    fig, ax = plt.subplots(figsize=(9, 7))
    kmfs = []

    for group_id, label in enumerate(labels):
        mask = groups == group_id
        kmf = KaplanMeierFitter()
        kmf.fit(times[mask], events[mask], label=label)
        kmf.plot_survival_function(ax=ax, ci_show=True, linewidth=2.2)
        kmfs.append(kmf)

    p_text = "p < 0.001" if lr.p_value < 0.001 else f"p = {lr.p_value:.4f}"
    ax.set_title(f"Kaplan-Meier Analysis ({p_text})", fontsize=14, fontweight="bold")
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival Probability")
    ax.grid(True, alpha=0.25)
    ax.set_ylim(0, 1.05)

    add_at_risk_counts(*kmfs, ax=ax, rows_to_show=["At risk"])
    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "prediction_file": str(pred_file),
        "n_patients": int(len(df)),
        "n_events": int(events.sum()),
        "logrank_chi2": float(lr.test_statistic),
        "logrank_p": float(lr.p_value),
    }


def main():
    parser = argparse.ArgumentParser(description="Run Kaplan-Meier analysis.")
    parser.add_argument("--predictions-dir", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--method", default="median", choices=["median", "tertiles"])
    args = parser.parse_args()

    pred_root = Path(args.predictions_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_files = sorted(pred_root.glob("fold_*/**/predictions.csv"))
    if not pred_files:
        pred_files = sorted(pred_root.glob("fold_*/predictions.csv"))

    if not pred_files:
        raise FileNotFoundError(f"No predictions.csv files found under {pred_root}")

    rows = []
    for pred_file in pred_files:
        fold_name = pred_file.parent.name
        if not fold_name.startswith("fold_"):
            fold_name = pred_file.parent.parent.name

        out_file = out_dir / f"{fold_name}_km_curves.png"
        print(f"[KM] {pred_file} -> {out_file}")
        rows.append(analyze_predictions(pred_file, out_file, method=args.method))

    pd.DataFrame(rows).to_csv(out_dir / "km_summary.csv", index=False)
    print(f"[KM] Saved summary to {out_dir / 'km_summary.csv'}")


if __name__ == "__main__":
    main()
