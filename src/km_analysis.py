#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kaplan-Meier and log-rank analysis for TRIAGE-MIL held-out test predictions.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import logrank_test


REQUIRED_COLUMNS = {"patient_id", "time", "event", "risk"}


def _check_prediction_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction file: {path}")

    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")

    df = df.copy()
    df["patient_id"] = df["patient_id"].astype(str)
    df["time"] = df["time"].astype(float)
    df["event"] = df["event"].astype(int)
    df["risk"] = df["risk"].astype(float)

    return df


def assign_test_groups_using_train_threshold(fold_dir: Path) -> tuple[pd.DataFrame, dict]:
    """
    For one fold:
    - compute risk threshold from train_predictions.csv
    - apply threshold to test_predictions.csv
    """
    train_file = fold_dir / "train_predictions.csv"
    test_file = fold_dir / "test_predictions.csv"

    train_df = _check_prediction_file(train_file)
    test_df = _check_prediction_file(test_file)

    if train_df["risk"].nunique() < 2:
        raise ValueError(f"{train_file} has non-informative/constant risk values.")

    threshold = float(np.median(train_df["risk"].values))

    test_df = test_df.copy()
    test_df["fold"] = fold_dir.name
    test_df["threshold_from_train"] = threshold
    test_df["risk_group"] = np.where(test_df["risk"].values >= threshold, "High Risk", "Low Risk")

    summary = {
        "fold": fold_dir.name,
        "train_file": str(train_file),
        "test_file": str(test_file),
        "train_n": int(len(train_df)),
        "test_n": int(len(test_df)),
        "test_events": int(test_df["event"].sum()),
        "train_median_risk_threshold": threshold,
        "test_low_risk_n": int((test_df["risk_group"] == "Low Risk").sum()),
        "test_high_risk_n": int((test_df["risk_group"] == "High Risk").sum()),
    }

    return test_df, summary


def plot_pooled_km(pooled_df: pd.DataFrame, output_file: Path) -> dict:
    """
    Plot pooled held-out test KM curves and compute log-rank test.
    """
    low_df = pooled_df[pooled_df["risk_group"] == "Low Risk"]
    high_df = pooled_df[pooled_df["risk_group"] == "High Risk"]

    if len(low_df) == 0 or len(high_df) == 0:
        raise ValueError(
            f"Cannot compute KM: low/high groups are empty "
            f"(low={len(low_df)}, high={len(high_df)})."
        )

    lr = logrank_test(
        high_df["time"].values,
        low_df["time"].values,
        event_observed_A=high_df["event"].values,
        event_observed_B=low_df["event"].values,
    )

    fig, ax = plt.subplots(figsize=(9, 7))

    kmf_low = KaplanMeierFitter()
    kmf_high = KaplanMeierFitter()

    kmf_low.fit(
        low_df["time"].values,
        event_observed=low_df["event"].values,
        label=f"Low Risk (n={len(low_df)})",
    )
    kmf_high.fit(
        high_df["time"].values,
        event_observed=high_df["event"].values,
        label=f"High Risk (n={len(high_df)})",
    )

    kmf_low.plot_survival_function(ax=ax, ci_show=True, linewidth=2.2)
    kmf_high.plot_survival_function(ax=ax, ci_show=True, linewidth=2.2)

    p_text = "p < 0.001" if lr.p_value < 0.001 else f"p = {lr.p_value:.4f}"

    ax.set_title(f"Kaplan-Meier Analysis (Log-rank {p_text})", fontsize=14, fontweight="bold")
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival Probability")
    ax.grid(True, alpha=0.25)
    ax.set_ylim(0, 1.05)

    add_at_risk_counts(kmf_low, kmf_high, ax=ax, rows_to_show=["At risk"])
    plt.tight_layout()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "n_patients": int(len(pooled_df)),
        "n_events": int(pooled_df["event"].sum()),
        "low_risk_n": int(len(low_df)),
        "low_risk_events": int(low_df["event"].sum()),
        "high_risk_n": int(len(high_df)),
        "high_risk_events": int(high_df["event"].sum()),
        "logrank_chi2": float(lr.test_statistic),
        "logrank_p": float(lr.p_value),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run pooled held-out test Kaplan-Meier analysis for TRIAGE-MIL."
    )
    parser.add_argument(
        "--predictions-dir",
        required=True,
        type=str,
        help="Root results directory containing fold_0, fold_1, ... folders.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=str,
        help="Directory to save KM figure and CSV summaries.",
    )
    args = parser.parse_args()

    pred_root = Path(args.predictions_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_dirs = sorted([p for p in pred_root.glob("fold_*") if p.is_dir()])
    if not fold_dirs:
        raise FileNotFoundError(f"No fold_* directories found under {pred_root}")

    pooled_rows = []
    fold_summaries = []

    for fold_dir in fold_dirs:
        print(f"[KM] Processing {fold_dir.name}")
        test_df, fold_summary = assign_test_groups_using_train_threshold(fold_dir)
        pooled_rows.append(test_df)
        fold_summaries.append(fold_summary)

    pooled_df = pd.concat(pooled_rows, ignore_index=True)

    pooled_pred_file = out_dir / "pooled_heldout_test_predictions_with_groups.csv"
    fold_summary_file = out_dir / "km_fold_threshold_summary.csv"
    km_summary_file = out_dir / "km_pooled_summary.csv"
    km_figure_file = out_dir / "pooled_heldout_test_km_curves.png"

    pooled_df.to_csv(pooled_pred_file, index=False)
    pd.DataFrame(fold_summaries).to_csv(fold_summary_file, index=False)

    km_summary = plot_pooled_km(pooled_df, km_figure_file)
    pd.DataFrame([km_summary]).to_csv(km_summary_file, index=False)

    print(f"[KM] Saved grouped pooled predictions: {pooled_pred_file}")
    print(f"[KM] Saved fold threshold summary: {fold_summary_file}")
    print(f"[KM] Saved pooled KM summary: {km_summary_file}")
    print(f"[KM] Saved pooled KM figure: {km_figure_file}")
    print(f"[KM] Log-rank p-value: {km_summary['logrank_p']:.4e}")


if __name__ == "__main__":
    main()
