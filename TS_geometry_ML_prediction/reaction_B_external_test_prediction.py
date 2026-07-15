#!/usr/bin/env python3
"""Apply fixed Reaction B geometry models to an external test set.
The script does not refit either model.
The script writes pointwise predictions, test-set metrics, and parity plots.
No compound is excluded unless requested explicitly.

Models:
  bond  = 0.0036*reaA_bond  + 0.0830*sigma_p_plus + 2.2087
  angle = 0.5407*reaA_angle - 1.5256*sigma_p_plus + 39.9721

Example:
  python reaction_B_external_test_prediction.py \
      --input reaction_B_external-set.csv
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DPI = 600
WIDTH = 9.0 / 2.54
REQUIRED = {
    "substituent", "substituent", "sigma-p+", "reaA_Bond_Length", "reaA_Angle",
    "reaB_Bond_Length", "reaB_Angle",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply fixed Reaction B models to an external test set."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument(
        "--output", type=Path,
        default=Path("reaction_B_external_test_predictions"),
    )
    parser.add_argument(
        "--exclude-substituent", action="append", default=[],
        help="Optional pre-specified exclusion; may be repeated.",
    )
    parser.add_argument("--bond-geom-coef", type=float, default=0.0036)
    parser.add_argument("--bond-desc-coef", type=float, default=0.0830)
    parser.add_argument("--bond-intercept", type=float, default=2.2087)
    parser.add_argument("--angle-geom-coef", type=float, default=0.5407)
    parser.add_argument("--angle-desc-coef", type=float, default=-1.5256)
    parser.add_argument("--angle-intercept", type=float, default=39.9721)
    parser.add_argument(
        "--formats", nargs="+", choices=("png", "pdf", "svg"),
        default=("png", "pdf"),
    )
    return parser.parse_args()


def clean_label(value):
    return re.sub(r"^\d+-", "-", str(value).strip())


def load_data(path):
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")

    data = pd.read_csv(path)
    data.columns = data.columns.str.strip()
    missing = sorted(REQUIRED - set(data.columns))
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    data["substituent"] = data["substituent"].map(clean_label)
    numeric = [
        "sigma-p+", "reaA_Bond_Length", "reaA_Angle",
        "reaB_Bond_Length", "reaB_Angle",
    ]
    data[numeric] = data[numeric].apply(pd.to_numeric, errors="coerce")
    data = data.dropna(subset=numeric).reset_index(drop=True)
    if len(data) < 2:
        raise ValueError("At least two complete test compounds are required.")
    return data


def apply_exclusions(data, requested):
    requested = {clean_label(x) for x in requested}
    if not requested:
        return data.copy(), []

    present = sorted(requested & set(data["substituent"]))
    absent = sorted(requested - set(data["substituent"]))
    if absent:
        print("Warning: exclusions not found: " + ", ".join(absent))

    data = data.loc[~data["substituent"].isin(requested)].reset_index(drop=True)
    if len(data) < 2:
        raise ValueError("Fewer than two compounds remain after exclusions.")
    return data, present


def make_predictions(data, args):
    out = data.copy()
    out["predicted_reaB_bond"] = (
        args.bond_geom_coef * out["reaA_Bond_Length"]
        + args.bond_desc_coef * out["sigma-p+"] + args.bond_intercept
    )
    out["predicted_reaB_angle"] = (
        args.angle_geom_coef * out["reaA_Angle"]
        + args.angle_desc_coef * out["sigma-p+"] + args.angle_intercept
    )
    out["bond_residual"] = out["reaB_Bond_Length"] - out["predicted_reaB_bond"]
    out["angle_residual"] = out["reaB_Angle"] - out["predicted_reaB_angle"]
    return out


def metrics_table(data):
    rows = []
    specs = [
        ("Bond length", "reaB_Bond_Length", "predicted_reaB_bond", "angstrom"),
        ("Angle", "reaB_Angle", "predicted_reaB_angle", "degree"),
    ]
    for target, actual_col, pred_col, unit in specs:
        actual = data[actual_col].to_numpy(float)
        pred = data[pred_col].to_numpy(float)
        r2 = np.nan if np.allclose(actual, actual[0]) else r2_score(actual, pred)
        rows.append({
            "target": target, "n_test": len(actual),
            "mae": mean_absolute_error(actual, pred),
            "rmse": mean_squared_error(actual, pred) ** 0.5,
            "r2": r2, "mean_residual": np.mean(actual - pred), "unit": unit,
        })
    return pd.DataFrame(rows)


def limits(actual, predicted):
    values = np.r_[actual, predicted]
    low, high = values.min(), values.max()
    pad = 0.1 * (high - low) if high > low else 0.05
    return low - pad, high + pad


def parity_plot(data, metrics, output, formats):
    plt.rcParams.update({"font.size": 8, "font.family": "sans-serif"})
    fig, axes = plt.subplots(2, 1, figsize=(WIDTH, WIDTH * 1.65))
    specs = [
        ("Bond length", "reaB_Bond_Length", "predicted_reaB_bond", "Bond length (A)", "%.2f"),
        ("Angle", "reaB_Angle", "predicted_reaB_angle", "Angle (degree)", "%.1f"),
    ]
    for ax, (target, actual_col, pred_col, label, tick_fmt) in zip(axes, specs):
        actual = data[actual_col].to_numpy(float)
        pred = data[pred_col].to_numpy(float)
        lim = limits(actual, pred)
        row = metrics.loc[metrics["target"] == target].iloc[0]
        ax.scatter(actual, pred, s=45, edgecolors="black", linewidths=0.7)
        ax.plot(lim, lim, "k--", linewidth=1.0)
        for x, y, name in zip(actual, pred, data["substituent"]):
            ax.annotate(name, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7)
        r2_text = "undefined" if pd.isna(row["r2"]) else f'{row["r2"]:.2f}'
        ax.text(
            0.05, 0.95,
            f'n = {int(row["n_test"])}\nR2 = {r2_text}\nMAE = {row["mae"]:.3f}\nRMSE = {row["rmse"]:.3f}',
            transform=ax.transAxes, va="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
        )
        ax.set(
            xlim=lim, ylim=lim,
            xlabel=f"DFT-optimized {label}",
            ylabel=f"Model-predicted {label}",
            title=target,
        )
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)
        ax.xaxis.set_major_formatter(FormatStrFormatter(tick_fmt))
        ax.yaxis.set_major_formatter(FormatStrFormatter(tick_fmt))
        
    fig.suptitle("Reaction B: external test-set predictions", y=1.01)
    fig.tight_layout()
    for fmt in formats:
        fig.savefig(output / f"external_test_parity.{fmt}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data, excluded = apply_exclusions(load_data(args.input), args.exclude_substituent)
    predictions = make_predictions(data, args)
    metrics = metrics_table(predictions)

    predictions.to_csv(args.output / "external_test_predictions.csv", index=False)
    metrics.to_csv(args.output / "external_test_metrics.csv", index=False)
    parity_plot(predictions, metrics, args.output, args.formats)

    print(f"Evaluated compounds: {len(predictions)}")
    if excluded:
        print("Explicit exclusions: " + ", ".join(excluded))
    print(f"Results directory: {args.output.resolve()}")


if __name__ == "__main__":
    main()
