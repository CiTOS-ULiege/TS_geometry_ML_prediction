#!/usr/bin/env python3
"""Descriptor-assisted ridge models for Reaction B TS-geometry prediction.

Each model uses the corresponding Reaction A geometry and one user-selected
numeric descriptor to predict a Reaction B bond length, angle, or dihedral.
Models are fitted with Ridge regression and evaluated by leave-one-out cross-
validation (LOOCV). By default, available Hammett constants are tested, but
other descriptor columns can be supplied with --descriptors.

Example
-------
python reaction_B_machine_learning_model.py \
  --input reaction_B_dataset.csv \
  --output reaction_B_ml_results --alpha 0.1
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut

ALIASES = {
    "name":  ["substituent", "Substituent"],
    "substituent": ["substituent", "Substituent"],
    "reaA_bond": ["reaA-Bond-Length", "reaA_Bond_Length", "reaA_Bond", "reaA-bond"],
    "reaA_angle": ["reaA-Angle", "reaA_Angle", "reaA-angle", "reaA_angle"],
    "reaA_dihedral": ["reaA-Dihedral", "reaA_Dihedral", "reaA-dihedral", "reaA_dihedral"],
    "reaB_bond": ["reaB-Bond-Length", "reaB_Bond_Length", "reaB_Bond", "reaB-bond"],
    "reaB_angle": ["reaB-Angle", "reaB_Angle", "reaB-angle", "reaB_angle"],
    "reaB_dihedral": ["reaB-Dihedral", "reaB_Dihedral", "reaB-dihedral", "reaB_dihedral"],
}
TARGETS = {
    "bond": ("reaA_bond", "reaB_bond", "Bond length", "Å", "%.2f"),
    "angle": ("reaA_angle", "reaB_angle", "Angle", "°", "%.1f"),
    "dihedral": ("reaA_dihedral", "reaB_dihedral", "Dihedral", "°", "%.1f"),
}
DEFAULT_DESCRIPTORS = ["σp+", "σp", "σp-"]


def arguments():
    p = argparse.ArgumentParser(description="Reaction B ridge-regression and LOOCV analysis.")
    p.add_argument("--input", required=True, type=Path, help="Input CSV file.")
    p.add_argument("--output", type=Path, default=Path("reaction_B_ml_results"))
    p.add_argument("--alpha", type=float, default=0.1, help="Ridge alpha (default: 0.1).")
    p.add_argument("--descriptors", nargs="*", help="Descriptor columns to evaluate.")
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def find_column(columns, aliases):
    for name in aliases:
        if name in columns:
            return name
    raise KeyError(f"Missing required column; accepted names: {aliases}")


def load_data(path, requested):
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = pd.read_csv(path)
    raw.columns = raw.columns.str.strip()
    data = pd.DataFrame({key: raw[find_column(raw.columns, vals)] for key, vals in ALIASES.items()})
    descriptors = requested or [d for d in DEFAULT_DESCRIPTORS if d in raw.columns]
    missing = [d for d in descriptors if d not in raw.columns]
    if missing:
        raise KeyError(f"Descriptor columns not found: {missing}")
    if not descriptors:
        raise ValueError("No descriptor columns selected or recognized.")
    numeric = [c for c in data if c not in ("name", "substituent")]
    for col in numeric:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    for desc in descriptors:
        data[desc] = pd.to_numeric(raw[desc], errors="coerce")
    geometry = [c for c in data if c.startswith(("reaA_", "reaB_"))]
    data = data.dropna(subset=geometry).reset_index(drop=True)
    if len(data) < 4:
        raise ValueError("At least four complete geometry records are required.")
    return data, descriptors


def q2(y, pred):
    tss = np.sum((y - y.mean()) ** 2)
    return np.nan if np.isclose(tss, 0) else 1 - np.sum((y - pred) ** 2) / tss


def fit_model(data, descriptor, target, alpha):
    source, response, _, _, _ = TARGETS[target]
    subset = data.dropna(subset=[descriptor, source, response]).reset_index(drop=True)
    if len(subset) < 4:
        return None, None
    X = subset[[source, descriptor]].to_numpy(float)
    y = subset[response].to_numpy(float)
    model = Ridge(alpha=alpha).fit(X, y)
    fitted = model.predict(X)
    loo_pred = np.empty(len(y))
    for train, test in LeaveOneOut().split(X):
        loo_pred[test] = Ridge(alpha=alpha).fit(X[train], y[train]).predict(X[test])
    metrics = {
        "descriptor": descriptor, "target": target, "n": len(y), "alpha": alpha,
        "train_r2": r2_score(y, fitted), "loocv_q2": q2(y, loo_pred),
        "train_mae": mean_absolute_error(y, fitted),
        "train_rmse": mean_squared_error(y, fitted) ** 0.5,
        "loocv_mae": mean_absolute_error(y, loo_pred),
        "loocv_rmse": mean_squared_error(y, loo_pred) ** 0.5,
        "coef_reference_geometry": model.coef_[0],
        "coef_descriptor": model.coef_[1], "intercept": model.intercept_,
    }
    predictions = pd.DataFrame({
        "name": subset["name"], "substituent": subset["substituent"],
        "descriptor": descriptor, "descriptor_value": subset[descriptor],
        "target": target, "reference_geometry": subset[source], "actual": y,
        "fitted_prediction": fitted, "loocv_prediction": loo_pred,
        "loocv_residual": y - loo_pred,
    })
    return metrics, predictions


def run(data, descriptors, alpha):
    metrics, predictions = [], []
    for descriptor in descriptors:
        for target in TARGETS:
            result, pred = fit_model(data, descriptor, target, alpha)
            if result is not None:
                metrics.append(result)
                predictions.append(pred)
    if not metrics:
        raise RuntimeError("No model could be fitted.")
    return pd.DataFrame(metrics), pd.concat(predictions, ignore_index=True)


def write_equations(metrics, path):
    lines = ["Reaction B ridge-regression models", "y = a*(Reaction A geometry) + b*(descriptor) + c", ""]
    for row in metrics.sort_values(["target", "loocv_q2"], ascending=[True, False]).itertuples():
        lines += [
            f"{row.target} | {row.descriptor}",
            f"y = {row.coef_reference_geometry:.8g}*x_reference + {row.coef_descriptor:.8g}*x_descriptor + {row.intercept:.8g}",
            f"n={row.n}; training R2={row.train_r2:.6f}; LOOCV Q2={row.loocv_q2:.6f}; LOOCV MAE={row.loocv_mae:.6f}", "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_best(metrics, predictions, output, dpi):
    best = metrics.loc[metrics.groupby("target")["loocv_q2"].idxmax()]
    fig, axes = plt.subplots(1, 3, figsize=(9, 3), constrained_layout=True)
    for ax, target in zip(axes, TARGETS):
        row = best[best["target"] == target].iloc[0]
        part = predictions[(predictions.target == target) & (predictions.descriptor == row.descriptor)]
        actual, pred = part.actual.to_numpy(), part.loocv_prediction.to_numpy()
        lo, hi = min(actual.min(), pred.min()), max(actual.max(), pred.max())
        pad = 0.08 * (hi - lo) if hi > lo else 0.1
        lim = (lo - pad, hi + pad)
        ax.plot(lim, lim, color="black", lw=1)
        ax.scatter(actual, pred, facecolors="white", edgecolors="black", s=30)
        for _, point in part.iterrows():
            ax.annotate(str(point.substituent), (point.actual, point.loocv_prediction), xytext=(3, 3), textcoords="offset points", fontsize=6)
        _, _, label, unit, fmt = TARGETS[target]
        ax.set(xlim=lim, ylim=lim, xlabel=f"DFT {label} ({unit})", ylabel=f"LOOCV prediction ({unit})", title=f"{label}: {row.descriptor}")
        ax.set_aspect("equal", adjustable="box")
        ax.text(0.04, 0.96, f"Q²={row.loocv_q2:.3f}\nMAE={row.loocv_mae:.3f} {unit}", transform=ax.transAxes, va="top", fontsize=7)
        ax.xaxis.set_major_formatter(FormatStrFormatter(fmt)); ax.yaxis.set_major_formatter(FormatStrFormatter(fmt))
    fig.savefig(output / "best_model_parity.png", dpi=dpi)
    fig.savefig(output / "best_model_parity.pdf")
    plt.close(fig)


def main():
    args = arguments()
    if args.alpha < 0:
        raise ValueError("Ridge alpha must be non-negative.")
    data, descriptors = load_data(args.input, args.descriptors)
    args.output.mkdir(parents=True, exist_ok=True)
    metrics, predictions = run(data, descriptors, args.alpha)
    metrics = metrics.sort_values(["target", "loocv_q2"], ascending=[True, False])
    metrics.to_csv(args.output / "model_metrics.csv", index=False)
    predictions.to_csv(args.output / "model_predictions.csv", index=False)
    write_equations(metrics, args.output / "model_equations.txt")
    plot_best(metrics, predictions, args.output, args.dpi)
    versions = {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__, "matplotlib": matplotlib.__version__, "scikit-learn": sklearn.__version__}
    (args.output / "software_versions.json").write_text(json.dumps(versions, indent=2), encoding="utf-8")
    print(f"Results written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
