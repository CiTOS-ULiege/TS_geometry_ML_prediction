#!/usr/bin/env python3
"""Compare linear regression methods for Reaction B TS-geometry prediction.

Each model uses the corresponding Reaction A geometry and one numerical
chemical descriptor to predict a Reaction B bond length, angle, or dihedral.
Fixed Ridge, Lasso, Elastic Net, and Bayesian Ridge configurations are compared
by full-data fitting and leave-one-out cross-validation (LOOCV).

Example
-------
python reaction_B_method_comparison.py \
  --input reaction_B_dataset.csv \
  --output reaction_B_method_comparison --descriptor "σp+"
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import BayesianRidge, ElasticNet, Lasso, Ridge
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
    "bond": ("reaA_bond", "reaB_bond", "Bond length", "Å"),
    "angle": ("reaA_angle", "reaB_angle", "Angle", "°"),
    "dihedral": ("reaA_dihedral", "reaB_dihedral", "Dihedral", "°"),
}
METHODS = {
    "Ridge (alpha=0.1)": Ridge(alpha=0.1),
    "Ridge (alpha=1.0)": Ridge(alpha=1.0),
    "Lasso (alpha=0.01)": Lasso(alpha=0.01, max_iter=10000),
    "Lasso (alpha=0.1)": Lasso(alpha=0.1, max_iter=10000),
    "Elastic Net (alpha=0.1, l1_ratio=0.3)": ElasticNet(alpha=0.1, l1_ratio=0.3, max_iter=10000),
    "Elastic Net (alpha=0.1, l1_ratio=0.8)": ElasticNet(alpha=0.1, l1_ratio=0.8, max_iter=10000),
    "Bayesian Ridge": BayesianRidge(),
}


def parse_args():
    p = argparse.ArgumentParser(description="Compare regression methods for Reaction B.")
    p.add_argument("--input", required=True, type=Path, help="Input CSV file.")
    p.add_argument("--output", type=Path, default=Path("reaction_B_method_comparison"))
    p.add_argument("--descriptor", default="σp+", help="Numerical descriptor column.")
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def find_column(columns, aliases):
    for name in aliases:
        if name in columns:
            return name
    raise KeyError(f"Missing required column; accepted names: {aliases}")


def load_data(path, descriptor):
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = pd.read_csv(path).replace(r"^\s*$", np.nan, regex=True)
    selected = {key: raw[find_column(raw.columns, names)] for key, names in ALIASES.items()}
    if descriptor not in raw.columns:
        raise KeyError(f"Descriptor column not found: {descriptor}")
    selected["descriptor"] = raw[descriptor]
    data = pd.DataFrame(selected)
    numeric = [c for c in data.columns if c not in ("name", "substituent")]
    for col in numeric:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    before = len(data)
    data = data.dropna(subset=numeric).reset_index(drop=True)
    if len(data) < 4:
        raise ValueError("At least four complete observations are required.")
    print(f"Loaded {before} rows; retained {len(data)} complete observations.")
    return data


def q2(y, pred):
    tss = np.sum((y - y.mean()) ** 2)
    return np.nan if np.isclose(tss, 0) else 1 - np.sum((y - pred) ** 2) / tss


def loocv(model, X, y):
    pred = np.empty(len(y))
    for train, test in LeaveOneOut().split(X):
        fitted = clone(model).fit(X[train], y[train])
        pred[test] = fitted.predict(X[test])
    return pred


def run_models(data):
    metrics, predictions, coefficients = [], [], []
    for method_name, model in METHODS.items():
        for target, (source, response, _, _) in TARGETS.items():
            X = data[[source, "descriptor"]].to_numpy(float)
            y = data[response].to_numpy(float)
            fitted_model = clone(model).fit(X, y)
            fitted = fitted_model.predict(X)
            loo = loocv(model, X, y)
            metrics.append({
                "method": method_name, "target": target, "n": len(y),
                "train_r2": r2_score(y, fitted), "loocv_q2": q2(y, loo),
                "train_mae": mean_absolute_error(y, fitted),
                "train_rmse": mean_squared_error(y, fitted) ** 0.5,
                "loocv_mae": mean_absolute_error(y, loo),
                "loocv_rmse": mean_squared_error(y, loo) ** 0.5,
            })
            coef = np.asarray(fitted_model.coef_).ravel()
            coefficients.append({
                "method": method_name, "target": target,
                "coefficient_reaction_A_geometry": coef[0],
                "coefficient_descriptor": coef[1],
                "intercept": float(np.asarray(fitted_model.intercept_).ravel()[0]),
            })
            for i, row in data.iterrows():
                predictions.append({
                    "method": method_name, "target": target, "name": row["name"],
                    "substituent": row["substituent"], "actual": y[i],
                    "fitted_prediction": fitted[i], "loocv_prediction": loo[i],
                    "loocv_residual": y[i] - loo[i],
                })
    return pd.DataFrame(metrics), pd.DataFrame(predictions), pd.DataFrame(coefficients)


def make_summary(metrics):
    q = metrics.pivot(index="method", columns="target", values="loocv_q2")
    summary = q.rename(columns=lambda c: f"loocv_q2_{c}")
    summary["mean_loocv_q2_all"] = q.mean(axis=1)
    summary["mean_loocv_q2_bond_angle"] = q[["bond", "angle"]].mean(axis=1)
    return summary.reset_index().sort_values("mean_loocv_q2_bond_angle", ascending=False)


def save_plots(metrics, summary, out, dpi):
    order = summary["method"].tolist()
    short = [m.replace("Elastic Net", "ElasticNet").replace("Bayesian Ridge", "Bayesian") for m in order]
    table = metrics.set_index(["method", "target"])
    x = np.arange(len(order)); width = 0.36
    fig, axes = plt.subplots(3, 1, figsize=(7, 8), constrained_layout=True)
    for ax, (target, (_, _, label, _)) in zip(axes, TARGETS.items()):
        rows = table.loc[[(m, target) for m in order]]
        ax.bar(x - width / 2, rows["train_r2"], width, label="Training R²")
        ax.bar(x + width / 2, rows["loocv_q2"], width, label="LOOCV Q²")
        ax.axhline(0, color="black", lw=0.8); ax.set_title(label); ax.set_ylabel("Score")
        ax.set_xticks(x); ax.set_xticklabels(short, rotation=35, ha="right", fontsize=7)
    axes[0].legend(frameon=False, ncol=2)
    fig.savefig(out / "method_comparison.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(out / "method_comparison.pdf", bbox_inches="tight"); plt.close(fig)

    ranked = summary.sort_values("mean_loocv_q2_bond_angle")
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.barh(ranked.method, ranked.mean_loocv_q2_bond_angle)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Mean LOOCV Q² for bond and angle")
    ax.set_title("Reaction B method ranking")
    fig.savefig(out / "method_ranking.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(out / "method_ranking.pdf", bbox_inches="tight"); plt.close(fig)


def main():
    args = parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    data = load_data(args.input, args.descriptor)
    metrics, predictions, coefficients = run_models(data)
    summary = make_summary(metrics)
    metrics.to_csv(args.output / "method_target_metrics.csv", index=False)
    summary.to_csv(args.output / "method_summary.csv", index=False)
    predictions.to_csv(args.output / "loocv_predictions.csv", index=False)
    coefficients.to_csv(args.output / "fitted_model_coefficients.csv", index=False)
    save_plots(metrics, summary, args.output, args.dpi)
    print(f"Results written to: {args.output.resolve()}")
    print(f"Highest-ranked method: {summary.iloc[0]['method']}")


if __name__ == "__main__":
    main()
