#!/usr/bin/env python3
"""Screen descriptors for Reaction C transition-state geometry models.

This script compares candidate nucleophile descriptors for Reaction C TS-geometry
prediction, including Hammett-related parameters (for example σp+, σp, σp-) and
other DFT-derived descriptors (for example EHOMO, ELUMO, η, ω, N, B1, B5, L,
v%, f-c, Nc, and ωc).

For each descriptor, separate Ridge models combine the corresponding Reaction A
geometry with that descriptor to predict Reaction C bond length, angle, and
dihedral. Models are evaluated by leave-one-out cross-validation (LOOCV).
Descriptors are ranked by dihedral LOOCV Q².
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut

DEFAULT_DESCRIPTORS = [
    "σp+", "σp", "σp-", "EHOMO", "ELUMO", "η", "ω", "N",
    "B1", "B5", "L", "v%", "f-c", "Nc", "ωc",
]
ALIASES = {
    "name": ["substituent", "Substituent"],
    "substituent": ["substituent", "Substituent"],
    "reaA_bond": ["reaA-Bond-Length", "reaA_Bond_Length", "reaA_Bond", "reaA-bond"],
    "reaA_angle": ["reaA-Angle", "reaA_Angle", "reaA-angle", "reaA_angle"],
    "reaA_dihedral": ["reaA-Dihedral", "reaA_Dihedral", "reaA-dihedral", "reaA_dihedral"],
    "reaC_bond": ["reaC-Bond-Length", "reaC_Bond_Length", "reaC_Bond", "reaC-bond"],
    "reaC_angle": ["reaC-Angle", "reaC_Angle", "reaC-angle", "reaC_angle"],
    "reaC_dihedral": ["reaC-Dihedral", "reaC_Dihedral", "reaC-dihedral", "reaC_dihedral"],
}
TARGETS = {
    "bond": ("reaA_bond", "reaC_bond"),
    "angle": ("reaA_angle", "reaC_angle"),
    "dihedral": ("reaA_dihedral", "reaC_dihedral"),
}
RANKING_TARGET = "dihedral"
RANKING_LABEL = r"LOOCV $Q^2$ (dihedral)"
FIGURE_TITLE = "Reaction C descriptor ranking"
DEFAULT_INPUT = Path(__file__).resolve().parent / "datasets/reaction_C_dataset.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results/reaction_C_descriptor_screening"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT,
        help=f"Input CSV file (default: {DEFAULT_INPUT.name})",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--alpha", type=float, default=0.1, help="Ridge alpha (default: 0.1).")
    parser.add_argument(
        "--descriptors", nargs="*", default=None,
        help="Descriptor columns to screen; default: recognized descriptors.",
    )
    parser.add_argument("--top-n", type=int, default=10, help="Maximum descriptors shown.")
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def find_column(columns, aliases):
    match = next((name for name in aliases if name in columns), None)
    if match is None:
        raise KeyError("Missing required column; accepted names: " + ", ".join(aliases))
    return match


def load_data(path, requested):
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")

    raw = pd.read_csv(path).replace(r"^\s*$", np.nan, regex=True)
    raw.columns = raw.columns.str.replace("\u00a0", "", regex=False).str.strip()
    data = pd.DataFrame({
        key: raw[find_column(raw.columns, aliases)]
        for key, aliases in ALIASES.items()
    })

    descriptors = requested or [name for name in DEFAULT_DESCRIPTORS if name in raw.columns]
    missing = [name for name in descriptors if name not in raw.columns]
    if missing:
        raise KeyError("Descriptor columns not found: " + ", ".join(missing))
    if not descriptors:
        raise ValueError("No descriptor columns were selected or recognized.")

    geometry = [column for column in data if column.startswith(("reaA_", "reaC_"))]
    for column in geometry:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    for descriptor in descriptors:
        data[descriptor] = pd.to_numeric(raw[descriptor], errors="coerce")

    data = data.dropna(subset=geometry).reset_index(drop=True)
    if len(data) < 4:
        raise ValueError("At least four complete geometry records are required.")
    return data, descriptors


def q2(y, pred):
    tss = np.sum((y - y.mean()) ** 2)
    return np.nan if np.isclose(tss, 0.0) else 1.0 - np.sum((y - pred) ** 2) / tss


def loo_predictions(X, y, alpha):
    pred = np.empty(len(y), dtype=float)
    for train, test in LeaveOneOut().split(X):
        pred[test] = Ridge(alpha=alpha).fit(X[train], y[train]).predict(X[test])
    return pred


def screen_descriptors(data, descriptors, alpha):
    rows = []
    for descriptor in descriptors:
        for target, (source, response) in TARGETS.items():
            subset = data.dropna(subset=[descriptor, source, response]).reset_index(drop=True)
            if len(subset) < 4:
                continue

            X = subset[[source, descriptor]].to_numpy(float)
            y = subset[response].to_numpy(float)
            model = Ridge(alpha=alpha).fit(X, y)
            fitted = model.predict(X)
            loo = loo_predictions(X, y, alpha)
            rows.append({
                "descriptor": descriptor,
                "target": target,
                "n_samples": len(y),
                "alpha": alpha,
                "train_r2": r2_score(y, fitted),
                "loocv_q2": q2(y, loo),
                "loocv_mae": mean_absolute_error(y, loo),
                "loocv_rmse": mean_squared_error(y, loo) ** 0.5,
            })

    metrics = pd.DataFrame(rows)
    if metrics.empty:
        raise RuntimeError("No descriptor-target model could be evaluated.")

    q2_table = metrics.pivot(index="descriptor", columns="target", values="loocv_q2")
    n_table = metrics.pivot(index="descriptor", columns="target", values="n_samples")
    complete = q2_table.dropna(subset=list(TARGETS))
    if complete.empty:
        raise RuntimeError("No descriptor has complete bond, angle, and dihedral results.")

    ranking = complete.rename(columns=lambda name: f"loocv_q2_{name}")
    ranking["n_samples"] = n_table.loc[ranking.index, RANKING_TARGET]
    ranking = ranking.reset_index()[[
        "descriptor", "n_samples", "loocv_q2_bond",
        "loocv_q2_angle", "loocv_q2_dihedral",
    ]].sort_values(f"loocv_q2_{RANKING_TARGET}", ascending=False)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    return metrics.sort_values(["target", "loocv_q2"], ascending=[True, False]), ranking


def descriptor_label(name):
    labels = {
        "σp+": r"$\sigma_{\mathrm{p}}^{+}$",
        "σp": r"$\sigma_{\mathrm{p}}$",
        "σp-": r"$\sigma_{\mathrm{p}}^{-}$",
        "EHOMO": r"$E_{\mathrm{HOMO}}$",
        "ELUMO": r"$E_{\mathrm{LUMO}}$",
        "η": r"$\eta$",
        "ω": r"$\omega$",
        "N": r"${\mathrm{N}}$",
        "v%": r"${\mathrm{v\%}}$",
        "f-c": r"$f_{\mathrm{c}}$",
        "Nc": r"$N_{\mathrm{c}}$",
        "ωc": r"$\omega_{\mathrm{c}}$",
    }
    return labels.get(name, name)


def save_figure(ranking, output, top_n, dpi):
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "axes.linewidth": 0.8})
    shown = ranking.head(max(1, top_n)).sort_values(f"loocv_q2_{RANKING_TARGET}")
    values = shown[f"loocv_q2_{RANKING_TARGET}"].to_numpy(float)
    labels = [descriptor_label(name) for name in shown["descriptor"]]

    fig, ax = plt.subplots(figsize=(3.5, max(2.5, 0.34 * len(shown) + 1.2)))
    ax.barh(np.arange(len(shown)), values)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_yticks(np.arange(len(shown)))
    ax.set_yticklabels(labels)
    ax.set_xlabel(RANKING_LABEL)
    ax.set_ylabel("Descriptor")
    ax.set_title(FIGURE_TITLE)
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output / "figure.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output / "figure.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    if args.alpha < 0:
        raise ValueError("Ridge alpha must be non-negative.")
    if args.top_n < 1:
        raise ValueError("--top-n must be at least 1.")

    args.output.mkdir(parents=True, exist_ok=True)
    data, descriptors = load_data(args.input, args.descriptors)
    metrics, ranking = screen_descriptors(data, descriptors, args.alpha)
    metrics.to_csv(args.output / "descriptor_target_metrics.csv", index=False)
    ranking.to_csv(args.output / "descriptor_ranking.csv", index=False)
    save_figure(ranking, args.output, args.top_n, args.dpi)
    print(ranking.to_string(index=False))
    print(f"Results written to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
