"""Parity and CV-curve plots in the paper's style (matplotlib; extra [plots])."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ion_insertion_modeling import config


def parity_plot(y_true, y_pred, ax=None, threshold: float | None = config.ABOVE_HULL_THRESHOLD,
                lim_meV: tuple[float, float] = (-100, 400), label: str | None = None, color: str = "tab:blue",
                annotate: bool = True, s: float = 8):
    """Predicted vs true Edecomp of the (de)lithiated materials in meV/atom. Inputs in eV/atom; non-finite
    predictions (finals P1 could not assign) are left out. Dashed lines mark the stability cutoff; the four
    quadrants are the confusion matrix of the stable / unstable call. Returns the Axes."""
    import matplotlib.pyplot as plt

    y, p = np.asarray(y_true, float) * 1000, np.asarray(y_pred, float) * 1000
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    if ax is None:
        _, ax = plt.subplots(figsize=(3.6, 3.4))
    ax.scatter(np.clip(y, *lim_meV), np.clip(p, *lim_meV), s=s, alpha=0.45, color=color, edgecolor="none",
               label=label, rasterized=True)
    ax.plot(lim_meV, lim_meV, color="0.3", lw=0.8)
    if threshold is not None:
        t = threshold * 1000
        ax.axvline(t, color="0.5", ls="--", lw=0.7)
        ax.axhline(t, color="0.5", ls="--", lw=0.7)
    ax.set_xlim(lim_meV)
    ax.set_ylim(lim_meV)
    ax.set_aspect("equal")
    ax.set_xlabel("DFT Edecomp (meV/atom)")
    ax.set_ylabel("Predicted Edecomp (meV/atom)")
    if annotate and len(y):
        mae = float(np.mean(np.abs(y - p)))
        txt = f"MAE {mae:.1f} meV/atom\nn = {len(y)}"
        if threshold is not None:
            from sklearn.metrics import f1_score
            f1 = f1_score(y > threshold * 1000, p > threshold * 1000, average="macro")
            txt += f"\nF1 {f1:.3f}"
        ax.text(0.04, 0.96, txt, transform=ax.transAxes, va="top", fontsize=8)
    return ax


def parity_from_cv(result, n_features="all", **kwargs):
    """parity_plot of a CVResult's out-of-fold predictions at one n_features."""
    p = result.predictions
    p = p[(p.n_features == str(n_features)) & (p.scored_by == "model")]
    return parity_plot(p.Edecomp_final_true, p.Edecomp_final_pred, **kwargs)


def cv_curve(summary: pd.DataFrame, ax=None, metric: str = "mae_meV", label: str | None = None):
    """Metric (fold mean ± std) against the number of features, from CVResult.summary() over integer k."""
    import matplotlib.pyplot as plt

    s = summary[summary.n_features != "all"].copy()
    s["k"] = s.n_features.astype(int)
    s = s.sort_values("k")
    if ax is None:
        _, ax = plt.subplots(figsize=(3.6, 3.0))
    ax.errorbar(s.k, s[f"{metric}_mean"], yerr=s[f"{metric}_std"], marker="o", capsize=3, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("Number of features (k)")
    ax.set_ylabel({"mae_meV": "MAE (meV/atom)", "f1_macro": "Macro F1"}.get(metric, metric))
    return ax
