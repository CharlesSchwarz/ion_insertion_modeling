"""Regression and classification metrics of predicted Edecomp (eV/atom; *_meV copies in meV/atom).

The classification is derived from the regression: a material is predicted unstable if Ê_final > threshold
(config.ABOVE_HULL_THRESHOLD = 75 meV/atom). Materials P1 could not assign get Ê = +inf: no energy error, and they
count as predicted unstable.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from ion_insertion_modeling import config

SUMMARY_KEYS = ("mae_meV", "rmse_meV", "r2", "f1_macro", "precision_stable", "recall_stable")


def regression_metrics(y, yhat) -> dict:
    """n, MAE and RMSE (eV/atom and meV/atom), R²."""
    y, yhat = np.asarray(y, dtype=float), np.asarray(yhat, dtype=float)
    if len(y) == 0:
        return {"n": 0, **dict.fromkeys(("mae", "rmse", "mae_meV", "rmse_meV", "r2"), float("nan"))}
    resid = y - yhat
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    mae, rmse = float(np.mean(np.abs(resid))), float(np.sqrt(np.mean(resid ** 2)))
    return {"n": int(len(y)), "mae": mae, "rmse": rmse, "mae_meV": 1000 * mae, "rmse_meV": 1000 * rmse,
            "r2": float(1 - np.sum(resid ** 2) / ss_tot) if ss_tot > 0 else float("nan")}


def classification_metrics(e_true, e_pred, threshold: float = config.ABOVE_HULL_THRESHOLD) -> dict:
    """Macro F1 (the paper's headline), weighted F1, per-class precision/recall/F1 and the confusion matrix for
    the label Edecomp > threshold (eV/atom)."""
    y = np.asarray(e_true, dtype=float) > threshold
    yhat = np.asarray(e_pred, dtype=float) > threshold
    labels = [False, True]
    p, r, f, s = precision_recall_fscore_support(y, yhat, labels=labels, zero_division=0)
    cm = confusion_matrix(y, yhat, labels=labels)
    return {"f1_macro": float(f1_score(y, yhat, labels=labels, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(y, yhat, labels=labels, average="weighted", zero_division=0)),
            "precision_stable": float(p[0]), "recall_stable": float(r[0]), "f1_stable": float(f[0]),
            "precision_above": float(p[1]), "recall_above": float(r[1]), "f1_above": float(f[1]),
            "n_true_above": int(s[1]), "n_true_stable": int(s[0]),
            "tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1])}


def metrics(g: pd.DataFrame, threshold: float = config.ABOVE_HULL_THRESHOLD) -> dict:
    """Regression metrics on the rows with a finite prediction and classification metrics on every row, for a
    predictions frame with Edecomp_final_true and Edecomp_final_pred."""
    pred, true = g.Edecomp_final_pred.to_numpy(dtype=float), g.Edecomp_final_true.to_numpy(dtype=float)
    scored = np.isfinite(pred)
    return {**regression_metrics(true[scored], pred[scored]), "n_assumed_unstable": int((~scored).sum()),
            **classification_metrics(true, pred, threshold)}


def summarise_folds(per_fold: list[dict], keys=SUMMARY_KEYS) -> dict:
    """Mean and standard deviation (ddof = 1) over folds of each metric."""
    df = pd.DataFrame(per_fold)
    out = {}
    for m in keys:
        out[f"{m}_mean"] = float(df[m].mean())
        out[f"{m}_std"] = float(df[m].std(ddof=1)) if len(df) > 1 else 0.0
    return out


def summary(preds: pd.DataFrame, threshold: float = config.ABOVE_HULL_THRESHOLD) -> pd.DataFrame:
    """One row per n_features: fold mean ± std of SUMMARY_KEYS and the pooled values."""
    out = []
    for k, g in preds.groupby("n_features", sort=False):
        per = [metrics(h, threshold) for _, h in g.groupby("context")]
        out.append({"n_features": k, "n_folds": len(per), **summarise_folds(per),
                    **{f"pooled_{c}": v for c, v in metrics(g, threshold).items() if c in SUMMARY_KEYS}})
    return pd.DataFrame(out)


def format_pm(mean: float, std: float, digits: int = 1) -> str:
    """'23.4 ± 1.4'."""
    return f"{mean:.{digits}f} ± {std:.{digits}f}"
