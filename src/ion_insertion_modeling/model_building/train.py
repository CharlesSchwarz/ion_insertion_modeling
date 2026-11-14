"""Random-forest models for Eform, Edecomp and ΔEdecomp, cross-validated on grouped folds.

Targets (all eV/atom) and how a prediction becomes the predicted Edecomp of the final material, Ê_final:

    target           rows                         y                Ê_final
    "delta_edecomp"  transitions                  ΔEdecomp         Edecomp_initial + ŷ
    "edecomp"        materials (all) or finals    Edecomp_final    ŷ
    "eform"          materials (all) or finals    formation energy ŷ − ef_ref_final (hull without the material)

trained_on="all" trains on every material of the training electrodes (initial materials included);
"final" on the (de)lithiated materials only. Every target is scored on the same rows: the (de)lithiated material of
each transition, by Ê_final. No leakage: per context (CV fold k, or "test"), feature ranking, hyperparameters and
fitting use that context's training electrodes only.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score

from ion_insertion_modeling import config
from ion_insertion_modeling.model_building import evaluate, splits

N_JOBS = -1
TARGETS = {"delta_edecomp": "delta_Edecomp", "edecomp": "Edecomp_final", "eform": "ef_final"}
CLASSIFIER_GRID = {"min_samples_leaf": [1, 3, 5], "max_features": [0.33, 0.66, 1.0], "class_weight": [None, "balanced"]}


def make_estimator(name: str = "rf", seed: int = config.SEED, task: str = "regression", **params):
    """Unfitted random forest (config.RF_N_ESTIMATORS trees; NaN handled natively)."""
    if name != "rf":
        raise ValueError(f"unknown model {name!r}; this package ships random forests ('rf')")
    cls = RandomForestClassifier if task == "classification" else RandomForestRegressor
    return cls(n_estimators=config.RF_N_ESTIMATORS, random_state=seed, n_jobs=N_JOBS, **params)


def loss(y_true, y_pred, task: str = "regression") -> float:
    """MAE (eV/atom) for regression, 1 − macro F1 for classification."""
    if task == "classification":
        return 1.0 - float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def tune(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, grid: dict | None = None,
         n_inner: int = config.INNER_FOLDS, seed: int = config.SEED, task: str = "regression") -> dict:
    """Grid point with the lowest inner grouped-CV loss on these (training) rows."""
    grid = (CLASSIFIER_GRID if task == "classification" else config.RF_GRID) if grid is None else grid
    points = [dict(zip(grid, v)) for v in itertools.product(*grid.values())]
    if len(points) == 1:
        return points[0]
    folds = splits.assign_folds(groups, n_inner, seed)
    best, best_loss = None, np.inf
    for params in points:
        losses = []
        for k in range(n_inner):
            tr, va = folds != k, folds == k
            pred = make_estimator("rf", seed, task, **params).fit(X[tr], y[tr]).predict(X[va])
            losses.append(loss(y[va], pred, task))
        if np.mean(losses) < best_loss:
            best, best_loss = params, float(np.mean(losses))
    return best


def fit_predict(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, tr: np.ndarray, ev: np.ndarray,
                n_features=("all",), seed: int = config.SEED, task: str = "regression",
                tune_params: bool = True) -> tuple[dict, dict]:
    """({k: predictions on the rows ev}, info) after ranking, tuning and fitting on the rows tr only.
    k = "all" uses every column of X without ranking; integer k uses the top k of the ranking."""
    from ion_insertion_modeling.model_building.selection import rank_features

    X_tr, y_tr, g_tr = X[tr].reset_index(drop=True), y[tr], groups[tr]
    ints = [k for k in n_features if k != "all"]
    ranking = rank_features(X_tr, y_tr, g_tr, seed=seed, task=task) if ints else []
    preds, info = {}, {"ranking": ranking, "per_k": {}}
    for k in n_features:
        use = list(X.columns) if k == "all" else ranking[:k]
        params = tune(X_tr[use], y_tr, g_tr, seed=seed, task=task) if tune_params else {}
        model = make_estimator("rf", seed, task, **params).fit(X_tr[use], y_tr)
        preds[k] = model.predict(X.loc[ev, use])
        info["per_k"][str(k)] = {"n_used": len(use), "features": use, "params": params}
    return preds, info


# --- Cross-validation on a Dataset ---
@dataclass
class CVResult:
    """Out-of-fold predictions of one model configuration.

    predictions: one row per evaluated transition × n_features × fold: transition_id, battery_id, Edecomp_initial,
        Edecomp_final_true, Edecomp_final_pred (eV/atom; +inf for finals P1 could not assign: predicted unstable),
        scored_by ("model" or "assumed_unstable"), n_features, context (fold).
    row_predictions: the target itself per evaluated row (y_true, y_pred).
    selection: per fold, the ranking and per-k features and tuned parameters.
    """
    target: str
    trained_on: str
    features: list[str]
    predictions: pd.DataFrame
    row_predictions: pd.DataFrame
    selection: dict = field(repr=False)
    threshold: float = config.ABOVE_HULL_THRESHOLD

    def summary(self) -> pd.DataFrame:
        """Mean ± std over folds of MAE (meV/atom), R², macro F1 and stable precision/recall, per n_features."""
        return evaluate.summary(self.predictions, self.threshold)

    def capacity(self, dataset, cutoff: float | None = None) -> pd.DataFrame:
        """Per battery and fold: true and predicted stable-range capacity (discrete, mAh/g and Δx)."""
        from ion_insertion_modeling.model_building.capacity_metrics import battery_capacities

        return battery_capacities(self.predictions, dataset.capacity_materials,
                                  cutoff=self.threshold if cutoff is None else cutoff, ion=dataset.ion)

    def capacity_summary(self, dataset, cutoff: float | None = None) -> pd.DataFrame:
        """Capacity MAE (mAh/g), mean ± std over folds, per n_features."""
        from ion_insertion_modeling.model_building.capacity_metrics import capacity_summary

        return capacity_summary(self.capacity(dataset, cutoff))


def resolve_features(features, dataset=None, kind: str = "transition") -> list[str]:
    """A feature list, or the name of one of FEATURE_SETS."""
    from ion_insertion_modeling.featurizers import FEATURE_SETS

    cols = list(FEATURE_SETS[features]) if isinstance(features, str) else list(features)
    if dataset is not None:
        rows = dataset.rows(kind)
        missing = [c for c in cols if c not in rows.columns]
        if missing:
            raise KeyError(f"features not in the dataset: {missing[:5]}")
    return cols


def _rows_kind(target: str, trained_on: str) -> str:
    if target not in TARGETS:
        raise ValueError(f"unknown target {target!r}; choose from {list(TARGETS)}")
    if trained_on not in ("all", "final"):
        raise ValueError("trained_on must be 'all' (every material) or 'final' ((de)lithiated materials only)")
    if target == "delta_edecomp":
        return "transition"
    if target == "edecomp" and trained_on == "final":
        return "transition"
    return "material"


def to_edecomp(target: str, yhat: np.ndarray, ev_rows: pd.DataFrame, ev_t: pd.DataFrame) -> np.ndarray:
    """Ê_final (eV/atom) of the evaluated transitions ev_t from row predictions yhat on ev_rows."""
    pred = pd.Series(yhat, index=ev_rows.row_id.to_numpy()).reindex(ev_t.transition_id).to_numpy()
    if target == "delta_edecomp":
        return ev_t.Edecomp_initial.to_numpy() + pred
    if target == "edecomp":
        return pred
    if target == "eform":
        return pred - ev_t.ef_ref_final.to_numpy()
    raise ValueError(target)


def run_context(dataset, target: str, cols: list[str], context, n_features=("all",), trained_on: str = "final",
                seed: int = config.SEED) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(Ê predictions on the evaluated transitions, row-level predictions, info) for one context: an int CV fold,
    or "test" (training electrodes vs the held-out test set; only through evaluate_test)."""
    kind = _rows_kind(target, trained_on)
    data, t = dataset.rows(kind), dataset.transitions
    train_b, eval_b = splits.context_groups(dataset.battery_split, context)
    tr, ev = data.battery_id.isin(train_b).to_numpy(), data.battery_id.isin(eval_b).to_numpy()
    if kind == "material":
        ev &= ~data.is_identity.to_numpy()  # initial materials are trained on, never scored
        if trained_on == "final":
            tr &= ~data.is_identity.to_numpy()
    ev_rows = data[ev].reset_index(drop=True)
    ev_t = t[t.battery_id.isin(eval_b)].reset_index(drop=True)
    y = data[TARGETS[target]].to_numpy(dtype=float)
    preds, info = fit_predict(data[cols], y, data.battery_id.to_numpy(), tr, ev, n_features, seed)
    e_hat = {k: to_edecomp(target, p, ev_rows, ev_t) for k, p in preds.items()}
    rows = pd.concat([pd.DataFrame({"row_id": ev_rows.row_id, "battery_id": ev_rows.battery_id,
                                    "n_features": str(k), "context": str(context),
                                    "y_true": ev_rows[TARGETS[target]].to_numpy(float), "y_pred": p})
                      for k, p in preds.items()], ignore_index=True)
    return _long(ev_t, e_hat, context, dataset.context_unscored(context)), rows, info


def _long(ev_t: pd.DataFrame, e_hat: dict, context, unscored: pd.DataFrame | None) -> pd.DataFrame:
    cols = ["transition_id", "battery_id", "Edecomp_initial", "Edecomp_final"]
    base = ev_t[cols].rename(columns={"Edecomp_final": "Edecomp_final_true"}).assign(scored_by="model")
    extra = None
    if unscored is not None and len(unscored):
        extra = unscored[cols].rename(columns={"Edecomp_final": "Edecomp_final_true"}).assign(
            scored_by="assumed_unstable", Edecomp_final_pred=np.inf)
    parts = []
    for k, e in e_hat.items():
        parts.append(base.assign(Edecomp_final_pred=e, n_features=str(k), context=str(context)))
        if extra is not None:
            parts.append(extra.assign(n_features=str(k), context=str(context)))
    return pd.concat(parts, ignore_index=True)


def cross_validate(dataset, target: str = "delta_edecomp", features="reference", n_features=("all",),
                   trained_on: str = "final", folds=None, seed: int = config.SEED) -> CVResult:
    """Grouped 5-fold CV on the training electrodes (the test set is never used).

    features: a list of columns or a FEATURE_SETS name ("reference" = the paper's full set, "interpretable5", ...).
    n_features: "all" (every feature, no selection) and/or integers k (top k of each fold's own permutation
        ranking). With a fixed short list such as interpretable5, use ("all",)."""
    kind = _rows_kind(target, trained_on)
    cols = resolve_features(features, dataset, kind)
    folds = range(config.N_FOLDS) if folds is None else folds
    preds, rows, sel = [], [], {}
    for k in folds:
        p, r, info = run_context(dataset, target, cols, k, n_features, trained_on, seed)
        preds.append(p)
        rows.append(r)
        sel[str(k)] = info
    return CVResult(target, trained_on, cols, pd.concat(preds, ignore_index=True), pd.concat(rows, ignore_index=True),
                    sel)


def evaluate_test(dataset, target: str = "delta_edecomp", features="interpretable5", n_features=("all",),
                  trained_on: str = "final", seed: int = config.SEED, confirm: bool = False) -> CVResult:
    """Train on every training electrode and score the held-out test set. The paper's test numbers are final;
    pass confirm=True to run this deliberately (every look at the test set is a use of it)."""
    if not confirm:
        raise PermissionError("evaluate_test scores the held-out test set; pass confirm=True if you mean to")
    kind = _rows_kind(target, trained_on)
    cols = resolve_features(features, dataset, kind)
    p, r, info = run_context(dataset, target, cols, "test", n_features, trained_on, seed)
    return CVResult(target, trained_on, cols, p, r, {"test": info})


def fit_final(dataset, target: str = "delta_edecomp", features="interpretable5", n_features="all",
              trained_on: str = "final", seed: int = config.SEED, include_test: bool = False):
    """TrainedModel fit on every training electrode (and the test electrodes too if include_test, for a model
    meant for applications after the evaluation is done). Hyperparameters are tuned by inner grouped CV."""
    from ion_insertion_modeling.model_building.bundle import TrainedModel
    from ion_insertion_modeling.model_building.selection import rank_features

    kind = _rows_kind(target, trained_on)
    cols = resolve_features(features, dataset, kind)
    data = dataset.rows(kind)
    b = dataset.battery_split
    keep = set(b.battery_id) if include_test else set(b.battery_id[~b.is_test])
    tr = data.battery_id.isin(keep).to_numpy()
    if kind == "material" and trained_on == "final":
        tr &= ~data.is_identity.to_numpy()
    X, y, g = data.loc[tr, cols].reset_index(drop=True), data.loc[tr, TARGETS[target]].to_numpy(float), \
        data.battery_id[tr].to_numpy()
    use = cols if n_features == "all" else rank_features(X, y, g, seed=seed)[:int(n_features)]
    params = tune(X[use], y, g, seed=seed)
    est = make_estimator("rf", seed, **params).fit(X[use], y)
    return TrainedModel(estimator=est, features=list(use), target=target, trained_on=trained_on, ion=dataset.ion,
                        params=params, threshold=config.ABOVE_HULL_THRESHOLD, n_train_rows=int(tr.sum()),
                        includes_test=include_test, data_fingerprint=dataset.fingerprint())
