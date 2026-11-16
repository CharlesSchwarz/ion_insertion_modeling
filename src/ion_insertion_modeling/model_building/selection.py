"""Feature ranking and the interpretable-set walk. Both run on one context's training rows only.

rank_features: drop constant columns, keep the top config.SELECT_PRESELECT by random-forest impurity importance,
then rank those by permutation importance (increase in MAE, eV/atom; or loss of macro F1 for a classifier) of the
model on a grouped inner hold-out of config.SELECT_HOLDOUT of the training groups. A model with k features uses the
first k entries.

interpretable_walk: go down a ranking keeping features that are not redundant with one already kept (normalised
mutual information above max_nmi) and do not merely encode the polyanion classes (AUC above max_auc; such a feature
is swapped for an unconfounded substitute measuring the same quantity, or dropped), starting from has_polyanion.
"""
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupShuffleSplit

from ion_insertion_modeling import config
from ion_insertion_modeling.model_building.train import make_estimator

PERM_REPEATS = 5
INTERP_FIXED = ("has_polyanion",)
INTERP_MAX_NMI = 0.5
INTERP_MAX_AUC = 0.7
NMI_BINS = 10
INTERP_SUBSTITUTES = {**dict.fromkeys(("mag_initial_mean_NdValence", "mag_final_mean_NdValence", "fw_cation_mean_NdValence",
                                       "nval_d_mean_initial", "nval_d_mean_final"), "nval_d_redox_initial"),
                      **dict.fromkeys(("p_ox_score_final", "p_ox_score_initial"), "delta_p_ox_score")}


def rank_features(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, model: str = "rf",
                  preselect: int = config.SELECT_PRESELECT, holdout: float = config.SELECT_HOLDOUT,
                  seed: int = config.SEED, task: str = "regression", forced=()) -> list[str]:
    """Columns of X from most to least useful (at most `preselect` plus `forced`), from these (training) rows only.
    forced: columns always ranked first, in that order, and kept in every scoring model; the rest are ranked
    around them. The interpretable walk forces has_polyanion (interpretable_ranking)."""
    forced = list(forced)
    missing = [c for c in forced if c not in X.columns]
    if missing:
        raise KeyError(f"forced features not among the candidates: {missing}")
    X = X.loc[:, X.nunique(dropna=True) > 1]
    rest = X.drop(columns=forced, errors="ignore")
    if rest.shape[1] == 0:
        raise ValueError("no non-constant candidate features")
    imp = make_estimator("rf", seed, task).fit(rest, y).feature_importances_
    cols = list(rest.columns[np.argsort(-imp, kind="stable")[:preselect]])
    tr, va = next(GroupShuffleSplit(n_splits=1, test_size=holdout, random_state=seed).split(X, y, groups))
    use = forced + cols
    est = make_estimator(model, seed, task).fit(X.iloc[tr][use], y[tr])
    scoring = "f1_macro" if task == "classification" else "neg_mean_absolute_error"
    perm = permutation_importance(est, X.iloc[va][use], y[va], scoring=scoring, n_repeats=PERM_REPEATS,
                                  random_state=seed, n_jobs=1)
    imps = perm.importances_mean[len(forced):]
    order = np.argsort(-np.asarray(imps), kind="stable")
    return forced + [cols[i] for i in order]


def interpretable_ranking(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, seed: int = config.SEED) -> list[str]:
    """The ranking the interpretable walk starts from: permutation ranking with has_polyanion forced in first."""
    return rank_features(X, y, groups, seed=seed, forced=INTERP_FIXED)


def similarity(X: pd.DataFrame, metric: str = "nmi") -> pd.DataFrame:
    """Pairwise feature similarity in [0, 1]: |Spearman ρ| ("spearman") or normalised mutual information of
    quantile bins ("nmi"; features with few values keep their values as bins). Rows missing either value are
    dropped per pair."""
    if metric == "spearman":
        sim = X.corr(method="spearman").abs().fillna(0.0)
    elif metric == "nmi":
        from sklearn.metrics import normalized_mutual_info_score

        def bins(s: pd.Series) -> pd.Series:
            if s.nunique() <= NMI_BINS:
                return s.rank(method="dense")
            return pd.qcut(s.rank(method="first"), NMI_BINS, labels=False)

        cols = list(X.columns)
        sim = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
        for i, a in enumerate(cols):
            for b in cols[i + 1:]:
                ok = X[a].notna() & X[b].notna()
                v = normalized_mutual_info_score(bins(X.loc[ok, a]), bins(X.loc[ok, b])) if ok.sum() > 1 else 0.0
                sim.loc[a, b] = sim.loc[b, a] = v
    else:
        raise ValueError(f"unknown similarity metric {metric!r}")
    np.fill_diagonal(sim.values, 1.0)
    return sim


def class_auc(x: pd.Series, classes: pd.Series) -> float:
    """How well x alone separates two classes: ROC AUC folded to [0.5, 1] (rows with x missing dropped)."""
    from sklearn.metrics import roc_auc_score

    ok = x.notna() & classes.notna()
    a = roc_auc_score(classes[ok].astype(int), x[ok])
    return float(max(a, 1 - a))


def interpretable_walk(ranking: list[str], X: pd.DataFrame, k: int = 5, fixed=INTERP_FIXED,
                       max_nmi: float = INTERP_MAX_NMI, max_auc: float = INTERP_MAX_AUC,
                       substitutes: dict | None = None,
                       max_rho: float | None = None) -> tuple[list[str], list[dict]]:
    """Walk down `ranking` on the training rows X and return (features, one log row per feature looked at).
    Start from `fixed`; a feature separating fixed[0]'s classes with AUC > max_auc is replaced by its substitute
    (or dropped); a feature with NMI > max_nmi (or, if max_rho is set, |Spearman ρ| > max_rho) to a kept one is
    skipped. Stops at k features."""
    substitutes = INTERP_SUBSTITUTES if substitutes is None else substitutes
    kept, log = list(fixed), []
    for col in ranking:
        if len(kept) >= k:
            break
        if col in kept:
            continue
        cand, auc = col, class_auc(X[col], X[fixed[0]])
        if auc > max_auc:
            cand = substitutes.get(col)
            if cand is None or cand in kept:
                log.append({"feature": col, "auc": auc, "outcome": "confounded, dropped"})
                continue
            if class_auc(X[cand], X[fixed[0]]) > max_auc:
                log.append({"feature": col, "auc": auc, "outcome": f"confounded; substitute {cand} also confounded"})
                continue
        nmi = similarity(X[kept + [cand]], "nmi").loc[cand, kept]
        if nmi.max() > max_nmi:
            log.append({"feature": col, "auc": auc, "outcome": f"redundant with {nmi.idxmax()} (NMI {nmi.max():.2f})"})
            continue
        if max_rho is not None:
            rho = X[kept + [cand]].astype(float).corr(method="spearman").loc[cand, kept].abs()
            if rho.max() > max_rho:
                log.append({"feature": col, "auc": auc,
                            "outcome": f"redundant with {rho.idxmax()} (Spearman |rho| {rho.max():.2f})"})
                continue
        kept.append(cand)
        log.append({"feature": col, "auc": auc, "outcome": "kept" if cand == col else f"replaced by {cand}"})
    return kept, log
