"""Grouped train/test split and cross-validation folds.

Rows are never split at random: every material of one electrode (battery_id) goes to the same side, because the
(de)lithiated materials of one framework are near-copies of each other and a random split would leak them across.
A split depends only on the row ids, the group labels and the seed, never on a target or feature.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from ion_insertion_modeling import config


def assign_folds(groups: np.ndarray, n_folds: int = config.N_FOLDS, seed: int = config.SEED) -> np.ndarray:
    """Grouped K-fold labels (0..n_folds-1) for rows with the given group labels (GroupKFold shuffled with the seed).
    Rows keep their order; the result depends only on the groups and the seed."""
    folds = np.empty(len(groups), dtype=int)
    gkf = GroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (_, val_idx) in enumerate(gkf.split(np.zeros(len(groups)), groups=groups)):
        folds[val_idx] = fold
    return folds


def grouped_split(ids: pd.DataFrame, id_col: str = "transition_id", group_col: str = "battery_id",
                  test_fraction: float = config.TEST_FRACTION, n_folds: int = config.N_FOLDS,
                  seed: int = config.SEED) -> pd.DataFrame:
    """Grouped held-out test set plus grouped CV folds on the rest.

    ids must hold only the id and group columns (so a target cannot leak into the split). Returns id, group,
    is_test, cv_fold (0..n_folds-1 for training rows, <NA> for test rows), sorted by id."""
    allowed = {id_col, group_col}
    if set(ids.columns) != allowed:
        raise ValueError(f"grouped_split takes only {sorted(allowed)}; got {sorted(ids.columns)}")
    if ids[id_col].duplicated().any():
        raise ValueError(f"duplicate {id_col}")
    ids = ids.sort_values(id_col).reset_index(drop=True)
    groups = ids[group_col].to_numpy()
    train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
                               .split(ids, groups=groups))
    cv_fold = pd.array([pd.NA] * len(ids), dtype="Int64")
    cv_fold[train_idx] = assign_folds(groups[train_idx], n_folds, seed)
    is_test = np.zeros(len(ids), dtype=bool)
    is_test[test_idx] = True
    out = pd.DataFrame({id_col: ids[id_col], group_col: groups, "is_test": is_test, "cv_fold": cv_fold})
    check_grouped(out, group_col)
    return out


def check_grouped(split: pd.DataFrame, group_col: str = "battery_id") -> None:
    """Raise if any group has rows on both sides of the test split or in two CV folds."""
    per = split.groupby(group_col).agg(t=("is_test", "nunique"), f=("cv_fold", lambda s: s.dropna().nunique()))
    if (per.t > 1).any() or (per.f > 1).any():
        raise ValueError("a group is split across the test set or across CV folds")


def context_groups(battery_split: pd.DataFrame, context, group_col: str = "battery_id") -> tuple[set, set]:
    """(training groups, evaluated groups) for CV fold `context` (int) or "test" (all training vs the test set).
    battery_split has one row per group with is_test and cv_fold."""
    b = battery_split.set_index(group_col) if group_col in battery_split.columns else battery_split
    if context == "test":
        return set(b.index[~b.is_test]), set(b.index[b.is_test])
    fold = b.cv_fold.fillna(-1).astype(int)
    return set(b.index[~b.is_test & (fold != context)]), set(b.index[~b.is_test & (fold == context)])
