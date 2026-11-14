import numpy as np
import pandas as pd
import pytest

from ion_insertion_modeling.model_building.splits import assign_folds, context_groups, grouped_split


def _ids(n=300, groups=60, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"transition_id": [f"t{i:04d}" for i in range(n)],
                         "battery_id": [f"b{g:03d}" for g in rng.integers(0, groups, n)]})


def test_grouped_and_deterministic():
    a, b = grouped_split(_ids()), grouped_split(_ids().sample(frac=1, random_state=1))
    pd.testing.assert_frame_equal(a, b)
    per = a.groupby("battery_id").agg(t=("is_test", "nunique"), f=("cv_fold", lambda s: s.dropna().nunique()))
    assert (per.t == 1).all() and (per.f <= 1).all()
    assert 0.03 < a.is_test.mean() < 0.2


def test_refuses_other_columns():
    with pytest.raises(ValueError):
        grouped_split(_ids().assign(delta_Edecomp=0.0))


def test_assign_folds_groups_together():
    g = np.array(list("aabbccddeeff"))
    f = assign_folds(g, 3)
    assert all(len(set(f[g == x])) == 1 for x in set(g))


def test_context_groups():
    b = pd.DataFrame({"battery_id": ["a", "b", "c", "d"], "is_test": [False, False, False, True],
                      "cv_fold": pd.array([0, 1, 1, pd.NA], dtype="Int64")})
    assert context_groups(b, 1) == ({"a"}, {"b", "c"})
    assert context_groups(b, "test") == ({"a", "b", "c"}, {"d"})
