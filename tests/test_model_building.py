import numpy as np
import pytest

from ion_insertion_modeling import config


@pytest.mark.release
def test_dataset_counts(li_dataset):
    s = li_dataset.summary()
    assert s["electrodes"] == 2238 and s["test_electrodes"] == 227 and s["transitions"] == 2741


@pytest.mark.release
def test_test_rows_need_confirmation(li_dataset):
    from ion_insertion_modeling.model_building import evaluate_test

    with pytest.raises(PermissionError):
        li_dataset.test()
    with pytest.raises(PermissionError):
        evaluate_test(li_dataset)


@pytest.mark.release
@pytest.mark.slow
def test_interpretable_five_cv_runs(li_dataset):
    """Two grouped folds of the interpretable-five ΔEdecomp RF: sensible MAE, F1 and capacity error."""
    from ion_insertion_modeling.model_building import cross_validate

    r = cross_validate(li_dataset, "delta_edecomp", "interpretable5", ("all",), folds=range(2))
    s = r.summary().iloc[0]
    assert 15 < s.mae_meV_mean < 40 and 0.6 < s.f1_macro_mean < 0.9
    assert set(r.predictions.context) == {"0", "1"}
    assert not set(r.predictions.battery_id) & li_dataset.test_batteries
    assert 0 < r.capacity_summary(li_dataset).iloc[0].cap_mae_mAhg_mean < 80


@pytest.mark.release
def test_bundle_roundtrip(li_dataset, tmp_path):
    from ion_insertion_modeling.model_building import TrainedModel
    from ion_insertion_modeling.model_building.train import make_estimator

    rows = li_dataset.train().head(200)
    cols = list(config.INTERPRETABLE_FEATURES)
    est = make_estimator("rf").set_params(n_estimators=10).fit(rows[cols], rows.delta_Edecomp)
    m = TrainedModel(est, cols, "delta_edecomp", "final", "Li", {})
    m.save(tmp_path / "m")
    m2 = TrainedModel.load(tmp_path / "m")
    assert np.allclose(m.predict(rows[cols]), m2.predict(rows[cols]))
    assert m2.features == cols and m2.target == "delta_edecomp"

