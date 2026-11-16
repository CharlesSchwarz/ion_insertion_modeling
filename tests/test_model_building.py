import numpy as np
import pytest

from ion_insertion_modeling import config


@pytest.mark.release
def test_dataset_counts(li_dataset):
    s = li_dataset.summary()
    assert s["electrodes"] == 2238 and s["test_electrodes"] == 227 and s["transitions"] == 2741


def test_test_rows_need_confirmation(demo_dataset):
    from ion_insertion_modeling.model_building import evaluate_test

    with pytest.raises(PermissionError):
        demo_dataset.test()
    with pytest.raises(PermissionError):
        evaluate_test(demo_dataset)


@pytest.mark.slow
def test_interpretable_five_cv_runs(demo_dataset):
    """Two grouped folds of the interpretable-five ΔEdecomp RF: sensible MAE, F1 and capacity error."""
    from ion_insertion_modeling.model_building import cross_validate

    r = cross_validate(demo_dataset, "delta_edecomp", "interpretable5", ("all",), folds=range(2))
    s = r.summary().iloc[0]
    assert 10 < s.mae_meV_mean < 60 and 0.5 < s.f1_macro_mean < 0.95
    assert set(r.predictions.context) == {"0", "1"}
    assert 0 < r.capacity_summary(demo_dataset).iloc[0].cap_mae_mAhg_mean < 100


def test_bundle_roundtrip(demo_dataset, tmp_path):
    from ion_insertion_modeling.model_building import TrainedModel
    from ion_insertion_modeling.model_building.train import make_estimator

    rows = demo_dataset.train().head(200)
    cols = list(config.INTERPRETABLE_FEATURES)
    est = make_estimator("rf").set_params(n_estimators=10).fit(rows[cols], rows.delta_Edecomp)
    m = TrainedModel(est, cols, "delta_edecomp", "final", "Li", {})
    m.save(tmp_path / "m")
    m2 = TrainedModel.load(tmp_path / "m")
    assert np.allclose(m.predict(rows[cols]), m2.predict(rows[cols]))
    assert m2.features == cols and m2.target == "delta_edecomp"



def test_demo_has_no_test_electrodes(demo_dataset):
    s = demo_dataset.summary()
    assert s["test_electrodes"] == 0 and s["electrodes"] == 301
    assert set(demo_dataset.transitions.battery_id) <= demo_dataset.train_batteries


def test_pretrained_model_ships_with_package():
    from ion_insertion_modeling.model_building import load_pretrained

    m = load_pretrained()
    assert m.target == "delta_edecomp" and m.features == list(config.INTERPRETABLE_FEATURES)
    assert m.n_train_rows == 2470 and not m.includes_test
