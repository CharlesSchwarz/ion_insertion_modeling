import numpy as np
import pytest
from pymatgen.core import Composition, Element

from ion_insertion_modeling import formulas, thermo
from ion_insertion_modeling.insertion_stability import (energy_density, grav_capacity, lower_envelope,
                                                        stable_range, VoltageProfile)


def test_lifepo4_theoretical_capacity():
    cap = grav_capacity(1.0, 1.0, Composition("FePO4").weight, float(Element("Li").atomic_mass))
    assert cap == pytest.approx(169.9, abs=0.3)


def test_stable_range_contiguous():
    assert stable_range([True, False, True, True, False], 2) == (2, 3)
    assert stable_range([True, False, True, True, False], 2, contiguous=False) == (0, 3)
    assert stable_range([False, False, False], 1) == (1, 1)  # the initial state always counts


def test_lower_envelope():
    x = np.array([0.0, 0.5, 1.0])
    assert lower_envelope(x, np.array([0.0, -1.0, 0.0])) == [0, 1, 2]
    assert lower_envelope(x, np.array([0.0, 1.0, 0.0])) == [0, 2]


def test_formulas():
    assert formulas.framework_formula("LiMn2O4") == "MnO2"
    assert formulas.ion_per_fu("LiMn2O4", "MnO2") == pytest.approx(0.5)
    assert formulas.with_ion_content("FePO4", 0.25).replace(" ", "") == "Li1Fe4P4O16"
    with pytest.raises(ValueError):
        formulas.framework_units("LiFeO2", "FePO4")


def test_energy_density_two_steps():
    import pandas as pd

    steps = pd.DataFrame({"x_from": [0.0, 0.5], "x_to": [0.5, 1.0], "V": [4.0, 3.0]})
    p = VoltageProfile("FePO4", "Li", 1.0, steps, pd.DataFrame(), -1.9, None)
    e = energy_density(p)
    cap = grav_capacity(1.0, 1.0, Composition("FePO4").weight, float(Element("Li").atomic_mass))
    assert e.capacity_mAh_g == pytest.approx(cap)
    assert e.wh_per_kg == pytest.approx(3.5 * cap)
    assert e.mean_voltage == pytest.approx(3.5)


def test_edecomp_convention():
    docs = [{"mpid": "a", "comp": "Li", "energy_per_atom": 0.0}, {"mpid": "b", "comp": "O", "energy_per_atom": 0.0},
            {"mpid": "c", "comp": "Li2O", "energy_per_atom": -2.0}]
    assert thermo.edecomp_of("Li2O", -1.9, docs) == pytest.approx(0.1)   # above the hull
    assert thermo.edecomp_of_mpid("c", docs) == pytest.approx(-2.0)        # stable: distance below Li + O


@pytest.mark.slow
def test_lifepo4_workflow(demo_dataset, lifepo4):
    """LiFePO4 with the pretrained model: stable over x = 0-1, ~170 mAh/g, one step near 3.47 V."""
    from ion_insertion_modeling import insertion_stability as ins
    from ion_insertion_modeling.model_building import load_pretrained

    model = load_pretrained()
    li_dataset = demo_dataset
    grid = [0, 0.25, 0.5, 0.75, 1.0, 1.25]
    cap = ins.predict_capacity(lifepo4, -0.008219, model, x_grid=grid)
    assert (cap.x_lo, cap.x_hi) == (0.0, 1.0)
    assert cap.capacity_mAh_g == pytest.approx(169.9, abs=0.3)
    docs = li_dataset.hull_docs(thermo.chemsys("LiFePO4"))
    finals = set(li_dataset.transitions.query("battery_id == 'mp-20361_Li'").mpid_final)
    e0 = next(d["energy_per_atom"] for d in docs if d["mpid"] == "mp-19017")
    vp = ins.predict_voltage_profile(lifepo4, -0.008219, model, docs=docs, exclude=finals, x_grid=grid[:5],
                                     initial_energy_per_atom=e0)
    assert len(vp.steps) == 1 and vp.steps.V.iloc[0] == pytest.approx(3.47, abs=0.05)
    e = ins.energy_density(vp)
    assert e.wh_per_kg == pytest.approx(589, abs=10)
