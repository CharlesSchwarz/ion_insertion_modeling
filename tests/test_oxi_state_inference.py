import math

import pytest

from ion_insertion_modeling import oxi_state_inference as oxi


def test_reproduces_stored_p1_guesses(p1_golden):
    """P1 from composition + environments reproduces the stored guesses and scores of the released dataset exactly."""
    for r in p1_golden:
        g = oxi.guess_from_environments(r["comp"], r["envs"], oxide_type=r["oxide_type"], fixed=r["fixed"] or None)
        assert g is not None, r["mpid"]
        assert set(g.states) == set(r["states"]), r["mpid"]
        for el, v in r["states"].items():
            assert g.states[el] == pytest.approx(v, abs=1e-6), r["mpid"]
        assert g.score == pytest.approx(r["score"], abs=1e-9), r["mpid"]


def test_per_site_split_mixed_valence():
    """Mn7(P2O7)4: Mn2.29+ with 2 of 14 Mn sites 8-coordinate (Mn3+ unseen there) scores ~0.66 with the split."""
    g = oxi.guess_from_environments("Mn7(P2O7)4", [("Mn", "O:6", 12), ("Mn", "C:8", 2), ("P", "T:4", 16)],
                                    fixed={"P": 5})
    assert g.states["Mn"] == pytest.approx(16 / 7)
    assert g.score == pytest.approx(0.66, abs=0.01)


def test_magnetite():
    g = oxi.guess_from_environments("Fe3O4", [("Fe", "T:4", 1), ("Fe", "O:6", 2)])
    assert g.states["Fe"] == pytest.approx(8 / 3)
    assert g.score == pytest.approx(0.50, abs=0.01)


def test_single_valence_and_single_environment_unchanged():
    """With one environment per element the split equals the geometric interpolation."""
    envs = [("Fe", "O:6", 4), ("P", "T:4", 4)]
    split = oxi.score_guess({"Fe": 2.5, "P": 5, "O": -2}, envs, scoring="P_ox_env_sites_split")
    plain = oxi.score_guess({"Fe": 2.5, "P": 5, "O": -2}, envs, scoring="P_ox_env_sites")
    assert split == pytest.approx(plain)


def test_probabilities():
    assert 0 < oxi.p_ox("Fe", 3) <= 1
    assert oxi.p_ox("Li", 1) == 1.0  # fixed-state element
    assert oxi.p_ox_given_env("Fe", 2, "O:6") > 0
    p25 = oxi.p_ox("Fe", 2.5)
    assert p25 == pytest.approx(math.sqrt(oxi.p_ox("Fe", 2) * oxi.p_ox("Fe", 3)))  # geometric interpolation


def test_candidates_ranked_and_fixed_states():
    g = oxi.guess_from_environments("LiFePO4", [("Fe", "O:6", 4), ("P", "T:4", 4)], fixed={"P": 5})
    assert g.states["Fe"] == pytest.approx(2.0) and g.states["Li"] == 1
    scores = [s for _, s in g.candidates]
    assert scores == sorted(scores, reverse=True)
    assert g.fixed["O"] == -2 and g.fixed["P"] == 5


def test_no_balanced_guess_returns_none():
    """Li and O are both fixed (Li+, O2-): LiO2 cannot balance, and no free element can absorb the charge."""
    assert oxi.guess_from_environments("LiO2", [], oxide_type="oxide", fallback=False) is None


def test_uniform_environments_disordered_formula():
    comp = "Li1.2Mn0.4Ti0.4O2"
    g = oxi.guess_from_environments(comp, oxi.uniform_environments(comp, "O:6"))
    assert g is not None
    charge = 1.2 + 0.4 * g.states["Mn"] + 0.4 * g.states["Ti"] - 4
    assert abs(charge) < 1e-6


@pytest.mark.slow
def test_guess_from_structure_lifepo4(lifepo4):
    g = oxi.guess_from_structure(lifepo4)
    assert g.states["Fe"] == pytest.approx(2.0) and g.states["P"] == pytest.approx(5.0)


@pytest.mark.slow
def test_guess_from_framework_matches_stored(lifepo4, feature_golden):
    """FePO4 scored on the LiFePO4 environments gives the stored final P1 guess and score."""
    rec = next(r for r in feature_golden if r["transition_id"].startswith("mp-20361_Li"))
    g = oxi.guess_from_framework(rec["comp_final"], lifepo4)
    for el, v in rec["oxi_final"].items():
        assert g.states[el] == pytest.approx(v)
    assert g.score == pytest.approx(rec["p1_score_final"], abs=1e-9)


@pytest.mark.slow
def test_best_ion_content_lifepo4(lifepo4):
    x, g, table = oxi.best_ion_content(lifepo4, x_grid=[0, 0.5, 1.0, 1.5])
    assert x in (0.0, 0.5, 1.0) and table.is_best.sum() == 1
    assert len(table) == 4 and g is not None
