import math

import pandas as pd
import pytest
from pymatgen.core import Structure

from ion_insertion_modeling import config
from ion_insertion_modeling.featurizers import (FEATURE_SETS, IonRedox, PaperFeaturizer,
                                                PolyanionFlag, Transition, manifest)


def _transition(r):
    return Transition.from_states(Structure.from_dict(r["structure"]), r["comp_final"], r["oxi_initial"],
                                  r["oxi_final"], r["p_ox_score_initial"], r["p_ox_score_final"],
                                  comp_initial=r["comp_initial"], framework_formula=r["framework_formula"],
                                  oxo_bonded=r["oxo_bonded"])


@pytest.mark.slow
def test_reproduces_released_features(feature_golden):
    feat = PaperFeaturizer()
    labels = feat.feature_labels()
    for r in feature_golden:
        got = dict(zip(labels, feat.featurize(_transition(r))))
        for c, v in r["features"].items():
            if v is None:
                assert math.isnan(got[c]), (r["transition_id"], c)
            else:
                assert got[c] == pytest.approx(v, rel=1e-6, abs=1e-8), (r["transition_id"], c)


def test_ion_redox_lifepo4(feature_golden):
    r = next(r for r in feature_golden if r["transition_id"].startswith("mp-20361_Li"))
    f = dict(zip(IonRedox().feature_labels(), IonRedox().featurize(_transition(r))))
    assert f["x_initial"] == 1 and f["x_final"] == 0 and f["delta_x"] == -1
    assert f["delta_ox_redox"] == pytest.approx(1.0)  # Fe2+ -> Fe3+
    assert f["delta_li_frac"] == pytest.approx(-1 / 7)


def test_polyanion_flag(feature_golden):
    r = next(r for r in feature_golden if r["transition_id"].startswith("mp-20361_Li"))
    assert PolyanionFlag().featurize(_transition(r)) == [1.0]


def test_labels_match_manifest_and_no_forbidden_names():
    labels = PaperFeaturizer().feature_labels()
    man = manifest()
    assert set(labels) <= set(man)
    assert not [c for c in labels if any(p in c.lower() for p in config.FORBIDDEN_FEATURE_PATTERNS)]
    assert len(labels) == len(set(labels))


def test_feature_sets():
    assert list(FEATURE_SETS["interpretable5"]) == list(config.INTERPRETABLE_FEATURES)
    assert len(FEATURE_SETS["reference"]) == 364
    assert set(FEATURE_SETS["reference"]) <= set(PaperFeaturizer().feature_labels())
    with pytest.raises(KeyError):
        FEATURE_SETS["nope"]


def test_featurize_dataframe(feature_golden):
    r = feature_golden[0]
    df = pd.DataFrame({"transition": [_transition(r)]})
    out = IonRedox().featurize_dataframe(df, "transition", pbar=False)
    assert out["delta_p_ox_score"].iloc[0] == pytest.approx(r["features"]["delta_p_ox_score"])


def test_featurize_rejects_other_inputs():
    with pytest.raises(TypeError):
        IonRedox().featurize("LiFePO4")


@pytest.mark.slow
def test_from_structure_matches_from_states(lifepo4, feature_golden):
    """Transition.from_structure (ChemEnv + P_ox from scratch) gives the same interpretable features."""
    r = next(r for r in feature_golden if r["transition_id"].startswith("mp-20361_Li"))
    t = Transition.from_structure(lifepo4, r["comp_final"])
    feat = PaperFeaturizer()
    got = dict(zip(feat.feature_labels(), feat.featurize(t)))
    for c in FEATURE_SETS["interpretable5"]:
        assert got[c] == pytest.approx(r["features"][c], rel=1e-6), c


@pytest.mark.slow
def test_disordered_framework_sites_and_structure():
    """A disordered garnet (mixed Mn/Mo sites) gets site and structure features from its ordered stand-in:
    9 insertion sites per Pr3(Mn,Mo)2O12 unit (24d + 48g-type garnet sites), edge-sharing framework."""
    from pathlib import Path

    from ion_insertion_modeling.featurizers import _sites, _structure

    s = Structure.from_file(Path(__file__).resolve().parents[1] / "notebooks" / "data" / "disordered_Li7PrMoMnO.cif")
    s.remove_oxidation_states()
    oxi = {"Li": 1, "Pr": 3, "Mn": 2, "Mo": 14 / 3, "O": -2}
    f = _sites.site_features(s, oxi, "Pr3Mo1.5Mn0.5O12")
    assert f["sites_per_fu"] == pytest.approx(9.0)
    assert f["n_fu_cell"] == pytest.approx(8.0)
    g = _structure.structure_features(s, oxi)
    assert not math.isnan(g["frac_edge_sharing"])


def test_ordered_stand_in_leaves_ordered_structures(lifepo4):
    from ion_insertion_modeling.featurizers._sites import ordered_stand_in

    assert ordered_stand_in(lifepo4) == lifepo4
