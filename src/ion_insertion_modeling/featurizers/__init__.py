"""Featurizers for ion insertion/removal transitions, following matminer's BaseFeaturizer.

Every featurizer takes one Transition (see transition.py), which holds the initial structure, the two compositions
and their P1 oxidation states. Use them one at a time, together through PaperFeaturizer, or on a DataFrame column of
Transition objects with featurize_dataframe:

    t = Transition.from_structure(initial_structure, "FePO4")
    PaperFeaturizer().featurize(t)
    df = PaperFeaturizer().featurize_dataframe(df, "transition")

Feature names match the paper's tables. li_frac columns hold the working ion's fraction whatever the ion.
Units are in data/manifest.json (manifest()).
"""
from __future__ import annotations

import json
import warnings
from functools import lru_cache

import numpy as np
from matminer.featurizers.base import BaseFeaturizer, MultipleFeaturizer

from ion_insertion_modeling import config
from ion_insertion_modeling import environments as envs_mod
from ion_insertion_modeling.featurizers import _composition, _li_redox, _oxi_features, _sites, _structure
from ion_insertion_modeling.featurizers.transition import Transition

_PAPER_CITATION = ("@article{schwarz_ion_insertion, title = {Predicting the stability of ion-insertion electrode "
                   "materials on ion insertion and removal}, author = {Schwarz, Charles}, note = {in preparation}}")
_MAGPIE_CITATION = ("@article{ward_magpie_2016, title = {A general-purpose machine learning framework for predicting "
                    "properties of inorganic materials}, author = {Ward, Logan and Agrawal, Ankit and Choudhary, Alok and "
                    "Wolverton, Christopher}, journal = {npj Computational Materials}, volume = {2}, pages = {16028}, "
                    "year = {2016}}")
_CHEMENV_CITATION = ("@article{waroquiers_chemenv_2020, title = {ChemEnv: a fast and robust coordination environment "
                     "identification tool}, author = {Waroquiers, David and others}, journal = {Acta Crystallographica "
                     "Section B}, volume = {76}, pages = {683--695}, year = {2020}}")
_OXI_STATS_CITATION = ("@article{waroquiers_statistical_2017, title = {Statistical analysis of coordination "
                       "environments in oxides}, author = {Waroquiers, David and others}, journal = {Chemistry of "
                       "Materials}, volume = {29}, pages = {8346--8360}, year = {2017}}")


class _TransitionFeaturizer(BaseFeaturizer):
    """Base class: featurize(transition) -> list of floats in feature_labels() order."""

    labels: list[str] = []

    def _features(self, t: Transition) -> dict[str, float]:
        raise NotImplementedError

    def featurize(self, transition: Transition) -> list[float]:
        if not isinstance(transition, Transition):
            raise TypeError("featurize takes a Transition (Transition.from_structure / from_states)")
        f = self._features(transition)
        return [float(f.get(k, np.nan)) for k in self.feature_labels()]

    def feature_labels(self) -> list[str]:
        return list(self.labels)

    def citations(self) -> list[str]:
        return [_PAPER_CITATION]

    def implementors(self) -> list[str]:
        return ["Charles Schwarz"]


class FrameworkMagpie(_TransitionFeaturizer):
    """Magpie statistics (mean, minimum, maximum; amount-weighted) of the framework's cations (fw_cation_*) and
    anions (fw_anion_*), split by the sign of the initial P1 state, plus element-presence flags has_P, has_S,
    has_Si, has_B, has_C, has_N. Framework = initial composition without the working ion. Magpie units."""

    labels = (_composition.magpie_labels("fw_cation") + _composition.magpie_labels("fw_anion")
              + [f"has_{el}" for el in config.POLYANION_ELEMENTS])

    def _features(self, t):
        return _composition.composition_features(t.framework_formula, t.oxi_initial)

    def citations(self):
        return [_PAPER_CITATION, _MAGPIE_CITATION]


class PolyanionFlag(_TransitionFeaturizer):
    """has_polyanion: 1 if any P, S, Si, B, C, N site (or As, Se site with at most 4 neighbours) of the initial
    structure has a first coordination shell made only of O, else 0. Redox-active d-block oxyanions (vanadates,
    chromates, molybdates, tungstates) are deliberately not counted."""

    labels = ["has_polyanion"]

    def _features(self, t):
        oxo = t.oxo_bonded if t.oxo_bonded is not None else envs_mod.oxo_bonded_fraction(t.initial_structure)
        return {"has_polyanion": envs_mod.has_polyanion(oxo)}


class CompositionMagpie(_TransitionFeaturizer):
    """Magpie statistics of the full initial and final compositions (ion included, no oxidation states) and their
    difference: mag_initial_*, mag_final_*, mag_delta_* (final − initial). Magpie units."""

    _stems = [label.split("_", 1)[1] for label in _composition.magpie_labels("x")]
    labels = ([f"mag_initial_{s}" for s in _stems] + [f"mag_final_{s}" for s in _stems]
              + [f"mag_delta_{s}" for s in _stems])

    def _features(self, t):
        return _composition.full_composition_features(t.comp_initial, t.comp_final)

    def citations(self):
        return [_PAPER_CITATION, _MAGPIE_CITATION]


class IonRedox(_TransitionFeaturizer):
    """Ion content and P1 redox change. x_* (ions per framework f.u.), li_frac_* (ion atomic fraction), their
    deltas and |deltas|; delta_ox_mean / _max_abs / _redox (e per atom, final − initial, framework cations);
    n_couples; electrons_per_redox_ion (e⁻ per redox-active atom); p1_score_* (0-1) and delta_p1_score."""

    labels = list(_li_redox.LABELS)

    def _features(self, t):
        return _li_redox.li_redox_features(t.oxi_initial, t.oxi_final, t.framework_formula, t.couples, t.n_couples,
                                           t.x_initial, t.x_final, t.li_frac_initial, t.li_frac_final,
                                           t.p1_score_initial, t.p1_score_final)

    def citations(self):
        return [_PAPER_CITATION, _OXI_STATS_CITATION]


class OxidationStateProperties(_TransitionFeaturizer):
    """Properties of the framework cations at their P1 states, at both ends: ionic radius (Å), next ionization
    energy ie_next (eV), s/p/d/f valence-electron counts nval_* (electrons), as mean / min / max over framework
    cations and for the primary redox element (_redox), plus the largest next electron affinity ea_next_max (eV).
    _delta = final − initial."""

    def feature_labels(self):
        return _oxi_labels()

    def _features(self, t):
        return {**_oxi_features.oxi_features(t.oxi_initial, t.oxi_final, t.framework_formula, t.couples),
                **_oxi_features.valence_features(t.oxi_initial, t.oxi_final, t.framework_formula, t.couples),
                **_oxi_features.ea_features(t.oxi_initial, t.oxi_final, t.framework_formula)}


class FrameworkStructure(_TransitionFeaturizer):
    """Framework (initial structure without the ion, decorated with initial P1 states) descriptors: fractions of
    cation-polyhedron pairs sharing corners / edges / faces, polyhedral volume ratio, framework volume per atom
    (Å³), anion packing ratio, Larsen dimensionality of the bonded network."""

    labels = list(_structure.LABELS)

    def _features(self, t):
        return _cached_structure(t.initial_structure, t.oxi_initial, t.ion)


class InsertionSites(_TransitionFeaturizer):
    """Predicted insertion sites in the initial framework (Voronoi nodes with clearance above the ion radius),
    each measured with one ion inserted: sites_per_fu, Voronoi volume site_vpv_* (Å³), CrystalNN coordination
    site_cn_*, minimum ion–anion distance (Å). At both ends: site_fill_* = x / sites_per_fu and the relative
    Voronoi-volume mismatch rel_vpv_mismatch_* of the worst site needed to hold x ions."""

    labels = list(_sites.LABELS) + ["site_fill_initial", "site_fill_final"] + list(_sites.VPV_MISMATCH_LABELS)

    def _features(self, t):
        rec = _cached_sites(t.initial_structure, t.oxi_initial, t.framework_formula, t.ion)
        out = {k: rec[k] for k in _sites.LABELS}
        spf = rec["sites_per_fu"]
        med = _sites.median_vpv(t.ion)
        for end, x in (("initial", t.x_initial), ("final", t.x_final)):
            out[f"site_fill_{end}"] = x / spf if spf > 0 else np.nan
            out[f"rel_vpv_mismatch_{end}"] = _sites.rel_vpv_mismatch(rec["site_vpvs"], x, rec["n_fu_cell"], med)
        return out


class HullEnergies(_TransitionFeaturizer):
    """Formation energy (eV/atom) of the competing-phase convex hull at the initial and final compositions and
    their difference: hull_ef_initial, hull_ef_final, delta_hull_ef. Needs entries on the MP2020 GGA/GGA+U scale
    for the chemical system, without the final material (and without any other (de)lithiated material of the same
    framework): pass `docs` as [{"mpid", "comp", "energy_per_atom"}] (thermo.mixed_docs) or a PhaseDiagram."""

    labels = ["hull_ef_initial", "hull_ef_final", "delta_hull_ef"]

    def __init__(self, docs=None, phase_diagram=None, exclude: set[str] | None = None):
        self.docs, self.phase_diagram, self.exclude = docs, phase_diagram, set(exclude or ())

    def _features(self, t):
        from ion_insertion_modeling import thermo

        pd_ = self.phase_diagram or thermo.phase_diagram(self.docs, self.exclude)[0]
        ei = thermo.hull_formation_energy(pd_, t.comp_initial)
        ef = thermo.hull_formation_energy(pd_, t.comp_final)
        return {"hull_ef_initial": ei, "hull_ef_final": ef, "delta_hull_ef": ef - ei}


class PaperFeaturizer(MultipleFeaturizer):
    """Every featurizer of the paper except HullEnergies (which needs DFT entries): pass include_hull=True and the
    HullEnergies arguments to add it."""

    def __init__(self, include_hull: bool = False, **hull_kwargs):
        feats = [FrameworkMagpie(), PolyanionFlag(), CompositionMagpie(), IonRedox(), OxidationStateProperties(),
                 FrameworkStructure(), InsertionSites()]
        if include_hull:
            feats.append(HullEnergies(**hull_kwargs))
        super().__init__(feats)

    def featurize_dict(self, transition: Transition) -> dict[str, float]:
        return dict(zip(self.feature_labels(), self.featurize(transition)))


class InterpretableFeaturizer(MultipleFeaturizer):
    """Only what the interpretable-five model needs (fast path for applications): PolyanionFlag, IonRedox,
    OxidationStateProperties and InsertionSites, then select FEATURE_SETS['interpretable5'] from the output."""

    def __init__(self):
        super().__init__([PolyanionFlag(), IonRedox(), OxidationStateProperties(), InsertionSites()])

    def featurize_dict(self, transition: Transition) -> dict[str, float]:
        return dict(zip(self.feature_labels(), self.featurize(transition)))


# --- Caches for the per-structure (slow) groups ---
_STRUCT_CACHE: dict = {}
_SITE_CACHE: dict = {}


def _key(structure, oxi, *extra) -> str:
    return json.dumps([structure.as_dict(), {k: round(float(v), 9) for k, v in sorted(oxi.items())}, *extra],
                      sort_keys=True, default=str)


def _cached_structure(structure, oxi, ion):
    k = _key(structure, oxi, ion)
    if k not in _STRUCT_CACHE:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                _STRUCT_CACHE[k] = _structure.structure_features(structure, oxi, ion)
            except Exception:  # noqa: BLE001 - a failing structure gives NaNs, as in the paper's pipeline
                _STRUCT_CACHE[k] = dict.fromkeys(_structure.LABELS, np.nan)
    return _STRUCT_CACHE[k]


def _cached_sites(structure, oxi, framework_formula, ion):
    k = _key(structure, oxi, framework_formula, ion)
    if k not in _SITE_CACHE:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                _SITE_CACHE[k] = _sites.site_features(structure, oxi, framework_formula, ion)
            except Exception:  # noqa: BLE001
                _SITE_CACHE[k] = {**dict.fromkeys(_sites.LABELS, np.nan), "site_vpvs": None, "n_fu_cell": np.nan}
    return _SITE_CACHE[k]


@lru_cache(maxsize=1)
def _oxi_labels() -> list[str]:
    """Labels of OxidationStateProperties, in the order its functions return them (from a dummy transition)."""
    oi, of = {"Fe": 2.0, "P": 5.0, "O": -2.0}, {"Fe": 3.0, "P": 5.0, "O": -2.0}
    cs = [{"element": "Fe", "ox_from": 2, "ox_to": 3, "label": "Fe2+/Fe3+", "electrons": 1.0, "electrons_per_atom": 1.0}]
    return list({**_oxi_features.oxi_features(oi, of, "FePO4", cs),
                 **_oxi_features.valence_features(oi, of, "FePO4", cs),
                 **_oxi_features.ea_features(oi, of, "FePO4")})


# --- Feature sets and manifest ---
@lru_cache(maxsize=1)
def manifest() -> dict[str, dict]:
    """{feature: {group, level, units}} for every feature the package computes."""
    return json.loads((config.PACKAGE_DATA_DIR / "manifest.json").read_text())


@lru_cache(maxsize=1)
def _feature_sets() -> dict[str, tuple[str, ...]]:
    return {k: tuple(v) for k, v in json.loads((config.PACKAGE_DATA_DIR / "feature_sets.json").read_text()).items()}


class _FeatureSets(dict):
    def __missing__(self, key):
        sets = _feature_sets()
        if key not in sets:
            raise KeyError(f"unknown feature set {key!r}; choose from {sorted(sets)}")
        return sets[key]

    def keys(self):
        return _feature_sets().keys()

    def __iter__(self):
        return iter(_feature_sets())

    def __repr__(self):
        return repr(_feature_sets())


FEATURE_SETS = _FeatureSets()

__all__ = ["Transition", "FrameworkMagpie", "PolyanionFlag", "CompositionMagpie", "IonRedox",
           "OxidationStateProperties", "FrameworkStructure", "InsertionSites", "HullEnergies", "PaperFeaturizer",
           "InterpretableFeaturizer", "FEATURE_SETS", "manifest"]
