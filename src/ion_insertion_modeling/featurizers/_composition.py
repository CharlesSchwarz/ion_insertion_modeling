"""Groups 1–2: Magpie statistics over the framework's cations and anions, plus polyanion flags.

The framework is the initial material with Li removed (framework_formula, per battery). Cations are elements with
initial P1 oxidation state >= 0 (polyanion centres such as P5+ included; decided 2026-09-25), anions < 0.
Magpie = matminer ElementProperty, magpie data (minus config.MAGPIE_EXCLUDE), stats config.MAGPIE_STATS,
amount-weighted.
"""
import warnings
from functools import lru_cache

from pymatgen.core import Composition

from ion_insertion_modeling import config


@lru_cache(maxsize=1)
def _magpie():
    from matminer.featurizers.composition import ElementProperty

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        preset = ElementProperty.from_preset("magpie")
        feats = [f for f in preset.features if f not in config.MAGPIE_EXCLUDE]
        return ElementProperty(data_source="magpie", features=feats, stats=list(config.MAGPIE_STATS),
                               impute_nan=False)


def magpie_labels(prefix: str) -> list[str]:
    ep = _magpie()
    return [f"{prefix}_{stat}_{feat}" for feat in ep.features for stat in ep.stats]


def split_ions(framework_formula: str, oxi: dict,
               working_ion: str = config.DEFAULT_ION) -> tuple[Composition, Composition]:
    """(cation, anion) parts of the framework composition per f.u., by the sign of the P1 oxidation state."""
    comp = Composition(framework_formula)
    missing = [el.symbol for el in comp if el.symbol not in oxi]
    if missing:
        raise ValueError(f"{framework_formula}: no P1 oxidation state for {missing}")
    cat = {el: amt for el, amt in comp.items() if el.symbol != working_ion and float(oxi[el.symbol]) >= 0}
    an = {el: amt for el, amt in comp.items() if el.symbol != working_ion and float(oxi[el.symbol]) < 0}
    return Composition(cat), Composition(an)


def magpie(comp: Composition, prefix: str) -> dict[str, float]:
    labels = magpie_labels(prefix)
    if comp.num_atoms == 0:
        return dict.fromkeys(labels, float("nan"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        values = _magpie().featurize(comp)
    return dict(zip(labels, map(float, values)))


def polyanion_flags(framework_formula: str) -> dict[str, float]:
    present = {el.symbol for el in Composition(framework_formula)}
    return {f"has_{el}": float(el in present) for el in config.POLYANION_ELEMENTS}


def composition_features(framework_formula: str, oxi: dict) -> dict[str, float]:
    """Group 1 (fw_cation_*) and group 2 (fw_anion_*, has_*) for one battery."""
    cat, an = split_ions(framework_formula, oxi)
    return {**magpie(cat, "fw_cation"), **magpie(an, "fw_anion"), **polyanion_flags(framework_formula)}


# --- Group mag: Magpie of the full compositions, no oxidation states ---
MAG_PREFIX = "mag"


@lru_cache(maxsize=None)
def _full_magpie(comp: str) -> tuple[float, ...]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return tuple(map(float, _magpie().featurize(Composition(comp))))


def full_composition_features(comp_initial: str, comp_final: str) -> dict[str, float]:
    """Magpie stats (config.MAGPIE_STATS, amount-weighted, working ion included) of the initial and final
    compositions and final − initial: mag_initial_*, mag_final_*, mag_delta_*. Uses no oxidation states."""
    i, f = _full_magpie(comp_initial), _full_magpie(comp_final)
    stems = [label.split("_", 1)[1] for label in magpie_labels("x")]
    out = {}
    for stem, a, b in zip(stems, i, f):
        out[f"{MAG_PREFIX}_initial_{stem}"] = a
        out[f"{MAG_PREFIX}_final_{stem}"] = b
        out[f"{MAG_PREFIX}_delta_{stem}"] = b - a
    return out
