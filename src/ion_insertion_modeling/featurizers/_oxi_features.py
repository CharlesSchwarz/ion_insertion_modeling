"""Group 3: oxidation-state-dependent cation features at both ends of a transition.

Ionic radius (Å) and next ionization energy IE_next (eV) of each framework cation at its P_ox oxidation state,
computed at the initial and final material; delta = final − initial (so Li insertion, which reduces, gives a
positive radius delta for the redox element).
Framework cations are the framework elements with P_ox state > 0 at that end (Li is not in the framework).
"""
import math
from functools import lru_cache

import numpy as np
from pymatgen.core import Element

from ion_insertion_modeling import config
from ion_insertion_modeling import _redox as redox

PROPS = ("radius", "ie_next")
STATS = ("mean", "min", "max", "redox")


@lru_cache(maxsize=None)
def radius_table(el: str) -> tuple[tuple[int, float], ...]:
    """(oxidation state, ionic radius Å) sorted by state. Per state: pymatgen's default ionic radius, else the
    high-spin, else the low-spin value (the order Species.ionic_radius uses). Element.ionic_radii alone only
    holds the high-spin table for transition metals, so it misses e.g. Mn4+, Cr3+ and Ni2+."""
    data = Element(el).data
    table: dict[int, float] = {}
    for key in ("Ionic radii ls", "Ionic radii hs", "Ionic radii"):  # later sources win
        for ox, r in (data.get(key) or {}).items():
            table[int(ox)] = float(r)
    return tuple(sorted(table.items()))


def ionic_radius(el: str, ox: float) -> float:
    """Ionic radius (Å) at oxidation state ox, linearly interpolated between tabulated integer states.
    NaN outside the tabulated range (decided 2026-09-25) or if the element has no radii."""
    table = radius_table(el)
    if not table:
        return float("nan")
    states = np.array([s for s, _ in table], dtype=float)
    radii = np.array([r for _, r in table])
    if ox < states[0] - 1e-9 or ox > states[-1] + 1e-9:
        return float("nan")
    return float(np.interp(ox, states, radii))


def ie_next(el: str, ox: float) -> float:
    """Energy (eV) to oxidize one more step from oxidation state ox: the (ox+1)-th ionization energy,
    Element.ionization_energies[ox]. Fractional ox interpolates between floor and ceil. NaN if missing."""
    ies = Element(el).ionization_energies
    lo, hi = math.floor(ox + 1e-9), math.ceil(ox - 1e-9)

    def at(i: int) -> float:
        return float(ies[i]) if 0 <= i < len(ies) and ies[i] is not None else float("nan")

    if lo == hi:
        return at(lo)
    return at(lo) + (at(hi) - at(lo)) * (ox - lo)


PROP_FUNCS = {"radius": ionic_radius, "ie_next": ie_next}


def endpoint_stats(oxi: dict, framework_formula: str, redox_element: str | None) -> dict[str, float]:
    """{prop}_{stat} at one end: amount-weighted mean, min, max over framework cations (any NaN value makes the
    stat NaN), and the value of the redox element (NaN if None or not a cation there)."""
    amounts = redox.element_amounts(framework_formula)
    cations = [el for el in sorted(amounts) if float(oxi[el]) > 0]
    out = {}
    for prop, func in PROP_FUNCS.items():
        vals = np.array([func(el, float(oxi[el])) for el in cations])
        w = np.array([amounts[el] for el in cations])
        empty = len(vals) == 0
        out[f"{prop}_mean"] = float("nan") if empty else float(np.sum(w * vals) / np.sum(w))
        out[f"{prop}_min"] = float("nan") if empty else float(np.min(vals))
        out[f"{prop}_max"] = float("nan") if empty else float(np.max(vals))
        ok = redox_element in cations
        out[f"{prop}_redox"] = func(redox_element, float(oxi[redox_element])) if ok else float("nan")
    return out


def redox_element(couples: list[dict]) -> str | None:
    """Element of the primary P_ox couple (Stage 3 rule), or None if no oxidising couple."""
    label = redox.primary_couple(couples)
    return next((c["element"] for c in couples if c["label"] == label), None)


def oxi_features(oxi_initial: dict, oxi_final: dict, framework_formula: str, couples: list[dict]) -> dict[str, float]:
    """All group 3 columns for one transition: {prop}_{stat}_{initial|final|delta}, delta = final − initial."""
    el = redox_element(couples)
    i = endpoint_stats(oxi_initial, framework_formula, el)
    f = endpoint_stats(oxi_final, framework_formula, el)
    out = {}
    for key in i:
        out[f"{key}_initial"] = i[key]
        out[f"{key}_final"] = f[key]
        out[f"{key}_delta"] = f[key] - i[key]
    return out


# --- s/p/d/f valence electrons at the P_ox state (Draft 4, decided 2026-10-19; also group oxi) ---
SHELLS = ("s", "p", "d", "f")
REMOVAL_ORDER = ("p", "s", "d", "f")  # outer np, then ns, then (n-1)d, then (n-2)f
ADDITION_ORDER = (("s", 2), ("p", 6))  # anions (ox < 0): fill ns, then np (Au− 6s2, Bi3− 6p6); d and f unchanged
VALENCE_PREFIX = "nval"


@lru_cache(maxsize=None)
def neutral_valence(el: str) -> tuple[float, ...]:
    """Magpie NsValence, NpValence, NdValence, NfValence of the neutral atom (filled d10 / f14 shells counted)."""
    data = _magpie_data()
    return tuple(float(data.get_elemental_property(Element(el), f"N{sh}Valence")) for sh in SHELLS)


@lru_cache(maxsize=1)
def _magpie_data():
    import warnings

    from matminer.utils.data import MagpieData

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return MagpieData(impute_nan=False)


def _ion_valence_int(el: str, ox: int) -> tuple[float, ...]:
    """s, p, d, f electron counts of el at integer state ox: ox > 0 removes electrons in REMOVAL_ORDER, ox < 0 adds
    them in ADDITION_ORDER (NaN if ox is above the valence count or below a filled s2 p6 shell)."""
    counts = dict(zip(SHELLS, neutral_valence(el)))
    if any(math.isnan(v) for v in counts.values()):
        return (float("nan"),) * len(SHELLS)
    if ox < 0:
        left = -ox
        for sh, cap in ADDITION_ORDER:
            add = min(max(cap - counts[sh], 0.0), left)
            counts[sh] += add
            left -= add
        return (float("nan"),) * len(SHELLS) if left > 1e-9 else tuple(counts[sh] for sh in SHELLS)
    left = ox
    for sh in REMOVAL_ORDER:
        take = min(counts[sh], left)
        counts[sh] -= take
        left -= take
    if left > 1e-9:
        return (float("nan"),) * len(SHELLS)
    return tuple(counts[sh] for sh in SHELLS)


def ion_valence(el: str, ox: float) -> dict[str, float]:
    """{shell: electrons} of el at P_ox state ox: the neutral atom's Magpie counts minus ox electrons, removed in
    REMOVAL_ORDER (Fe3+ d5, Pb2+ s2 d10 f14, Ce4+ f0), or plus |ox| for an anion in ADDITION_ORDER (Sb3− s2 p6 d10).
    Fractional ox interpolates linearly between floor and ceil."""
    lo, hi = math.floor(ox + 1e-9), math.ceil(ox - 1e-9)
    v_lo = np.array(_ion_valence_int(el, lo))
    v = v_lo if lo == hi else v_lo + (np.array(_ion_valence_int(el, hi)) - v_lo) * (ox - lo)
    return dict(zip(SHELLS, map(float, v)))


def valence_stats(oxi: dict, framework_formula: str, redox_el: str | None) -> dict[str, float]:
    """{shell}_{mean|min|max|redox} at one end: amount-weighted mean, min, max over framework cations (P_ox state > 0;
    any NaN makes the stat NaN) and the redox element's count at its P_ox state, also when that state is 0 or negative
    (anion redox, e.g. Bi3−/Bi2−); NaN if None or not in the framework."""
    amounts = redox.element_amounts(framework_formula)
    cations = [el for el in sorted(amounts) if float(oxi[el]) > 0]
    vals = {el: ion_valence(el, float(oxi[el])) for el in cations}
    if redox_el in amounts and redox_el not in vals:
        vals[redox_el] = ion_valence(redox_el, float(oxi[redox_el]))   # not a cation: kept out of mean/min/max
    w = np.array([amounts[el] for el in cations])
    out = {}
    for sh in SHELLS:
        x = np.array([vals[el][sh] for el in cations])
        empty = len(x) == 0
        out[f"{sh}_mean"] = float("nan") if empty else float(np.sum(w * x) / np.sum(w))
        out[f"{sh}_min"] = float("nan") if empty else float(np.min(x))
        out[f"{sh}_max"] = float("nan") if empty else float(np.max(x))
        out[f"{sh}_redox"] = vals[redox_el][sh] if redox_el in vals else float("nan")
    return out


def valence_features(oxi_initial: dict, oxi_final: dict, framework_formula: str,
                     couples: list[dict]) -> dict[str, float]:
    """nval_{shell}_{mean|min|max}_{initial|final|delta} over framework cations and nval_{shell}_redox_{initial|final}
    for the primary redox element (electrons; delta = final − initial)."""
    el = redox_element(couples)
    i = valence_stats(oxi_initial, framework_formula, el)
    f = valence_stats(oxi_final, framework_formula, el)
    out = {}
    for key in i:
        out[f"{VALENCE_PREFIX}_{key}_initial"] = i[key]
        out[f"{VALENCE_PREFIX}_{key}_final"] = f[key]
        if not key.endswith("_redox"):
            out[f"{VALENCE_PREFIX}_{key}_delta"] = f[key] - i[key]
    return out


# --- Next electron affinity (group 3 alongside ie_next; 2026-10-21, Charlie) ---
EA_PREFIX = "ea_next"


def ea_next(el: str, ox: float) -> float:
    """Energy (eV) released when el is reduced one step from oxidation state ox: the ox-th ionization energy, i.e.
    ie_next at ox − 1 (fractional ox interpolates). NaN below ox = 1 or if the energy is missing."""
    return ie_next(el, ox - 1) if ox >= 1 - 1e-9 else float("nan")


def ea_max(oxi: dict, framework_formula: str, excluded=config.EA_EXCLUDED_ELEMENTS) -> float:
    """Largest next electron affinity (eV) over the framework cations that can be reduced (P_ox state > 0, element not in
    `excluded`): the most easily reduced cation. NaN if there is none."""
    amounts = redox.element_amounts(framework_formula)
    vals = [ea_next(el, float(oxi[el])) for el in amounts if float(oxi[el]) > 0 and el not in excluded]
    vals = [v for v in vals if np.isfinite(v)]
    return max(vals) if vals else float("nan")


def ea_features(oxi_initial: dict, oxi_final: dict, framework_formula: str) -> dict[str, float]:
    """ea_next_max_{initial|final|delta} (eV; delta = final − initial)."""
    i, f = ea_max(oxi_initial, framework_formula), ea_max(oxi_final, framework_formula)
    return {f"{EA_PREFIX}_max_initial": i, f"{EA_PREFIX}_max_final": f, f"{EA_PREFIX}_max_delta": f - i}
