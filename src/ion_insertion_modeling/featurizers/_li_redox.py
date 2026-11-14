"""Group li_redox: Li content and P1 oxidation-state change per transition.

x is Li per framework f.u., li_frac the Li atomic fraction of the full composition; deltas are final − initial.
p1_score is the P1 guesser's score of each end's oxidation states (Stage 1: site-weighted geometric mean of
P(ox | env), 0-1). Both ends are scored on the initial structure's coordination environments (every material of a
battery gets them), so the final's score uses only its composition + the initial structure (rule 3).
Δox is the change in P1 oxidation state (final − initial, e per atom) of framework cations (elements with P1 state
>= 0 at either end; Li is not in the framework). Electrons per redox ion is in e⁻ per redox-active atom.
"""
import numpy as np

from ion_insertion_modeling import _redox as redox
from ion_insertion_modeling.featurizers._oxi_features import redox_element

# abs_delta_x / abs_delta_li_frac (Draft 4, 2026-10-19): ΔEdecomp rises on both sides of the initial material, so it
# is V-shaped in the signed change; linear models need |Δ|.
ION_CONC_LABELS = ["x_initial", "x_final", "delta_x", "li_frac_initial", "li_frac_final", "delta_li_frac",
                   "abs_delta_x", "abs_delta_li_frac"]
P1_LABELS = ["delta_ox_mean", "delta_ox_max_abs", "delta_ox_redox", "n_couples", "electrons_per_redox_ion",
             "p1_score_initial", "p1_score_final", "delta_p1_score"]
LABELS = ION_CONC_LABELS + P1_LABELS


def electrons_per_redox_ion(couples: list[dict], framework_formula: str) -> float:
    """Oxidising electrons per framework f.u. divided by the atoms per f.u. of the elements carrying them
    (e.g. Mo2P2O9 -> LiMo2P2O9: 1 e⁻ over 2 Mo = 0.5). NaN if no oxidising couple."""
    positive = [c for c in couples if c["electrons"] > 0]
    if not positive:
        return float("nan")
    amounts = redox.element_amounts(framework_formula)
    atoms = sum(amounts[el] for el in {c["element"] for c in positive})
    return sum(c["electrons"] for c in positive) / atoms


def oxi_change(oxi_initial: dict, oxi_final: dict, framework_formula: str, couples: list[dict]) -> dict[str, float]:
    """delta_ox_mean (amount-weighted over framework cations), delta_ox_max_abs and delta_ox_redox (primary redox
    element; NaN if none). All final − initial, in e per atom."""
    amounts = redox.element_amounts(framework_formula)
    cations = [el for el in sorted(amounts) if max(float(oxi_initial[el]), float(oxi_final[el])) >= 0]
    d = np.array([float(oxi_final[el]) - float(oxi_initial[el]) for el in cations])
    w = np.array([amounts[el] for el in cations])
    el = redox_element(couples)
    empty = len(cations) == 0
    return {"delta_ox_mean": float("nan") if empty else float(np.sum(w * d) / np.sum(w)),
            "delta_ox_max_abs": float("nan") if empty else float(np.max(np.abs(d))),
            "delta_ox_redox": float(oxi_final[el]) - float(oxi_initial[el]) if el else float("nan")}


def li_redox_features(oxi_initial: dict, oxi_final: dict, framework_formula: str, couples: list[dict],
                      n_couples: int, x_initial: float, x_final: float, li_frac_initial: float,
                      li_frac_final: float, p1_score_initial: float, p1_score_final: float) -> dict[str, float]:
    """All li_redox columns for one transition (see module docstring for units)."""
    return {"x_initial": float(x_initial), "x_final": float(x_final), "delta_x": float(x_final - x_initial),
            "li_frac_initial": float(li_frac_initial), "li_frac_final": float(li_frac_final),
            "delta_li_frac": float(li_frac_final - li_frac_initial),
            "abs_delta_x": abs(float(x_final - x_initial)),
            "abs_delta_li_frac": abs(float(li_frac_final - li_frac_initial)),
            **oxi_change(oxi_initial, oxi_final, framework_formula, couples),
            "n_couples": float(n_couples), "electrons_per_redox_ion": electrons_per_redox_ion(couples, framework_formula),
            "p1_score_initial": float(p1_score_initial), "p1_score_final": float(p1_score_final),
            "delta_p1_score": float(p1_score_final - p1_score_initial)}
