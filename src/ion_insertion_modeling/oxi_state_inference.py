"""P1 oxidation-state inference: pick the charge-balanced oxidation-state assignment that is most probable given the
cation coordination environments.

Candidates are pymatgen's charge-balanced guesses (all oxidation states, on the largest multiple of the reduced
formula below config.OXI_GUESS_MAX_ATOMS atoms), with O fixed at -2 in oxides, the elements of
config.OXI_STATE_OVERRIDES fixed (Li+, Na+, Mg2+, F-, ...) and P fixed at +5 when every P site bonds only to O.
Each candidate is scored with ICSD statistics of P(ox | env, el) over the cation environments, and the best one wins.

The paper's score (config.P1_SCORING = "P_ox_env_sites_split"): the geometric mean of P(ox | env, el) over every
cation environment entry, weighted by its site count. An element with a fractional average state (Mn 2.29+) is scored
by the better of (a) the geometric interpolation between the neighbouring integers and (b) its best split over its
sites into the two adjacent integers (Mn3+ in some environments, Mn2+ in others), with charge balance fixing how
many sites take the higher state.

Probabilities: P(ox | el) from config.OXI_DATA_SOURCE (Waroquiers et al. 2017 by default), and
P(ox | env, el) = P(env | ox, el) P(ox | el) / P(env | el) from the ICSD statistics in data/species_data.json.
"""
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable

import numpy as np
import pandas as pd
from pymatgen.core import Composition, Element, Species, Structure

from ion_insertion_modeling import config
from ion_insertion_modeling import environments as envs_mod

log = logging.getLogger(__name__)

States = dict[str, float]  # element -> average oxidation state
DATA_SOURCE_SUFFIXES = {"Waroquiers": "DW", "Davies": "DD", "PMG": "PMG"}
SCORING_MAP = {
    "P_ox_avg":     {"use_environment": False, "mixing_method": "average",   "averaging_method": "product"},
    "P_ox":         {"use_environment": False, "mixing_method": "geometric", "averaging_method": "geometric"},
    "P_ox_env_avg": {"use_environment": True,  "mixing_method": "average",   "averaging_method": "product"},
    "P_ox_env":     {"use_environment": True,  "mixing_method": "geometric", "averaging_method": "geometric"},
    "P4":           {"use_environment": True,  "mixing_method": "geometric", "averaging_method": "product"},
    "P1":           {"use_environment": True,  "mixing_method": "geometric", "averaging_method": "geometric"},
    "P_ox_env_sites": {"use_environment": True, "mixing_method": "geometric", "averaging_method": "site_weighted"}, # Geometric mean weighted by each environment entry's site count, so an element on many sites
    # P_ox_env_sites, but an element with a fractional average state is scored by its best split over its sites into
    # the two adjacent integers (split_log_score); Draft 7 rerun, Charlie 2026-11-11.
    "P_ox_env_sites_split": {"use_environment": True, "mixing_method": "geometric", "averaging_method": "site_weighted",
                             "split": True},
}


@lru_cache(maxsize=1)
def species_data() -> pd.DataFrame:
    """Per (element, oxidation state) probabilities P(ox), P(env|ox), P(env|element) from the ICSD."""
    return pd.read_json(config.SPECIES_DATA)


@lru_cache(maxsize=None)
def _element_table(el: str) -> pd.DataFrame:
    df = species_data()
    return df[df.element == el]


def _is_override(el: str, ox: float) -> bool:
    return el in config.OXI_STATE_OVERRIDES and config.OXI_STATE_OVERRIDES[el][0] == ox


# --- Probabilities ---
def _interpolate_prob(el: str, ox: float, prob_func: Callable[..., float], **kwargs) -> float:
    """Probability of a (possibly fractional) oxidation state, mixed from the neighbouring integers."""
    if ox % 1 == 0:
        p = prob_func(el, ox, **kwargs)
        if p > 0:  # otherwise (e.g. Bi4+) fall through to interpolation between neighbours
            return p
    ox_low, ox_high = np.floor(ox), np.ceil(ox)
    p_low = prob_func(el, ox_low, **kwargs)
    p_high = prob_func(el, ox_high, **kwargs)
    # Fall back one step further if the immediate neighbour has no data
    if not p_low:
        ox_low = ox_low - 1
        p_low = prob_func(el, ox_low, **kwargs)
    if not p_high:
        ox_high = ox_high + 1
        p_high = prob_func(el, ox_high, **kwargs)
    try:
        mixing = kwargs.get("mixing_method", "average")
        weight = (ox - ox_low) / (ox_high - ox_low)
        if mixing == "minimum":
            return min(p_low, p_high)
        if mixing == "geometric":
            return (p_low ** (1 - weight)) * (p_high ** weight)
        return (1 - weight) * p_low + weight * p_high
    except Exception:
        return 0.0


def prob_int_ox(el: str, ox: float, data_source: str = config.OXI_DATA_SOURCE, **kwargs) -> float:
    """P(ox) for an integer oxidation state."""
    if data_source not in DATA_SOURCE_SUFFIXES:
        raise ValueError(f"Invalid data_source {data_source!r}; expected one of {list(DATA_SOURCE_SUFFIXES)}")
    if _is_override(el, ox):
        return 1.0
    el_df = _element_table(el)
    match = el_df[el_df.oxidation_state == ox]
    return float(match[f"oxi_prob_{DATA_SOURCE_SUFFIXES[data_source]}"].iloc[0]) if not match.empty else 0.0


def prob_int_ox_given_env(el: str, ox: float, env: str, data_source: str = config.OXI_DATA_SOURCE, **kwargs) -> float:
    """P(ox | env) = P(env | ox) P(ox) / P(env) for an integer oxidation state."""
    try:
        row = _element_table(el).set_index("oxidation_state")
        p_env_given_ox = row.at[ox, "env_probs"][env]
        p_env = row.at[ox, "el_env_probs"][env]
        return (p_env_given_ox * prob_int_ox(el, ox, data_source=data_source, **kwargs)) / p_env
    except (KeyError, ZeroDivisionError):
        return 0.0


def prob_ox(el: str, ox: float, **kwargs) -> float:
    if _is_override(el, ox):
        return 1.0
    return _interpolate_prob(el, ox, prob_int_ox, **kwargs)


def prob_ox_given_env(el: str, ox: float, env: str, **kwargs) -> float:
    if _is_override(el, ox):
        return 1.0
    return _interpolate_prob(el, ox, prob_int_ox_given_env, env=env, **kwargs)


def split_log_score(el: str, avg: float, groups: list[tuple[str, float]],
                    data_source: str = config.OXI_DATA_SOURCE) -> float | None:
    """Best sum over an element's sites of log P(ox | env) when its fractional average state `avg` is split into the
    adjacent integers lo = floor(avg), hi = ceil(avg). groups = [(env, n_sites)]. Charge balance on the element's sites
    fixes the number of hi sites; they go to the groups with the largest log P(hi|env) - log P(lo|env) first (exact,
    the sum is linear in the counts). None if avg is an integer, the site counts or the total charge are not integers,
    or no split exists. -inf if every split has a zero-probability site."""
    if abs(avg - round(avg)) < 1e-9 or any(abs(n - round(n)) > 1e-6 for _, n in groups):
        return None
    n_tot = int(round(sum(n for _, n in groups)))
    total = avg * n_tot
    if abs(total - round(total)) > 1e-6:
        return None
    lo, hi = int(np.floor(avg)), int(np.ceil(avg))
    n_hi = int(round(total)) - lo * n_tot
    if not 0 < n_hi < n_tot:
        return None
    log = lambda p: float(np.log(p)) if p > 0 else -np.inf
    la = [log(prob_int_ox_given_env(el, lo, env, data_source=data_source)) for env, _ in groups]
    lb = [log(prob_int_ox_given_env(el, hi, env, data_source=data_source)) for env, _ in groups]
    gain = [0.0 if np.isnan(b - a) else b - a for a, b in zip(la, lb)]
    left, out = n_hi, 0.0
    for i in sorted(range(len(groups)), key=lambda i: -gain[i]):
        m = int(round(groups[i][1]))
        h = min(m, left)
        left -= h
        out += (h * lb[i] if h else 0.0) + ((m - h) * la[i] if m - h else 0.0)
    return out


def get_score_from_guess(guess: States, envs: list, data_source: str = config.OXI_DATA_SOURCE,
                         use_environment: bool = True, mixing_method: str = "geometric",
                         averaging_method: str = "product", split: bool = False) -> float:
    """Score of one oxidation-state guess: combined probability over all environment entries.

    averaging_method: 'product', 'geometric' (unweighted mean over entries), 'minimum', or
    'site_weighted' (geometric mean with each entry weighted by its site count: prod f_i^(m_i / sum m)).
    split (site_weighted with environments only): an element whose guessed state is fractional is scored by the
    better of its averaged factors and its best per-site split into adjacent integers (split_log_score).
    """
    if not envs:
        return 1.0
    factors, counts, elements = [], [], []
    for spec, env_symbol, mult in envs:
        el = str(Species(spec).element)
        ox = guess[el]
        if use_environment:
            factor = prob_ox_given_env(el, ox, env_symbol, data_source=data_source, mixing_method=mixing_method)
        else:
            factor = prob_ox(el, ox, data_source=data_source, mixing_method=mixing_method)
        factors.append(1 if el == "O" else factor)
        counts.append(mult)
        elements.append(el)
    if averaging_method == "site_weighted":
        weights = np.asarray(counts, dtype=float) / np.sum(counts)
        plain = float(np.prod(np.power(factors, weights)))
        if not (split and use_environment):
            return plain
        logs = np.array([m * (np.log(f) if f > 0 else -np.inf) for f, m in zip(factors, counts)], dtype=float)
        improved = False
        for el in dict.fromkeys(elements):
            if el == "O" or _is_override(el, guess[el]):
                continue
            idx = [i for i, e in enumerate(elements) if e == el]
            best = split_log_score(el, float(guess[el]), [(envs[i][1], float(counts[i])) for i in idx], data_source)
            if best is not None and best > logs[idx].sum() + 1e-12:
                logs[idx] = 0.0
                logs[idx[0]] = best
                improved = True
        if not improved:
            return plain
        total = logs.sum()
        return float(np.exp(total / np.sum(counts))) if np.isfinite(total) else 0.0
    raw_product = np.prod(factors)
    if averaging_method == "geometric":
        return float(raw_product ** (1 / len(factors)))
    if averaging_method == "minimum":
        return float(min(factors))
    if averaging_method == "product":
        return float(raw_product)
    raise NotImplementedError(f"Averaging method {averaging_method!r} is not recognized.")


# --- Guess generation ---
def _overrides_for(oxide_type: str | None, fixed: dict[str, int] | None = None) -> dict[str, list[int]]:
    overrides = dict(config.OXI_STATE_OVERRIDES)
    if oxide_type in ("oxide", "hydroxide"):
        overrides["O"] = [-2]
    overrides.update({el: [int(ox)] for el, ox in (fixed or {}).items()})
    return overrides


def fixed_states(oxo_bonded: dict[str, float] | None,
                 rules: dict[str, int] = config.OXI_OXO_FIXED_STATES) -> dict[str, int]:
    """Oxidation states fixed for a battery (Draft 4): each element of `rules` whose every site in the initial
    structure bonds only to O (oxo_bonded fraction 1, envs.oxo_bonded_fraction) gets its rule state, e.g. {"P": 5}."""
    return {el: ox for el, ox in rules.items() if (oxo_bonded or {}).get(el, 0.0) >= 1.0 - 1e-9}


def fast_fractional_oxi_states(comp: Composition, overrides: dict) -> States:
    """Charge neutrality with a single non-fixed element."""
    fixed = {str(el): overrides[str(el)][0] for el in comp.elements if str(el) in overrides}
    free = [str(el) for el in comp.elements if str(el) not in fixed]
    total_fixed_charge = sum(fixed[el] * comp[el] for el in fixed)
    n_free = sum(comp[el] for el in free)
    val_free = -total_fixed_charge / n_free if n_free > 0 else 0
    return {**fixed, **{el: val_free for el in free}}


def most_probable_state(el: str, data_source: str = config.OXI_DATA_SOURCE) -> int | None:
    """The integer oxidation state with the highest P(ox) in the ICSD statistics (None without data)."""
    if el in config.OXI_STATE_OVERRIDES:
        return config.OXI_STATE_OVERRIDES[el][0]
    t = _element_table(el)
    col = f"oxi_prob_{DATA_SOURCE_SUFFIXES[data_source]}"
    t = t[t[col] > 0] if col in t else t.iloc[0:0]
    return None if t.empty else int(t.loc[t[col].idxmax(), "oxidation_state"])


def fractional_fallback_guesses(comp: Composition, overrides: dict) -> list[States]:
    """Guesses for a composition the enumeration could not balance: one per non-fixed element e, in which e takes
    the (possibly fractional) average state that balances the charge and every other non-fixed element its most
    probable integer state. Kept only if e's average lies within its known oxidation states (pymatgen
    Element.oxidation_states). P1 scoring then picks among them with the coordination environments."""
    fixed = {str(el): overrides[str(el)][0] for el in comp.elements if str(el) in overrides}
    free = [str(el) for el in comp.elements if str(el) not in fixed]
    common = {el: most_probable_state(el) for el in free}
    out = []
    for e in free:
        others = {el: common[el] for el in free if el != e}
        if any(v is None for v in others.values()):
            continue
        charge = sum(fixed[el] * comp[el] for el in fixed) + sum(others[el] * comp[el] for el in others)
        avg = -charge / comp[e]
        states = Element(e).oxidation_states
        if states and min(states) - 1e-9 <= avg <= max(states) + 1e-9:
            out.append({**fixed, **others, e: float(avg)})
    return out


def integer_reduced(formula: str, max_denominator: int = config.OXI_INTEGER_MAX_DENOMINATOR) -> Composition:
    """Reduced composition with integer amounts. A formula with non-integer amounts (a disordered structure's
    partial occupancies) is scaled to the smallest integer multiple, e.g. Li1.2Mn0.4Ti0.4O2 -> Li3MnTiO5."""
    comp = Composition(formula)
    if all(abs(a - round(a)) < 1e-6 for a in comp.values()):
        return Composition(comp.reduced_formula)
    return Composition(comp.get_integer_formula_and_factor(max_denominator=max_denominator)[0])


def get_pmg_oxi_state_guesses(formula: str, oxide_type: str | None = None,
                              max_atoms: int = config.OXI_GUESS_MAX_ATOMS,
                              fallback: bool = config.OXI_FRACTIONAL_FALLBACK,
                              fixed: dict[str, int] | None = None) -> list[States]:
    """All charge-balanced pymatgen guesses (all_oxi_states=True) for the largest multiple of the
    reduced formula with < max_atoms atoms (pymatgen's common states only above that size). Non-integer formulas are
    first scaled to integers (integer_reduced). O is fixed at -2 for oxides and hydroxides, and each element of
    `fixed` at its given state (fixed_states). If none balance and fallback is on, fractional_fallback_guesses."""
    reduced = integer_reduced(formula)
    overrides = _overrides_for(oxide_type, fixed)
    if sum(1 for el in reduced.elements if str(el) not in overrides) == 1:
        return [fast_fractional_oxi_states(reduced, overrides)]
    if reduced.num_atoms > max_atoms:
        guesses = list(reduced.oxi_state_guesses(all_oxi_states=False, oxi_states_override=overrides))
    else:
        factor = max(1, int((max_atoms - 1) // reduced.num_atoms))
        work = Composition(reduced.formula * factor)
        guesses = list(work.oxi_state_guesses(all_oxi_states=True, oxi_states_override=overrides))
    if not guesses and fallback:
        guesses = fractional_fallback_guesses(reduced, overrides)
    return guesses


def candidate_guesses(formula: str, oxide_type: str | None, fallback: bool = config.OXI_FRACTIONAL_FALLBACK,
                      fixed: dict[str, int] | None = None) -> tuple[list[States], States | None]:
    """(P1 candidates, pymatgen's first enumerated guess). The candidates are get_pmg_oxi_state_guesses with the
    fallback switch and the fixed states; the first guess never comes from the fallback or the fixed states (it is
    PMG_oxi_state_guess, which feeds only the oxi_pmg ablation, spec.md §5)."""
    enumerated = get_pmg_oxi_state_guesses(formula, oxide_type, fallback=False)
    first = enumerated[0] if enumerated else None
    if fixed:
        return get_pmg_oxi_state_guesses(formula, oxide_type, fallback=fallback, fixed=fixed), first
    if enumerated or not fallback:
        return enumerated, first
    return get_pmg_oxi_state_guesses(formula, oxide_type, fallback=True), None


def get_best_guess_and_score(envs: list, all_guesses: list[States] | None,
                             scoring: str = config.P1_SCORING,
                             data_source: str = config.OXI_DATA_SOURCE) -> tuple[States | None, float | None]:
    """Highest-scoring guess and its score; (None, None) if there are no guesses."""
    if scoring not in SCORING_MAP:
        raise NotImplementedError(f"Scoring method {scoring!r} is not recognized.")
    if not all_guesses:
        return None, None
    scored = [(g, get_score_from_guess(g, envs, data_source=data_source, **SCORING_MAP[scoring])) for g in all_guesses]
    return max(scored, key=lambda x: x[1])


def p1_guess(envs: list | None, pmg_guesses: list[States] | None,
             scoring: str = config.P1_SCORING) -> tuple[States | None, float | None]:
    """P1 guess and score (P1_oxi_state_guess, P1_score); (None, None) without guesses or envs.
    O entries are skipped: O always scores 1, so they would only dilute the score, never change the guess."""
    if isinstance(pmg_guesses, list) and len(pmg_guesses) > 0 and isinstance(envs, list):
        scored_envs = [e for e in envs if str(e[0]) != "O"]
        return get_best_guess_and_score(scored_envs, pmg_guesses, scoring=scoring)
    return None, None




# =====================================================================================================================
# Public API
# =====================================================================================================================
@dataclass(frozen=True)
class OxiGuess:
    """Result of the P1 guesser.

    states: element -> average oxidation state (fractional for mixed valence, e.g. {"Fe": 2.6667, "O": -2}).
    score: P1 score of `states` (0-1).
    candidates: every charge-balanced candidate with its score, best first (ties keep pymatgen's order).
    environments: (element, ChemEnv symbol, site count) entries the score used (O and fixed-state elements left out).
    fixed: states fixed by rule (O = -2 in oxides, oxo-bonded P = +5, plus anything passed in).
    scoring: name of the score (SCORING_MAP key).
    """
    states: dict[str, float]
    score: float
    candidates: list[tuple[dict[str, float], float]] = field(repr=False)
    environments: list[tuple[str, str, float]] = field(repr=False)
    fixed: dict[str, int]
    scoring: str

    def __getitem__(self, el: str) -> float:
        return self.states[el]

    def runner_up(self) -> tuple[dict[str, float], float] | None:
        """Second-best candidate and its score (None if there is only one)."""
        return self.candidates[1] if len(self.candidates) > 1 else None


def p_ox(el: str, ox: float, data_source: str = config.OXI_DATA_SOURCE, mixing_method: str = "geometric") -> float:
    """P(ox | el) from the ICSD statistics. A fractional `ox` is mixed from the neighbouring integers (geometric
    interpolation by default, as in the P1 score). Fixed-state elements (config.OXI_STATE_OVERRIDES) give 1 at
    their state."""
    return float(prob_ox(el, ox, data_source=data_source, mixing_method=mixing_method))


def p_ox_given_env(el: str, ox: float, env: str, data_source: str = config.OXI_DATA_SOURCE,
                   mixing_method: str = "geometric") -> float:
    """P(ox | env, el) = P(env | ox, el) P(ox | el) / P(env | el), with env a ChemEnv symbol such as "O:6" (octahedron)
    or "T:4" (tetrahedron). 0 if the (state, environment) pair never occurs in the statistics."""
    return float(prob_ox_given_env(el, ox, env, data_source=data_source, mixing_method=mixing_method))


def element_table(el: str) -> pd.DataFrame:
    """The ICSD statistics of one element: one row per oxidation state with P(ox) per data source and the
    environment distributions P(env | ox) (env_probs) and P(env | el) (el_env_probs)."""
    return _element_table(el).reset_index(drop=True).copy()


def _scored_envs(environments) -> list[list]:
    return [list(e) for e in envs_mod.clean_environments(environments) if str(e[0]) != "O"]


def score_guess(states: dict[str, float], environments, scoring: str = config.P1_SCORING,
                data_source: str = config.OXI_DATA_SOURCE) -> float:
    """P1 score (0-1) of an oxidation-state assignment on the given environments. O and fixed-state elements
    score 1 and are left out."""
    if scoring not in SCORING_MAP:
        raise ValueError(f"unknown scoring {scoring!r}; choose from {list(SCORING_MAP)}")
    return float(get_score_from_guess(dict(states), _scored_envs(environments), data_source=data_source,
                                      **SCORING_MAP[scoring]))


def _composition_formula(composition) -> str:
    comp = Composition(composition)
    comp = Composition({el.element if hasattr(el, "element") else el: amt for el, amt in comp.items()})
    return comp.formula


def guess_from_environments(composition, environments, oxide_type: str = "oxide", fixed: dict[str, int] | None = None,
                            scoring: str = config.P1_SCORING, fallback: bool = config.OXI_FRACTIONAL_FALLBACK,
                            data_source: str = config.OXI_DATA_SOURCE) -> OxiGuess | None:
    """P1 guess for a composition scored on a list of coordination environments.

    composition: formula or Composition (fractional amounts allowed: scaled to integers first).
    environments: [(element, ChemEnv symbol, site count), ...], e.g. from environments_from_structure, or
        uniform_environments(composition, "O:6") for a structure-free guess.
    oxide_type: "oxide" or "hydroxide" fixes O at -2; None leaves O free.
    fixed: extra element -> state constraints, e.g. {"P": 5}.
    Returns None when no candidate balances the charge.
    """
    formula = _composition_formula(composition)
    guesses, _ = candidate_guesses(formula, oxide_type, fallback=fallback, fixed=fixed or None)
    if not guesses:
        return None
    env = _scored_envs(environments)
    params = SCORING_MAP[scoring]
    scored = [(dict(g), float(get_score_from_guess(g, env, data_source=data_source, **params))) for g in guesses]
    ranked = sorted(scored, key=lambda gs: -gs[1])  # stable: ties keep pymatgen's order, as max() did
    best, best_score = ranked[0]
    overrides = _overrides_for(oxide_type, fixed)
    fixed_out = {el: v[0] for el, v in overrides.items() if el in best}
    return OxiGuess(states={k: float(v) for k, v in best.items()}, score=best_score, candidates=ranked,
                    environments=[tuple(e) for e in env], fixed=fixed_out, scoring=scoring)


def framework_constraints(structure: Structure) -> tuple[list, dict[str, int]]:
    """(cleaned ChemEnv environments, rule-fixed states) of a structure: the inputs P1 takes from a structure.
    Fixed states: config.OXI_OXO_FIXED_STATES for elements whose every site bonds only to O (P -> +5)."""
    s = structure.copy()
    s.remove_oxidation_states()
    env = envs_mod.environments_from_structure(s)
    return env, fixed_states(envs_mod.oxo_bonded_fraction(s))


def guess_from_framework(composition, framework_structure: Structure, oxide_type: str = "oxide",
                         scoring: str = config.P1_SCORING, **kwargs) -> OxiGuess | None:
    """P1 guess for `composition` scored on the environments of another structure of the same framework, usually
    the initial (pristine) material when `composition` is a lithiated or delithiated state of it. This is how
    every final material is assigned in the paper: changing the ion content is assumed to leave the other ions'
    environments unchanged, and the final material's own structure is never needed."""
    env, fixed = framework_constraints(framework_structure)
    return guess_from_environments(composition, env, oxide_type=oxide_type, fixed=fixed or None, scoring=scoring,
                                   **kwargs)


def guess_from_structure(structure: Structure, oxide_type: str = "oxide", scoring: str = config.P1_SCORING,
                         **kwargs) -> OxiGuess | None:
    """P1 guess for a structure, ordered or disordered (partial occupancies: ChemEnv counts each species of a
    mixed site by its occupancy, and the formula is scaled to integers before candidates are enumerated).
    Oxidation states already on the structure are ignored."""
    s = structure.copy()
    s.remove_oxidation_states()
    return guess_from_framework(s.composition, s, oxide_type=oxide_type, scoring=scoring, **kwargs)


def default_framework(structure: Structure, ion: str = config.DEFAULT_ION) -> str:
    """Framework formula of a structure: its composition without `ion`, scaled to the smallest integer formula
    (Pr6MnMo3O24 for the disordered Li7Pr3(Mn0.25Mo0.75)2O12 garnet)."""
    s = structure.copy()
    s.remove_oxidation_states()
    return integer_reduced(Composition({el: a for el, a in s.composition.items() if el.symbol != ion}).formula).formula


def scan_ion_content(structure: Structure, x_grid, ion: str = config.DEFAULT_ION, framework_formula: str | None = None,
                     oxide_type: str = "oxide", scoring: str = config.P1_SCORING) -> pd.DataFrame:
    """P1 guess and score of ion_x + framework for each x in x_grid, on the environments and fixed states of
    `structure`. x is ions per framework formula unit (default framework: default_framework; pass e.g.
    "Pr3Mn0.5Mo1.5O12" to count x per 12 O instead). ChemEnv runs once.
    Columns: x, formula, states (None if nothing balances), score (NaN then), is_best (highest score; ties go to
    the first x in grid order)."""
    env, fixed = framework_constraints(structure)
    framework = Composition(framework_formula or default_framework(structure, ion))
    rows = []
    for x in x_grid:
        comp = framework + Composition({ion: float(x)}) if x > 0 else framework
        g = guess_from_environments(comp, env, oxide_type=oxide_type, fixed=fixed or None, scoring=scoring)
        rows.append({"x": float(x), "formula": comp.formula, "states": None if g is None else g.states,
                     "score": np.nan if g is None else g.score})
    out = pd.DataFrame(rows)
    out["is_best"] = False
    if out.score.notna().any():
        out.loc[out.score.idxmax(), "is_best"] = True
    return out


def best_ion_content(structure: Structure, x_grid=None, ion: str = config.DEFAULT_ION,
                     framework_formula: str | None = None, oxide_type: str = "oxide",
                     scoring: str = config.P1_SCORING) -> tuple[float | None, OxiGuess | None, pd.DataFrame]:
    """The ion content x (ions per framework f.u.) whose composition gets the highest P1 score on the structure's
    environments: (x_best, its OxiGuess, the scan table from scan_ion_content). Useful for a disordered or
    hypothetical framework whose ion content is not known. Default grid: 0 to 2·x0 (or 0-2 when the structure has
    no ion) in steps of config.X_STEP, where x0 is the structure's own ion content."""
    framework = framework_formula or default_framework(structure, ion)
    if x_grid is None:
        s = structure.copy()
        s.remove_oxidation_states()
        from ion_insertion_modeling.formulas import framework_units
        x0 = float(s.composition[ion]) / framework_units(s.composition, framework, ion) if s.composition[ion] else 0.0
        x_max = 2 * x0 if x0 > 0 else 2.0
        x_grid = np.round(np.arange(0.0, x_max + 1e-9, config.X_STEP), 6)
    table = scan_ion_content(structure, x_grid, ion=ion, framework_formula=framework, oxide_type=oxide_type,
                             scoring=scoring)
    if not table.is_best.any():
        return None, None, table
    x_best = float(table.loc[table.is_best, "x"].iloc[0])
    env, fixed = framework_constraints(structure)
    comp = Composition(framework) + Composition({ion: x_best}) if x_best > 0 else Composition(framework)
    return x_best, guess_from_environments(comp, env, oxide_type=oxide_type, fixed=fixed or None, scoring=scoring), table


environments_from_structure = envs_mod.environments_from_structure
uniform_environments = envs_mod.uniform_environments

__all__ = ["OxiGuess", "p_ox", "p_ox_given_env", "element_table", "score_guess", "guess_from_environments",
           "guess_from_framework", "guess_from_structure", "scan_ion_content", "best_ion_content",
           "environments_from_structure", "uniform_environments", "framework_constraints", "SCORING_MAP"]
