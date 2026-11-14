"""Coordination environments (pymatgen ChemEnv) and the oxo-bonded test behind has_polyanion.

An environment list is [(element, chemenv_symbol, count), ...], where count is the number of sites of that element
in that environment in the given cell, e.g. [("Fe", "O:6", 4.0), ("P", "T:4", 4.0)]. On a disordered structure
ChemEnv counts each species of a mixed site by its occupancy.
"""
import json
import warnings
from collections import defaultdict
from collections.abc import Iterable

from pymatgen.core import Species, Structure

from ion_insertion_modeling import config

Env = tuple[str, str, float]
_CACHE: dict[str, list[Env]] = {}


def default_ignore() -> set[str]:
    """Elements dropped from environment lists: fixed-oxidation-state elements (Li, Na, Mg, ...), and O unless
    config.ENV_INCLUDE_O."""
    return set(config.OXI_STATE_OVERRIDES) | (set() if config.ENV_INCLUDE_O else {"O"})


def element_symbol(spec: str) -> str:
    """'Fe', 'Fe2+' -> 'Fe'."""
    return Species(spec).element.symbol if any(ch in spec for ch in "+-") else str(spec)


def raw_environments(structure: Structure) -> list[Env]:
    """ChemEnv environments of every element (O included: without oxidation states ChemEnv cannot tell cations from
    anions), retried with a tighter distance cutoff if any come back UNKNOWN. Slow: seconds to minutes."""
    from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import SimplestChemenvStrategy
    from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import LocalGeometryFinder
    from pymatgen.analysis.chemenv.coordination_environments.structure_environments import (
        LightStructureEnvironments,
    )

    s = structure.copy()
    s.remove_oxidation_states()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lgf = LocalGeometryFinder()
        lgf.setup_structure(s)
        se = lgf.compute_structure_environments(only_cations=True)
        data = {}
        for cutoff in config.ENV_DISTANCE_CUTOFFS:
            strategy = SimplestChemenvStrategy(distance_cutoff=cutoff, angle_cutoff=config.ENV_ANGLE_CUTOFF)
            lse = LightStructureEnvironments.from_structure_environments(strategy, se)
            data = lse.get_statistics()["atom_coordination_environments_present"]
            if not any("UNKNOWN" in env for envs in data.values() for env in envs):
                break
    return [(el, env, float(num)) for el, envs in data.items() for env, num in envs.items()]


def clean_environments(envs: Iterable, ignore: Iterable[str] | None = None) -> list[Env]:
    """Strip oxidation states, drop ignored elements (default_ignore), merge duplicate (element, environment)."""
    ignore = default_ignore() if ignore is None else set(ignore)
    sums: dict[tuple[str, str], float] = defaultdict(float)
    for spec, env, mult in envs:
        el = element_symbol(str(spec))
        if el not in ignore:
            sums[(el, env)] += mult
    return [(el, env, mult) for (el, env), mult in sums.items()]


def environments_from_structure(structure: Structure, cache: bool = True) -> list[Env]:
    """Cleaned ChemEnv environments of a structure (ordered or disordered). Cached in memory per structure."""
    s = structure.copy()
    s.remove_oxidation_states()
    key = json.dumps(s.as_dict(), sort_keys=True, default=str)
    if cache and key in _CACHE:
        return list(_CACHE[key])
    envs = clean_environments(raw_environments(s))
    if cache:
        _CACHE[key] = envs
    return list(envs)


def uniform_environments(composition, env: str = "O:6") -> list[Env]:
    """One environment for every cation with no fixed state: a structure-free stand-in, e.g. every cation
    octahedral in a cation-disordered rock salt. Counts are the amounts in the composition."""
    from pymatgen.core import Composition

    comp = Composition(composition)
    skip = default_ignore() | {"O"}
    return [(el.symbol, env, float(amt)) for el, amt in comp.items() if el.symbol not in skip]


def oxo_bonded_fraction(structure: Structure, elements: Iterable[str] = config.OXO_ELEMENTS,
                        tol: float = config.OXO_SHELL_TOL, radius: float = config.OXO_SEARCH_RADIUS,
                        max_cn: dict[str, int] | None = None) -> dict[str, float]:
    """{element: fraction of its sites (0-1) that bond only to O}, for each of `elements` present in the structure.
    A site's first shell is every neighbour within (1 + tol) × its nearest-neighbour distance (Å, searched out to
    `radius`). An element in `max_cn` (default As, Se: 4) counts only sites with at most that many shell neighbours."""
    max_cn = config.OXO_MAX_CN if max_cn is None else max_cn
    present = {el.symbol for el in structure.composition.elements}
    out = {}
    for el in sorted(set(elements) & present):
        sites = [s for s in structure if el in {e.symbol for e in s.species.elements}]
        n_oxo = 0
        for site in sites:
            neighbours = structure.get_neighbors(site, radius)
            if not neighbours:
                continue
            d_min = min(n.nn_distance for n in neighbours)
            shell = [n for n in neighbours if n.nn_distance <= (1 + tol) * d_min]
            n_oxo += (all({e.symbol for e in n.species.elements} == {"O"} for n in shell)
                      and len(shell) <= max_cn.get(el, len(shell)))
        out[el] = n_oxo / len(sites)
    return out


def has_polyanion(oxo_bonded: dict[str, float] | None) -> float:
    """1.0 if any site of a polyanion-forming element (config.OXO_ELEMENTS) bonds only to O, else 0.0; NaN if
    unknown. Light main-group, redox-inactive oxyanions only: vanadates, chromates, molybdates and tungstates are
    deliberately not counted."""
    if oxo_bonded is None:
        return float("nan")
    return float(any(v > 0 for v in oxo_bonded.values()))
