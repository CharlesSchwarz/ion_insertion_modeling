"""Group 4: framework structure features of the initial material.

Port of ElectrodeStructureFeaturizer / structure_utility in old_example_code/. The framework is the initial
structure with Li removed, decorated with the initial material's P1 oxidation states. Cations (P1 state >= 0,
polyanion centres included) define the polyhedra; neighbours come from one CrystalNN bonded-structure pass.
Polyhedron sharing is counted in the unit cell by matching periodic images, instead of the old code's O(n²)
comparison on a >= 10 Å supercell (same fractions; see tests).
"""
import warnings
from collections import Counter, defaultdict

import numpy as np
from pymatgen.analysis.dimensionality import get_dimensionality_larsen
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core import Structure
from scipy.spatial import ConvexHull

from ion_insertion_modeling import config
from ion_insertion_modeling.featurizers._oxi_features import ionic_radius

Neighbours = dict[int, set[tuple[int, tuple[int, int, int]]]]
LABELS = ["frac_corner_sharing", "frac_edge_sharing", "frac_face_sharing", "poly_vol_ratio",
          "framework_vol_per_atom", "anion_packing_ratio", "larsen_dimensionality"]


def framework_structure(structure: Structure, oxi: dict | None = None,
                        working_ion: str = config.DEFAULT_ION) -> Structure:
    """Copy of `structure` without the working ion; decorated with `oxi` (element -> state) if given."""
    s = structure.copy()
    s.remove_oxidation_states()
    if working_ion in {el.symbol for el in s.composition.elements}:
        s.remove_species([working_ion])
    if oxi is not None:
        s.add_oxidation_state_by_element({el.symbol: float(oxi[el.symbol]) for el in s.composition.elements})
    return s


def cation_indices(framework: Structure, oxi: dict) -> list[int]:
    return [i for i, site in enumerate(framework) if float(oxi[site.specie.element.symbol]) >= 0]


def bonded(framework: Structure):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return CrystalNN().get_bonded_structure(framework)


def neighbour_sets(graph, indices: list[int]) -> Neighbours:
    """For each site i (at image 0): {(neighbour index, neighbour image)}."""
    return {i: {(cs.index, tuple(int(round(v)) for v in cs.jimage)) for cs in graph.get_connected_sites(i)}
            for i in indices}


def shared_counts(nbrs: Neighbours) -> list[int]:
    """Number of shared neighbours for every pair of polyhedra (i at image 0, j at image t) that share at least
    one, excluding i with itself at t = 0. Each unordered pair is listed twice (from both sides)."""
    owners: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
    for j, ns in nbrs.items():
        for k, b in ns:
            owners[k].append((j, np.array(b)))
    counts: list[int] = []
    for i, ns in nbrs.items():
        c: Counter = Counter()
        for k, a in ns:
            for j, b in owners[k]:
                t = tuple(int(v) for v in np.array(a) - b)
                if not (j == i and t == (0, 0, 0)):
                    c[(j, t)] += 1
        counts += list(c.values())
    return counts


def sharing_fractions(nbrs: Neighbours) -> tuple[float, float, float]:
    """Fractions of sharing polyhedron pairs that share 1 (corner), 2 (edge) or >= 3 (face) atoms; 0s if none."""
    counts = Counter(shared_counts(nbrs))
    total = sum(counts.values())
    if total == 0:
        return 0.0, 0.0, 0.0
    return counts[1] / total, counts[2] / total, sum(v for k, v in counts.items() if k > 2) / total


def poly_vol_ratio(framework: Structure, nbrs: Neighbours) -> float:
    """Σ convex-hull volumes of cation polyhedra with >= 4 neighbours / cell volume (dimensionless)."""
    total = 0.0
    for ns in nbrs.values():
        if len(ns) < 4:
            continue
        pts = framework.lattice.get_cartesian_coords([framework[k].frac_coords + np.array(img) for k, img in ns])
        total += ConvexHull(pts, qhull_options="QJ").volume
    return total / framework.volume


def anion_packing_ratio(framework: Structure, oxi: dict) -> float:
    """Σ over anion sites of 4/3·π·r³ (ionic radius at the P1 state, Å) / cell volume. NaN if a radius is missing."""
    vol = 0.0
    for site in framework:
        el = site.specie.element.symbol
        if float(oxi[el]) < 0:
            vol += 4 / 3 * np.pi * ionic_radius(el, float(oxi[el])) ** 3
    return vol / framework.volume


def structure_features(structure: Structure, oxi: dict, ion: str = config.DEFAULT_ION) -> dict[str, float]:
    """Group 4 for one initial structure. Volumes in Å³."""
    fw = framework_structure(structure, working_ion=ion)
    if not fw.is_ordered:  # new disordered inputs: one representative ordering (see _sites.ordered_stand_in)
        from ion_insertion_modeling.featurizers._sites import ordered_stand_in
        fw = ordered_stand_in(fw)
    fw.add_oxidation_state_by_element({el.symbol: float(oxi[el.symbol]) for el in fw.composition.elements})
    graph = bonded(fw)
    nbrs = neighbour_sets(graph, cation_indices(fw, oxi))
    corner, edge, face = sharing_fractions(nbrs)
    return {"frac_corner_sharing": corner, "frac_edge_sharing": edge, "frac_face_sharing": face,
            "poly_vol_ratio": poly_vol_ratio(fw, nbrs), "framework_vol_per_atom": fw.volume / len(fw),
            "anion_packing_ratio": anion_packing_ratio(fw, oxi),
            "larsen_dimensionality": float(get_dimensionality_larsen(graph))}
