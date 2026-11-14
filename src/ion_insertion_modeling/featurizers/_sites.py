"""Group 5: predicted Li insertion sites in the initial material's framework .

Port of structure_utility.get_interstitial_sites and the VPV featurizers in old_example_code/. Li is removed
first, so lithiated and delithiated initial materials are treated alike (occupied Li sites come back as voids).
Sites: TopographyAnalyzer Voronoi nodes (framework ions = anions), clustered within the ion radius (config.ION_RADIUS), kept
if their clearance (distance − atomic radius of the nearest atom) exceeds that radius. Each kept site is
measured with one Li inserted: Voronoi volume (Å³), CrystalNN coordination number, min Li–anion distance (Å).
"""
import warnings

import numpy as np
from pymatgen.analysis.local_env import CrystalNN, VoronoiNN
from pymatgen.core import Structure

from ion_insertion_modeling import config
from ion_insertion_modeling import formulas
from ion_insertion_modeling.featurizers._structure import framework_structure

LABELS = ["sites_per_fu", "site_vpv_mean", "site_vpv_min", "site_vpv_max", "site_cn_mean", "site_cn_min",
          "site_cn_max", "site_li_anion_dist_min", "site_li_anion_dist_mean"]


def candidate_sites(framework: Structure, anions: list[str], r_li: float = config.ION_RADIUS[config.DEFAULT_ION]) -> np.ndarray:
    """Fractional coordinates of Li site candidates in an undecorated, Li-free framework."""
    from pymatgen.analysis.defects.utils import TopographyAnalyzer, cluster_nodes

    cations = [el.symbol for el in framework.composition.elements if el.symbol not in anions]
    topo = TopographyAnalyzer(structure=framework, framework_ions=anions, cations=cations, check_volume=False)
    nodes = [v.frac_coords for v in topo.vnodes]
    if not nodes:
        return np.zeros((0, 3))
    clustered = cluster_nodes(nodes, lattice=framework.lattice, tol=r_li)
    dists = framework.lattice.get_all_distances(clustered, framework.frac_coords)
    radii = np.array([float(site.specie.atomic_radius) for site in framework])
    clearance = (dists - radii[None, :]).min(axis=1)
    return np.asarray(clustered)[clearance > r_li]


def site_descriptors(framework: Structure, frac: np.ndarray, anions: list[str],
                     working_ion: str = config.DEFAULT_ION) -> tuple[float, float, float]:
    """(Voronoi volume Å³, CrystalNN CN, min Li–anion distance Å) of one Li inserted at `frac`."""
    s = framework.copy()
    s.append(working_ion, frac)
    idx = len(s) - 1
    poly = VoronoiNN().get_voronoi_polyhedra(s, idx)
    vpv = float(sum(f.get("volume", 0.0) for f in poly.values()))
    cn = float(CrystalNN().get_cn(s, idx))
    anion_fc = [site.frac_coords for site in framework if site.specie.symbol in anions]
    d = float(framework.lattice.get_all_distances([frac], anion_fc).min()) if anion_fc else float("nan")
    return vpv, cn, d


def ordered_stand_in(framework: Structure) -> Structure:
    """Ordered copy of a disordered framework for the site search: each site takes its majority species at full
    occupancy (Mn0.25Mo0.75 -> Mo). The paper's training structures are all ordered, so this only applies to new,
    disordered inputs; the sites found are those of one representative cation arrangement."""
    s = framework.copy()
    for i, site in enumerate(s):
        if not site.is_ordered:
            el = max(site.species.items(), key=lambda kv: kv[1])[0]
            s.replace(i, el.element if hasattr(el, "element") else el, properties=site.properties)
    return s


def site_features(structure: Structure, oxi: dict, framework_formula: str,
                  ion: str = config.DEFAULT_ION) -> dict[str, float]:
    """Group 5 for one initial structure. sites_per_fu = candidate sites per framework formula unit. Also returns
    site_vpvs (every candidate site's Voronoi volume, Å³, sorted) and n_fu_cell (framework formula units in the cell)
    for the relative VPV mismatch (not features themselves)."""
    fw = framework_structure(structure, working_ion=ion)  # undecorated
    anions = sorted({el.symbol for el in fw.composition.elements if float(oxi[el.symbol]) < 0})
    n_fu = formulas.framework_units(fw.composition.formula.replace(" ", ""), framework_formula, ion)
    if not fw.is_ordered:  # n_fu above uses the true (fractional) composition
        fw = ordered_stand_in(fw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sites = candidate_sites(fw, anions, config.ION_RADIUS[ion])
        desc = np.array([site_descriptors(fw, f, anions, ion) for f in sites]).reshape(-1, 3)
    out = {"sites_per_fu": len(sites) / n_fu, "site_vpvs": sorted(map(float, desc[:, 0])), "n_fu_cell": float(n_fu)}
    for j, name in enumerate(("vpv", "cn")):
        col = desc[:, j]
        empty = len(col) == 0
        out[f"site_{name}_mean"] = float("nan") if empty else float(col.mean())
        out[f"site_{name}_min"] = float("nan") if empty else float(col.min())
        out[f"site_{name}_max"] = float("nan") if empty else float(col.max())
    d = desc[:, 2]
    out["site_li_anion_dist_min"] = float("nan") if len(d) == 0 else float(d.min())
    out["site_li_anion_dist_mean"] = float("nan") if len(d) == 0 else float(d.mean())
    return out



# --- Relative VPV mismatch (2026-10-20; port of old_example_code/featurizers/VPV_featurizers.get_vpv_mismatch_cutoff) ---
VPV_MISMATCH_LABELS = ["rel_vpv_mismatch_initial", "rel_vpv_mismatch_final"]


def median_vpv(element: str = config.DEFAULT_ION) -> float:
    """Median Voronoi polyhedron volume (Å³) of `element` in the ICSD (config.VPV_STATS)."""
    import pandas as pd

    t = pd.read_csv(config.VPV_STATS)
    return float(t.loc[t.Element.eq(element), "Median"].iloc[0])


def rel_vpv_mismatch(vpvs: list[float], x: float, n_fu_cell: float, median: float) -> float:
    """Relative Voronoi-volume mismatch (dimensionless) of the worst candidate site needed to hold x working ions per
    framework f.u.: sites sorted by |VPV − median| / median, the k-th best taken with k = floor(x · n_fu_cell) ions per
    cell (Tekliye et al. 2026). 1.0 if the structure has fewer than k sites (including none); 0.0 if k = 0 (no ion to
    place: the old code took index −1 here, i.e. the worst site); NaN without site information."""
    if vpvs is None or not np.isfinite(n_fu_cell) or not np.isfinite(x):
        return float("nan")
    k = int(np.floor(x * n_fu_cell + 1e-9))
    if k <= 0:
        return 0.0
    if len(vpvs) < k:
        return 1.0
    return float(np.sort(np.abs(np.asarray(vpvs, dtype=float) - median) / median)[k - 1])
