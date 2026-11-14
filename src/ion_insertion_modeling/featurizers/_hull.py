"""Group hull: competing-phase hull energies. eV/atom, on the locally rebuilt GGA/GGA+U
hull (thermo.py), the same hull every Edecomp comes from (CLAUDE.md rule 5).

Per battery, the hull is built from every mixed entry in the battery's chemical system (framework elements + Li)
except the battery's non-initial materials, so no row's final-material energy (or that of any other final in the
battery) enters its features (rule 3). The initial material stays in: its energy is allowed.

- hull_ef_initial / hull_ef_final: formation energy per atom of that hull at the initial / final composition.
- delta_hull_ef = hull_ef_final − hull_ef_initial.
"""
import warnings

from ion_insertion_modeling import thermo

LABELS = ["hull_ef_initial", "hull_ef_final", "delta_hull_ef"]


def battery_features(docs: list[dict], exclude: set[str], comp_initial: str,
                     comp_finals: dict[str, str]) -> dict[str, dict[str, float]]:
    """{transition_id: hull features} for one battery. comp_finals maps transition_id -> final composition."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pd_, _ = thermo.phase_diagram(docs, exclude)
        e_init = thermo.hull_formation_energy(pd_, comp_initial)
        out = {}
        for tid, comp in comp_finals.items():
            e_fin = thermo.hull_formation_energy(pd_, comp)
            out[tid] = {"hull_ef_initial": e_init, "hull_ef_final": e_fin, "delta_hull_ef": e_fin - e_init}
    return out


def battery_job(docs: list[dict], members: list[str], initial_mpid: str, comp_initial: str,
                comp_finals: dict[str, str]) -> dict:
    """Features per transition for one battery (members = all the battery's mpids); run in a worker."""
    return battery_features(docs, set(members) - {initial_mpid}, comp_initial, comp_finals)
