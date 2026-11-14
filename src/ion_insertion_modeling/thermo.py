"""MP2020-corrected GGA/GGA+U convex hulls, Edecomp and hull energies (eV/atom).

The paper rebuilds one hull per chemical system from Materials Project GGA/GGA+U entries (MP2020 corrections),
because MP's stored energies mix GGA(+U) and r2SCAN across subsystems. Every Edecomp in the paper, and every
Edecomp this package expects as input, is on that scale. Hull entries are passed around as "docs":
[{"mpid", "comp", "energy_per_atom"}] (corrected total energy per atom).

Edecomp convention: the energy above the hull when the material is above it (> 0); otherwise (on the hull) the
negative distance to the hull of the other phases: the larger (less negative) of the phase-separation energy and
the energy relative to the most stable other polymorph of the same composition.
"""
from __future__ import annotations

import itertools
import warnings

from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core import Composition

from ion_insertion_modeling import config


def chemsys(comp, ion: str = config.DEFAULT_ION) -> str:
    """Chemical system of a composition plus the working ion: 'FePO4' -> 'Fe-Li-O-P'."""
    return "-".join(sorted({el.symbol for el in Composition(comp)} | {ion}))


def subsystems(system: str) -> list[str]:
    els = system.split("-")
    return ["-".join(sorted(c)) for r in range(1, len(els) + 1) for c in itertools.combinations(els, r)]


def phase_diagram(docs: list[dict], exclude=frozenset()) -> tuple[PhaseDiagram, dict[str, PDEntry]]:
    """Phase diagram of the docs, leaving out the mpids in `exclude`."""
    entries = {d["mpid"]: PDEntry(Composition(d["comp"]), d["energy_per_atom"] * Composition(d["comp"]).num_atoms,
                                  name=d["mpid"]) for d in docs if d["mpid"] not in set(exclude)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return PhaseDiagram(list(entries.values())), entries


def edecomp(entry: PDEntry, pd_: PhaseDiagram, entries: dict[str, PDEntry]) -> float:
    """Edecomp (eV/atom) of an entry of the diagram (see the module docstring for the convention)."""
    e_hull = pd_.get_e_above_hull(entry, allow_negative=True)
    if e_hull is not None and e_hull > 1e-12:
        return float(e_hull)
    psep = float(pd_.get_phase_separation_energy(entry))
    red = entry.composition.reduced_formula
    others = [e.energy_per_atom for k, e in entries.items() if k != entry.name and e.composition.reduced_formula == red]
    return max(psep, entry.energy_per_atom - min(others)) if others else psep


def edecomp_of(composition, energy_per_atom: float, docs: list[dict], exclude=frozenset(),
               name: str = "candidate") -> float:
    """Edecomp (eV/atom) of a new material (composition, corrected energy per atom on the MP2020 GGA/GGA+U scale)
    against the docs' hull."""
    pd_, entries = phase_diagram(docs, exclude)
    comp = Composition(composition)
    e = PDEntry(comp, energy_per_atom * comp.num_atoms, name=name)
    pd2 = PhaseDiagram(list(entries.values()) + [e])
    return edecomp(e, pd2, {**entries, name: e})


def edecomp_of_mpid(mpid: str, docs: list[dict]) -> float:
    """Edecomp (eV/atom) of a material that is in the docs."""
    pd_, entries = phase_diagram(docs)
    return edecomp(entries[mpid], pd_, entries)


def hull_energy_per_atom(pd_: PhaseDiagram, comp) -> float:
    """Energy per atom (eV/atom) of the convex hull at `comp`."""
    return float(pd_.get_hull_energy_per_atom(Composition(comp)))


def hull_formation_energy(pd_: PhaseDiagram, comp) -> float:
    """Formation energy per atom (eV/atom, vs the diagram's elemental references) of the convex hull at `comp`."""
    c = Composition(comp)
    ref = sum(c.get_atomic_fraction(el) * pd_.el_refs[el].energy_per_atom for el in c.elements)
    return float(pd_.get_hull_energy_per_atom(c) - ref)


def ion_chemical_potential(pd_: PhaseDiagram, ion: str = config.DEFAULT_ION) -> float:
    """μ of the ion metal (eV/atom) on the diagram: its elemental reference energy per atom."""
    from pymatgen.core import Element

    return float(pd_.el_refs[Element(ion)].energy_per_atom)


def docs_from_mp(system: str, api_key: str | None = None) -> list[dict]:
    """Docs of a chemical system and all its subsystems from the Materials Project (needs mp-api, extra [mp], and
    MP_API_KEY in the environment or api_key): GGA/GGA+U thermo entries passed through
    MaterialsProjectDFTMixingScheme (which leaves a GGA/GGA+U-only set unchanged), one entry per material (the
    lower energy if both runs are kept). Energies can drift between MP database versions; the paper's hull docs
    are in the data release (Dataset.hull_docs)."""
    import os

    from mp_api.client import MPRester
    from pymatgen.entries.mixing_scheme import MaterialsProjectDFTMixingScheme

    key = api_key or os.environ.get("MP_API_KEY")
    if not key:
        raise RuntimeError("set MP_API_KEY (or pass api_key=) to fetch entries from the Materials Project")
    with MPRester(key, mute_progress_bars=True) as mpr:
        docs = mpr.materials.thermo.search(chemsys=subsystems(system), thermo_types=["GGA_GGA+U"],
                                           fields=["material_id", "entries"])
    entries = []
    for d in docs:
        for e in (d.entries or {}).values():
            e.data["material_id"] = str(d.material_id)
            entries.append(e)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mixed = MaterialsProjectDFTMixingScheme().process_entries(entries, verbose=False)
    best: dict = {}
    for e in mixed:
        m = e.data["material_id"]
        if m not in best or e.energy_per_atom < best[m].energy_per_atom:
            best[m] = e
    return [{"mpid": m, "comp": e.composition.formula, "energy_per_atom": float(e.energy_per_atom)}
            for m, e in sorted(best.items())]
