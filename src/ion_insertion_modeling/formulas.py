"""Composition helpers: framework formula, ion content x and atoms per framework formula unit.

The framework is the composition with the working ion removed. x is the number of working-ion atoms per framework
formula unit (f.u.), e.g. LiMn2O4 on the framework MnO2 has 2 f.u. per formula and x = 0.5.
"""
from fractions import Fraction

import numpy as np
from pymatgen.core import Composition

from ion_insertion_modeling import config


def framework_formula(comp: str | Composition, ion: str = config.DEFAULT_ION) -> str:
    """Reduced formula with the working ion removed: 'LiMn2O4' -> 'MnO2'."""
    c = Composition(comp)
    return Composition({el: amt for el, amt in c.items() if el.symbol != ion}).reduced_formula


def framework_units(comp: str | Composition, framework: str, ion: str = config.DEFAULT_ION,
                    tol: float = config.FRAMEWORK_TOL) -> float:
    """Number of framework formula units in `comp` (LiMn2O4 per MnO2 -> 2).
    Raises ValueError if the non-ion part of comp is not a multiple of the framework."""
    c, f = Composition(comp), Composition(framework)
    host = {el.symbol: amt for el, amt in c.items() if el.symbol != ion}
    ref = {el.symbol: amt for el, amt in f.items()}
    if set(host) != set(ref):
        raise ValueError(f"{c.formula}: framework elements {sorted(host)} differ from {framework} {sorted(ref)}")
    ratios = [host[el] / ref[el] for el in ref]
    if max(ratios) - min(ratios) > tol * max(ratios):
        raise ValueError(f"{c.formula}: framework {framework} not conserved (ratios {ratios})")
    return float(np.mean(ratios))


def ion_per_fu(comp: str | Composition, framework: str, ion: str = config.DEFAULT_ION) -> float:
    """x: working-ion atoms per framework formula unit."""
    return float(Composition(comp)[ion] / framework_units(comp, framework, ion))


def atoms_per_fu(comp: str | Composition, framework: str, ion: str = config.DEFAULT_ION) -> float:
    """Atoms (framework + ion) per framework formula unit; eV/atom x this = eV per f.u."""
    return float(Composition(comp).num_atoms / framework_units(comp, framework, ion))


def ion_fraction(comp: str | Composition, ion: str = config.DEFAULT_ION) -> float:
    """Working-ion atomic fraction of the full composition (dimensionless). Feature columns call it li_frac
    whatever the ion."""
    c = Composition(comp)
    return float(c[ion] / c.num_atoms)


def with_ion_content(framework: str, x: float, ion: str = config.DEFAULT_ION, max_denominator: int = 64) -> str:
    """Integer formula of ion_x(framework): x = 0.25 on FePO4 -> 'Li1 Fe4 P4 O16'. A fractional framework
    (Pr3W1.75Mn0.25O12) is scaled up to whole numbers."""
    f = Fraction(float(x)).limit_denominator(max_denominator)
    fw = Composition(framework)
    comp = (Composition({ion: f.numerator}) + fw * f.denominator) if f else fw
    for k in range(1, max_denominator + 1):
        scaled = comp * k
        if all(abs(a - round(a)) < 1e-6 for a in scaled.values()):
            return Composition({el: round(a) for el, a in scaled.items()}).formula
    return comp.formula


def molar_mass(comp: str | Composition) -> float:
    """g/mol of the formula as written."""
    return float(Composition(comp).weight)
