"""Transition: the one input every featurizer takes.

A transition is a change of ion content in one framework: from the initial (reference) material, whose structure is
known, to a final composition, whose structure is not. Featurizers may use only the initial structure and the two
compositions with their P_ox oxidation states (the application setting). A Transition has no field for an energy or
for the final structure, so no featurizer can use one.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from pymatgen.core import Composition, Structure

from ion_insertion_modeling import _redox as redox
from ion_insertion_modeling import config, formulas
from ion_insertion_modeling import environments as envs_mod
from ion_insertion_modeling import oxi_state_inference as oxi


@dataclass
class Transition:
    """Prepared inputs of one transition. Build with Transition.from_structure (runs ChemEnv and P_ox) or
    Transition.from_states (P_ox states already known, e.g. from the data release).

    x_* are ions per framework formula unit; li_frac_* the ion's atomic fraction (named li_frac for any ion);
    oxi_* are P_ox average oxidation states; p_ox_score_* their P_ox scores on the initial structure's environments;
    couples are the redox couples from the lower-x to the higher-x end (e⁻ per framework f.u.).
    """
    initial_structure: Structure
    final_composition: Composition
    ion: str
    framework_formula: str
    comp_initial: str
    comp_final: str
    x_initial: float
    x_final: float
    li_frac_initial: float
    li_frac_final: float
    oxi_initial: dict[str, float]
    oxi_final: dict[str, float]
    p_ox_score_initial: float
    p_ox_score_final: float
    couples: list[dict]
    oxo_bonded: dict[str, float] | None = None
    environments: list | None = field(default=None, repr=False)

    @property
    def n_couples(self) -> int:
        return len(self.couples)

    @property
    def delta_x(self) -> float:
        return self.x_final - self.x_initial

    @classmethod
    def from_states(cls, initial_structure: Structure, final_composition, oxi_initial: dict, oxi_final: dict,
                    p_ox_score_initial: float, p_ox_score_final: float, ion: str = config.DEFAULT_ION,
                    comp_initial: str | None = None, framework_formula: str | None = None,
                    oxo_bonded: dict | None = None) -> "Transition":
        """Transition from known P_ox states and scores. comp_initial defaults to the structure's reduced formula;
        oxo_bonded (for has_polyanion) is computed from the structure when not given."""
        s = initial_structure.copy()
        s.remove_oxidation_states()
        ci = comp_initial or s.composition.reduced_formula
        cf = Composition(final_composition).formula.replace(" ", "") if not isinstance(final_composition, str) \
            else final_composition
        fw = framework_formula or formulas.framework_formula(ci, ion)
        low, high = (oxi_initial, oxi_final)
        xi, xf = formulas.ion_per_fu(ci, fw, ion), formulas.ion_per_fu(cf, fw, ion)
        if xi > xf:
            low, high = oxi_final, oxi_initial
        couples = redox.assign_couples(low, high, fw)
        oxo = envs_mod.oxo_bonded_fraction(s) if oxo_bonded is None else oxo_bonded
        return cls(initial_structure=s, final_composition=Composition(cf), ion=ion, framework_formula=fw,
                   comp_initial=ci, comp_final=cf, x_initial=xi, x_final=xf,
                   li_frac_initial=formulas.ion_fraction(ci, ion), li_frac_final=formulas.ion_fraction(cf, ion),
                   oxi_initial=dict(oxi_initial), oxi_final=dict(oxi_final), p_ox_score_initial=float(p_ox_score_initial),
                   p_ox_score_final=float(p_ox_score_final), couples=couples, oxo_bonded=oxo)

    @classmethod
    def from_structure(cls, initial_structure: Structure, final_composition, ion: str = config.DEFAULT_ION,
                       environments: list | None = None, oxide_type: str = "oxide") -> "Transition":
        """Transition from the initial structure and the final composition: ChemEnv environments of the initial
        structure (or the ones given), P_ox at both ends on those environments, redox couples between them.
        Raises ValueError if P_ox finds no charge-balanced assignment at either end."""
        s = initial_structure.copy()
        s.remove_oxidation_states()
        oxo = envs_mod.oxo_bonded_fraction(s)
        fixed = oxi.fixed_states(oxo) or None
        env = envs_mod.environments_from_structure(s) if environments is None else environments
        gi = oxi.guess_from_environments(s.composition, env, oxide_type=oxide_type, fixed=fixed)
        gf = oxi.guess_from_environments(final_composition, env, oxide_type=oxide_type, fixed=fixed)
        if gi is None or gf is None:
            end = "initial" if gi is None else "final"
            raise ValueError(f"P_ox found no charge-balanced oxidation states for the {end} composition")
        t = cls.from_states(s, final_composition, gi.states, gf.states, gi.score, gf.score, ion=ion,
                            comp_initial=s.composition.reduced_formula, oxo_bonded=oxo)
        t.environments = env
        return t

    @classmethod
    def material(cls, structure: Structure, ion: str = config.DEFAULT_ION, **kwargs) -> "Transition":
        """A material-level row (for the Eform and direct-Edecomp targets): initial = final."""
        s = structure.copy()
        s.remove_oxidation_states()
        return cls.from_structure(s, s.composition, ion=ion, **kwargs)
