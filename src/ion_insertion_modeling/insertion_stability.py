"""Predictions for a new electrode material: stable ion-content range, capacity, voltage profile, energy density.

Inputs follow the paper's application setting: the initial (pristine) material's structure and its Edecomp, and a
trained ΔEdecomp model. For every ion content x on a grid (ions per framework formula unit), the composition
ion_x(framework) gets P1 oxidation states on the initial structure's environments, features, and a predicted
Ê(x) = Edecomp_initial + ΔÊ(x). The stable range is the unbroken run of grid states around the initial material with
Ê <= cutoff (75 meV/atom by default) and a P1 score > 0.

edecomp_initial must be the signed Edecomp (eV/atom) on the MP2020 GGA/GGA+U hull: negative for a material on the
hull (its distance below the hull of the other phases). Do not pass e_above_hull: with 0 instead of the negative
Edecomp of a stable material every Ê is biased upward and the predicted capacity comes out too small.
thermo.edecomp_of / thermo.edecomp_of_mpid compute it.

Voltages need energies, not just stability: E(x) = E_hull(c_x) + Ê(x), with E_hull the hull energy per atom of the
competing phases at c_x (thermo docs: passed in, or fetched from MP). G(x) = N(x)·E(x) − x·μ_ion per framework f.u.;
the voltage steps are V = −ΔG/Δx (V vs the ion metal) along the lower convex envelope of G, cut where the first
V <= 0 step begins. Energy density = Σ V·ΔQ over the steps (Wh/kg), with ΔQ the gravimetric capacity of each step on
the mass of framework + x_hi ions (the same basis as the capacity).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pymatgen.core import Composition, Element, Structure

from ion_insertion_modeling import config, formulas, thermo
from ion_insertion_modeling import environments as envs_mod
from ion_insertion_modeling import oxi_state_inference as oxi


# --- Small physics helpers (also used by model_building.capacity_metrics) ---
def grav_capacity(dx: float, x_hi: float, framework_mass: float, ion_mass: float, z: int = 1) -> float:
    """Gravimetric capacity (mAh/g) of dx ions per framework f.u., per gram of framework + x_hi ions."""
    return dx * z * config.FARADAY / (3.6 * (framework_mass + x_hi * ion_mass)) if dx > 0 else 0.0


def stable_range(stable, i0: int, contiguous: bool = config.CAPACITY_CONTIGUOUS) -> tuple[int, int]:
    """(lo, hi) indices (states sorted by x) of the stable range containing state i0 (always counted stable)."""
    stable = np.asarray(stable, dtype=bool).copy()
    stable[i0] = True
    if not contiguous:
        idx = np.flatnonzero(stable)
        return int(idx.min()), int(idx.max())
    lo = hi = i0
    while lo > 0 and stable[lo - 1]:
        lo -= 1
    while hi < len(stable) - 1 and stable[hi + 1]:
        hi += 1
    return lo, hi


def lower_envelope(x: np.ndarray, g: np.ndarray) -> list[int]:
    """Indices of the lower convex envelope of the points (x, g), sorted by x."""
    order = np.argsort(x, kind="stable")
    hull: list[int] = []
    for i in order:
        while len(hull) >= 2:
            a, b = hull[-2], hull[-1]
            if (g[b] - g[a]) * (x[i] - x[a]) >= (g[i] - g[a]) * (x[b] - x[a]):
                hull.pop()
            else:
                break
        hull.append(i)
    return hull


def default_grid(x0: float, x_max: float | None = None, n: int = 41) -> np.ndarray:
    """n evenly spaced ion contents from 0 to x_max (default max(2·x0, x0 + 1)), plus x0 itself."""
    x_max = max(2 * x0, x0 + 1.0) if x_max is None else x_max
    g = set(np.round(np.linspace(0.0, x_max, n), 6)) | {round(float(x0), 6)}
    return np.array(sorted(g))


# --- The stability curve ---
@dataclass
class StabilityCurve:
    """Predicted stability over ion content for one initial structure.

    states: one row per grid x: x, formula, p1_states, p1_score, delta_pred and edecomp_pred (eV/atom; the initial
        state keeps edecomp_initial), stable (Ê <= cutoff and P1 score > 0), in_range (in the unbroken stable run
        around the initial state), is_initial.
    """
    framework: str
    ion: str
    x_initial: float
    edecomp_initial: float
    cutoff: float
    states: pd.DataFrame
    features: pd.DataFrame = field(repr=False)

    @property
    def x_range(self) -> tuple[float, float]:
        r = self.states[self.states.in_range]
        return float(r.x.min()), float(r.x.max())

    def capacity(self) -> "CapacityResult":
        x_lo, x_hi = self.x_range
        mass = Composition(self.framework).weight
        ion_mass = float(Element(self.ion).atomic_mass)
        cap = grav_capacity(x_hi - x_lo, x_hi, mass, ion_mass, config.ION_CHARGE[self.ion])
        return CapacityResult(self.framework, self.ion, self.x_initial, x_lo, x_hi, x_hi - x_lo, cap, self.cutoff,
                              self)


@dataclass
class CapacityResult:
    """Stable-range capacity: x_lo..x_hi (ions per framework f.u.), dx, mAh/g (framework + x_hi ions)."""
    framework: str
    ion: str
    x_initial: float
    x_lo: float
    x_hi: float
    dx: float
    capacity_mAh_g: float
    cutoff: float
    curve: StabilityCurve = field(repr=False)


def _state_transitions(structure: Structure, framework: str, xs, ion: str, environments=None, oxide_type="oxide"):
    from ion_insertion_modeling.featurizers.transition import Transition

    s = structure.copy()
    s.remove_oxidation_states()
    oxo = envs_mod.oxo_bonded_fraction(s)
    fixed = oxi.fixed_states(oxo) or None
    env = envs_mod.environments_from_structure(s) if environments is None else environments
    if isinstance(env, str):
        env = envs_mod.uniform_environments(s.composition, env)
    gi = oxi.guess_from_environments(s.composition, env, oxide_type=oxide_type, fixed=fixed)
    if gi is None:
        raise ValueError("P1 found no charge-balanced oxidation states for the initial structure")
    ci = s.composition.formula.replace(" ", "")
    out = []
    for x in xs:
        comp = formulas.with_ion_content(framework, x, ion).replace(" ", "")
        g = oxi.guess_from_environments(comp, env, oxide_type=oxide_type, fixed=fixed)
        t = None if g is None else Transition.from_states(s, comp, gi.states, g.states, gi.score, g.score, ion=ion,
                                                            comp_initial=ci, framework_formula=framework,
                                                            oxo_bonded=oxo)
        out.append((x, comp, g, t))
    return gi, out


def stability_curve(initial_structure: Structure, edecomp_initial: float, model, x_grid=None,
                    ion: str = config.DEFAULT_ION, cutoff: float = config.ABOVE_HULL_THRESHOLD,
                    framework: str | None = None, environments=None) -> StabilityCurve:
    """Predicted Ê(x) over an ion-content grid (see the module docstring).

    model: a TrainedModel with target "delta_edecomp" (or "edecomp": Ê = ŷ).
    x_grid: ions per framework f.u. (default default_grid around the structure's own x).
    framework: the formula unit x counts ions per (default: the structure's composition without the ion, scaled to
        the smallest integer formula).
    environments: precomputed environments, or a ChemEnv symbol such as "O:6" to treat every cation as in that
        environment (structure-free; e.g. a cation-disordered rock salt). Default: ChemEnv on the structure.
    """
    from ion_insertion_modeling.featurizers import PaperFeaturizer

    if model.target not in ("delta_edecomp", "edecomp"):
        raise ValueError("stability_curve needs a ΔEdecomp or Edecomp model (an Eform model needs hull energies)")
    s = initial_structure.copy()
    s.remove_oxidation_states()
    framework = framework or oxi.default_framework(s, ion)
    x0 = formulas.ion_per_fu(s.composition, framework, ion) if s.composition[ion] else 0.0
    xs = default_grid(x0) if x_grid is None else np.array(sorted(set(np.round(list(x_grid), 6)) | {round(x0, 6)}))
    gi, states = _state_transitions(s, framework, xs, ion, environments)
    feat = PaperFeaturizer()
    labels = feat.feature_labels()
    rows, frows = [], []
    for x, comp, g, t in states:
        is_init = abs(x - x0) < 1e-6
        f = dict(zip(labels, feat.featurize(t))) if t is not None else {}
        frows.append({"x": x, **f})
        rows.append({"x": float(x), "formula": comp, "p1_states": None if g is None else g.states,
                     "p1_score": np.nan if g is None else g.score, "is_initial": is_init})
    st = pd.DataFrame(rows)
    fx = pd.DataFrame(frows)
    ok = st.p1_score.notna().to_numpy()
    pred = np.full(len(st), np.nan)
    if ok.any():
        pred[ok] = model.predict(fx.loc[ok, model.features])
    if model.target == "delta_edecomp":
        st["delta_pred"] = pred
        st["edecomp_pred"] = edecomp_initial + pred
    else:
        st["delta_pred"] = pred - edecomp_initial
        st["edecomp_pred"] = pred
    st.loc[st.is_initial, ["delta_pred", "edecomp_pred"]] = [0.0, edecomp_initial]
    st["stable"] = (st.edecomp_pred <= cutoff) & (st.p1_score > 0)
    st.loc[st.is_initial, "stable"] = True
    i0 = int(np.flatnonzero(st.is_initial.to_numpy())[0])
    lo, hi = stable_range(st.stable.to_numpy(), i0)
    st["in_range"] = False
    st.loc[lo:hi, "in_range"] = True
    return StabilityCurve(framework, ion, x0, float(edecomp_initial), cutoff, st, fx)


def predict_capacity(initial_structure: Structure, edecomp_initial: float, model,
                     cutoff: float = config.ABOVE_HULL_THRESHOLD, **kwargs) -> CapacityResult:
    """Stable-range capacity of one initial structure (see stability_curve for the keyword arguments)."""
    return stability_curve(initial_structure, edecomp_initial, model, cutoff=cutoff, **kwargs).capacity()


def predict_capacities(structures, edecomp_initials, model, cutoff: float = config.ABOVE_HULL_THRESHOLD,
                       ids=None, n_jobs: int = 1, **kwargs) -> pd.DataFrame:
    """predict_capacity for many structures. One row per structure: id, framework, x_initial, x_lo, x_hi, dx,
    capacity_mAh_g, error (the failure message when a structure could not be processed)."""
    structures, edecomp_initials = list(structures), list(edecomp_initials)
    ids = list(ids) if ids is not None else [s.properties.get("id", i) if hasattr(s, "properties") else i
                                             for i, s in enumerate(structures)]

    def one(i, s, e):
        try:
            c = predict_capacity(s, e, model, cutoff=cutoff, **kwargs)
            return {"id": i, "framework": c.framework, "x_initial": c.x_initial, "x_lo": c.x_lo, "x_hi": c.x_hi,
                    "dx": c.dx, "capacity_mAh_g": c.capacity_mAh_g, "error": None}
        except Exception as exc:  # noqa: BLE001 - one bad structure must not stop a screen
            return {"id": i, "framework": None, "x_initial": np.nan, "x_lo": np.nan, "x_hi": np.nan, "dx": np.nan,
                    "capacity_mAh_g": np.nan, "error": f"{type(exc).__name__}: {exc}"[:300]}

    if n_jobs == 1:
        rows = [one(i, s, e) for i, s, e in zip(ids, structures, edecomp_initials)]
    else:
        from joblib import Parallel, delayed
        rows = Parallel(n_jobs=n_jobs)(delayed(one)(i, s, e) for i, s, e in zip(ids, structures, edecomp_initials))
    return pd.DataFrame(rows)


# --- Voltage profile and energy density ---
@dataclass
class VoltageProfile:
    """Voltage steps (V vs the ion metal) over the predicted stable range.

    steps: x_from, x_to (ions per framework f.u.), V. states: the stability-curve states with E (eV/atom) and G
    (eV per framework f.u.) and whether each lies on the lower envelope.
    """
    framework: str
    ion: str
    x_initial: float
    steps: pd.DataFrame
    states: pd.DataFrame
    mu_ion: float
    curve: StabilityCurve = field(repr=False)

    @property
    def x_range(self) -> tuple[float, float]:
        return float(self.steps.x_from.min()), float(self.steps.x_to.max())

    @property
    def mean_voltage(self) -> float:
        dx = (self.steps.x_to - self.steps.x_from).to_numpy()
        return float(np.sum(self.steps.V.to_numpy() * dx) / dx.sum()) if dx.sum() > 0 else float("nan")


@dataclass
class EnergyDensityResult:
    wh_per_kg: float
    mean_voltage: float
    capacity_mAh_g: float
    x_lo: float
    x_hi: float
    profile: VoltageProfile = field(repr=False)


def predict_voltage_profile(initial_structure: Structure, edecomp_initial: float, model, docs: list[dict] | None = None,
                            exclude=frozenset(), ion: str = config.DEFAULT_ION,
                            cutoff: float = config.ABOVE_HULL_THRESHOLD,
                            initial_energy_per_atom: float | None = None, **kwargs) -> VoltageProfile:
    """Voltage profile over the predicted stable range.

    initial_energy_per_atom: the initial material's corrected energy (eV/atom, MP2020 GGA/GGA+U), if known (e.g. its
        doc energy). Default: the hull of the other phases at its composition plus edecomp_initial (exact when the
        phase-separation energy sets Edecomp, i.e. no lower-energy polymorph of the same formula).

    docs: hull entries of the chemical system on the MP2020 GGA/GGA+U scale ([{mpid, comp, energy_per_atom}];
        Dataset.hull_docs for the paper's electrodes, thermo.docs_from_mp otherwise; None fetches from MP).
    exclude: mpids to leave out of the hull (e.g. known (de)lithiated phases of this framework, which a real screen
        would not know).
    """
    curve = stability_curve(initial_structure, edecomp_initial, model, ion=ion, cutoff=cutoff, **kwargs)
    s = initial_structure.copy()
    s.remove_oxidation_states()
    if docs is None:
        docs = thermo.docs_from_mp(thermo.chemsys(s.composition, ion))
    st = curve.states.copy()
    st = st[st.in_range].reset_index(drop=True)
    init_formula = st.formula[st.is_initial].iloc[0]
    e_init = initial_energy_per_atom
    if e_init is None:  # the hull of the other phases at the initial composition, plus edecomp_initial
        red = Composition(init_formula).reduced_formula
        others = [d for d in docs if d["mpid"] not in set(exclude) and Composition(d["comp"]).reduced_formula != red]
        e_init = thermo.hull_energy_per_atom(thermo.phase_diagram(others)[0], init_formula) + edecomp_initial
    # Ê of every other state is relative to the hull with the initial material on it (as in training)
    from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram

    _, entries = thermo.phase_diagram(docs, exclude)
    c0 = Composition(init_formula)
    pd_ = PhaseDiagram(list(entries.values()) + [PDEntry(c0, e_init * c0.num_atoms, name="initial")])
    mu = thermo.ion_chemical_potential(pd_, ion)
    n_fw = Composition(curve.framework).num_atoms
    st["N"] = n_fw + st.x
    st["E_hull"] = [thermo.hull_energy_per_atom(pd_, f) for f in st.formula]
    st["E"] = np.where(st.is_initial, e_init, st.E_hull + st.edecomp_pred)
    st["G"] = st.N * st.E - st.x * mu
    x0 = curve.x_initial

    def envelope(h):
        xx, g = h.x.to_numpy(float), h.G.to_numpy(float)
        env = lower_envelope(xx, g)
        segs = [{"x_from": xx[a], "x_to": xx[b], "V": -(g[b] - g[a]) / (xx[b] - xx[a])} for a, b in zip(env[:-1], env[1:])]
        return env, segs

    cur = st
    while True:  # cut at the envelope vertex where the first V <= 0 step starts, on either side of x0
        _, segs = envelope(cur)
        neg = [q for q in segs if q["V"] <= 0]
        if not neg:
            break
        keep = np.ones(len(cur), bool)
        up = [q["x_from"] for q in neg if q["x_from"] >= x0 - 1e-9]
        down = [q["x_to"] for q in neg if q["x_to"] <= x0 + 1e-9]
        if up:
            keep &= cur.x.to_numpy() <= min(up) + 1e-9
        if down:
            keep &= cur.x.to_numpy() >= max(down) - 1e-9
        cur = cur[keep].reset_index(drop=True)
    env, segs = envelope(cur)
    cur["on_envelope"] = [i in env for i in range(len(cur))]
    return VoltageProfile(curve.framework, ion, x0, pd.DataFrame(segs, columns=["x_from", "x_to", "V"]), cur, mu, curve)


def energy_density(profile: VoltageProfile) -> EnergyDensityResult:
    """Wh/kg of a voltage profile: Σ V·ΔQ with ΔQ in mAh/g on the mass of framework + x_hi ions."""
    steps = profile.steps
    if steps.empty:
        return EnergyDensityResult(0.0, float("nan"), 0.0, profile.x_initial, profile.x_initial, profile)
    x_lo, x_hi = profile.x_range
    mass = Composition(profile.framework).weight
    ion_mass = float(Element(profile.ion).atomic_mass)
    z = config.ION_CHARGE[profile.ion]
    dq = np.array([grav_capacity(r.x_to - r.x_from, x_hi, mass, ion_mass, z) for r in steps.itertuples()])
    wh = float(np.sum(steps.V.to_numpy() * dq))  # mAh/g × V = Wh/kg
    return EnergyDensityResult(wh, profile.mean_voltage, float(dq.sum()), x_lo, x_hi, profile)


def predict_energy_density(initial_structure: Structure, edecomp_initial: float, model, docs: list[dict] | None = None,
                           **kwargs) -> EnergyDensityResult:
    """Predicted energy density (Wh/kg) from the voltage profile over the stable range (see
    predict_voltage_profile for the arguments)."""
    return energy_density(predict_voltage_profile(initial_structure, edecomp_initial, model, docs=docs, **kwargs))


__all__ = ["stability_curve", "predict_capacity", "predict_capacities", "predict_voltage_profile",
           "predict_energy_density", "energy_density", "StabilityCurve", "CapacityResult", "VoltageProfile",
           "EnergyDensityResult", "grav_capacity", "stable_range", "lower_envelope", "default_grid"]
