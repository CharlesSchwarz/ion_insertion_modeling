"""Write the example notebooks in notebooks/ (then execute them: tools/build_notebooks.py --execute).

The notebooks are generated from this file so their code stays reviewable in one place.
"""
import argparse
from pathlib import Path

import nbformat as nbf

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"

SETUP = '''%matplotlib inline
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
'''


def md(s):
    return nbf.v4.new_markdown_cell(s.strip())


def code(s):
    return nbf.v4.new_code_cell(s.strip())


def quickstart():
    return [
        md("""
# Quickstart: capacity of a cathode from its structure

Predict the stable lithium range and capacity of a cathode material from its structure and its decomposition energy,
with the paper's pretrained model (the interpretable-five ΔEdecomp random forest).

Inputs:
- the pristine structure (here olivine LiFePO4, mp-19017);
- its Edecomp on the MP2020 GGA/GGA+U hull: −8.2 meV/atom (negative means it sits below the hull of the other phases).
"""),
        code(SETUP),
        code('''
from pymatgen.core import Structure
from ion_insertion_modeling.model_building import load_pretrained
from ion_insertion_modeling.insertion_stability import predict_capacity

model = load_pretrained()  # ships with the package
structure = Structure.from_file("../tests/data/LiFePO4_mp-19017.cif")
cap = predict_capacity(structure, edecomp_initial=-0.0082, model=model, x_grid=np.arange(0, 1.51, 0.25))
print(f"stable range x = {cap.x_lo:g} to {cap.x_hi:g} Li per FePO4, capacity {cap.capacity_mAh_g:.0f} mAh/g")'''),
        md("""
Every grid composition Li$_x$FePO$_4$ gets P_ox oxidation states (scored on the LiFePO4 coordination environments),
five features and a predicted Edecomp. The stable range is the unbroken run around the initial material whose
predicted Edecomp is at most 75 meV/atom and whose P_ox score is above 0. Beyond x = 1 no charge-balanced assignment
is plausible (Fe would go below 2+), so the P_ox score drops to 0:
"""),
        code('cap.curve.states[["x", "formula", "p_ox_score", "edecomp_pred", "stable", "in_range"]].round(4)'),
    ]


def oxi_notebook():
    return [
        md("""
# Oxidation-state inference (P_ox)

The P_ox guesser assigns oxidation states from a composition and the coordination environments of its cations. It
needs no DFT and no bond valences, so it works on the (de)lithiated compositions whose structures are unknown.

1. **Candidates.** pymatgen enumerates every charge-balanced assignment of the composition. Some states are fixed
   by rule: O at −2 in oxides; Li, Na, Mg, F and similar at their only common state; P at +5 when every P site
   bonds only to O.
2. **Score.** Each candidate is scored with ICSD statistics of how often an element in a given state sits in a
   given environment: P(ox | env, el) = P(env | ox, el) P(ox | el) / P(env | el). The score is the geometric mean
   over the cation environments, weighted by site count.
3. **Pick.** The highest-scoring candidate wins.

An element with a fractional average state (Mn 2.29+) is scored by its best split over its sites into the two
adjacent integers (Mn3+ in some environments, Mn2+ in others).
"""),
        code(SETUP),
        code("from ion_insertion_modeling import oxi_state_inference as oxi"),
        md("## 1. The statistics: P(ox | el) and P(ox | env, el)"),
        code('''
rows = []
for el in ("Fe", "Mn", "Co", "V"):
    t = oxi.element_table(el)
    for ox in sorted(t.oxidation_state):
        if 0 < ox <= 7:
            rows.append({"element": el, "ox": int(ox), "P(ox|el)": oxi.p_ox(el, ox),
                         "P(ox|O:6,el)": oxi.p_ox_given_env(el, ox, "O:6"),
                         "P(ox|T:4,el)": oxi.p_ox_given_env(el, ox, "T:4")})
probs = pd.DataFrame(rows)
probs.round(3)'''),
        code('''
fig, axes = plt.subplots(1, 2, figsize=(9, 3), sharey=True)
for ax, el in zip(axes, ("Mn", "Co")):
    p = probs[probs.element == el].set_index("ox")[["P(ox|el)", "P(ox|O:6,el)", "P(ox|T:4,el)"]]
    p.plot.bar(ax=ax, rot=0, width=0.8)
    ax.set_title(el); ax.set_xlabel("oxidation state"); ax.set_ylabel("probability")
plt.tight_layout()'''),
        md("""
Environment matters: tetrahedral Mn is mostly Mn2+ (or Mn5+, Mn6+, Mn7+ oxo-anions), octahedral Mn mostly Mn3+/Mn4+.
ChemEnv symbols: O:6 octahedron, T:4 tetrahedron, C:8 cube, S:4 square plane, etc.

## 2. Scoring a composition with known environments

LiFePO4: Fe in octahedra, P in tetrahedra (4 of each per cell). Every candidate is listed with its score:
"""),
        code('''
envs = [("Fe", "O:6", 4), ("P", "T:4", 4)]
g = oxi.guess_from_environments("LiFePO4", envs, fixed={"P": 5})
print(g)
pd.DataFrame([{**states, "score": s} for states, s in g.candidates])'''),
        md("Delithiate it: the same environments, composition FePO4. Fe goes to 3+:"),
        code('oxi.guess_from_environments("FePO4", envs, fixed={"P": 5})'),
        md("""
## 3. Mixed valence and the per-site split

Magnetite Fe3O4 has one tetrahedral and two octahedral Fe per formula. Its average state is Fe 2.67+, scored as the
best split: Fe3+ on the tetrahedral site and Fe2+/Fe3+ on the octahedral ones. In Mn7(P2O7)4, 2 of the 14 Mn sites
are 8-coordinate, where Mn3+ was never observed. Interpolating to Mn 2.29+ on every site would give a score of 0;
the split puts Mn2+ there and Mn3+ in the octahedra.
"""),
        code('''
for comp, e, fixed in [("Fe3O4", [("Fe", "T:4", 1), ("Fe", "O:6", 2)], None),
                       ("Mn7(P2O7)4", [("Mn", "O:6", 12), ("Mn", "C:8", 2), ("P", "T:4", 16)], {"P": 5})]:
    split = oxi.guess_from_environments(comp, e, fixed=fixed)
    plain = oxi.guess_from_environments(comp, e, fixed=fixed, scoring="P_ox_env_sites")
    print(f"{comp:12s} states {split.states}   score with split {split.score:.3f}, averaged only {plain.score:.3f}")'''),
        md("""
## 4. From a structure

`guess_from_structure` runs ChemEnv on the structure (a few seconds for small cells), then scores the candidates.
`guess_from_framework` scores *another* composition of the same framework on those environments. That is how every
(de)lithiated material is assigned without its own structure.
"""),
        code('''
from pymatgen.core import Structure

lfp = Structure.from_file("../tests/data/LiFePO4_mp-19017.cif")
print(oxi.environments_from_structure(lfp))  # O entries are kept for reference; P_ox skips them (O is fixed at -2)
print(oxi.guess_from_structure(lfp))
print(oxi.guess_from_framework("Li0.5FePO4", lfp))'''),
        md("""
## 5. A disordered structure

The cation-disordered garnet Li7Pr3(Mn0.25Mo0.75)2O12 has Mn/Mo sharing octahedral sites and partly occupied Li
sites. ChemEnv counts each species of a mixed site by its occupancy; the formula is scaled to integers
(Li56Pr24Mn4Mo12O96) before candidates are enumerated. ChemEnv takes about a minute on this 208-site cell.

The CIF was written by pymatgen with oxidation states attached (Mn4+, Mo4+). Those states are only one charge-balanced
guess; P_ox ignores them and decides from the environments:
"""),
        code('''
garnet = Structure.from_file("data/disordered_Li7PrMoMnO.cif")
print("ordered:", garnet.is_ordered, "| sites:", len(garnet), "|", garnet.composition.reduced_formula)
print("states written in the CIF:", sorted({str(sp) for site in garnet for sp in site.species}))
g = oxi.guess_from_structure(garnet)
print(g)
pd.DataFrame([{**s, "score": sc} for s, sc in g.candidates[:6]]).round(3)'''),
        md("""
Every octahedral candidate is close (0.54, 0.47, 0.44): this composition really is ambiguous between Mn2+/Mo4.67+
and Mn4+/Mo4+. A structure-free alternative treats every cation as octahedral (`uniform_environments`), which is how
compositions without a structure (e.g. disordered rock salts) can be handled:
"""),
        code('oxi.guess_from_environments(garnet.composition, oxi.uniform_environments(garnet.composition, "O:6"))'),
        md("""
## 6. Which Li content is most plausible?

`best_ion_content` scans the Li content of a framework and returns the content whose composition gets the highest
P_ox score on the structure's environments. Here x counts Li per Pr3Mo1.5Mn0.5O12 (12 O):
"""),
        code('''
x_best, g_best, table = oxi.best_ion_content(garnet, x_grid=np.arange(3, 8.01, 0.25), framework_formula="Pr3Mo1.5Mn0.5O12")
print("highest P_ox score at x =", x_best, g_best.states)
ax = table.plot(x="x", y="score", marker="o", legend=False, figsize=(5, 3))
ax.axvline(7, color="0.6", ls="--", label="x in the CIF")
ax.set_xlabel("x in Li$_x$Pr$_3$Mo$_{1.5}$Mn$_{0.5}$O$_{12}$"); ax.set_ylabel("P_ox score"); ax.legend()'''),
        md("""
The score peaks where the oxidation states are most typical for these environments (here octahedral Mo6+ with Mn2+).
It measures how plausible the oxidation states are, not thermodynamic stability; the stable Li range comes from the
model (notebook 03). In the paper, the change in this score between the initial and (de)lithiated material
(`delta_p_ox_score`) is one of the strongest predictors of ΔEdecomp.
"""),
    ]


def training_notebook():
    return [
        md("""
# Example model training: ΔEdecomp random forest

This walks through how the paper's models are built, on the **demo dataset** that ships with the package: 300
randomly chosen training electrodes (plus olivine LiFePO4) of the full Li dataset, with every table of the full
release and no held-out test electrode. The same code runs on the full dataset with `load_dataset("Li")` once the
data release is downloaded. Numbers on the demo are noisier than, and differ from, the full dataset's.

Steps:
1. the data and how it was cleaned;
2. the grouped train/test split, and why rows are never split at random;
3. feature selection inside each fold;
4. cross-validation and parity plots;
5. fitting and saving the final model.

The held-out test set is never touched here.
"""),
        code(SETUP),
        code('''
from ion_insertion_modeling.model_building import (load_demo_dataset, cross_validate, fit_final, rank_features,
                                                   interpretable_ranking, interpretable_walk)
from ion_insertion_modeling.model_building.plots import parity_from_cv
from ion_insertion_modeling.featurizers import FEATURE_SETS

ds = load_demo_dataset()   # full data: load_dataset("Li") after downloading the release
ds.summary()'''),
        md("""
## 1. Data and cleaning

Materials Project InsertionElectrode electrodes with Li as the working ion and O-containing frameworks. Each
electrode's initial (reference) material is its lowest-Edecomp material. Every other material of the electrode gives
one transition (initial → (de)lithiated), with target ΔEdecomp = Edecomp(final) − Edecomp(initial) in eV/atom on
the rebuilt MP2020 GGA/GGA+U hull.

Removed before the split was fixed:
"""),
        code('''
import json
log = json.loads((ds.root / "cleaning_log.json").read_text())
print("electrodes in the MP database:", log["ingest"]["n_batteries"], "| materials:", log["ingest"]["n_materials"])
dropped = pd.read_parquet(ds.root / "batteries_dropped.parquet")
print("\\nelectrodes removed:"); print(dropped.reason.value_counts().to_string())
tdrop = pd.read_parquet(ds.root / "transitions_dropped.parquet")
print("\\n(de)lithiated materials removed:"); print(tdrop.drop_reason.value_counts().to_string())
print("\\nDFT outlier electrode left out:", json.loads((ds.root / "checksums.json").read_text())["outlier_batteries"])'''),
        md("""
- Electrodes are dropped when every material is more than 75 meV/atom above the hull, or when their materials do not
  share one framework.
- Materials are dropped for implausible Fe/Mn magnetic moments, a missing energy, or no cation–anion bond.
- (De)lithiated materials P_ox cannot assign (`missing_p_ox`) stay in the evaluation: they count as predicted unstable.
"""),
        code('''
t = ds.transitions
fig, ax = plt.subplots(1, 2, figsize=(9, 3))
ax[0].hist(1000 * t.delta_Edecomp.clip(-200, 600), bins=80); ax[0].set_xlabel("ΔEdecomp (meV/atom)")
ax[1].hist(1000 * t.Edecomp_final.clip(-200, 600), bins=80); ax[1].axvline(75, color="k", ls="--")
ax[1].set_xlabel("Edecomp of the (de)lithiated material (meV/atom)")
print(f"{(t.Edecomp_final > 0.075).mean():.0%} of (de)lithiated materials are more than 75 meV/atom above the hull")'''),
        md("""
## 2. Grouped split

The (de)lithiated materials of one electrode share its framework and are near-copies of each other. A random row
split puts siblings on both sides and overstates accuracy. So 10% of the **electrodes** are held out as the test set,
and the other 90% are split into 5 cross-validation folds by electrode. The split was fixed before any target was
looked at; `grouped_split` refuses any column besides the id and the group:
"""),
        code('''
b = ds.battery_split
print(b.is_test.value_counts().rename({False: "train", True: "test"}).to_string())
print(b[~b.is_test].cv_fold.value_counts().sort_index().to_string())'''),
        code('''
# What a random row split would report instead (for illustration only; same model, same folds count)
from sklearn.model_selection import KFold, cross_val_predict
from ion_insertion_modeling.model_building.train import make_estimator
rows = ds.train()
X, y = rows[list(FEATURE_SETS["interpretable5"])], rows.delta_Edecomp
pred_random = cross_val_predict(make_estimator("rf", min_samples_leaf=1, max_features=0.33), X, y,
                                cv=KFold(5, shuffle=True, random_state=0))
print(f"random row split MAE: {1000 * np.mean(np.abs(pred_random - y)):.1f} meV/atom")'''),
        md("""
Compare this with the grouped CV below. With only five coarse features the gap is small; it grows with richer,
more material-specific features, which let a model recognise a sibling it has seen.
The grouped split is used throughout so the numbers hold for frameworks the model has never seen.

## 3. Feature selection inside each fold

Candidate features: the 364 columns of the paper's reference set (Magpie statistics of the compositions, ion content,
P_ox-state properties, P_ox scores, framework structure and insertion sites; all computable from the initial structure
and the two compositions). Within each fold's training rows only:
- a random forest pre-screens the top 40 by impurity importance;
- permutation importance on a grouped inner hold-out ranks them;
- a model with k features uses the top k.

The ranking on all (demo) training electrodes:
"""),
        code('''
cols = list(FEATURE_SETS["reference"])
rows = ds.train()
ranking = rank_features(rows[cols], rows.delta_Edecomp.to_numpy(), rows.battery_id.to_numpy())
pd.Series(ranking[:15], name="feature").to_frame()'''),
        md("""
The interpretable five come from walking down a ranking made with `has_polyanion` forced into every
scoring model (`interpretable_ranking`):
- start from `has_polyanion`;
- skip a feature that is redundant with one already kept (normalised mutual information > 0.5);
- swap or drop a feature that mostly separates polyanion from non-polyanion frameworks (AUC > 0.7).
"""),
        code('''
walk_ranking = interpretable_ranking(rows[cols], rows.delta_Edecomp.to_numpy(), rows.battery_id.to_numpy())
five, walk = interpretable_walk(walk_ranking, rows[cols], k=5)
print(five)
pd.DataFrame(walk).head(12)'''),
        md("""
On a sample this small the walk is noisy, so its picks can differ from the paper's frozen five
(`has_polyanion, delta_p_ox_score, nval_d_redox_initial, delta_li_frac, site_cn_mean`), which the rest of this
notebook uses.

## 4. Cross-validation

`cross_validate` runs the 5 grouped folds. Per fold it ranks, tunes (inner grouped CV over `min_samples_leaf` and
`max_features`) and fits on that fold's training electrodes only, then predicts the held-out fold. Predictions become
Edecomp of the (de)lithiated material: Ê = Edecomp(initial) + ΔÊ.
- **Regression:** MAE in meV/atom.
- **Classification:** macro F1 of the call Ê > 75 meV/atom.
- **Capacity:** MAE of the stable-range capacity of each electrode (mAh/g).
"""),
        code('''
cv5 = cross_validate(ds, "delta_edecomp", "interpretable5", ("all",))
display(cv5.summary()[["n_features", "mae_meV_mean", "mae_meV_std", "f1_macro_mean", "f1_macro_std"]].round(3))
cv5.capacity_summary(ds)[["cap_mae_mAhg_mean", "cap_mae_mAhg_std"]].round(1)'''),
        md("The full reference set with fold-internal selection of k features (a few minutes):"),
        code('''
QUICK = bool(os.environ.get("IIM_QUICK"))  # CI: two folds only
cvk = cross_validate(ds, "delta_edecomp", "reference", (3, 5, 10), folds=range(2) if QUICK else None)
cvk.summary()[["n_features", "mae_meV_mean", "mae_meV_std", "f1_macro_mean"]].round(3)'''),
        md("### Parity plots"),
        code('''
fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
parity_from_cv(cv5, "all", ax=axes[0], color="tab:blue"); axes[0].set_title("Interpretable five")
parity_from_cv(cvk, 5, ax=axes[1], color="tab:red"); axes[1].set_title("Permutation five (per fold)")
plt.tight_layout()'''),
        md("""
The dashed lines are the 75 meV/atom cutoff, so the four quadrants are the stable/unstable confusion matrix. Most
errors are large Edecomp values predicted too low: very unstable (de)lithiated phases are rare, and the forest
cannot extrapolate past what it saw.

## 5. Other targets

The same call fits the other targets. Formation energy (`"eform"`) is converted to Edecomp with the hull of
the other phases. Edecomp itself (`"edecomp"`) can be trained on every material or only on the (de)lithiated ones:
"""),
        code('''
cv_direct = cross_validate(ds, "edecomp", "interpretable5", ("all",), trained_on="final")
cv_direct.summary()[["mae_meV_mean", "f1_macro_mean"]].round(3)'''),
        md("## 6. Final model"),
        code('''
model = fit_final(ds, "delta_edecomp", "interpretable5")
print(model)
path = model.save("output/interpretable5_delta_Li")
print("saved", path)'''),
    ]


def cathode_notebook():
    return [
        md("""
# Cathode analysis: a cation-disordered garnet

Use the pretrained ΔEdecomp model on a structure that is not in the training data: the disordered garnet
Li$_7$Pr$_3$(Mn$_{0.25}$Mo$_{0.75}$)$_2$O$_{12}$ (Mn and Mo share the octahedral sites, Li partly occupies its
sites). We predict:
1. its oxidation states;
2. the stable Li range (capacity);
3. the voltage profile over that range;
4. the energy density.

Inputs besides the structure:
- **Edecomp of the pristine garnet:** 13.6 meV/atom. This is a MACE estimate on the MP2020 GGA/GGA+U scale; for a
  new material it would come from a DFT or machine-learned energy and `thermo.edecomp_of`.
- **Competing phases of the Li–Mn–Mo–O–Pr system** for the voltages: MP2020-corrected GGA/GGA+U Materials Project
  entries, in `data/hull_docs_Li-Mn-Mo-O-Pr.json`. Without them, `thermo.docs_from_mp` fetches them with an MP API key.
"""),
        code(SETUP),
        code('''
import json
from pymatgen.core import Structure
from ion_insertion_modeling import oxi_state_inference as oxi, insertion_stability as ins
from ion_insertion_modeling.model_building import load_pretrained

model = load_pretrained()  # ships with the package
garnet = Structure.from_file("data/disordered_Li7PrMoMnO.cif")
docs = json.loads(open("data/hull_docs_Li-Mn-Mo-O-Pr.json").read())
EDECOMP_INITIAL = 0.01364   # eV/atom (MACE estimate, MP2020 GGA/GGA+U scale)
FRAMEWORK = "Pr3Mo1.5Mn0.5O12"  # x counts Li per 12 O
print(model.features)'''),
        md("""
## 1. Oxidation states

ChemEnv runs once on the 208-site cell (about a minute) and is cached for every later call.
"""),
        code('''
g = oxi.guess_from_structure(garnet)
print(g.states, f"score {g.score:.3f}")
print(g.environments)'''),
        md("""
## 2. Stability over Li content

Every grid composition Li$_x$Pr$_3$Mo$_{1.5}$Mn$_{0.5}$O$_{12}$ is scored on the garnet's environments and
featurized; the model predicts ΔEdecomp from the pristine material.
"""),
        code('''
grid = np.arange(3, 8.01, 0.25)
curve = ins.stability_curve(garnet, EDECOMP_INITIAL, model, x_grid=grid, framework=FRAMEWORK)
cap = curve.capacity()
print(f"stable range x = {cap.x_lo:g} to {cap.x_hi:g} (initial x = {curve.x_initial:g}); capacity {cap.capacity_mAh_g:.1f} mAh/g")
curve.states[["x", "formula", "p_ox_score", "edecomp_pred", "stable", "in_range"]].assign(
    edecomp_pred=lambda d: (1000 * d.edecomp_pred).round(1)).rename(columns={"edecomp_pred": "Ê (meV/atom)"})'''),
        code('''
st = curve.states
fig, ax = plt.subplots(figsize=(5.5, 3.2))
ax.plot(st.x, 1000 * st.edecomp_pred, "o-", color="0.4")
ax.scatter(st.x[st.in_range], 1000 * st.edecomp_pred[st.in_range], color="tab:green", zorder=3, label="stable range")
ax.axhline(75, color="k", ls="--", lw=0.8, label="75 meV/atom")
ax.axvline(curve.x_initial, color="tab:blue", ls=":", label="pristine")
ax.set_xlabel("x in Li$_x$Pr$_3$Mo$_{1.5}$Mn$_{0.5}$O$_{12}$"); ax.set_ylabel("predicted Edecomp (meV/atom)"); ax.legend(fontsize=8)'''),
        md("""
## 3. Voltage profile

The predicted energy of each state is E(x) = E$_{hull}$(x) + Ê(x), with E$_{hull}$ from the competing phases.
G(x) = N(x)·E(x) − x·μ$_{Li}$ per framework formula unit. The voltage steps are V = −ΔG/Δx along the lower convex
envelope of G over the stable range.
"""),
        code('''
vp = ins.predict_voltage_profile(garnet, EDECOMP_INITIAL, model, docs=docs, x_grid=grid, framework=FRAMEWORK)
print(vp.steps.round(3))
fig, ax = plt.subplots(figsize=(5, 3.2))
steps = vp.steps.sort_values("x_from").reset_index(drop=True)
for r in steps.itertuples():
    ax.plot([r.x_from, r.x_to], [r.V, r.V], color="tab:red", lw=2.5)
for a, b in zip(steps.itertuples(), steps.iloc[1:].itertuples()):  # vertical jump between adjacent steps
    ax.plot([a.x_to, a.x_to], [a.V, b.V], color="tab:red", lw=1.5)
ax.set_ylim(bottom=0)
ax.set_xlabel("x in Li$_x$Pr$_3$Mo$_{1.5}$Mn$_{0.5}$O$_{12}$"); ax.set_ylabel("V vs Li/Li$^+$")
ax.set_title(f"Mean voltage {vp.mean_voltage:.2f} V")'''),
        md("## 4. Energy density"),
        code('''
e = ins.energy_density(vp)
print(f"{e.wh_per_kg:.0f} Wh/kg = {e.capacity_mAh_g:.1f} mAh/g × {e.mean_voltage:.2f} V (mass: framework + x_hi Li)")'''),
        md("""
## 5. Environment assumption and screening many structures

The same capacity with every cation assumed octahedral (`environments="O:6"`, no ChemEnv), the shortcut for
compositions without a reliable structure:
"""),
        code('''
cap_o6 = ins.predict_capacity(garnet, EDECOMP_INITIAL, model, x_grid=grid, framework=FRAMEWORK, environments="O:6")
print(f"ChemEnv environments: x = {cap.x_lo:g}-{cap.x_hi:g}, {cap.capacity_mAh_g:.1f} mAh/g")
print(f"all octahedral:       x = {cap_o6.x_lo:g}-{cap_o6.x_hi:g}, {cap_o6.capacity_mAh_g:.1f} mAh/g")'''),
        md("""
`predict_capacities` screens a list of structures (default grid: 41 contents from 0 to twice the pristine content);
a structure that fails is reported in the `error` column instead of stopping the run:
"""),
        code('''
lfp = Structure.from_file("../tests/data/LiFePO4_mp-19017.cif")
ins.predict_capacities([lfp, lfp.copy().scale_lattice(lfp.volume * 1.02)], [-0.0082, -0.0082], model,
                       ids=["LiFePO4", "LiFePO4 (+2% volume)"])'''),
        md("""
Caveats:
- The model was trained on ordered Materials Project electrodes. For a disordered structure, the environments are
  occupancy-weighted averages, and the predicted range is a screening estimate, not a DFT result.
- The voltages use predicted Edecomp values on a real hull, so their errors scale with the Edecomp MAE (about
  25 meV/atom per state).
"""),
    ]


NOTEBOOKS = {"00_quickstart.ipynb": quickstart, "01_oxi_state_inference.ipynb": oxi_notebook,
             "02_model_training.ipynb": training_notebook, "03_cathode_analysis.ipynb": cathode_notebook}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    for name, fn in NOTEBOOKS.items():
        if args.only and args.only not in name:
            continue
        nb = nbf.v4.new_notebook()
        nb.cells = fn()
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        path = NB_DIR / name
        if args.execute:
            from nbclient import NotebookClient

            NotebookClient(nb, timeout=3600, kernel_name="python3",
                           resources={"metadata": {"path": str(NB_DIR)}}).execute()
        nbf.write(nb, path)
        print("wrote", path)


if __name__ == "__main__":
    main()
