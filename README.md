# ion_insertion_modeling

Companion code for "Interpretable, Data-driven Approaches to Modeling Stability of Insertion Electrodes" (in
preparation).

**The Problem:** Given some oxygen-based ionic material, how much lithium can we add to/remove from the material's structure with it maintaining good stability? How can we leverage machine learning to predict this stability range?

This repo contains code to:
- Predict changes in element oxidation states upon lithiation/delithiation.
- Calculate physical features of a material's crystal structure, using topographical analysis and CrystalNN.
- Design and train scikit-learn models to predict changes in a materials stability upon lithiation/delithiation, drawing on data from the Materials Project Insertion Electrode Database.
- Apply models to predict quantities like capacity, voltage, and energy density for any oxide cathode material.

```python
from pymatgen.core import Structure
from ion_insertion_modeling.model_building import load_pretrained
from ion_insertion_modeling.insertion_stability import predict_capacity

model = load_pretrained()                                   # 5-feature random forest model, predicting ΔE_decomp
structure = Structure.from_file("tests/data/LiFePO4_mp-19017.cif")
cap = predict_capacity(structure, edecomp_initial=-0.0082, model=model)
cap.x_lo, cap.x_hi, cap.capacity_mAh_g                      # (0.0, 1.0, 169.9)
```

## Install

```bash
git clone https://github.com/CharlesSchwarz/ion_insertion_modeling
cd ion_insertion_modeling
python -m venv .venv && source .venv/bin/activate
pip install -e ".[plots,shap]"        # add mp for Materials Project access, dev for tests and notebooks
```

Python ≥ 3.10. Dependencies are pinned to the versions behind the paper. pymatgen's ChemEnv decides the coordination
environments, and so the oxidation states and features, so a different pymatgen version can change results slightly.

## Data

Two things ship with the package, so the quickstart and notebooks run right after install:
- **The pretrained model** (`load_pretrained()`): the 5-feature random forest predicting ΔE_decomp, trained on the
  2,011 training electrodes of the Li dataset.
- **A demo dataset** (`load_demo_dataset()`): 300 randomly chosen training electrodes of the Li dataset (plus olivine
  LiFePO4), with features, targets, cross-validation folds, structures and competing-phase energies. It has no
  held-out test electrodes. It is meant for trying the model-building tools; metrics on it are noisier than on the
  full dataset.

The full dataset (Li, Na, Mg, K) is not yet available. Once it is, `load_dataset("Li")` reads it from the folder in
`ION_INSERTION_MODELING_DATA`.

## What is in the package

| Module | What it does | Contents |
|---|---|---|
| `oxi_state_inference` | Custom oxidation-state guessing algorithm. Given an oxide crystal structure, or just a list of coordination environments, predict | guess_from_environments(composition, environments), guess_from_framework(composition, structure), guess_from_structure(structure), p_ox(el, ox), p_ox_given_env(el, ox, env), best_ion_content(structure)
| `featurizers` |  Calculators for a range of feature, based on the matminer `BaseFeaturizer` framework | `FrameworkMagpie`, `PolyanionFlag`, `CompositionMagpie`, `IonRedox`, `OxidationStateProperties`, `FrameworkStructure`, `InsertionSites`, `HullEnergies`, `PaperFeaturizer`; the paper's feature lists in `FEATURE_SETS` |
| `model_building` | gather data, select features, and train regression models for stability| `load_dataset`, `load_demo_dataset`, `grouped_split`, `rank_features`, `interpretable_walk`, `cross_validate` (targets `delta_edecomp`, `edecomp`, `eform`), `fit_final`, `TrainedModel`, parity plots |
| `insertion_stability` | predict higher-level properties, like capacity and energy density| `stability_curve`, `predict_capacity`, `predict_capacities`, `predict_voltage_profile`, `predict_energy_density` |
| `thermo` | thermodynamics helpers| MP2020 GGA/GGA+U hulls, `edecomp_of`, `docs_from_mp` |

Conventions:
- Energies are eV/atom on the MP2020-corrected GGA/GGA+U hull.
- x is ions per framework formula unit (the composition without the working ion).
- `edecomp_initial` is the **signed** decomposition energy: negative for a material on the hull. Do not pass
  e_above_hull; with 0 instead of the negative value every prediction is biased upward. `thermo.edecomp_of`
  computes it.
- Features use only the initial structure and the two compositions, never the (de)lithiated material's structure or
  energy.
- Splits are grouped by electrode. The held-out test set needs `confirm=True` to be read or scored.

## Notebooks

| Notebook | |
|---|---|
| `notebooks/00_quickstart.ipynb` | capacity of LiFePO4 from its structure, in a few lines |
| `notebooks/01_oxi_state_inference.ipynb` | how the oxidation state guessing algorithm works: the ICSD statistics, scoring, mixed valence, structures, a disordered garnet, scanning Li content |
| `notebooks/02_model_training.ipynb` | data and cleaning, the grouped split, feature selection, cross-validation, parity plots, the final model |
| `notebooks/03_cathode_analysis.ipynb` | a cation-disordered garnet: oxidation states, stable range, voltage profile, energy density |

## Citation

See `CITATION.cff`. Please also cite the Materials Project (Jain et al., APL Materials 2013) for the data, and
Waroquiers et al. (Chem. Mater. 2017; Acta Cryst. B 2020) for the oxidation-state statistics and ChemEnv.

## License

MIT (code). The demo dataset and the data release are CC BY 4.0 (derived from the Materials Project).
