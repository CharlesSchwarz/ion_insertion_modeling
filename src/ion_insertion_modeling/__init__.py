"""ion_insertion_modeling: predicting the stability of ion-insertion electrode materials on ion insertion and removal.

Modules
-------
oxi_state_inference   the P_ox oxidation-state guesser (composition + coordination environments)
featurizers           matminer featurizers for an ion insertion/removal transition
model_building        dataset, grouped splits, feature selection, random-forest training and CV
insertion_stability   stable range, capacity, voltage profile and energy density of a new material
thermo                MP2020 GGA/GGA+U hulls and Edecomp

Energies are eV/atom on the MP2020-corrected GGA/GGA+U hull; x is ions per framework formula unit.
"""
__version__ = "0.1.0"

from ion_insertion_modeling import config  # noqa: F401
