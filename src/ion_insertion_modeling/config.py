"""Defaults for every threshold and switch in the package (the values behind the paper).

Functions take these as keyword defaults, so a value can be changed per call; nothing here is read from the
environment except the data-cache directory.
"""
import os
from pathlib import Path

PACKAGE_DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_CACHE_DIR = Path(os.environ.get("ION_INSERTION_MODELING_DATA", Path.home() / ".cache" / "ion_insertion_modeling"))

SEED = 42

# --- Working ions -------------------------------------------------------------------------------------------------
ION_CHARGE = {"Li": 1, "Na": 1, "Mg": 2, "K": 1}  # z: electrons per inserted ion
ION_RADIUS = {"Li": 0.76, "Na": 1.02, "Mg": 0.72, "K": 1.38}  # Å, Shannon 6-coordinate; insertion-site clearance
SUPPORTED_IONS = tuple(ION_CHARGE)
DEFAULT_ION = "Li"

# --- Coordination environments (ChemEnv) ---------------------------------------------------------------------------
ENV_DISTANCE_CUTOFFS = (1.4, 1.0)  # second value is the fallback when any environment comes back UNKNOWN
ENV_ANGLE_CUTOFF = 0.3
ENV_INCLUDE_O = True  # keep O environments (P_ox scoring skips them: O is fixed at -2 and scores 1)

# --- Oxidation states (P_ox guesser) ----------------------------------------------------------------------------------
SPECIES_DATA = PACKAGE_DATA_DIR / "species_data.json"  # ICSD statistics: P(ox|el), P(env|ox), P(env|el)
OXI_STATE_OVERRIDES = {"H": [1], "Li": [1], "F": [-1], "Na": [1], "Mg": [2], "Al": [3], "K": [1], "Ca": [2], "Zn": [2]}
OXI_GUESS_MAX_ATOMS = 40  # pymatgen enumerates all states below this cell size, common states above
OXI_INTEGER_MAX_DENOMINATOR = 100  # disordered formulas are scaled to integers with at most this factor
OXI_FRACTIONAL_FALLBACK = True  # no charge-balanced integer guess -> one fractional guess per free element
OXI_OXO_FIXED_STATES = {"P": 5}  # fixed when every site of the element bonds only to O
OXO_SHELL_TOL = 0.25  # first shell = neighbours within (1 + tol) x nearest-neighbour distance
OXO_SEARCH_RADIUS = 3.5  # Å
OXI_DATA_SOURCE = "Waroquiers"  # P(ox|el) table: "Waroquiers", "Davies" or "PMG"
P_OX_SCORING = "P_ox_env_sites_split"  # the paper's P_ox score

# --- Polyanions and features ----------------------------------------------------------------------------------------
POLYANION_ELEMENTS = ("P", "S", "Si", "B", "C", "N")
OXO_EXTRA_ELEMENTS = ("As", "Se")  # count only with an O-only shell of at most OXO_MAX_CN atoms
OXO_MAX_CN = {"As": 4, "Se": 4}
OXO_ELEMENTS = POLYANION_ELEMENTS + OXO_EXTRA_ELEMENTS
MAGPIE_STATS = ("mean", "minimum", "maximum")
MAGPIE_EXCLUDE = ("SpaceGroupNumber",)
EA_EXCLUDED_ELEMENTS = POLYANION_ELEMENTS + ("H", "Li", "F", "Na", "Mg", "Al", "K", "Ca", "Zn")
VPV_STATS = PACKAGE_DATA_DIR / "voronoi_volume_statistics_summary.csv"
OXI_CHANGE_TOL = 1e-3  # |Δ oxidation state| below this means the element does not change
OXI_INTEGER_EPS = 1e-6
ELECTRON_BALANCE_TOL = 0.05  # e⁻ per framework f.u.
SUSPECT_REDOX_ELEMENTS = ("H", "B", "C", "N", "Si", "P", "S", "Cl", "F", "Al", "Sc", "Y", "La")
FRAMEWORK_TOL = 1e-4  # max relative disagreement in framework formula units between elements of one composition
# Column-name patterns no feature may contain (targets, energies, non-P_ox oxidation states)
FORBIDDEN_FEATURE_PATTERNS = ("voltage", "vbar", "edecomp", "energy_above", "ehull", "above_hull", "pmg_oxi", "mp_oxi")

# --- Targets, splits, models ----------------------------------------------------------------------------------------
ABOVE_HULL_THRESHOLD = 0.075  # eV/atom: Ê_final above this is "unstable" (classification and capacity cutoff)
TEST_FRACTION = 0.1
N_FOLDS = 5
INNER_FOLDS = 3
N_FEATURES_GRID = (1, 3, 6, 10, 20, 30)
RF_N_ESTIMATORS = 300
RF_GRID = {"min_samples_leaf": [1, 3, 5], "max_features": [0.33, 0.66, 1.0]}  # tuned by inner grouped CV
SELECT_PRESELECT = 40  # candidates kept by RF impurity importance before permutation ranking
SELECT_HOLDOUT = 0.2  # fraction of training batteries held out for permutation importance
INTERPRETABLE_FEATURES = ("has_polyanion", "delta_p_ox_score", "nval_d_redox_initial", "delta_li_frac", "site_cn_mean")

# --- Capacity, voltage ----------------------------------------------------------------------------------------------
CAPACITY_CONTIGUOUS = True  # the stable range is the unbroken run containing the initial material
X_STEP = 0.05  # ions per framework formula unit between grid compositions
FARADAY = 96485.33212  # C/mol
