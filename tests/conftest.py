import json
import os
import warnings
from pathlib import Path

import pytest

warnings.filterwarnings("ignore")
DATA = Path(__file__).resolve().parent / "data"
REPO = Path(__file__).resolve().parents[1]


def release_root() -> Path | None:
    for cand in (os.environ.get("ION_INSERTION_MODELING_DATA"), REPO / "release"):
        if cand and (Path(cand) / "Li" / "transitions.parquet").exists():
            return Path(cand)
    return None


@pytest.fixture(scope="session")
def li_dataset():
    root = release_root()
    if root is None:
        pytest.skip("data release not available (set ION_INSERTION_MODELING_DATA)")
    from ion_insertion_modeling.model_building import load_dataset

    return load_dataset("Li", root=root)


@pytest.fixture(scope="session")
def demo_dataset():
    """The demo dataset shipped with the package (always available)."""
    from ion_insertion_modeling.model_building import load_demo_dataset

    return load_demo_dataset()


@pytest.fixture(scope="session")
def p1_golden():
    return json.loads((DATA / "p1_golden.json").read_text())


@pytest.fixture(scope="session")
def feature_golden():
    return json.loads((DATA / "feature_golden.json").read_text())


@pytest.fixture(scope="session")
def lifepo4():
    from pymatgen.core import Structure

    return Structure.from_file(DATA / "LiFePO4_mp-19017.cif")
