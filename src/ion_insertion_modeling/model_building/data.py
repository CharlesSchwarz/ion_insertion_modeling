"""The paper's dataset (Materials Project InsertionElectrode electrodes with O-containing frameworks).

load_dataset(ion) reads the data release (one folder per working ion) from `root`, or from the cache directory
(config.DATA_CACHE_DIR, overridable with the ION_INSERTION_MODELING_DATA environment variable). Tables:

    transitions          one row per (initial material -> (de)lithiated material) of an electrode: ids, compositions,
                         x, Edecomp_initial / Edecomp_final / delta_Edecomp (eV/atom, rebuilt MP2020 GGA/GGA+U hull),
                         ef_ref_final (formation energy of the hull without the final material)
    transition_rows      transitions + every feature (row_id = transition_id): training rows for ΔEdecomp
    material_rows        every material of every electrode (initial included, is_identity) + features + ef_final,
                         Edecomp_final: training rows for Eform and Edecomp
    unscored             (de)lithiated materials P1 could not assign: scored as predicted unstable
    battery_split        battery_id -> is_test, cv_fold (fixed before any target was looked at)
    materials            per material: P1 states and score, environments, Edecomp, MP's Edecomp, pymatgen/MP states
    initial_structures   battery_id, initial_mpid, structure_json (pymatgen Structure.as_dict JSON)
    capacity_materials   per electrode material: x, atoms per f.u., Edecomp, for the stable-range capacity
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import pandas as pd

from ion_insertion_modeling import config

FILES = ("transitions", "transition_rows", "material_rows", "unscored", "battery_split", "materials",
         "capacity_materials")
ZENODO_RECORD = None  # set at release: the Zenodo record id of the data release


@dataclass
class Dataset:
    ion: str
    root: Path
    transitions: pd.DataFrame = field(repr=False)
    transition_rows: pd.DataFrame = field(repr=False)
    material_rows: pd.DataFrame = field(repr=False)
    unscored: pd.DataFrame = field(repr=False)
    battery_split: pd.DataFrame = field(repr=False)
    materials: pd.DataFrame = field(repr=False)
    capacity_materials: pd.DataFrame = field(repr=False)
    manifest: dict = field(repr=False)

    def rows(self, kind: str = "transition") -> pd.DataFrame:
        """'transition' or 'material' training rows (features + targets)."""
        return {"transition": self.transition_rows, "material": self.material_rows}[kind]

    @property
    def train_batteries(self) -> set[str]:
        return set(self.battery_split.battery_id[~self.battery_split.is_test])

    @property
    def test_batteries(self) -> set[str]:
        return set(self.battery_split.battery_id[self.battery_split.is_test])

    def train(self, kind: str = "transition") -> pd.DataFrame:
        """Training rows only (the 90% used for cross-validation)."""
        r = self.rows(kind)
        return r[r.battery_id.isin(self.train_batteries)].reset_index(drop=True)

    def test(self, kind: str = "transition", confirm: bool = False) -> pd.DataFrame:
        """Held-out test rows. The paper's test evaluation is final; pass confirm=True to read them."""
        if not confirm:
            raise PermissionError("these are the held-out test rows; pass confirm=True if you mean to use them")
        r = self.rows(kind)
        return r[r.battery_id.isin(self.test_batteries)].reset_index(drop=True)

    def context_unscored(self, context) -> pd.DataFrame:
        """Unscored rows evaluated in a context (fold k's electrodes, or the test electrodes for "test")."""
        u = self.unscored
        if context == "test":
            return u[u.is_test.astype(bool)]
        return u[~u.is_test.astype(bool) & (u.cv_fold.fillna(-1).astype(int) == context)]

    @cached_property
    def initial_structures(self) -> pd.DataFrame:
        return pd.read_parquet(self.root / "initial_structures.parquet")

    def initial_structure(self, battery_id: str):
        """pymatgen Structure of an electrode's initial material."""
        from pymatgen.core import Structure

        s = self.initial_structures.set_index("battery_id").structure_json[battery_id]
        return Structure.from_dict(json.loads(s))

    def hull_docs(self, chemsys: str) -> list[dict]:
        """[{mpid, comp, energy_per_atom}] of a chemical system ('Fe-Li-O-P'): the rebuilt hull's entries."""
        return self._hull()[chemsys]

    @cached_property
    def _hull_cache(self) -> dict:
        import gzip

        with gzip.open(self.root / "hull_docs.json.gz", "rt") as f:
            return json.load(f)

    def _hull(self) -> dict:
        return self._hull_cache

    def fingerprint(self) -> str:
        """Short hash of the release checksums (stored in trained models)."""
        p = self.root / "checksums.json"
        return hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() else "unknown"

    def summary(self) -> dict:
        b = self.battery_split
        return {"ion": self.ion, "electrodes": len(b), "training_electrodes": int((~b.is_test).sum()),
                "test_electrodes": int(b.is_test.sum()), "transitions": len(self.transitions),
                "unscored_finals": len(self.unscored), "features": len(self.manifest)}


def data_dir(ion: str = config.DEFAULT_ION, root: str | Path | None = None) -> Path:
    base = Path(root) if root is not None else config.DATA_CACHE_DIR
    return base / ion if (base / ion).is_dir() else base


def load_dataset(ion: str = config.DEFAULT_ION, root: str | Path | None = None, verify: bool = True) -> Dataset:
    """The released dataset of one working ion ("Li", "Na", "Mg", "K")."""
    if ion not in config.SUPPORTED_IONS:
        raise ValueError(f"ion must be one of {config.SUPPORTED_IONS}")
    d = data_dir(ion, root)
    if not (d / "transitions.parquet").exists():
        raise FileNotFoundError(
            f"no data release for {ion} under {d}. Download the release (see README, 'Data') and unpack it there, "
            "or pass root= / set ION_INSERTION_MODELING_DATA to the folder holding Li/, Na/, Mg/, K/.")
    if verify:
        verify_release(d)
    tables = {name: pd.read_parquet(d / f"{name}.parquet") for name in FILES}
    manifest = json.loads((d / "manifest.json").read_text())
    return Dataset(ion=ion, root=d, manifest=manifest, **tables)


def verify_release(d: Path) -> None:
    """Check file sizes against checksums.json (full SHA-256 check: verify_release_sha)."""
    p = d / "checksums.json"
    if not p.exists():
        return
    meta = json.loads(p.read_text())
    for name, info in meta["files"].items():
        f = d / name
        if f.exists() and f.stat().st_size != info["bytes"]:
            raise OSError(f"{f} does not match the release (size {f.stat().st_size}, expected {info['bytes']})")


def verify_release_sha(d: Path) -> list[str]:
    """Names of files whose SHA-256 differs from checksums.json (empty if all match)."""
    meta = json.loads((d / "checksums.json").read_text())
    bad = []
    for name, info in meta["files"].items():
        h = hashlib.sha256((d / name).read_bytes()).hexdigest()
        if h != info["sha256"]:
            bad.append(name)
    return bad
