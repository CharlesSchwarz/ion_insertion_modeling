"""Build the data release from the research repository (run with the research repo's venv).

    cd <research repo>
    VOLTAGE_ION=Li .venv/bin/python <package>/tools/build_release.py --out <package>/release
    (repeat with VOLTAGE_ION=Na, Mg, K)

Writes <out>/<ion>/ with the tables ion_insertion_modeling.model_building.data.load_dataset reads, plus a checksum
manifest. Everything is taken from the research pipeline's own loaders, so the released rows, features, energies
and folds are exactly the ones behind the paper. Structures are stored as JSON dicts (never pickled).
"""
import argparse
import gzip
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path



def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# The research pipeline calls the oxidation-state score "P1"; the package calls it P_ox.
COLUMN_RENAMES = {"P1_oxi_state_guess": "P_ox_guess", "P1_score": "P_ox_score",
                  "p1_score_initial": "p_ox_score_initial", "p1_score_final": "p_ox_score_final",
                  "delta_p1_score": "delta_p_ox_score"}
VALUE_RENAMES = {"missing_p1": "missing_p_ox"}
GROUP_RENAMES = {"p1": "p_ox"}


def apply_renames(out: Path) -> None:
    """Rename the research pipeline's P1 columns, values and feature group in every table of `out`, and refresh
    the file entries of checksums.json if it exists."""
    import pandas as pd

    for p in sorted(out.glob("*.parquet")):
        d = pd.read_parquet(p)
        new = d.rename(columns=COLUMN_RENAMES)
        for c in new.columns:
            if new[c].dtype == object and new[c].isin(list(VALUE_RENAMES)).any():
                new[c] = new[c].replace(VALUE_RENAMES)
        if not new.equals(d) or list(new.columns) != list(d.columns):
            new.to_parquet(p, index=False)
    man_p = out / "manifest.json"
    if man_p.exists():
        man = json.loads(man_p.read_text())
        man = {COLUMN_RENAMES.get(c, c): {**m, "group": GROUP_RENAMES.get(m["group"], m["group"]),
                                          "units": str(m.get("units", "")).replace("P1 score", "P_ox score")}
               for c, m in man.items()}
        man_p.write_text(json.dumps(man, indent=1))
    log_p = out / "cleaning_log.json"
    if log_p.exists():
        t = log_p.read_text()
        for a, b in {**VALUE_RENAMES, "p1_scoring": "p_ox_scoring", "n_rows_p1_": "n_rows_p_ox_", "no_p1_guess": "no_p_ox_guess"}.items():
            t = t.replace(a, b)
        log_p.write_text(t)
    ck = out / "checksums.json"
    if ck.exists():
        meta = json.loads(ck.read_text())
        if "files" in meta:
            meta["files"] = {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size}
                             for p in sorted(out.iterdir()) if p.name != "checksums.json"}
            ck.write_text(json.dumps(meta, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--rename-only", action="store_true", help="only apply the P1 -> P_ox renames to <out>/<ion>/")
    args = ap.parse_args()
    if args.rename_only:
        for d in sorted(p for p in args.out.iterdir() if p.is_dir() and p.name != "models"):
            apply_renames(d)
            print("renamed", d)
        return
    sys.path.insert(0, str(Path.cwd() / "src"))
    from voltage import benchmark, capacity, config, transitions  # noqa: E402
    from voltage.io import read_parquet  # noqa: E402

    ion = config.WORKING_ION
    out = args.out / ion
    out.mkdir(parents=True, exist_ok=True)
    drop = transitions.drop_outlier_batteries

    # Row tables (targets + energies + features) exactly as the benchmark sees them
    tr = benchmark.load_inputs("delta")
    mat = benchmark.load_inputs("eform")
    tr["data"].to_parquet(out / "transition_rows.parquet", index=False)
    mat["data"].to_parquet(out / "material_rows.parquet", index=False)
    tr["transitions"].to_parquet(out / "transitions.parquet", index=False)
    tr["unscored"].to_parquet(out / "unscored.parquet", index=False)
    tr["bsplit"].reset_index().to_parquet(out / "battery_split.parquet", index=False)
    man = {c: m for c, m in tr["manifest"].items() if m["group"] not in ("oxi_pmg", "cgcnn")}
    (out / "manifest.json").write_text(json.dumps(man, indent=1))

    # Materials (P_ox states, environments) and initial structures
    m = drop(read_parquet(config.INTERIM_DIR / "materials.parquet"))
    stab = read_parquet(config.EDECOMP_INTERIM_DIR / "stability.parquet")[["battery_id", "mpid", "Edecomp", "Edecomp_mp"]]
    keep = ["battery_id", "mpid", "comp", "framework_formula", "oxide_type", "envs", "oxo_bonded", "oxi_fixed",
            "P1_oxi_state_guess", "P1_score", "PMG_oxi_state_guess", "MP_oxi_state_guess", "MP_oxi_method"]
    m = m[[c for c in keep if c in m.columns]].merge(stab, on=["battery_id", "mpid"], how="left", validate="one_to_one")
    m = m[m.battery_id.isin(set(tr["bsplit"].index) | set(tr["unscored"].battery_id))]
    m.to_parquet(out / "materials.parquet", index=False)
    s = drop(read_parquet(config.INTERIM_DIR / "initial_structures.parquet"))
    s = s[s.battery_id.isin(set(m.battery_id))]
    s.to_parquet(out / "initial_structures.parquet", index=False)

    # Cleaning record: electrodes and materials removed before the dataset was fixed, with reasons, and the logs
    for name in ("batteries_dropped", "transitions_dropped"):
        p = config.EDECOMP_INTERIM_DIR / f"{name}.parquet"
        if p.exists():
            d = read_parquet(p)
            d[[c for c in d.columns if "structure" not in c]].to_parquet(out / f"{name}.parquet", index=False)
    logs = {}
    for name, p in (("ingest", config.INTERIM_DIR / "ingest_log.json"),
                    ("transitions", config.EDECOMP_INTERIM_DIR / "transitions_log.json")):
        if p.exists():
            logs[name] = json.loads(p.read_text())
    (out / "cleaning_log.json").write_text(json.dumps(logs, indent=1, default=str))

    # Capacity inputs: battery materials and the DFT hull-convexity grid (interpolated capacities)
    cm = capacity.battery_materials()
    cm.to_parquet(out / "capacity_materials.parquet", index=False)
    if config.CAPACITY_HULL_GRID.exists():
        shutil.copy(config.CAPACITY_HULL_GRID, out / "capacity_hull_grid.parquet")

    # Hull: MP2020-corrected GGA/GGA+U energies per chemical system (the rebuilt hull every Edecomp comes from)
    if config.MIXED_ENERGIES.exists():
        with open(config.MIXED_ENERGIES, "rb") as f, gzip.open(out / "hull_docs.json.gz", "wb") as g:
            shutil.copyfileobj(f, g)

    apply_renames(out)
    files = sorted(p for p in out.iterdir() if p.name != "checksums.json")
    meta = {"ion": ion, "files": {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size} for p in files},
            "n_batteries": int(len(tr["bsplit"])), "n_test_batteries": int(tr["bsplit"].is_test.sum()),
            "n_transitions": int(len(tr["transitions"])), "outlier_batteries": list(config.OUTLIER_BATTERIES),
            "threshold_eV_per_atom": config.ABOVE_HULL_THRESHOLD, "seed": config.SEED,
            "source_commit": os.popen("git rev-parse HEAD").read().strip()}
    (out / "checksums.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k != "files"}, indent=1))
    print({k: v["bytes"] for k, v in meta["files"].items()})


if __name__ == "__main__":
    main()
