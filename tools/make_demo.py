"""Build the demo dataset and the packaged pretrained model from a local data release.

    .venv/bin/python tools/make_demo.py --data release [--n 300]

Demo: a random sample of the Li *training* electrodes (no held-out test electrode is included), with every table
load_dataset reads, restricted to those electrodes, written to src/ion_insertion_modeling/data/demo/Li/.
Model: the pretrained interpretable-five ΔEdecomp RF from <data>/models/, re-saved with zip compression to
src/ion_insertion_modeling/data/models/.
"""
import argparse
import gzip
import json
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
PKG = Path(__file__).resolve().parents[1] / "src" / "ion_insertion_modeling" / "data"
ALWAYS = ("mp-20361_Li",)  # olivine LiFePO4, used in the docs and tests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="release")
    ap.add_argument("--n", type=int, default=300, help="training electrodes in the demo")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    src = Path(args.data) / "Li"
    out = PKG / "demo" / "Li"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    split = pd.read_parquet(src / "battery_split.parquet")
    train = split[~split.is_test]
    rng = np.random.default_rng(args.seed)
    pick = set(rng.choice(sorted(train.battery_id), size=args.n, replace=False)) | set(ALWAYS)
    keep = lambda df, col="battery_id": df[df[col].isin(pick)].reset_index(drop=True)  # noqa: E731

    keep(split).to_parquet(out / "battery_split.parquet", index=False)
    for name in ("transitions", "transition_rows", "material_rows", "materials", "capacity_materials",
                 "initial_structures"):
        keep(pd.read_parquet(src / f"{name}.parquet")).to_parquet(out / f"{name}.parquet", index=False)
    u = pd.read_parquet(src / "unscored.parquet")
    keep(u[~u.is_test.astype(bool)]).to_parquet(out / "unscored.parquet", index=False)
    for name in ("batteries_dropped", "transitions_dropped"):
        if (src / f"{name}.parquet").exists():
            shutil.copy(src / f"{name}.parquet", out / f"{name}.parquet")
    for name in ("manifest.json", "cleaning_log.json"):
        shutil.copy(src / name, out / name)

    from ion_insertion_modeling import thermo
    t = pd.read_parquet(out / "transitions.parquet")
    systems = {thermo.chemsys(c) for c in t.drop_duplicates("battery_id").comp_initial}
    with gzip.open(src / "hull_docs.json.gz", "rt") as f:
        docs = json.load(f)
    with gzip.open(out / "hull_docs.json.gz", "wt") as f:
        json.dump({s: docs[s] for s in sorted(systems) if s in docs}, f)

    full = json.loads((src / "checksums.json").read_text())
    meta = {"ion": "Li", "demo": True, "outlier_batteries": full.get("outlier_batteries", []), "n_batteries": len(pick), "n_transitions": len(t),
            "note": "Random sample of training electrodes from the full release; contains no held-out test "
                    "electrode. For examples and tests only: metrics on it are not the paper's."}
    (out / "checksums.json").write_text(json.dumps(meta, indent=1))

    from ion_insertion_modeling.model_building import TrainedModel
    m = TrainedModel.load(Path(args.data) / "models" / "interpretable5_delta_Li")
    m.save(PKG / "models" / "interpretable5_delta_Li")
    sizes = {p.name: round(p.stat().st_size / 1e6, 2) for p in list(out.iterdir()) + list((PKG / "models").iterdir())}
    print(meta, sizes)


if __name__ == "__main__":
    main()
