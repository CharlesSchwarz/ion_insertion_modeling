"""Write the small golden files in tests/data/ from a local data release (run once; outputs are committed).

    .venv/bin/python tools/make_test_data.py --data release
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parents[1] / "tests" / "data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="release")
    args = ap.parse_args()
    from ion_insertion_modeling.featurizers import PaperFeaturizer
    from ion_insertion_modeling.model_building import load_dataset

    OUT.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("Li", root=args.data)

    # P1: stored guesses of 200 materials (composition + environments -> states, score)
    m = ds.materials.dropna(subset=["envs", "P1_oxi_state_guess"]).sample(200, random_state=0)
    p1 = [{"mpid": r.mpid, "comp": r.comp, "oxide_type": r.oxide_type, "envs": json.loads(r.envs),
           "fixed": json.loads(r.oxi_fixed) if isinstance(r.oxi_fixed, str) else {},
           "states": json.loads(r.P1_oxi_state_guess), "score": float(r.P1_score)} for r in m.itertuples()]
    (OUT / "p1_golden.json").write_text(json.dumps(p1))

    # Features: 6 transitions with their initial structures and every expected feature value
    t = ds.transitions
    pick = list(t.sample(5, random_state=3).transition_id) + list(t[t.battery_id == "mp-20361_Li"].transition_id[:1])
    rows = ds.transition_rows.set_index("row_id")
    mats = ds.materials.set_index(["battery_id", "mpid"])
    labels = PaperFeaturizer().feature_labels()
    feats = []
    for r in t[t.transition_id.isin(pick)].itertuples():
        s = ds.initial_structure(r.battery_id)
        feats.append({"transition_id": r.transition_id, "structure": s.as_dict(), "comp_initial": r.comp_initial,
                      "comp_final": r.comp_final, "framework_formula": r.framework_formula,
                      "oxi_initial": json.loads(r.oxi_initial), "oxi_final": json.loads(r.oxi_final),
                      "p1_score_initial": float(r.p1_score_initial), "p1_score_final": float(r.p1_score_final),
                      "oxo_bonded": json.loads(mats.loc[(r.battery_id, r.mpid_initial)].oxo_bonded),
                      "envs": json.loads(mats.loc[(r.battery_id, r.mpid_initial)].envs),
                      "edecomp_initial": float(r.Edecomp_initial), "edecomp_final": float(r.Edecomp_final),
                      "features": {c: (None if np.isnan(v) else float(v)) for c, v in
                                   ((c, float(rows.at[r.transition_id, c])) for c in labels)}})
    (OUT / "feature_golden.json").write_text(json.dumps(feats))

    # LiFePO4 (olivine) initial structure as a CIF for the quickstart
    ds.initial_structure("mp-20361_Li").to(filename=str(OUT / "LiFePO4_mp-19017.cif"))
    print("wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
