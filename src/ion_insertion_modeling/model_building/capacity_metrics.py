"""Stable-range capacity of each electrode in the dataset, from true and predicted Edecomp.

An electrode's materials (its initial material and every (de)lithiated material, ordered by x) form a stable range:
the unbroken run of materials with Edecomp <= cutoff that contains the initial material (whose true Edecomp is
always used). Capacity = x_hi − x_lo (ions per framework f.u.) and Δx·z·F / (3.6·M) mAh/g with M the molar mass of
framework + x_hi ions. This is the discrete capacity (the range ends at computed materials) reported in the paper.
"""
import numpy as np
import pandas as pd
from pymatgen.core import Composition, Element

from ion_insertion_modeling import config
from ion_insertion_modeling.insertion_stability import grav_capacity, stable_range
from ion_insertion_modeling.model_building.evaluate import summarise_folds


def _arrays(mats: pd.DataFrame) -> dict:
    out = {}
    for b, g in mats.groupby("battery_id", sort=False):
        g = g.sort_values("x")
        out[b] = {"x": g.x.to_numpy(float), "e": g.Edecomp.to_numpy(float), "tid": g.transition_id.to_numpy(),
                  "i0": int(np.flatnonzero(g.is_initial.to_numpy())[0]),
                  "mass": Composition(g.framework_formula.iloc[0]).weight}
    return out


def _cap(a: dict, e: np.ndarray, cutoff: float, ion_mass: float, z: int) -> dict:
    lo, hi = stable_range(e <= cutoff, a["i0"])
    x_lo, x_hi = float(a["x"][lo]), float(a["x"][hi])
    return {"x_lo": x_lo, "x_hi": x_hi, "cap_x": x_hi - x_lo,
            "cap_mAhg": grav_capacity(x_hi - x_lo, x_hi, a["mass"], ion_mass, z)}


def battery_capacities(preds: pd.DataFrame, mats: pd.DataFrame, cutoff: float = config.ABOVE_HULL_THRESHOLD,
                       ion: str = config.DEFAULT_ION) -> pd.DataFrame:
    """One row per (n_features, context, battery): true (*_true) and predicted (*_pred) range ends and capacities.
    preds: CVResult.predictions (every final of an evaluated battery must be there, including assumed-unstable ones).
    mats: Dataset.capacity_materials."""
    arrays = _arrays(mats[mats.battery_id.isin(set(preds.battery_id))])
    ion_mass, z = float(Element(ion).atomic_mass), config.ION_CHARGE[ion]
    out, true_cache = [], {}
    for (k, c), g in preds.groupby(["n_features", "context"], sort=False):
        pred = dict(zip(g.transition_id, g.Edecomp_final_pred.astype(float)))
        for b in g.battery_id.unique():
            a = arrays[b]
            finals = [i for i in range(len(a["x"])) if i != a["i0"]]
            missing = [a["tid"][i] for i in finals if a["tid"][i] not in pred]
            if missing:
                raise ValueError(f"battery {b} lacks predictions for {missing}")
            e_pred = a["e"].copy()
            e_pred[finals] = [pred[a["tid"][i]] for i in finals]
            if b not in true_cache:
                true_cache[b] = _cap(a, a["e"], cutoff, ion_mass, z)
            p = _cap(a, e_pred, cutoff, ion_mass, z)
            out.append({"n_features": k, "context": c, "battery_id": b, "n_materials": len(a["x"]),
                        **{f"{q}_true": v for q, v in true_cache[b].items()},
                        **{f"{q}_pred": v for q, v in p.items()}})
    return pd.DataFrame(out)


def capacity_summary(caps: pd.DataFrame) -> pd.DataFrame:
    """Capacity MAE (mAh/g and Δx), mean ± std over folds, per n_features."""
    out = []
    for k, g in caps.groupby("n_features", sort=False):
        per = [{"cap_mae_mAhg": float(np.abs(h.cap_mAhg_pred - h.cap_mAhg_true).mean()),
                "cap_mae_x": float(np.abs(h.cap_x_pred - h.cap_x_true).mean())} for _, h in g.groupby("context")]
        out.append({"n_features": k, "n_batteries": int(g.battery_id.nunique()),
                    **summarise_folds(per, ("cap_mae_mAhg", "cap_mae_x")),
                    "true_mean_mAhg": float(g.drop_duplicates("battery_id").cap_mAhg_true.mean())})
    return pd.DataFrame(out)
