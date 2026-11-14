"""TrainedModel: a fitted model with everything needed to use it later (features, target, cutoff, versions).

Saved with skops (a safe, inspectable format: loading never runs arbitrary code) plus a JSON sidecar of metadata.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ion_insertion_modeling import config


def _versions() -> dict:
    import pymatgen.core
    import sklearn

    import ion_insertion_modeling

    return {"ion_insertion_modeling": ion_insertion_modeling.__version__, "scikit-learn": sklearn.__version__,
            "pymatgen": pymatgen.core.__version__, "numpy": np.__version__}


@dataclass
class TrainedModel:
    estimator: object = field(repr=False)
    features: list[str]
    target: str  # "delta_edecomp", "edecomp" or "eform"
    trained_on: str
    ion: str
    params: dict
    threshold: float = config.ABOVE_HULL_THRESHOLD
    n_train_rows: int = 0
    includes_test: bool = False
    data_fingerprint: str = "unknown"
    versions: dict = field(default_factory=_versions)

    def predict(self, X: pd.DataFrame | dict | list[dict]) -> np.ndarray:
        """Predictions of the target (eV/atom) for feature rows (a DataFrame with these columns, or dicts)."""
        if isinstance(X, dict):
            X = [X]
        X = pd.DataFrame(X)
        missing = [c for c in self.features if c not in X.columns]
        if missing:
            raise KeyError(f"missing features: {missing}")
        return self.estimator.predict(X[self.features])

    def metadata(self) -> dict:
        d = asdict(self)
        d.pop("estimator")
        return d

    def save(self, path: str | Path) -> Path:
        """Write <path>.skops and <path>.json."""
        import skops.io as sio

        path = Path(path).with_suffix("")
        path.parent.mkdir(parents=True, exist_ok=True)
        sio.dump(self.estimator, path.with_suffix(".skops"))
        path.with_suffix(".json").write_text(json.dumps(self.metadata(), indent=1))
        return path.with_suffix(".skops")

    @classmethod
    def load(cls, path: str | Path) -> "TrainedModel":
        import skops.io as sio

        path = Path(path).with_suffix("")
        meta = json.loads(path.with_suffix(".json").read_text())
        trusted = sio.get_untrusted_types(file=path.with_suffix(".skops"))
        allowed = [t for t in trusted if t.startswith(("sklearn.", "numpy."))]
        est = sio.load(path.with_suffix(".skops"), trusted=allowed)
        saved = meta.get("versions", {})
        now = _versions()
        if saved.get("scikit-learn") != now["scikit-learn"]:
            warnings.warn(f"model saved with scikit-learn {saved.get('scikit-learn')}, running {now['scikit-learn']}")
        return cls(estimator=est, **meta)


def load_pretrained(name: str = "interpretable5_delta_Li", root: str | Path | None = None) -> TrainedModel:
    """The paper's pretrained model (the interpretable-five ΔEdecomp random forest, Li), from the data release
    folder models/ (config.DATA_CACHE_DIR or `root`)."""
    base = Path(root) if root is not None else config.DATA_CACHE_DIR
    for cand in (base / "models" / name, base.parent / "models" / name):
        if cand.with_suffix(".skops").exists():
            return TrainedModel.load(cand)
    raise FileNotFoundError(f"no pretrained model {name!r} under {base}/models (download the release; see README)")
