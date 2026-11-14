# Changelog

## 0.1.0 (unreleased)
- First version, ported from the research code behind Draft 7 (rerun 2, 2026-11-11).
- `oxi_state_inference`: P1 guesser (`P_ox_env_sites_split`), structure / framework / environment entry points,
  ion-content scan.
- `featurizers`: matminer featurizers over a `Transition`; the paper's feature sets.
- `model_building`: release loader, grouped splits, permutation ranking, interpretable walk, RF CV for ΔEdecomp,
  Edecomp and Eform, capacity metrics, parity plots, skops model bundles.
- `insertion_stability`: stability curve, capacity, voltage profile, energy density.
- Data release builder (`tools/build_release.py`) for Li, Na, Mg, K.
