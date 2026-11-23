# TL-GR-MCAE: code and benchmark for the CMC manuscript

Python >= 3.10; packages: numpy, scipy, scikit-learn, pywavelets, torch, h5py, matplotlib.

| File | Purpose |
|---|---|
| `dassim.py` | SV-DASBench simulator (Svelvik geometry: 2352 ch, dt = 0.2 ms, 1 s window) |
| `pipeline.py` | CMNR, DWT, onset alignment, window features (vectorised, exact), MC-Dropout AE, graph diffusion, metrics |
| `gen_cache.py` | Simulate all dates/scenarios and cache window features: `python gen_cache.py main 0 1 ... 9 100 101 102`, `python gen_cache.py sweep 0 1 2 3 4` |
| `gen_s5.py` | Regenerates scenario S5 (trench perturbation, x = 1060 m) for the given seeds (run after `gen_cache.py main`) |
| `analyze2.py` | `select` (alpha/sigma_g on validation seeds 100-102), `main`, `ablate`, `strength` |
| `tables.py`, `fig_*.py` | Summaries and manuscript figures |
| `run_field_tlgrmcae.py` | Applies TL-GR-MCAE to the Svelvik HDF5 field records (coarse onset detection + fine alignment) |
| `results/` | JSON outputs used in the manuscript (per-seed raw results, aggregates, Wilcoxon tests, ablations, grid) |

Order to reproduce: gen_cache (main, sweep) -> gen_s5 -> analyze2 select -> main / ablate / strength -> fig_results.py maps loss strength sens.
Runtime on a 2-core CPU: ~3.5 min per seed for the cache; analysis ~25 min.

Field re-analysis (required before claims about CO2):
`python run_field_tlgrmcae.py <DASGAS/.../Svelvik_standard> <out_dir> --alpha 0.1 --sigma 15`
(add `--no-dwt` to test the DWT finding of the ablation study).

## Real field data (Svelvik, POS_00)
Download only the POS_00 folder from `202507_GeoInquire_hammer_POS_20.zip` (Zenodo 21193610) with `remotezip`
(16 HDF5 files, ~2.2 GB). Then:
1. `python strikes.py` - detects every hammer strike in each 10 s record (results_field/strikes.json).
2. `python field2.py` - strike-level windows; original configuration vs TL-GR-MCAE with label-free metrics
   (held-out baseline FAR, flag rate, split-half Dice, Moran's I) -> results_field/field2_results.json.
Paths inside the scripts assume the extracted data sit in ./raw/.
