# TL-GR-MCAE: Time-Lapse-Referenced Graph-Regularized Monte Carlo Dropout Autoencoder for DAS Anomaly Detection

Code for the manuscript:

> Beg F, Aziz IA, Karim SN, Usmani UA. *Time-Lapse-Referenced Graph-Regularized Monte Carlo Dropout Autoencoder for Unsupervised Anomaly Detection in Distributed Acoustic Sensing Data for CO₂ Storage Monitoring.* Submitted to *Computers, Materials & Continua*, 2026.

## Overview

Distributed acoustic sensing (DAS) records from CO₂ storage sites rarely come with labels showing where the subsurface has changed. TL-GR-MCAE is an unsupervised framework that:

1. conditions each hammer shot with common-mode noise removal (CMNR) and wavelet (DWT) denoising, and aligns its onset to the baseline;
2. compares every window with the same window in the pre-injection baseline (time-lapse residuals), then applies robust re-centring and signed-log compression;
3. scores the residuals with a Monte Carlo (MC) Dropout autoencoder trained only on residuals between two baseline half-stacks (the null model);
4. smooths the scores with graph Laplacian diffusion and flags windows above the 95th percentile of the equally processed null scores.

## Data

**Simulated data (SV-DASBench).** All detection metrics (AUPRC, F1, false-alarm rate) come from simulated data. `dassim.py` generates DAS records on the Svelvik acquisition grid (2352 channels, 0.2 ms sampling, 1 s windows, 3237 windows per date). It includes realistic nuisances (common-mode noise, coupling variation, hum, spikes, dead channels, trigger shifts) and localized perturbations at known positions. There are 5 scenarios: 2 no-change and 3 perturbation. Seeds 0–9 are used for testing and seeds 100–102 for hyperparameter selection.

**Real field data (Svelvik CO₂ Field Lab, DASGAS project).** The field records are openly available on Zenodo: https://zenodo.org/records/21193610. We use hammer position `POS_00`, which sits inside `202507_GeoInquire_hammer_POS_20.zip` (16 HDF5 files, about 2.2 GB): 24 July 2025 (baseline), 26 July (injection) and 28 July (post-injection). The record has no labels of where CO₂ is in the DAS data, so the field data are evaluated only with label-free metrics:

- empirical false-alarm rate on held-out baseline partitions;
- flag rate per monitoring date;
- split-half reproducibility (Dice overlap between flag maps from disjoint strike subsets);
- spatial coherence (Moran's I).

The raw field data are not included in this repository because of their size.

## Installation

```bash
python -m pip install numpy scipy scikit-learn pywavelets torch h5py matplotlib remotezip
```

Tested with Python 3.13, PyTorch 2.15 (CPU), PyWavelets 1.8, scikit-learn 1.9 and SciPy 1.18. No GPU is required.

## Reproducing the results

### Simulated data

```bash
# 1. Simulate all scenarios and cache window features (~3.5 min per seed on 2 CPU cores)
python gen_cache.py main 0 1 2 3 4 5 6 7 8 9 100 101 102
python gen_cache.py sweep 0 1 2 3 4
python gen_s5.py 0 1 2 3 4 5 6 7 8 9 100 101 102

# 2. Select alpha / sigma_g on the validation seeds (100-102)
python analyze2.py select

# 3. Main comparison, ablation, strength sweep
python analyze2.py main
python analyze2.py ablate
python analyze2.py strength

# 4. Summaries and figures
python tables.py
python fig_framework.py
python fig_bench.py
python fig_results.py maps
python fig_combo.py
```

### Real field data

```python
# Download only the POS_00 folder (~2.2 GB) from the Zenodo zip
from remotezip import RemoteZip
url = "https://zenodo.org/api/records/21193610/files/202507_GeoInquire_hammer_POS_20.zip/content"
with RemoteZip(url) as z:
    for i in z.infolist():
        if i.filename.startswith("202507_GeoInquire_hammer_POS_00/") and i.filename.endswith(".hdf5"):
            z.extract(i.filename, "raw")
```

```bash
python strikes.py   # detect every hammer strike in each 10 s record
python field2.py    # original configuration vs TL-GR-MCAE, label-free metrics
```

| Paper item | Script | Output |
|---|---|---|
| Table 2 (main comparison) | `analyze2.py main` | `results/main_agg.json`, `results/wilcoxon.json` |
| Table 3 (ablation) | `analyze2.py ablate` | `results/ablations.json` |
| Fig. 4 (strength, sensitivity) | `analyze2.py strength`, `fig_combo.py` | `results/strength.json`, `results/grid.json` |
| Table 4, Fig. 5 (field) | `strikes.py`, `field2.py` | `results_field/*.json` |

## Repository structure

| File | Purpose |
|---|---|
| `dassim.py` | SV-DASBench simulator (Svelvik geometry) |
| `pipeline.py` | CMNR, DWT, onset alignment, window features, MC Dropout AE, graph diffusion, metrics |
| `gen_cache.py`, `gen_s5.py` | Simulate scenarios and cache features |
| `analyze2.py` | Hyperparameter selection, main comparison, ablation, strength sweep |
| `tables.py`, `fig_*.py` | Result summaries and manuscript figures |
| `strikes.py` | Hammer-strike detection in real field records |
| `field2.py` | Real-data analysis with label-free metrics |
| `run_field_tlgrmcae.py` | Stand-alone TL-GR-MCAE runner for DASGAS HDF5 data |
| `results/`, `results_field/` | JSON outputs used in the manuscript |

## Key results

Simulated data (mean over 10 seeds, perturbation scenarios S3–S5):

| Method | AUPRC | F1 | False-alarm rate (no-change scenarios) |
|---|---|---|---|
| Original configuration (absolute features) | 0.075 | 0.063 | 4.0%–4.3% |
| **TL-GR-MCAE (proposed)** | **0.308** | **0.404** | **5.1%–7.5%** |

Real Svelvik data (no labels):

| Method | False-alarm rate on held-out baseline splits | Reproducibility (Dice, 26 / 28 July) |
|---|---|---|
| Original configuration | 12.1% | 0.80 / 0.77 |
| **TL-GR-MCAE** | **4.7%** | **0.89 / 0.85** |

## Acknowledgements

Field data: DASGAS project, ECCSEL Svelvik CO₂ Field Lab, acquired under the EU Geo-INQUIRE project (Grant Agreement No. 101058518). This work was supported by the Graduate Research Assistantship Scheme at Universiti Teknologi PETRONAS and the TotalEnergies EP Malaysia Research Grant.

## License

MIT License (see `LICENSE`).
