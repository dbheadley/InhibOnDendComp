# InhibOnDendComp

Analysis code for: **Headley et al. (2026) "Spatially targeted inhibitory rhythms differentially affect neuronal integration." *eLife*.** doi: [10.7554/eLife.95562](https://doi.org/10.7554/eLife.95562)

## Overview

This repository contains post-processing and analysis code for biophysical simulations of layer 5 pyramidal neurons, examining how the spatial location of inhibitory inputs (perisomatic vs. distal dendritic) modulates dendritic spike (Na⁺, NMDA, Ca²⁺) integration and somatic action potential output.

## Project Structure

```
src/           – Reusable Python modules (data loading, cross-correlation, phase analysis, STA)
scripts/       – Jupyter notebooks that reproduce each figure in the paper
data/          – Pre-computed simulation outputs (spike times, dendritic events, summary CSVs)
figures/       – Output figure files
tests/         – Unit tests for src modules
```

## Installation

```bash
conda env create -f environment.yml
conda activate dend_comp
```

Requires Python 3.9. Key dependencies: `pandas`, `scipy`, `h5py`, `holoviews`, `statsmodels`, `scikit-learn`.

## Reproducing Figures

Run the Jupyter notebooks in `scripts/` to reproduce each figure:

| Notebook | Content |
|---|---|
| `Fig2_3.ipynb` | Spike-triggered averages revealing temporal coupling between dendritic spikes (Na⁺, NMDA, Ca²⁺) and somatic APs, stratified by electrotonic distance and branch type |
| `Fig4.ipynb` | Effects of varying E/I coupling lags (4–500 ms) on perisomatic vs. distal dendritic inhibition and modulation of AP firing |
| `Fig5.ipynb` | Phase-dependent modulation of dendritic spike rates and AP threshold by 16 Hz beta (distal) and 64 Hz gamma (perisomatic) rhythmic inhibition |
| `Fig6.ipynb` | Phase-dependent modulation of AP voltage threshold by beta and gamma inhibition, revealing shunting mechanisms at soma vs. dendrites |
| `Fig7.ipynb` | Frequency sweep (0.5–80 Hz) of rhythmic inhibition effects; beta (~20 Hz) optimality for dendritic spike entrainment vs. gamma effects on membrane voltage |
| `Fig8.ipynb` | Frequency- and phase-dependent effects of perisomatic rhythmic inhibition on somatic membrane potential and AP threshold across 11 frequencies (0.5–80 Hz) |
| `Fig9.ipynb` | Phase-dependent modulation of dendritic spikes and APs by oscillatory bursts of beta and gamma; effects emerge within the first few burst cycles |
| `Fig10.ipynb` | Location-dependent gating of clustered synaptic inputs: beta gates distal/apical inputs, gamma gates proximal/basal inputs, in a phase-dependent manner |

## Data

Data files are available for download from Dryad: https://datadryad.org/dataset/doi:10.5061/dryad.v6wwpzhb8

## Key Analysis Methods

- **Spike-Triggered Averages (STA)** – temporal coupling between dendritic and somatic spikes
- **Phase-Binned Analysis** – dendritic spike probability as a function of inhibitory oscillation phase
- **Pairwise Phase Consistency (PPC)** – unbiased measure of oscillatory entrainment (Vinck et al. 2010)
- **Electrotonic Distance Stratification** – results grouped by normalized distance from soma (deciles)
- **Point-Process Cross-Correlation** – time-lagged relationships between neural event trains

## Citation

Headley DB et al. (2026). Spatially targeted inhibitory rhythms differentially affect neuronal integration. *eLife*. https://doi.org/10.7554/eLife.95562
