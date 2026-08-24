# SpectraLab

Local desktop app for **Renishaw `.wdf` photoluminescence (PL)** measurements: single spectra, line scans, and maps.

Upload a file, clean it, then inspect spatial maps, unmix spectral components, or fit emission bands. Everything runs on your computer and opens in a browser.

Raman `.wdf` files can be opened and viewed. Cosmic-ray removal, denoising, decomposition, and peak fitting are currently available for PL only.

---

## Requirements

- Python 3.10–3.13 (Anaconda / Miniconda is fine)
- [wdfkit](https://github.com/dshirya/wdfkit) — reads Renishaw `.wdf` files

## Setup

```bash
git clone https://github.com/svi-lab/spectralab.git
cd spectralab

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install git+https://github.com/dshirya/wdfkit.git
```

With conda, create an environment (`conda create -n spectralab python=3.12` then `conda activate spectralab`) and run the same `pip` commands.

## Launch

```bash
streamlit run app.py
```

A browser tab opens at [http://localhost:8501](http://localhost:8501). Leave the terminal open while you work; press `Ctrl+C` to stop.

---

## Workflow

Drop one or more `.wdf` files in the **sidebar**. The top bar is the analysis path:

| Step | What it does |
|------|----------------|
| **Data** | File info, white-light scan image (if present), optional film/substrate optical model, export of processed spectra |
| **Preprocessing** | Drop dead pixels, remove cosmic rays, denoise, normalize, and exclude spectra you do not want later |
| **Decomposition** | Find recurring spectral shapes on a map — **NMF** (fast, statistical) or **MCR-ALS** (non-negative, more physically constrained) |
| **Deconvolution** | Fit Gaussian emission bands. Load ZnO:Al or TiO₂ presets, or click the chart to add centres. Fit the mean, a pixel, an NMF component, or the whole map |
| **Map Analysis** | Spatial heatmap of integrated intensity or deviation from the mean, in a chosen energy window |

**Quick Setup** on Preprocessing is the usual starting point:

- **1D preprocessing** — single spectra and small series (per-spectrum cosmic-ray removal and smoothing)
- **3D preprocessing** — maps and line scans (spatial cosmic-ray removal and PCA denoising)

Settings chosen there apply to every later page. Excluded spectra are left as empty pixels so the original scan grid is unchanged.

Processed spectra, decomposition results, and fitted bands can be downloaded as `.npz` files.
