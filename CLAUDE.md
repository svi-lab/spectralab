# SpectraLab
## 1. General Purpose

* Streamlit desktop app for Renishaw .wdf spectral file analysis
* Primary target: PL (photoluminescence) spectroscopy; Raman planned
* Parses files via wdfkit.WDFReader, wraps data in SpectralDataset
* Pipeline: load → clean → cosmic ray removal → PCA denoising → visualize



## 2. Structure
```
app/
  app.py                      — entry point (streamlit run)
  backend/
    pipeline.py               — load + staged processing orchestrator
    _shared/                  — dataset, clean_data, normalize, spectral utils
    cosmic_ray/               — CosmicRayRemover; mask_1d / mask_map / harmonic
    spectra_cleaner/          — SpectraCleaner (PCA denoising)
    spectra_smoother/         — SpectraSmoother (SavGol / Whittaker)
  frontend/
    left_panel.py             — sidebar: upload, file info, pipeline controls
    tabs.py                   — main area tabs (raw, processed, progress, map)
    controls.py               — reusable widgets (crr, sc, cd, norm, axis)
    charts.py / map_chart.py  — plotting helpers
```
## 3. Code Rules


* Stack
  * Streamlit app
  * Python (xarray, numpy, sklearn, and such)
  * Frontend: Streamlit elements with echarts/JS plotting
* Python env: /opt/anaconda3/envs/wdftest; run app with streamlit run app/app.py
* Backend modules use plain imports (from cosmic_ray import ...) — app/backend/ is on sys.path
* SpectralDataset is the typed container passed everywhere; never pass raw da across modules

## 4. Notes


* measurement_kind: "PL" for Nanometer/ElectronVolt axes; "Raman" for RamanShift/Wavenumber — CRR and SC only enabled for PL currently
* Broad CR detection: kernel = 4 × broad_spike_width + 1 (not 2×) — CR must be < 50% of kernel
* SNIP repair floor prevents negative peaks after CR removal on sloped PL features
* CleanData is a manual frontend step; it is NOT called automatically inside SpectraCleaner anymore
* @st.cache_data on file loading; session state keys reset when uploaded file set changes


### Pixel size on the map scan logic:

1. Convert the scan step size from µm to pixels — step_x / fov_x * image_width gives how many pixels one step occupies in the image
2. Multiply by 0.45 — so the dot fills ~90% of the step gap (leaving a small gap between neighbours)
3. Take the smaller of x and y directions so dots stay circular
4. Clamp to [1, 8] pixels hard

For example, if a 50×50 raster scan covers 100 µm on a 1000-pixel-wide image:

* step = 2 µm, fov_x = 100 µm → 2/100 * 1000 * 0.45 = 9 → clamped to 8 px

For a denser 200×200 scan over the same area:

* step = 0.5 µm → 0.5/100 * 1000 * 0.45 = 2.25 → 2 px

If you want to tune it, the three knobs are:

* The 0.45 fill factor (line 106–107) — increase toward 0.5 for touching dots, decrease for more gap
* The 8 upper clamp (line 108) — raise if dots look too small on sparse scans
* The 1 lower clamp — keep at 1 so even the densest scans show something visible

### Denoising

Two engines, selected in the frontend:

PCA — population-based (default, per_spectrum=False)

Treats all spectra as a dataset. Fits sklearn PCA on the full (n_spectra × n_channels) matrix, reconstructs each spectrum using only the leading k components. Those components capture correlated signal shared across spectra; the dropped components carry uncorrelated per-channel noise.

Requires ≥ 2 spectra — degenerate on a single spectrum
n_components controls how many components to keep (mle = auto via Minka's MLE, with fallback when n_spectra < n_channels)
subtract_min (Baseline handling): removes each spectrum's DC offset before fitting so PCA models shape, not absolute level; optionally restored after
NaN rows from CleanData are filled with the column-wise median of valid spectra before fitting, then re-marked as NaN after reconstruction
Smoother — per spectrum (per_spectrum=True)

Applies a 1D filter independently to every spectrum. Three methods:

Method	How it works	Knobs
Savitzky-Golay	Fits a polynomial to a sliding window via least squares	window_length, polyorder
Whittaker-Eilers	Minimises ‖y−z‖² + λ‖D²z‖² (sparse linear system)	λ (auto via GCV), d
Wavelet (VisuShrink)	DWT → threshold detail coefficients at σ√(2 log N) → iDWT	wavelet family, soft/hard threshold
NaN rows are detected upfront, skipped during smoothing, and preserved as NaN in the output. For Whittaker auto-λ, the GCV mean spectrum is computed from valid rows only.

When to use which:

PCA is more powerful for maps and line scans — it exploits the fact that all spectra share the same emission features, so it separates signal from noise statistically
Smoother is the only option for a single spectrum, or when you want spectrum-by-spectrum control independent of neighbours
Wavelet is the best single-spectrum option for PL with sharp peaks — it doesn't shift peak positions the way Savgol can at aggressive settings