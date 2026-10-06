# GeoFuse troubleshooting

---

## `No module named geofuse`

The package wasn't linked correctly during setup.

**Fix:** Re-run the installer (`install.bat` / `install.sh`) and confirm a success message at the end. If it still fails, manually run inside the activated environment:

```bash
pip install -e .
```

---

## `Conda Not Found`

The installer script fails immediately.

**Fix:**
* Confirm Miniforge/Anaconda is installed.
* Ensure the Conda `Scripts/` (Windows) or `bin/` (Linux/macOS) directory is on your `PATH`.
* If installed in a custom location, paste the correct path when prompted by the script.

---

## `[WinError 127]` or DLL Load Failure on Windows

A GDAL or GDAL-dependent library (GeoPandas, rasterio) fails to import.

**Cause:** `torch` / `geofuse` DLLs must be loaded before the GDAL stack on Windows.

**Fix:** Never import `geopandas`, `rasterio`, or similar libraries before `import torch` or `from geofuse import ...` in custom scripts. The application entry point (`ui/app.py` and `scripts/cli.py`) already enforces the correct order.

---

## `Windows fatal exception: code 0xc06d007f` from `numpy.linalg.cholesky` (or `MixedLM`)

A fatal exception terminates the interpreter with no Python traceback the moment a matrix operation runs. Most often surfaces when running the mixed-effects fusion (`statsmodels.regression.mixed_linear_model.MixedLM.fit`).

**Cause:** Conda-forge's MKL 2026.x ships an INTEL OpenMP threading runtime whose symbols clash with `numpy.linalg.cholesky` and matrix-multiply (`gemm`) on Windows. The mismatch causes the OS loader to fail entry-point resolution and kill the process.

**Fix:** Switch the env's BLAS metapackage from MKL to openblas, then reinstall `numpy` and `scipy` via conda so they bind to the new BLAS:

```bash
conda activate geofuse
conda install -c conda-forge "libblas=*=*openblas" "libcblas=*=*openblas" "liblapack=*=*openblas" -y
pip uninstall numpy scipy -y
conda install -c conda-forge numpy scipy --force-reinstall -y
```

Fresh installs run by the current `install.bat` / `install.sh` already pin openblas + conda-installed numpy/scipy (see `scripts/setup_env.py`), so this fix is only needed for envs built manually, or before this recent fix.

---

## `DecompressionBombWarning`

A warning appears in the terminal when downloading very high-resolution Street View images.

**This is safe to ignore.** GeoFuse automatically handles large image limits and resizes panoramas before segmentation.

---

## `Earth Engine authentication error` / `ee.Initialize` fails

Google Earth Engine credentials are missing or expired.

**Fix:**
1. Run `earthengine authenticate` in the activated environment.
2. Follow the browser prompt to authorize your Google account.
3. If using a service account, ensure the credentials JSON is accessible and the project ID is set correctly.

---

## Out-of-Memory (GPU) Errors

The segmentation model runs out of VRAM during batch processing.

**Fix:**
* Process fewer study areas simultaneously.
* The engine automatically falls back from CUDA → MPS → CPU if a device runs out of memory mid-job.

---

## `RuntimeError: main thread is not in main loop` when browsing for a file

Tk requires the thread that owns its interpreter to be the process's main
thread. Streamlit runs the script body, and every `on_click` callback, on a
per-rerun ScriptRunner thread, so a `tkinter` dialog opened in-process fails as
soon as any Tk state outlives the thread that created it.

The dialog therefore runs in a **child process** (`ui/file_picker_child.py`),
launched per click and gone before the next rerun. If you still see this error,
the app is running an older copy of `ui/file_picker.py`. Restart Streamlit so
the module is re-imported.

Related: if the picker reports *"The native file dialog could not open"* and
offers a path box instead, that is not this bug. The dialog opens on the machine
running the Streamlit **server**, so it cannot appear when the server is remote
and the browser is elsewhere. Paste the server-side absolute path into the box.

---

## GVI Job Shows as "Interrupted" After a Streamlit Restart

The job was running when Streamlit (or the machine) restarted. The job state was saved but the worker process was killed.

**Fix:** Switch to the tab the job came from (GVI or NDVI). Re-upload the **same** study area file you originally used. A restart panel appears above the input form with **Resume Job** and **Discard** buttons. **Resume Job** continues from the exact point where the job stopped; already-processed points and cached panoramas are skipped.

If the file you re-upload does not match the original geometry, the restart panel will refuse to resume. Discard the interrupted job and start a new run instead.

---

## Where Are Per-Job Logs Stored?

Every job writes a persistent text log to `logs/jobs/<job_id>.log`. The in-UI expander only keeps the last 100 lines in memory; the file on disk has the full log indefinitely.

**Quick access:** open the job's expander in the sidebar Job Monitor and click "📄 Open log file", which opens it in your OS default text editor.

External library logs (Google Earth Engine, Optuna) are routed into the same per-job file so they no longer appear in the host terminal.

---

## GeoTIFF File Is Huge or Mostly Empty (GVI)

This happens when a single bbox-wide raster is rendered for a study area whose points are clustered in a few small regions (e.g. neighbourhoods across multiple cities). The bbox contains millions of empty cells.

**Fix:** Switch to **GeoPackage** as the GVI output format; it is the recommended default for sparse or national-scale data. If you still want raster output, enable **Save GeoTIFF** in the GVI tab and GeoFuse will write one dense GeoTIFF per spatial cluster into `[Filename]_gvi_tiles/` along with a `tiles_index.json` describing each tile's bounds. No empty cells.
