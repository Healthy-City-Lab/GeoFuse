# GeoFuse — Troubleshooting

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
* Reduce the grid resolution (increase meters-per-point spacing in the GVI tab).
* Process fewer study areas simultaneously.
* The engine automatically falls back from CUDA → MPS → CPU if a device runs out of memory mid-job.

---

## Built With & References

* **Street View Download:** `geofuse/streetview.py` — a minimal custom port adapted from [streetlevel](https://github.com/sk-zk/streetlevel) (MIT). Uses Google's `SingleImageSearch` protobuf API for metadata and `streetviewpixels-pa.googleapis.com` for tiles. No external Street View package required.
* **Semantic Segmentation:** [DeepLabV3+](https://github.com/VainF/DeepLabV3Plus-Pytorch/tree/master) via PyTorch.
* **Training Data:** [Cityscapes Dataset](https://www.cityscapes-dataset.com/).
* **Satellite Imagery:** [Google Earth Engine](https://earthengine.google.com/) and [geemap](https://geemap.org).
