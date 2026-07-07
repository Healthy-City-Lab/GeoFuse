# GeoFuse — Installation Guide

## Prerequisites

* A Conda-based Python distribution. [Miniforge](https://github.com/conda-forge/miniforge) is recommended.

> [!TIP]
> For a significantly faster installation, install `mamba` in your base environment first:
>
> ```bash
> conda install mamba -n base -c conda-forge
> ```

---

## Automated Installation

The installer script creates a `geofuse` conda environment and installs all required packages. Detailed logs are written to `logs/install.log`.

### Windows

Double-click `install.bat`. It auto-detects your Conda installation and guides you through the process.

### Linux / macOS

Make the script executable (one-time only):

```bash
chmod +x install.sh
```

Then run it:

```bash
./install.sh
```

---

## Segmentation Model

Place your pre-trained `.pth` file inside `geofuse/model/` and rename it to `best_model.pth`:

```text
geofuse/model/best_model.pth
```

You can use your own model or download one from [VainF's DeepLabV3+ repo](https://github.com/VainF/DeepLabV3Plus-Pytorch/tree/master).

> [!IMPORTANT]
> **Custom Model Requirements**
>
> Your model must satisfy these constraints for GVI to work correctly:
>
> 1. **Architecture**: Compatible with the DeepLabV3+ implementation in `geofuse/dl_core/`.
> 2. **Classes**: Must output exactly **19 classes** (Cityscapes standard).
> 3. **Class IDs**: `Vegetation` = **8**, `Terrain` = **9**. Deviating from these IDs will produce incorrect GVI values.

---

## Verifying the Installation

Run the test suite after setup to confirm everything works.

### Logic Test (Fast, Offline)

Validates internal engine logic with mock data. No API key required.

```bash
conda activate geofuse
python tests/test_pipeline.py
```

Expected output (a successful run ends with unittest's `OK`):

```text
[PASS] GVI Pipeline processed N points
[PASS] NDVI Export Logic Verified
...
OK
```

### Full System Test

Connects to Google Street View and Earth Engine, downloads real data, and saves visual outputs.

```bash
conda activate geofuse
python tests/test_real_execution.py
```

Expected console output:

```text
[PASS] Sample GVI: veg=..., ter=... (pano=...)
[PASS] Large NDVI GeoTIFF exported (... KB)
```

Expected files in `tests/output/test2_system/`:

* `images/` — downloaded Street View panoramas
* `test_ndvi.tif` — NDVI tile

### Stability / Memory Test

Simulates thousands of processing cycles to confirm there are no memory leaks on GPU or RAM.

```bash
conda activate geofuse
python tests/test_memory_stability.py --iterations 5000
```

Expected output:

```text
[PASS] Memory usage appears stable.
```

A `memory_test.png` plot is saved to `tests/output/test3_stability/`.
