# GeoFuse installation guide

## Prerequisites

* A Conda-based Python distribution. [Miniforge](https://github.com/conda-forge/miniforge) is recommended; it includes `mamba`, which the installer uses when present.

> [!TIP]
> With Anaconda or Miniconda, install `mamba` in the base environment first for a much faster installation:
>
> ```bash
> conda install mamba -n base -c conda-forge
> ```

---

## Automated Installation

The installer script creates a `geofuse` conda environment with Python 3.13 and installs everything GeoFuse needs at tested versions. Packages come from conda-forge in a single solve, so compiled libraries stay consistent; pip adds only what conda-forge cannot provide: PyTorch (its CUDA builds), the Bayesian index's sampler (JAX and NumPyro) and `streamlit-sortables`. Detailed logs are written to `logs/install.log`.

If a `geofuse` environment already exists, the installer asks before removing it, since replacing it also removes packages you added by hand.

### Windows

Double-click `install.bat`. It finds your Conda installation (from an activated prompt, the usual install folders, or `PATH`) and asks for its folder only when it cannot.

### Linux / macOS

Make the script executable (one-time only):

```bash
chmod +x install.sh
```

Then run it:

```bash
./install.sh
```

### Installer options

Both installers take the same options:

| Option | Effect |
|--------|--------|
| `--yes` | Replace an existing environment without asking, and do not wait for a key at the end (unattended runs) |
| `--torch auto` / `cuda` / `cpu` | PyTorch build. `auto` (default) installs the CUDA build on Windows and Linux, which also runs on the CPU, and the standard build on macOS. `cpu` installs the smaller CPU-only build on machines without an NVIDIA GPU |
| `--env-name NAME` | Environment name (default `geofuse`) |

```bash
./install.sh --torch cpu          # Linux / macOS
install.bat --yes --torch cuda    # Windows, unattended
```

### The `geofuse` command

The installer makes `geofuse` available in the environment and ends with
`geofuse doctor --offline`. Run the full check once the model is in place:

```bash
conda activate geofuse
geofuse doctor
```

It reports the configuration, the state and cache folders, the credential
store, Street View, Earth Engine, the GPU, the segmentation model and the
monitor, and exits 3 when something is not ready. Into an existing environment,
`pip install -e .` from the repository root installs the command and any missing
dependency by name; the installer is what pins the tested versions.
`pip install -e ".[dev]"` adds the test and formatting tools.

---

## Segmentation Model

Place your pre-trained `.pth` file inside `geofuse/model/` and rename it to `best_model.pth`:

```text
geofuse/model/best_model.pth
```

You can use your own model or download one from [VainF's DeepLabV3+ repo](https://github.com/VainF/DeepLabV3Plus-Pytorch/tree/master).

The network is built from this file alone, so GVI needs no other download and runs on machines without internet access, such as cluster compute nodes.

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

* `images/`: downloaded Street View panoramas
* `test_ndvi.tif`: NDVI tile

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
