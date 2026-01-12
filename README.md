# GeoFuse: Multimodal Greenspace Profiling Toolbox

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![OS](https://img.shields.io/badge/OS-Linux%7CWindows%7CmacOS-blue)](#-installation)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![MPI](https://img.shields.io/badge/MPI-Parallel-blue)](#4-high-performance-computing-hpc-integration)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=Streamlit&logoColor=white)](#2-streamlit-web-ui)
[![GDAL](https://img.shields.io/badge/GDAL-5CAE58?logo=gdal&logoColor=white)](https://gdal.org/)
[![GeoPandas](https://img.shields.io/badge/GeoPandas-139C5A?logo=geopandas&logoColor=white)](https://geopandas.org/)
[![Google Earth Engine](https://img.shields.io/badge/Google%20Earth%20Engine-4285F4?logo=google-earth&logoColor=white)](#2-satellite-intelligence-ndvi)

**GeoFuse** is a comprehensive Python toolbox designed for digital health and urban planning research that utilizes greenspace exposure. It automates the sourcing, processing, and fusion of environmental exposure metrics, specifically focusing on **Green View Index (GVI)** from street-level imagery and **Normalized Difference Vegetation Index (NDVI)** from satellite data.

The toolbox can be run in two modes: a user-friendly **Streamlit Dashboard** for interactive analysis, or a **Command-Line Interface (CLI)** for scalable, parallel processing on high-performance computing (HPC) systems.

---

## 🌟 Key Features

### 1. Street View Intelligence (GVI)

* **Automated Sourcing**: Scrapes or downloads Google Street View panoramas for any study area (Shapefile/GeoJSON). It can operate _with_ or _without_ an API Key.
* **Deep Learning Segmentation**: Uses the **DeepLabV3+** model (PyTorch) trained on the **Cityscapes** dataset to identify Vegetation and Terrain greenery coverage.
* **Batch Processing**: Drag-and-drop multiple study areas (GeoJSON/SHP) to process distinct regions simultaneously.
* **Smart Caching**: Implements a shared panorama cache across all batch files to prevent redundant downloads of overlapping areas, significantly reducing processing time and API costs.

* **Robust Processing**:

  * **Async/Multi-threaded** downloading for speed.
  * **Image pre-processing**: Ensures 360° coverage, corrects panorama artifacts, detects corrupt panoramas, and standardizes image resolution.

* **Batch Processing & Recovery**:
  * **Multi-threaded Analysis**: Runs deep learning inference in the background while keeping the UI responsive.
  * **Crash Recovery**: If a job is interrupted (internet loss, crash, or manual cancel), simply re-upload the same input file and click "Start". The engine automatically detects existing progress and **resumes** from the last processed point.
  * **Job Monitor**: Track progress via the sidebar. Use the **Scan Output Folder** button to load and visualize results from previous sessions without re-running analysis.

* **Outputs**:

  * **Vector**: GeoJSON of sampling points with their respective GVI values (Vegetation and Terrain separated).
  * **Raster**: Heatmaps (GeoTIFF) aligned to the user's defined sampling grid.
  * **Raw Panoramas and Masks**: Google Street View panoramas and their segmentation results used to create the outputs.

### 2. Satellite Intelligence (NDVI)

* **Google Earth Engine Integration**: Fetches cloud-free Sentinel-2 or Landsat imagery.
* **Dynamic Calculation**: Computes NDVI (Vegetation Health) for the exact same timeframe as your street view data.

### 3. Fusion & Visualization

* **Interactive Map**: Visualize results immediately with Folium/Leaflet.
* **Results Inspector**: Toggle between multiple loaded datasets and visualize specific layers (Points, Vegetation Raster, Terrain Raster) with dynamic opacity controls.
* **Metric Fusion**: (In Development) Combines top-down (NDVI) and eye-level (GVI) metrics for a holistic "Composite Greenery Index" score, tailored towards a specific spatial outcome variable.

### 4. High-Performance Computing (HPC) Integration

GeoFuse is architected to scale from local laptops to High-Performance Computing (HPC) clusters, enabling city-wide or regional-scale analysis.

* **MPI-Enabled CLI**: The Command-Line Interface supports parallel execution via MPI (`mpiexec`/`mpirun`), allowing workloads to be distributed across multiple CPU cores or compute nodes.
* **Headless Batch Processing**: The core logic is decoupled from the UI into standalone engines. This allows for direct Python script execution, enabling automated batch processing pipelines without browser dependencies.
* **GPU Acceleration & Scaling**: Built on **PyTorch** with native **CUDA** support. The architecture is designed for Multi-GPU inference, allowing massive segmentation workloads to be distributed across available hardware nodes.
* **Optimized Resource Management**: Implements asynchronous I/O for non-blocking downloads and memory-efficient raster operations (windowed reading/writing) to maximize throughput on shared compute nodes.

---

## 🚀 Installation

GeoFuse uses an automated installer that handles all dependencies, including complex geospatial and deep learning libraries.

### Prerequisites

* A Conda-based Python distribution. We recommend **Miniforge**.
* (Optional) For significantly faster installation, make sure `mamba` is installed in your base conda environment: `conda install mamba -n base -c conda-forge`.

### Automated Installation

The installer script will automatically create a `geofuse` conda environment and install all required packages.

#### 1. Windows

Simply double-click the `instal.bat` script. It will auto-detect your Conda installation and guide you through the process.

#### 2. Linux / macOS

First, make the installer script (`install.sh`) executable:

```bash
# Give the script execute permissions
# (Only needs to be done once)
chmod +x install.sh
```

then run it:

```bash
# Run the installer
./install.sh
```

### Segmentation Model

Place your pre-trained segmentation model's `.pth` file inside `geofuse/model` and rename it to `best_model.pth`. The toolbox can automatically detect commonly-used backbones.

_Note: You can either use your own trained model, or find one from online sources such as [VainF's repo](https://github.com/VainF/DeepLabV3Plus-Pytorch/tree/master)._

---

## ✅ Usage

After installation, you can run GeoFuse using either the parallel Command-Line Interface or the interactive Streamlit Web UI.

### 1. Command-Line Interface (CLI) for Batch Processing

The CLI is designed for large-scale, automated, and parallel processing. It is the recommended method for running large study areas or multiple files on servers or HPC clusters.

1. **Activate the Environment**
    Open your Conda terminal and activate the environment:

    ```bash
    conda activate geofuse
    ```

2. **Run the CLI**
    Use `mpiexec` or `mpirun` to execute the main CLI script in parallel. The `-n` flag specifies the number of parallel processes.

    ```bash
    # Example: Run with 4 parallel processes
    mpiexec -n 4 python scripts/cli.py --config config.csv
    ```

    * `--config`: Path to a CSV file that defines your input files and parameters.
    * A template `config.csv` will be created for you if one is not found.

### 2. Streamlit Web UI

The Web UI is ideal for interactive exploration, visualization of results, and processing smaller study areas.

1. **Activate the Environment**

    ```bash
    conda activate geofuse
    ```

2. **Launch the App**

    ```bash
    streamlit run ui/app.py
    ```

---

## ✅ Verifying Installation

We provide a suite of test scripts to validate the installation, logic, and hardware stability. For each test, launch your conda terminal, activate the `geofuse` environment, and execute the provided scripts.

### Logic Test (Fast & Offline)

* **Script:** `tests/test_pipeline.py`
* **Purpose:** Validates the internal logic of the Fusion Engine and GVI pipeline using "mock" data. It does not require an API key or GPU.
* **Run:**

```bash
python tests/test_pipeline.py
```

* **Expected Output:**

``` bash
[PASS] GVI GeoTIFF Created at tests/output/test1_logic/gvi_distribution.tif
[PASS] NDVI Export Logic Verified (File created at tests/output/test1_logic/test_ndvi.tif)
OK
```

### Full System Test

* **Script:** `tests/test_real_execution.py`
* **Purpose:** Connects to Google Street View and Earth Engine to download real data, processes it on the GPU, and saves visual results. Use this to confirm your credentials and model are working.
* **Run:**

```bash
python tests/test_real_execution.py
```

* **Expected Output:**

  * **Console:**

  ```bash
  [PASS] Image found! Size: (1920, 960)
  [PASS] Segmentation complete. Mask Shape: (960, 1920)
  [PASS] Metrics: {'GVI_Vegetation': ...}
  [PASS] Large NDVI GeoTIFF exported (... KB)
  ```

  * **Files:** Check `tests/output/test2_system/` for:
    * `test_pano_rgb.jpg` (Street View)
    * `test_pano_mask.png` (Segmentation Mask)
    * `large_ndvi.tif` (NDVI Tile)

### Stability Test

* **Script:** `tests/test_memory_stability.py`
* **Purpose:** Simulates thousands of processing cycles to ensure there are no memory leaks in the GPU or RAM, which is critical for large city-wide runs.
* **Run:**

```bash
# Runs 5000 iterations (reduce iteration if you want faster runtime)
python tests/test_memory_stability.py --iterations 5000
```

* **Expected Output:**

  * **Console:**

  ```bash
  [PASS] Memory usage appears stable.
  ```

  * **Files:** Check `tests/output/test3_stability/` for `memory_test.png` plot which shows RAM/VRAM usage over time.

---

## 🛠️ Configuration & Troubleshooting

### **"No module named geofuse"**

If you see this error, it means the package wasn't linked correctly during setup.

* **Fix:** Rerun the installer script (`instal.bat` / `install.sh`) and ensure you see a success message at the end. Or, if installing manually, make sure you run `pip install -e .` after activating the environment.

### **"Conda Not Found"**

If the installer script fails immediately:

* Ensure Miniforge/Anaconda is installed and that its 'Scripts' or 'bin' directory is accessible from your terminal.
* If installed in a custom location, the script will prompt you to paste the correct path.

### **"DecompressionBombWarning"**

* **Info:** You may see this warning in the terminal if downloading very high-res Street View images.
* It is Safe to ignore. The toolbox automatically handles large image limits and resizes them for processing.

---

## 📄 Outputs

For every analysis run, GeoFuse generates:

* `[Filename]_gvi.geojson`: Point data containing `gvi_veg` (Vegetation) and `gvi_ter` (Terrain) scores, and the unique `pano_id` for every point.
* `[Filename]_gvi.tif`: A multi-band GeoTIFF raster (Band 1: Vegetation, Band 2: Terrain) generated for each input region.
* (Optional) `output_results/masks/{pano_id}.png` and `output_results/images/{pano_id}.jpg`: Raw panoramas and segmentation masks named by their unique Panorama ID.

---

## 🧩 Built With & References

This project relies on several open-source libraries and public datasets. We gratefully acknowledge:

* **Street View Download:** Adapted from the [streetview](https://github.com/robolyst/streetview) library by `@robolyst`.
* **Semantic Segmentation Model:** [DeepLabV3+](https://github.com/VainF/DeepLabV3Plus-Pytorch/tree/master) architecture implemented via PyTorch.
* **Segmentation Training Data:** The semantic segmentation model was pre-trained on the [Cityscapes Dataset](https://www.cityscapes-dataset.com/).
* **Satellite Imagery Analysis:** Powered by [Google Earth Engine](https://earthengine.google.com/) and [geemap](https://geemap.org).

---

## 📚 Citation

If you use GeoFuse in your research, please cite:

> **Sadigh, A. G.** (2025). _GeoFuse: Multimodal Greenspace Profiling Toolbox_. Healthy City Lab, University of Calgary. [https://github.com/Healthy-City-Lab/GeoFuse](https://github.com/Healthy-City-Lab/GeoFuse)

## 🎓 Credits & Acknowledgments

**Primary Developer and Maintainer:** [Armin Ghayur Sadigh](https://github.com/Armin-GS)

* **Academic Context:** This toolbox was developed as part of a Ph.D. research project at the Healthy City Lab, University of Calgary. It is designed to support ongoing research into environmental determinants of health and urban sensing.

* **Lab:** [Healthy City Lab](https://www.healthycitylab.ca/)
* **Institution:** [University of Calgary](https://ucalgary.ca/)

**License:** GNU General Public License v3.0 (See [LICENSE file](LICENSE))
