# GeoFuse: Multimodal Greenspace Profiling Toolbox

[![Installation & Test Suite](https://github.com/Healthy-City-Lab/GeoFuse/actions/workflows/setup-tests.yml/badge.svg)](https://github.com/Healthy-City-Lab/GeoFuse/actions/workflows/setup-tests.yml)
[![License](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![OS](https://img.shields.io/badge/OS-Linux|Windows|macOS-blue)](#installation)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Conda Env](https://img.shields.io/badge/Conda%20Env-geofuse-342B029.svg?logo=anaconda&logoColor=white)](#installation)
[![MPI](https://img.shields.io/badge/MPI-Parallel-blue)](#4-high-performance-computing-hpc-integration)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=Streamlit&logoColor=white)](#2-streamlit-web-ui)
[![Google Earth Engine](https://img.shields.io/badge/Google%20Earth%20Engine-4285F4?logo=google-earth&logoColor=white)](#2-satellite-intelligence-ndvi)
[![NumPy](https://img.shields.io/badge/NumPy-013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![GDAL](https://img.shields.io/badge/GDAL-5CAE58?logo=gdal&logoColor=white)](https://gdal.org/)
[![GeoPandas](https://img.shields.io/badge/GeoPandas-139C5A?logo=geopandas&logoColor=white)](https://geopandas.org/)

**GeoFuse** is a comprehensive Python toolbox designed for digital health and urban planning research that utilizes greenspace exposure. It automates the sourcing, processing, and fusion of environmental exposure metrics, specifically focusing on **Green View Index (GVI)** from street-level imagery and **Normalized Difference Vegetation Index (NDVI)** from satellite data.

The toolbox can be run in two modes: a user-friendly **Streamlit Dashboard** for interactive analysis, or a **Command-Line Interface (CLI)** for scalable, parallel processing on high-performance computing (HPC) systems.

---

## Key Features

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

* **Metric Fusion Engine**: Combines top-down (NDVI) and eye-level (GVI) metrics for a holistic "Composite Greenery Index" score, optimized towards spatial health/environmental outcomes.

  * **Automated Metric Alignment**: Automatically downloads and spatially aligns GVI (vegetation/terrain) and NDVI metrics within your study area if not provided.
  * **Dual Input Support**: Works with both **point-based** targets (GeoJSON with health/environmental data) and **raster-based** targets (GeoTIFF continuous surfaces).
  * **Bayesian Optimization**: Uses **Optuna** with TPE sampling to optimize 9 parameters:
    * **Weights**: Vegetation, Terrain, NDVI contribution (0-100%, sum=100)
    * **Spatial Aggregation**: Circular buffer radii (100m to user-defined buffer distance, step=50m)
    * **Statistical Functions**: Mean, median, or percentile-based aggregation
    * **Separate Controls**: Independent radius and aggregation settings for each metric component
  
  * **Robust Cross-Validation**: 
    * **Stratified K-Fold CV**: Ensures representative sampling across target value distribution
    * **Held-Out Test Set**: 20% of data reserved for final validation
    * **Statistical Validation**: Benjamini-Hochberg FDR correction for correlation-based metrics
    * **Pruning Support**: Median, Hyperband, or Successive Halving pruners to accelerate optimization
  
  * **Multi-Metric Optimization**: Supports multiple objective functions:
    * **Pearson Correlation**: Linear relationship strength (with p-value validation)
    * **Spearman Correlation**: Monotonic relationship strength (with p-value validation)
    * **R² Score**: Coefficient of determination
    * **RMSE**: Root Mean Squared Error (minimization)
    * **Mutual Information**: Non-linear dependency measure
  
  * **Intelligent Caching**: 
    * **Deterministic Filenames**: Cache files named based on target area hash
    * **Resume Support**: Reuses downloaded metrics for identical study areas
    * **Multi-Band Optimization**: Stores GVI vegetation and terrain in single raster for efficiency
  
  * **Comprehensive Reporting**: Automatically generates:
    * **Optuna Visualizations**: Optimization history, parameter importance, parallel coordinates, contour plots, EDF, slice plots, timeline
    * **Robust Trial Analysis**: Separate reports for statistically significant trials vs. all trials
    * **Best Parameters**: JSON export with optimal weights, radii, and aggregation functions
    * **Test Set Evaluation**: Final performance on held-out data with statistical significance
  
  * **Composite Map Generation**: 
    * **Ensemble Averaging**: Averages parameters from top 20% of robust trials
    * **Grid-Aligned Output**: Generates composite greenery raster matching target resolution
    * **Multi-Resolution Support**: 10m satellite resolution or custom grid spacing
    * **Parameter Tracking**: Saves final averaged parameters alongside composite map

### 4. High-Performance Computing (HPC) Integration

GeoFuse is architected to scale from local laptops to High-Performance Computing (HPC) clusters, enabling city-wide or regional-scale analysis.

* **MPI-Enabled CLI**: The Command-Line Interface supports parallel execution via MPI (`mpiexec`/`mpirun`), allowing workloads to be distributed across multiple CPU cores or compute nodes.
* **Headless Batch Processing**: The core logic is decoupled from the UI into standalone engines. This allows for direct Python script execution, enabling automated batch processing pipelines without browser dependencies.
<!-- TODO: MULTI_GPU_IMPLEMENTATION - Implement DataParallel/DistributedDataParallel for multi-GPU inference workloads -->
* **Intelligent GPU Acceleration**: Built on **PyTorch** with automatic device selection (prioritizing **CUDA** → **MPS** → **CPU**). The architecture supports Multi-GPU inference, enabling massive segmentation workloads to be distributed across available hardware nodes on any platform.
* **Optimized Resource Management**: Implements asynchronous I/O for non-blocking downloads and memory-efficient raster operations (windowed reading/writing) to maximize throughput on shared compute nodes.

---

## Installation

GeoFuse uses an automated installer that handles all dependencies, including complex geospatial and deep learning libraries.

### Prerequisites

* A Conda-based Python distribution. We recommend **Miniforge**.

> [!TIP]
> For a significantly faster installation, make sure `mamba` is installed in your base conda environment:
>
> ```bash
> conda install mamba -n base -c conda-forge
> ```

### Automated Installation

The installer script will automatically create a `geofuse` conda environment and install all required packages. The installation process produces clean console output, with the detailed logs automatically saved to `logs/install.log` for troubleshooting.

#### 1. Windows

Simply double-click the `install.bat` script. It will auto-detect your Conda installation and guide you through the process.

#### 2. Linux / macOS

First, make the installer script (`install.sh`) executable (This only needs to be done once):

```bash
chmod +x install.sh
```

Then run it:

```bash
./install.sh
```

### Segmentation Model

Place your pre-trained segmentation model's `.pth` file inside `geofuse/model` and rename it to `best_model.pth`.
You can either use your own trained model, or find one from online sources such as [VainF's repo](https://github.com/VainF/DeepLabV3Plus-Pytorch/tree/master). The toolbox can automatically detect commonly-used backbones.

> [!IMPORTANT]
> **Using a Custom Segmentation Model**
>
> While you can load your own pre-trained model, it must adhere to specific requirements for the GVI calculation and visualization to work correctly:
>
> 1. **Model Architecture**: The model must be compatible with the DeepLabV3+ architecture used in this toolbox.
> 2. **Number of Classes**: The model must be trained on **19 output classes**, matching the Cityscapes dataset standard.
> 3. **Class ID Mapping**: For GVI calculation, the model's output classes **must** use the following specific IDs:
>     * `Vegetation`: Class ID **8**
>     * `Terrain`: Class ID **9**
>
> Models that do not follow this structure will either fail to load or produce incorrect GVI results.

---

## Verifying Installation

We provide a suite of test scripts to validate the installation, logic, and hardware stability. For each test, launch your conda terminal, activate the `geofuse` environment, and execute the provided scripts.

### Logic Test (Fast & Offline)

* **Script:** `tests/test_pipeline.py`
* **Purpose:** Validates the internal logic of the Fusion Engine and GVI pipeline using "mock" data. It does not require an API key or credentials.
* **Run:**

```bash
python tests/test_pipeline.py
```

* **Expected Output:**

``` text
   [PASS] GVI Pipeline processed 100 points
...
   [PASS] NDVI Export Logic Verified
...

OK
```

### Full System Test

* **Script:** `tests/test_real_execution.py`
* **Purpose:** Connects to Google Street View and Earth Engine to download real data, and saves visual results. Use this to confirm your credentials and check if the segmentation model works as expected.
* **Run:**

```bash
python tests/test_real_execution.py
```

* **Expected Output:**

  * **Console:**

  ```text
  [PASS] Image found! Size: (1024, 512)
  [PASS] Segmentation complete. Mask Shape: (512, 1024)
  [PASS] Metrics: {'GVI_Vegetation': ..., 'GVI_Terrain': ..., 'GVI_Total': ...}
  ...
  [PASS] Large NDVI GeoTIFF exported (50.90 KB)
  ```

  * **Files:** Check `tests/output/test2_system/` for:
    * `test_pano_rgb.jpg` (Street View)
    * `test_pano_mask.png` (Segmentation Mask)
    * `test_ndvi.tif` (NDVI Tile)

### Stability Test

* **Script:** `tests/test_memory_stability.py`
* **Purpose:** Simulates thousands of processing cycles to ensure there are no memory leaks on GPU (CUDA/MPS) or RAM, which is critical for large scale runs.
* **Run (reduce iteration if you want faster runtime):**

```bash
python tests/test_memory_stability.py --iterations 5000
```

* **Expected Output:**

  * **Console:**

  ```text
  [PASS] Memory usage appears stable.
  ```

  * **Files:** Check `tests/output/test3_stability/` for `memory_test.png` plot which shows RAM/VRAM usage over time.

---

## Usage

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

> [!NOTE]
> A template `config.csv` will be created for you if one is not found.

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

## Outputs

GeoFuse generates different outputs depending on which engine is used. All files are saved to the `output_results/` directory by default.

### GVI Engine Outputs

For each input file processed through the GVI pipeline:

* **`[Filename]_gvi.geojson`**: Vector point data containing:
  * `gvi_veg`: Green View Index for Vegetation (%)
  * `gvi_ter`: Green View Index for Terrain (%)
  * `pano_id`: Unique Google Street View panorama identifier
  * `lat`, `lon`: Geographic coordinates
  * `row`, `col`: Grid position (if raster-aligned)

* **`[Filename]_gvi.tif`**: Multi-band GeoTIFF raster (EPSG:4326):
  * Band 1: Vegetation GVI heatmap
  * Band 2: Terrain GVI heatmap

* **Optional Outputs**:
  * `output_results/images/{pano_id}.jpg`: Original Street View panoramas
  * `output_results/masks/{pano_id}.png`: Semantic segmentation masks (Cityscapes palette)

### NDVI Engine Outputs

For each input file processed through the NDVI pipeline:

* **`[Filename]_ndvi.geojson`**: Vector point data containing:
  * `NDVI`: Normalized Difference Vegetation Index values (-1 to 1)
  * `x`, `y`: Geographic coordinates (EPSG:4326)

* **`[Filename]_ndvi.tif`**: Single-band GeoTIFF raster (EPSG:4326):
  * Band 1: NDVI values extracted from Sentinel-2 imagery
  * NoData values represented as -9999

### Fusion Engine Outputs

For each optimization run through the Metric Fusion pipeline:

* **Composite Greenery Map**:
  * **`composite_greenery.tif`**: Single-band GeoTIFF containing the optimized composite greenery index
  * **`composite_greenery_params.json`**: Final averaged parameters used to generate the composite map
    * Weights for vegetation, terrain, and NDVI components
    * Optimal buffer radii for each metric
    * Aggregation statistics (mean/median/percentile) for each component

* **Optimization Study Results** (`output_results/fusion/study_results/`):
  
  * **Robust Trials Analysis** (`robust_trials/`):
    * **`optimization_report.txt`**: Summary of statistically significant trials with performance metrics
    * **`best_params.json`**: Best performing parameter set from robust trials
    * **Optuna Visualizations**:
      * `optimization_history.html`: Trial value progression over time
      * `param_importances.html`: Feature importance analysis
      * `parallel_coordinates_all.html` & `parallel_coordinates_weights.html`: Multi-dimensional parameter visualization
      * `contour_weights.html`: 2D parameter interaction heatmaps
      * `slice_plot_weights.html`: 1D parameter impact plots
      * `edf.html`: Empirical Distribution Function of trial values
      * `rank.html`: Trial ranking visualization
      * `timeline.html`: Trial execution timeline
  
  * **All Trials Analysis** (`all_trials/`):
    * Contains same visualizations as robust trials for debugging and comparison
    * Includes all trials regardless of statistical significance

* **Cached Metrics** (`output_results/fusion_cache/`):
  * **`cache-veg-{hash}.geojson`**: Cached GVI vegetation component
  * **`cache-terrain-{hash}.geojson`**: Cached GVI terrain component  
  * **`cache-ndvi-{hash}.tif`**: Cached NDVI raster
  * **`cache-gvi_combined-{hash}.tif`**: Multi-band raster (Band 1: Vegetation, Band 2: Terrain)
  * Cache files are automatically reused for identical study areas to avoid redundant downloads

---

## Configuration & Troubleshooting

> 🛑 **"No module named geofuse"**

If you see this error, it means the package wasn't linked correctly during setup.

* **Fix:** Rerun the installer script (`install.bat` / `install.sh`) and ensure you see a success message at the end.

> 🛑 **"Conda Not Found"**

If the installer script fails immediately:

* Ensure Miniforge/Anaconda is installed and that its 'Scripts' or 'bin' directory is accessible from your terminal.
* If installed in a custom location, the script will prompt you to paste the correct path.

> 🛑 **"DecompressionBombWarning"**

You may see this warning in the terminal if downloading very high-res Street View images. It is safe to ignore. The toolbox automatically handles large image limits and resizes them for processing.

---

## Built With & References

This project relies on several open-source libraries and public datasets. We gratefully acknowledge:

* **Street View Download:** Adapted from the [streetview](https://github.com/robolyst/streetview) library by `@robolyst`.
* **Semantic Segmentation Model:** [DeepLabV3+](https://github.com/VainF/DeepLabV3Plus-Pytorch/tree/master) architecture implemented via PyTorch.
* **Segmentation Training Data:** The semantic segmentation model was pre-trained on the [Cityscapes Dataset](https://www.cityscapes-dataset.com/).
* **Satellite Imagery Analysis:** Powered by [Google Earth Engine](https://earthengine.google.com/) and [geemap](https://geemap.org).

---

## Citation

If you use GeoFuse in your research, please cite:

> **Sadigh, A. G.** (2025). _GeoFuse: Multimodal Greenspace Profiling Toolbox_. Healthy City Lab, University of Calgary. [https://github.com/Healthy-City-Lab/GeoFuse](https://github.com/Healthy-City-Lab/GeoFuse)

## Credits & Acknowledgments

**Primary Developer and Maintainer:** [Armin Ghayur Sadigh](https://github.com/Armin-GS)

* **Academic Context:** This toolbox was developed as part of a Ph.D. research project at the Healthy City Lab, University of Calgary. It is designed to support ongoing research into environmental determinants of health and urban sensing.

* **Lab:** [Healthy City Lab](https://www.healthycitylab.ca/)
* **Institution:** [University of Calgary](https://ucalgary.ca/)

**License:** GNU General Public License v3.0 (See [LICENSE file](LICENSE))
