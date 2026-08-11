# GeoFuse: Multimodal Greenspace Profiling Toolbox

[![Installation & Test Suite](https://github.com/Healthy-City-Lab/GeoFuse/actions/workflows/setup-tests.yml/badge.svg)](https://github.com/Healthy-City-Lab/GeoFuse/actions/workflows/setup-tests.yml)
[![License](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

**GeoFuse** is a Python toolbox for urban health and greenspace research. It
sources, processes and fuses two complementary greenery signals:

- **Green View Index (GVI)**, eye-level greenery from Google Street View
  panoramas, segmented with DeepLabV3+.
- **NDVI** (Normalized Difference Vegetation Index), overhead greenery from
  Google Earth Engine satellite imagery (Sentinel-2 and Landsat).

It combines them into a single **Composite Greenery Index (CGI)**, optimized
against a health or environmental outcome you supply. The channel weights, each
channel's spatial scale and aggregation statistic, and the functional form are
all discovered from the data: an exhaustive cross-validated sweep picks the
configuration, a Bayesian fit puts credible intervals on the weights, and the
effect is reported on a held-out test set the search never saw.

Run it either as an interactive Streamlit dashboard or as an MPI-parallel CLI
for HPC batch processing.

---

## Quick Start

```bash
# 1. Install (Windows: double-click install.bat instead)
conda install mamba -n base -c conda-forge
chmod +x install.sh && ./install.sh

# 2. Place your segmentation model
cp your_model.pth geofuse/model/best_model.pth

# 3. Launch the web UI
conda activate geofuse
streamlit run ui/app.py
```

---

## Documentation

| Guide                                      | Contents                                                                         |
| ------------------------------------------ | -------------------------------------------------------------------------------- |
| [Features](docs/FEATURES.md)               | What each engine does, what it expects, and why the fusion result is defensible  |
| [Installation](docs/INSTALLATION.md)       | Full setup instructions, segmentation model requirements, and verification tests |
| [Usage](docs/USAGE.md)                     | Web UI walkthrough and CLI reference                                             |
| [Outputs](docs/OUTPUTS.md)                 | All generated files and directory structure                                      |
| [Troubleshooting](docs/TROUBLESHOOTING.md) | Common errors and fixes                                                          |

---

## Citation

If you use GeoFuse in your research, please cite:

> **Ghayur Sadigh, A.** (2025). _GeoFuse: Multimodal Greenspace Profiling Toolbox_. Healthy City Lab, University of Calgary. [https://github.com/Healthy-City-Lab/GeoFuse](https://github.com/Healthy-City-Lab/GeoFuse)

## Credits & Acknowledgments

**Primary Developer:** [Armin Ghayur Sadigh](https://github.com/Armin-GS), Ph.D. research at the [Healthy City Lab](https://www.healthycitylab.ca/), [University of Calgary](https://ucalgary.ca/).

**License:** GNU General Public License v3.0. See [LICENSE](LICENSE).

**Built With & References**:

* **Street View Download:** `geofuse/streetview.py`, a minimal custom port adapted from [streetlevel](https://github.com/sk-zk/streetlevel) (MIT). Uses Google's `SingleImageSearch` protobuf API for metadata and `streetviewpixels-pa.googleapis.com` for tiles.
* **Semantic Segmentation:** [DeepLabV3+](https://github.com/VainF/DeepLabV3Plus-Pytorch/tree/master) via PyTorch.
* **Training Data:** [Cityscapes Dataset](https://www.cityscapes-dataset.com/).
* **Satellite Imagery:** [Google Earth Engine](https://earthengine.google.com/) and [geemap](https://geemap.org).
