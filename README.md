# GeoFuse: Multimodal Greenspace Profiling Toolbox

[![Installation & Test Suite](https://github.com/Healthy-City-Lab/GeoFuse/actions/workflows/setup-tests.yml/badge.svg)](https://github.com/Healthy-City-Lab/GeoFuse/actions/workflows/setup-tests.yml)
[![License](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![OS](https://img.shields.io/badge/OS-Linux|Windows|macOS-blue)](#)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Conda Env](https://img.shields.io/badge/Conda%20Env-geofuse-342B029.svg?logo=anaconda&logoColor=white)](#)
[![MPI](https://img.shields.io/badge/MPI-Parallel-blue)](#)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=Streamlit&logoColor=white)](#)
[![Google Earth Engine](https://img.shields.io/badge/Google%20Earth%20Engine-4285F4?logo=google-earth&logoColor=white)](#)

**GeoFuse** is a Python toolbox for urban health and greenspace research. It automates the sourcing, processing, and fusion of **Green View Index (GVI)** — measured from street-level imagery using DeepLabV3+ segmentation — and **NDVI** — fetched from Google Earth Engine satellite data — then uses **Bayesian optimization** (Optuna) to fuse them into a composite greenery index optimized against health or environmental outcomes.

Two deployment modes are available: an interactive **Streamlit web dashboard** for exploratory analysis, and an **MPI-parallel CLI** for large-scale HPC batch processing.

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

| Guide | Contents |
|-------|----------|
| [Features](docs/FEATURES.md) | Detailed breakdown of GVI, NDVI, Fusion, and HPC capabilities |
| [Installation](docs/INSTALLATION.md) | Full setup instructions, segmentation model requirements, and verification tests |
| [Usage](docs/USAGE.md) | Web UI walkthrough and CLI reference |
| [Outputs](docs/OUTPUTS.md) | All generated files and directory structure |
| [Troubleshooting](docs/TROUBLESHOOTING.md) | Common errors and fixes |

---

## Citation

If you use GeoFuse in your research, please cite:

> **Ghayur Sadigh, A.** (2025). _GeoFuse: Multimodal Greenspace Profiling Toolbox_. Healthy City Lab, University of Calgary. [https://github.com/Healthy-City-Lab/GeoFuse](https://github.com/Healthy-City-Lab/GeoFuse)

## Credits & Acknowledgments

**Primary Developer:** [Armin Ghayur Sadigh](https://github.com/Armin-GS) — Ph.D. research at the [Healthy City Lab](https://www.healthycitylab.ca/), [University of Calgary](https://ucalgary.ca/).

**License:** GNU General Public License v3.0 — see [LICENSE](LICENSE).
