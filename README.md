# K-means Image Segmentation


**Project Title:** *What Makes Pixels Similar? Investigating Colour and Spatial Features for K-means Image Segmentation*

---

## 📌 Overview

This project implements and investigates **K-means clustering** as an image segmentation technique, motivated by an agricultural robotics scenario where a robot camera must distinguish crops, soil, weeds, and shadows in field images.

The central research question is:

> **What makes two pixels similar, and how does changing the features used to represent a pixel affect the quality of image segmentation?**

Three feature representations are systematically compared across a 20-image agricultural dataset (CWFID):

| Representation | Features per pixel |
|---|---|
| RGB | `[R, G, B]` |
| CIELAB | `[L*, a*, b*]` |
| RGB + Spatial (RGBXY) | `[R, G, B, λx, λy]` |

---

## 🗂️ Project Structure

```
MCE401__KmeansProject/
│
├── src/                          # All Python source code
│   ├── segmentation.py           # Core module: image loading, feature extraction, K-means, reconstruction
│   ├── pilot_test.py             # Experiment A & B pilot — single image, visual check
│   ├── lab_pilot_and_repeatability.py  # CIELAB colour space pilot + repeatability test
│   ├── run_experiments.py        # Full experiment runner (Experiments A, B, C)
│   ├── validate_dataset.py       # Dataset integrity checks
│   ├── validate_termination.py   # K-means termination criteria sensitivity analysis
│   ├── reevaluate_threshold.py   # Re-runs threshold analysis with updated parameters
│   └── generate_final_figures.py # Generates publication-quality summary figures
│
├── data/
│   ├── images/                   # 20 selected agricultural field images (PNG)
│   ├── dataset_manifest.csv      # Image metadata and selection rationale
│   └── selected_images.txt       # List of selected image IDs
│
├── dataset-1.0/                  # CWFID dataset annotations and train/test split
│   ├── annotations/              # Ground-truth annotation PNGs + YAML metadata
│   └── train_test_split.yaml
│
├── outputs/
│   ├── segmentations/            # Early pilot segmentation PNGs
│   ├── full_experiment/
│   │   ├── segmentations/        # All segmentation output images (Exp A, B, C)
│   │   ├── labels/               # Cluster label arrays (.npy) — excluded from git
│   │   ├── figures/              # Final summary figures and plots
│   │   └── results.csv           # Numerical results (compactness, timing)
│   └── tables/
│       └── termination_sensitivity_results.csv
│
├── requirements.txt              # Python dependencies (pip-frozen)
├── PROJECT_CONTEXT.md            # Full project brief and experimental requirements
└── .gitignore
```

---

## ⚙️ Setup & Installation

### 1. Prerequisites
- Python 3.13+
- `pip`

### 2. Clone the repository
```bash
git clone https://github.com/<your-username>/MCE401__KmeansProject.git
cd MCE401__KmeansProject
```

### 3. Create a virtual environment and install dependencies
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

---

## 🔬 Experiments

### Experiment A — Influence of Number of Clusters (K)
K-means segmentation using **RGB** colour features.  
K values tested: `{2, 3, 4, 5, 8}`

### Experiment B — RGB vs. CIELAB
Segmentation repeated using **CIELAB** colour space.  
Comparison based on compactness (J/N) and visual output.

### Experiment C — Adding Spatial Coordinates (RGBXY)
Pixel represented as `[R, G, B, λx, λy]`.  
Spatial weighting λ tested: `{0.1, 0.5, 1.0}`

---

## ▶️ Running the Code

Run experiments from the project root:

```bash
# Full experiment suite (A, B, C)
python src/run_experiments.py

# Generate final summary figures
python src/generate_final_figures.py

# Termination criteria sensitivity
python src/validate_termination.py
```

All outputs are saved to the `outputs/` directory.

---

## 📦 Dependencies

| Package | Version |
|---|---|
| `numpy` | 2.5.4 |
| `opencv-python` | 5.0.0.93 |
| `matplotlib` | 3.11.2 |
| `pandas` | 3.1.1 |
| `pillow` | 13.0.0 |

See [`requirements.txt`](requirements.txt) for the full pinned list.

---

## 📊 Performance Metrics

| Metric | Description |
|---|---|
| **Compactness (J)** | Total within-cluster sum of squared distances: `Σ ‖xᵢ − μ_cᵢ‖²` |
| **J/N (per-pixel)** | Compactness normalised by number of pixels |
| **Execution Time** | Wall-clock time in seconds |
| **Reproducibility** | Controlled via fixed random seed (`seed=42`) and fixed termination criteria |

---

## 📄 Dataset

**Crop/Weed Field Image Dataset (CWFID)**  
Haug, S. & Ostermann, J. (2014/2015)  
20 outdoor agricultural field images selected from the full dataset.

---

## 🏫 Academic Context

This project was developed as **Task 1** of MCE 401 (Computer Vision and Image Processing) in the Department of Mechatronics Engineering, Bowen University.

It demonstrates both practical Python/OpenCV programming skills and an understanding of the computer vision principles underlying K-means clustering for image segmentation.

