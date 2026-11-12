# Frozen Experimental Protocol: K-Means Agricultural Image Segmentation

**Course:** MCE 401 — Computer Vision and Image Processing  
**Department:** Mechatronics Engineering, Bowen University  
**Project Title:** What Makes Pixels Similar? Investigating Colour and Spatial Features for K-means Image Segmentation  
**Protocol Version:** 1.0 (Frozen)  
**Date:** November 2026  

---

## 1. Experimental Overview and Objectives

This document establishes the frozen experimental protocol for evaluating unsupervised K-means image segmentation across 20 representative agricultural images from the Crop/Weed Field Image Dataset (CWFID). 

The protocol systematically investigates three core independent variables:
1. **Cluster Count ($K$):** Level of feature-space granularity ($K \in \{2, 3, 4, 5, 8\}$).
2. **Color Space Representation:** Perceptual differences between standard RGB and colorimetric CIELAB.
3. **Spatial Feature Integration:** Incorporating spatial pixel coordinates ($X, Y$) under varying spatial weighting factors ($\lambda \in \{0.1, 0.5, 1.0\}$).

Total planned experimental runs:
$$20 \text{ images} \times 25 \text{ configurations} = 500 \text{ clustering runs}$$

---

## 2. Dataset and Preprocessing Specifications

- **Dataset Source:** Crop/Weed Field Image Dataset (CWFID) (Haug & Ostermann, 2015).
- **Selected Cohort:** Exactly 20 outdoor field images listed in [`data/selected_images.txt`](file:///c:/Users/USER/Documents/Bowen%20files/400%20Lvl/MCE%20401/Projects/Task%201/MCE401__KmeansProject/data/selected_images.txt) and documented in [`data/dataset_manifest.csv`](file:///c:/Users/USER/Documents/Bowen%20files/400%20Lvl/MCE%20401/Projects/Task%201/MCE401__KmeansProject/data/dataset_manifest.csv).
- **Original Resolution:** $1296 \times 966$ pixels, 24-bit color (3 channels).
- **Image Resizing Protocol:**
  - The longest dimension is scaled down to **$320\text{ pixels}$** (`MAX_SIDE = 320`), maintaining the original aspect ratio ($320 \times 239$ pixels, $N = 76,480$ pixels).
  - Resizing interpolation: Area interpolation (`cv2.INTER_AREA`), which prevents moiré aliasing and accurately preserves average pixel intensities during downsampling.
  - Raw original files in `dataset-1.0/` remain read-only and unmodified.

---

## 3. Feature Representations and Mathematics

### 3.1 Experiment A: RGB Color-Only Representation
- **Raw Channels:** Input OpenCV BGR converted to RGB:
  $$[R_i, G_i, B_i] = \text{cvtColor}(I_{\text{BGR}}, \text{COLOR\_BGR2RGB})$$
- **Normalization:** Scaled by $\frac{1}{255}$ into $[0.0, 1.0]$:
  $$x_i^{\text{RGB}} = \left[ \frac{R_i}{255}, \frac{G_i}{255}, \frac{B_i}{255} \right] \in [0, 1]^3$$
- **Tested $K$ Values:** $K \in \{2, 3, 4, 5, 8\}$ (5 configurations per image).

### 3.2 Experiment B: CIELAB Color-Only Representation
- **Transformation:** Float32 BGR in $[0.0, 1.0]$ is converted to true floating-point CIELAB using `cv2.COLOR_BGR2LAB`:
  $$[L^*_i, a^*_i, b^*_i] = \text{cvtColor}(I_{\text{BGR\_float}}, \text{COLOR\_BGR2LAB})$$
  Where $L^* \in [0, 100]$, $a^* \in [-127, 127]$, and $b^* \in [-127, 127]$.
- **Normalization:** Scaled by a single common divisor of $100.0$:
  $$x_i^{\text{Lab}} = \left[ \frac{L^*_i}{100}, \frac{a^*_i}{100}, \frac{b^*_i}{100} \right]$$
  *Justification:* Using a uniform scale factor strictly preserves the aspect ratio and perceptual metric ($\Delta E^*_{ab} / 100$) without artificially weighting chromatic channels over lightness.
- **Tested $K$ Values:** $K \in \{2, 3, 4, 5, 8\}$ (5 configurations per image).

### 3.3 Experiment C: RGB Plus Spatial Coordinates (RGB+XY)
- **Spatial Grid:** Normalized pixel coordinates:
  $$x_{n, i} = \frac{x_i}{W - 1} \in [0.0, 1.0], \quad y_{n, i} = \frac{y_i}{H - 1} \in [0.0, 1.0]$$
- **Spatial Weighting ($\lambda$):**
  $$x_{w, i} = \lambda \cdot x_{n, i}, \quad y_{w, i} = \lambda \cdot y_{n, i}$$
- **5D Hybrid Feature Vector:**
  $$x_i^{\text{RGBXY}} = \left[ \frac{R_i}{255}, \frac{G_i}{255}, \frac{B_i}{255}, \lambda x_{n, i}, \lambda y_{n, i} \right] \in \mathbb{R}^5$$
- **Tested Spatial Weights:** $\lambda \in \{0.1, 0.5, 1.0\}$.
- **Tested $K$ Values:** $K \in \{2, 3, 4, 5, 8\}$ for each $\lambda$ (15 configurations per image).

---

## 4. K-Means Algorithm & Execution Parameters

All clustering runs use OpenCV's native C++ implementation (`cv2.kmeans`) under strictly controlled hyperparameters:
- **Initialization Method:** `cv2.KMEANS_PP_CENTERS` (K-means++ algorithm, ensuring probabilistic dispersion of initial seeds).
- **Random Seed:** Fixed to **`42`** via `cv2.setRNGSeed(42)` prior to each execution, guaranteeing 100% deterministic reproducibility.
- **Maximum Iterations:** Fixed to **`100`** (`TERM_CRITERIA_MAX_ITER = 100`).
- **Stopping Epsilon Tolerance:** Fixed to **`0.001`** (`TERM_CRITERIA_EPS = 0.001`), ensuring centroid movements stabilize before termination without premature aborts.
- **Clustering Attempts:** Fixed to **`3`** (`attempts = 3`), running 3 independent initializations and selecting the run with minimum compactness.
- **Sample Processing:** All $N = 76,480$ processed pixels are directly clustered by `cv2.kmeans`.

---

## 5. Measured Dependent Variables

For every run $(image, \text{representation}, K, \lambda)$, record:
1. **Clustering Compactness ($J$):**
   $$J = \sum_{j=1}^K \sum_{x_i \in C_j} \|x_i - \mu_j\|^2$$
   Sum of squared Euclidean distances in the active feature space.
2. **Compactness per Pixel ($J_{\text{pixel}}$):**
   $$J_{\text{pixel}} = \frac{J}{N}$$
   Normalizes for pixel count across resolutions.
3. **Execution Time ($t_{\text{exec}}$):** Measured in seconds via `time.perf_counter()`, strictly isolating the execution of `cv2.kmeans()`.
4. **Cluster Breakdown:** Pixel count, percentage coverage, and mean BGR color per cluster.

---

## 6. Ground-Truth Mask Evaluation Protocol

> [!IMPORTANT]
> **Strict Separation of Unsupervised and Supervised Domains:**
> Ground-truth masks are used **strictly for post-hoc validation and quantitative scoring**. They are **never** used during feature construction, initialization, or cluster optimization.

### 6.1 Verified Ground-Truth Semantics (CWFID)
- **Mask Value $0$:** Foreground Vegetation (Sugar beet crops and weeds).
- **Mask Value $255$:** Background (Soil, stones, and shadows).

### 6.2 Mask-Independent Unsupervised Cluster Interpretation Rule
In an operational agricultural robotics deployment, ground-truth masks are unavailable at runtime. Therefore, cluster interpretation must be completely mask-independent and physically grounded:

1. **Pixel-Level Examination:** After K-means assigns cluster labels to all pixels, examine the original RGB color values of the pixels allocated to each cluster $j$.
2. **Mean Color Calculation:** Compute the empirical mean red ($\bar{R}_j$), green ($\bar{G}_j$), and blue ($\bar{B}_j$) in $[0, 255]$ for cluster $j$.
3. **Excess Green Index ($\text{ExG}$):** Compute the classical Woebbecke et al. (1995) Excess Green index:
   $$\text{ExG}_j = 2\bar{G}_j - \bar{R}_j - \bar{B}_j$$
   *Note:* This rule is applied consistently to all experiments (RGB, CIELAB, and RGB+XY) using the original RGB image, regardless of the feature space used during clustering.
4. **Binary Semantic Classification Rule:**
   $$\text{Semantic Class}(j) = \begin{cases} \text{Vegetation (Foreground)}, & \text{if } \text{ExG}_j > 0 \\ \text{Background (Soil / Shadow)}, & \text{if } \text{ExG}_j \le 0 \end{cases}$$
5. **Predicted Vegetation Mask ($M_{\text{pred}}$):**
   $$M_{\text{pred}}(x, y) = \begin{cases} 1, & \text{if pixel } (x, y) \text{ belongs to any cluster with } \text{ExG} > 0 \\ 0, & \text{otherwise} \end{cases}$$

#### Strict Independence Guarantees:
- Ground-truth masks **never** influence K-means clustering.
- Ground-truth masks **never** influence or determine which clusters are labeled vegetation.
- The $\text{ExG} > 0$ threshold is fixed *a priori* based on colorimetric vegetative reflectance principles and is never tuned against the masks.
- Reference masks are used solely post-prediction to compute benchmark metrics (IoU, Precision, Recall, F1, Accuracy).

#### Methodological Limitations of Fixed ExG Thresholding:
- **Transitional/Boundary Pixels:** Clusters capturing blurred leaf edges or mossy/wet soil clods may exhibit slight positive ExG, causing minor false positives ($FP$).
- **Severe Shadowing:** Heavily shadowed leaves with low signal-to-noise ratio in green reflectance may have $\text{ExG} \le 0$, resulting in false negatives ($FN$).
- **Non-Vegetative Green Objects:** While absent in CWFID, synthetic green field markers would be grouped as vegetation under pure ExG.

### 6.3 Evaluation Metrics Formulation
Let $A$ be the predicted vegetation region and $B$ be the ground-truth vegetation region:

1. **Intersection over Union (IoU / Jaccard Index):**
   $$\text{IoU} = \frac{|A \cap B|}{|A \cup B|} = \frac{TP}{TP + FP + FN}$$
2. **Precision:**
   $$\text{Precision} = \frac{TP}{TP + FP}$$
   Measures the proportion of segmented vegetation pixels that are genuinely vegetation.
3. **Recall (Sensitivity):**
   $$\text{Recall} = \frac{TP}{TP + FN}$$
   Measures the proportion of true vegetation correctly captured by the cluster.
4. **F1-Score (Dice Coefficient):**
   $$\text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$
5. **Pixel Accuracy:**
   $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

---

## 7. Output Directory Structure

The full experimental pipeline will store all outputs systematically:
```
outputs/
├── tables/
│   ├── dataset_manifest.csv
│   ├── experiment_results.csv
│   ├── summary_statistics.csv
│   └── ground_truth_metrics.csv
├── segmentations/
│   └── {stem}_{feature_space}_lambda{lam}_K{k}.png
├── figures/
│   ├── {stem}_RGB_K_panel.jpg
│   ├── {stem}_Lab_K_panel.jpg
│   ├── {stem}_RGBXY_lambda{lam}_K4.jpg
│   └── dataset_qc_panel.png
└── plots/
    ├── compactness_vs_K.png
    └── execution_time_vs_K.png
```
