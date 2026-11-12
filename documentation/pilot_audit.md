# Technical Audit and Validation Report: K-Means Segmentation Pilot (Corrected)

**Course:** MCE 401 — Computer Vision and Image Processing  
**Department:** Mechatronics Engineering, Bowen University  
**Project Title:** What Makes Pixels Similar? Investigating Colour and Spatial Features for K-means Image Segmentation  
**Document:** Pilot Technical Audit, Methodology Corrections & Convergence Validation  
**Date:** November 2026  
**Status:** Audited, Corrected & Validated  

---

## 1. Executive Summary

This report documents the rigorous technical audit and methodological corrections applied to the initial K-means image segmentation pipeline implemented in [`src/segmentation.py`](file:///c:/Users/USER/Documents/Bowen%20files/400%20Lvl/MCE%20401/Projects/Task%201/MCE401__KmeansProject/src/segmentation.py). 

The audit reviewed 11 core mathematical, algorithmic, and software components. Two critical corrections were identified and resolved:
1. **Convergence Criterion & Epsilon Tolerance:** The initial stopping epsilon threshold ($\epsilon = 0.2$) represents a cluster-center movement tolerance. Because centroid movements between early iterations in normalized feature space quickly become small, $\epsilon = 0.2$ triggers premature termination before adequate cluster boundary optimization. Setting the movement tolerance to $\epsilon = 0.001$ ensures proper convergence while preserving high computational throughput.
2. **Perceptually Consistent Floating-Point CIELAB:** The initial 8-bit Lab normalization created an asymmetric dynamic range ($L \in [0, 1]$ vs. $a, b \in [-1, 1]$), artificially magnifying chromatic distances by $4\times$ in Euclidean metric calculations. This has been updated to a true floating-point CIELAB transformation with a uniform scale factor of $100.0$, preserving isotropic perceptual color differences ($\Delta E^*_{ab}$).

---

## 2. Component-by-Component Audit (Task A)

| No. | Audit Item | Implementation in `src/segmentation.py` | Audit Finding & Assessment | Status |
| :---: | :--- | :--- | :--- | :---: |
| **1** | **BGR-to-RGB Conversion** | `cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)` | **Correct.** OpenCV reads images in BGR byte order; explicit conversion ensures channel 0 is Red, 1 is Green, and 2 is Blue. | Verified |
| **2** | **RGB Normalisation** | `.astype(np.float32) / 255.0` | **Correct.** Scales 8-bit integer intensities to continuous $[0.0, 1.0]$ float32 values, preventing integer overflow during distance calculation. | Verified |
| **3** | **CIELAB Conversion & Normalisation** | Float32 BGR $\to$ `cv2.COLOR_BGR2LAB` $\to$ divided uniformly by $100.0$ | **Corrected.** Converts BGR to true CIE $L^*a^*b^*$ and applies a single common divisor ($100.0$), maintaining perceptual uniformity ($\Delta E^*_{ab}$). | Corrected & Verified |
| **4** | **Spatial Coordinate Normalisation** | $x_n = x / (W-1)$, $y_n = y / (H-1)$ | **Correct.** Uses `np.indices((H, W))` and normalizes both dimensions strictly to $[0.0, 1.0]$, independent of aspect ratio or pixel count. | Verified |
| **5** | **Lambda ($\lambda$) Weighting** | $x_w = \lambda \cdot x_n$, $y_w = \lambda \cdot y_n$ | **Correct.** Appropriately modulates the relative contribution of spatial coordinates in the 5D feature space $[R, G, B, \lambda x, \lambda y]$. | Verified |
| **6** | **K-means++ Initialisation** | `cv2.KMEANS_PP_CENTERS` | **Correct.** Implements Arthur & Vassilvitskii (2007) probabilistic centroid seeding, ensuring well-separated initial cluster seeds. | Verified |
| **7** | **Random Seed Control** | `cv2.setRNGSeed(random_seed)` | **Correct.** Sets OpenCV internal C++ RNG seed prior to `cv2.kmeans()`, ensuring deterministic repeatability across identical runs. | Verified |
| **8** | **Stopping Criteria** | `(TERM_CRITERIA_EPS + MAX_ITER, 100, 0.001)` | **Corrected.** Movement tolerance $\epsilon = 0.001$ prevents premature termination on normalized data while bounding runtime at 100 iterations. | Corrected & Verified |
| **9** | **Execution-Time Measurement** | `time.perf_counter()` wrapping `cv2.kmeans` | **Correct.** Isolates algorithmic execution time from disk I/O, resizing, and rendering. Uses high-resolution OS timer. | Verified |
| **10** | **Compactness Calculation** | $J = \sum_{j} \sum_{i} \|x_i - \mu_j\|^2$, $J_{\text{pixel}} = J / N$ | **Correct.** Evaluates total within-cluster sum-of-squared errors ($J$) and compactness per pixel ($J_{\text{pixel}}$) within the designated feature space. | Verified |
| **11** | **Cluster Reconstruction** | `image[mask].mean(axis=0)` per cluster | **Correct.** Paints each pixel with the empirical mean BGR color of its assigned cluster, preserving natural agronomic tones for visual inspection. | Verified |

---

## 3. Convergence Sensitivity & Termination Validation (Task B)

### 3.1 Experimental Setup and Metric Clarification
Tests were executed on image `001_image.png` using [`src/validate_termination.py`](file:///c:/Users/USER/Documents/Bowen%20files/400%20Lvl/MCE%20401/Projects/Task%201/MCE401__KmeansProject/src/validate_termination.py):
- **Image:** `001_image.png` (resized to $320 \times 239$, $N = 76,480$ pixels)
- **Feature Space:** Normalized RGB $[0.0, 1.0]$
- **Cluster Count ($K$):** 3
- **Random Seed:** 42 | **Attempts:** 3 | **Max Iterations:** 100
- **Epsilon Thresholds Tested:** $[0.2, 0.01, 0.001, 0.0001]$

> [!NOTE]
> **OpenCV API Clarification on Iteration Counts:**
> OpenCV's Python wrapper for `cv2.kmeans()` returns a 3-element tuple: `(compactness, labels, centres)`. It **does not natively return the number of iterations executed**. Any iteration counts previously mentioned were estimated via an external, instrumented step-search script. To maintain strict scientific integrity, only the directly measured physical quantities ($J$, $J_{\text{pixel}}$, and execution time) are presented in the primary verification record.

### 3.2 Measured Quantitative Results

| Epsilon Movement Tolerance ($\epsilon$) | Measured Compactness ($J$) | Compactness / Pixel ($J_{\text{pixel}}$) | Median Time (s) | Marginal $\Delta J$ Relative to Coarser Setting |
| :---: | :---: | :---: | :---: | :---: |
| **0.2** (Coarse Tolerance) | $691.0499$ | $0.009036$ | $0.0534$ | Baseline |
| **0.01** | $614.8571$ | $0.008039$ | $0.1062$ | $-76.1928$ ($-11.03\%$) |
| **0.001** (Provisional Setting) | **613.5188** | **0.008022** | **0.1854** | **-1.3383** (**-0.22%**) |
| **0.0001** (Ultra-Fine Tolerance) | $613.4826$ | $0.008021$ | $0.3272$ | $-0.0362$ ($-0.006\%$) |

### 3.3 Physical Interpretation of Epsilon ($\epsilon$)
In `cv2.kmeans()`, the parameter `epsilon` defines the **maximum allowable cluster-center displacement** between consecutive iterations:
$$\max_{j \in \{1 \dots K\}} \|\mu_j^{(t+1)} - \mu_j^{(t)}\| < \epsilon$$
When centroids move by less than $\epsilon$, the algorithm declares convergence and terminates.
- At $\epsilon = 0.2$, the displacement threshold is relatively large compared to the scale of normalized color features. Early iterations often produce centroid movements just under $0.2$, causing the routine to stop prematurely.
- At $\epsilon = 0.001$, centroids must stabilize until inter-iteration movement is under $0.001$ units, ensuring that clusters reach a stable local minimum.

### 3.4 Convergence Recommendation: $\epsilon = 0.001$ vs. $\epsilon = 0.0001$
- Tightening $\epsilon$ from $0.001$ to $0.0001$ yields a negligible compactness improvement of just $0.0362$ across $76,480$ pixels ($0.0000005$ per pixel), with zero change in the number of segmented vegetation pixels ($10,765$ pixels, $14.08\%$).
- However, execution time increases by $\sim 76\%$ (from $0.1854\text{ s}$ to $0.3272\text{ s}$).
- Over a 500-run experiment ($20\text{ images} \times 25\text{ configurations}$), using $\epsilon = 0.0001$ would add substantial cumulative runtime with no observable difference in visual segmentation quality or cluster semantics.
- **Recommendation:** **$\epsilon = 0.001$** provides a practical, defensible engineering balance between convergence depth and computational cost.

---

## 4. Evaluation and Correction of CIELAB (Task C)

### 4.1 Explanation of the Former 8-Bit Normalisation
The initial code used:
```python
lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
lab[:, :, 0] /= 255.0               # L in [0, 1] (span = 1.0)
lab[:, :, 1] = (lab[:, :, 1] - 128.0) / 127.0  # a in [-1, 1] (span = 2.0)
lab[:, :, 2] = (lab[:, :, 2] - 128.0) / 127.0  # b in [-1, 1] (span = 2.0)
```
- **Methodological Flaw:** This scaling produced an anisotropic feature space where chromatic channels $a$ and $b$ spanned $[-1, 1]$ (width $2.0$) while lightness $L$ spanned $[0, 1]$ (width $1.0$). In squared Euclidean distance:
  $$d^2 = (\Delta L)^2 + (\Delta a)^2 + (\Delta b)^2$$
  Because $\Delta a$ and $\Delta b$ can reach up to $2.0$, chromatic differences carried up to **$4\times$ the weight of lightness differences**. This distorted the perceptual uniformity that CIELAB was designed to offer.

### 4.2 Updated Floating-Point CIELAB Implementation
To restore perceptual isotropy, `src/segmentation.py` was updated to:
```python
# Convert BGR uint8 to float32 in [0.0, 1.0]
bgr_float = image_bgr.astype(np.float32) / 255.0
# Convert using cv2.COLOR_BGR2LAB to true floating-point CIELAB
# (L* in [0, 100], a* in [-127, 127], b* in [-127, 127])
lab = cv2.cvtColor(bgr_float, cv2.COLOR_BGR2LAB)
# Divide all three Lab coordinates by 100 using one common scale factor
features = lab / 100.0
```
- **Scientific Justification:** Dividing all three coordinates by the same constant factor ($100.0$) scales the values to manageable numerical magnitudes ($L^* \in [0, 1]$, $a^*, b^* \in [-1.27, 1.27]$) while **strictly preserving the aspect ratio and perceptual metric ($\Delta E^*_{ab} / 100$)**.

---

## 5. Mathematical Interpretation of Clustering Compactness ($J$)

OpenCV calculates compactness as:
$$J = \sum_{j=1}^K \sum_{x_i \in C_j} \|x_i - \mu_j\|^2$$
It measures the sum of squared Euclidean distances between each pixel's feature vector $x_i$ and its assigned centroid $\mu_j$ **in the specific coordinate space used during clustering**:
- In **RGB**, features reside in $[0, 1]^3$ (dimension $D=3$).
- In **CIELAB**, features represent scaled colorimetric coordinates $(L^*/100, a^*/100, b^*/100)$ (dimension $D=3$).
- In **RGB+XY**, features reside in a 5-dimensional hybrid space $[R, G, B, \lambda x, \lambda y]$ (dimension $D=5$).

> [!WARNING]
> **Methodological Boundary:**
> Because the physical units, coordinate ranges, and geometric volumes differ between feature spaces, **raw compactness ($J$) values from RGB and CIELAB cannot be compared directly as quality scores**. A lower $J$ in one space does not prove superior segmentation over another. Compactness is only valid for comparing different $K$ values within the *same* feature representation.

---

## 6. Pilot Repeatability Test Results

A repeatability test was executed on `001_image.png` using floating-point CIELAB ($K=3$, $\epsilon=0.001$, seed=42, 3 attempts) across two consecutive runs:
- **Compactness Run 1:** $731.55034837$
- **Compactness Run 2:** $731.55034837$ ($\Delta J = 0.0$)
- **Centroids Difference:** Exactly $0.0000000000\text{ units}$ across all dimensions.
- **Pixel Label Agreement:** $76,480 / 76,480\text{ pixels}$ ($100.00\%$ bitwise agreement).
- **Conclusion:** The pipeline is completely deterministic and reproducible.
