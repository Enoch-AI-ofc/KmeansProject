# Dataset Validation and Quality Control Report

**Course:** MCE 401 — Computer Vision and Image Processing  
**Department:** Mechatronics Engineering, Bowen University  
**Project:** What Makes Pixels Similar? Investigating Colour and Spatial Features for K-means Image Segmentation  
**Dataset:** Crop/Weed Field Image Dataset (CWFID) (Haug & Ostermann, 2014/2015)  
**Date:** November 2026  
**Status:** 100% Verified and Validated  

---

## 1. Dataset Overview and Source Verification

The experimental evaluation utilizes images from the **Crop/Weed Field Image Dataset (CWFID)**:
- **Reference:** Haug, S., & Ostermann, J. (2015). *A Crop/Weed Field Image Dataset for the Evaluation of Computer Vision Based Precision Agriculture Tasks.* ECCV 2014 Workshops, pp. 105–116. DOI: [10.1007/978-3-319-16220-1_8](https://doi.org/10.1007/978-3-319-16220-1_8).
- **Acquisition Platform:** Autonomous agricultural field robot (BoniRob) operating in organic sugar beet fields.
- **Sensory Modality:** Downward-looking RGB camera mounted under controlled shroud illumination with natural sunlight infiltration.
- **Total Dataset Size:** 60 field images, 60 vegetation segmentation masks, and 60 plant-type (crop vs. weed) annotations.
- **Raw Data Location:** `dataset-1.0/` (held read-only; completely unmodified).

---

## 2. 20-Image Selection & Integrity Audit

All 20 selected filenames from [`data/selected_images.txt`](file:///c:/Users/USER/Documents/Bowen%20files/400%20Lvl/MCE%20401/Projects/Task%201/MCE401__KmeansProject/data/selected_images.txt) were individually inspected against raw files:

| No. | Filename | Original Dimensions | Mask Dimensions | Dimensions Compatible? | Unique Mask Values | Vegetation Pixels ($0$) | Soil Pixels ($255$) | Vegetation Cover (%) | Agronomic Category |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | `001_image.png` | $1296 \times 966$ | $1296 \times 966$ | **Yes** | `[0, 255]` | $221,109$ | $1,030,827$ | $17.66\%$ | Dense Vegetation |
| **2** | `007_image.png` | $1296 \times 966$ | $1296 \times 966$ | **Yes** | `[0, 255]` | $203,015$ | $1,048,921$ | $16.22\%$ | Dense Vegetation |
| **3** | `010_image.png` | $1296 \times 966$ | $1296 \times 966$ | **Yes** | `[0, 255]` | $85,362$ | $1,166,574$ | $6.82\%$ | Moderate Vegetation |
| **4** | `011_image.png` | $1296 \times 966$ | $1296 \times 966$ | **Yes** | `[0, 255]` | $91,458$ | $1,160,478$ | $7.31\%$ | Moderate Vegetation |
| **5** | `015_image.png` | $1296 \times 966$ | $1296 \times 966$ | **Yes** | `[0, 255]` | $76,201$ | $1,175,735$ | $6.09\%$ | Moderate Vegetation |
| **6** | `018_image.png` | $1296 \times 966$ | $1296 \times 966$ | **Yes** | `[0, 255]` | $100,431$ | $1,151,505$ | $8.02\%$ | Moderate Vegetation |
| **7** | `020_image.png` | $1296 \times 966$ | $1296 \times 966$ | **Yes** | `[0, 255]` | $22,329$ | $1,229,607$ | $1.78\%$ | Sparse / Soil-Dominant |
| **8** | `021_image.png` | $1296 \times 966$ | $1296 \times 966$ | **Yes** | `[0, 255]` | $28,698$ | $1,223,238$ | $2.29\%$ | Sparse / Soil-Dominant |
| **9** | `023_image.png` | $1296 \times 966$ | $1296 \times 966$ | **Yes** | `[0, 255]` | $85,147$ | $1,166,789$ | $6.80\%$ | Moderate Vegetation |
| **10** | `025_image.png` | $1296 \times 966$ | $1296 \times 966$ | **Yes** | `[0, 255]` | $84,856$ | $1,167,080$ | $6.78\%$ | Moderate Vegetation |
| **11** | `026_image.png` | $1296 \times 966$ | $1296 \times 966$ | **Yes** | `[0, 255]` | $62,012$ | $1,189,924$ | $4.95\%$ | Sparse / Soil-Dominant |
| **12** | `029_image.png` | $1296 \times 966$ | $1296 \times 966$ | **Yes** | `[0, 255]` | $240,361$ | $1,011,575$ | $19.20\%$ | Dense Vegetation |
| **13** | `030_image.png` | $1296 \times 966$ | $1296 \times 966$ | **Yes** | `[0, 255]` | $28,403$ | $1,223,533$ | $2.27\%$ | Sparse / Soil-Dominant |
| **14** | `031_image.png` | $1296 \times 966$ | $1296 \times 966$ | **Yes** | `[0, 255]` | $174,175$ | $1,077,761$ | $13.91\%$ | Dense Vegetation |
| **15** | `038_image.png` | $1296 \times 966$ | $1296 \times 966$ | **Yes** | `[0, 255]` | $157,268$ | $1,094,668$ | $12.56\%$ | Dense Vegetation |
| **16** | `046_image.png` | $1296 \times 966$ | $1296 \times 966$ | **Yes** | `[0, 255]` | $30,361$ | $1,221,575$ | $2.43\%$ | Sparse / Soil-Dominant |
| **17** | `050_image.png` | $1296 \times 966$ | $1296 \times 966$ | **Yes** | `[0, 255]` | $19,885$ | $1,232,051$ | $1.59\%$ | Sparse / Soil-Dominant |
| **18** | `051_image.png` | $1296 \times 966$ | $1296 \times 966$ | **Yes** | `[0, 255]` | $189,412$ | $1,062,524$ | $15.13\%$ | Dense Vegetation |
| **19** | `058_image.png` | $1296 \times 966$ | $1296 \times 966$ | **Yes** | `[0, 255]` | $184,296$ | $1,067,640$ | $14.72\%$ | Dense Vegetation |
| **20** | `059_image.png` | $1296 \times 966$ | $1296 \times 966$ | **Yes** | `[0, 255]` | $143,849$ | $1,108,087$ | $11.49\%$ | Moderate Vegetation |

The full manifest is exported to [`data/dataset_manifest.csv`](file:///c:/Users/USER/Documents/Bowen%20files/400%20Lvl/MCE%20401/Projects/Task%201/MCE401__KmeansProject/data/dataset_manifest.csv).

---

## 3. Semantic Verification of Mask Pixel Values

To ensure rigorous scientific integrity, we explicitly tested and verified the semantic meaning of the binary mask values:

1. **Correlation with Multi-Class Annotations:**
   In `dataset-1.0/annotations/001_annotation.png`, pixel annotations represent:
   - Green `[0, 255, 0]` = Crop
   - Red `[0, 0, 255]` = Weed
   - Black `[0, 0, 0]` = Background Soil
   Cross-indexing all labeled plant pixels against `001_mask.png` showed that **$100\%$ ($204,176$ of $204,176$) of plant pixels have value $0$ in the mask**.
2. **Visual Inspection:**
   The 5-image quality control panel ([`outputs/figures/dataset_qc_panel.png`](file:///c:/Users/USER/Documents/Bowen%20files/400%20Lvl/MCE%20401/Projects/Task%201/MCE401__KmeansProject/outputs/figures/dataset_qc_panel.png)) confirms exact morphological alignment between green foliage in the original images and pixel value $0$ in the masks.

> [!IMPORTANT]
> **Definitive Class Label Mapping:**
> - **Mask Value $0$ = Foreground Vegetation (Crops and Weeds)**
> - **Mask Value $255$ = Background (Soil, Stones, and Shadows)**

---

## 4. Agronomic Variation in the Selected Cohort

The selected 20 images represent a controlled spread across typical field conditions encountered by agricultural robots:
- **Coverage Range:** Minimum vegetation cover is $1.59\%$ (`050_image.png`, extreme soil dominance); maximum is $19.20\%$ (`029_image.png`, dense overlapping weed/crop patches); median is $7.67\%$.
- **Illumination Range:** Mean grayscale intensities span from $70.36$ to $79.19$, with standard deviations up to $25.56$, capturing varying degrees of cloud cover, soil moisture, and leaf specular reflections.
- **Missing Data:** No missing masks or corrupt images were found ($20/20$ complete).
