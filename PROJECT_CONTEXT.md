# MCE 401 — PROJECT CONTEXT

## 1. Project Identity

**Course:** MCE 401 — Computer Vision and Image Processing  
**Department:** Mechatronics Engineering  
**Institution:** Bowen University  
**Project Title:** What Makes Pixels Similar? Investigating Colour and Spatial Features for K-means Image Segmentation.

**Development Environment:** Antigravity IDE  
**Programming Language:** Python  
**Operating System:** Windows  
**Python Version:** 3.13.14

This is an academic engineering project. The objective is not simply to produce images using Python. It is to implement, investigate, compare and explain K-means clustering as an image segmentation technique.

The project must demonstrate both practical programming ability and an understanding of the computer vision principles behind the implementation.

---

## 2. Engineering Problem and Motivation

Imagine an agricultural robot moving through a farm.

The robot has a camera that captures images of the field. These images contain several visual elements:

- Green vegetation and crops.
- Soil.
- Dry leaves.
- Shadows.
- Stones and other background objects.
- Weeds and other unwanted vegetation.

For the robot to understand its environment, it needs to distinguish different visual regions in the captured images.

One possible approach is image segmentation.

Image segmentation involves dividing an image into meaningful regions or groups of pixels based on selected characteristics.

In this project, we will investigate how K-means clustering can group image pixels based on their colour and spatial characteristics.

The central research question is:

**What makes two pixels similar, and how does changing the features used to represent a pixel affect the quality of image segmentation?**

We will investigate three major representations:

1. Colour information using RGB.
2. Colour information using CIELAB.
3. Colour information combined with spatial coordinates (X and Y).

---

## 3. Main Project Objectives

The project aims to:

1. Understand and implement K-means clustering for image segmentation.
2. Investigate the influence of the number of clusters (K) on segmentation.
3. Compare segmentation using RGB and CIELAB colour spaces.
4. Investigate the effect of adding spatial coordinates to pixel features.
5. Examine how spatial weighting affects segmentation.
6. Measure and compare clustering compactness and execution time.
7. Interpret the results in the context of an agricultural robot.
8. Recommend an appropriate feature representation for a crop-monitoring application.

---

## 4. Understanding the Core Algorithm

K-means is an unsupervised machine learning algorithm.

In image segmentation, every pixel can be represented as a numerical data point.

For example, in RGB colour space:

Pixel = [R, G, B]

K-means groups pixels into K clusters according to their distances in the selected feature space.

The algorithm works through the following process:

1. Select the number of clusters, K.
2. Initialise K cluster centres (centroids).
3. Calculate the distance between each pixel and the cluster centres.
4. Assign each pixel to its nearest cluster.
5. Recalculate the cluster centres.
6. Repeat the assignment and updating process until the stopping criteria are satisfied.

The final output is an image in which pixels have been grouped according to their assigned clusters.

Important: K-means does not inherently understand which cluster represents vegetation, soil, weeds or crops. It only groups numerical feature vectors. Interpretation of the resulting clusters must be done separately.

---

## 5. Experimental Requirements

### Experiment A: Influence of the Number of Clusters
Implement K-means segmentation using RGB colour features.
Test K in {2, 3, 4, 5, 8}.

### Experiment B: RGB versus CIELAB
Repeat colour-based segmentation using CIELAB colour space.
Compare RGB and CIELAB results.

### Experiment C: Adding Spatial Coordinates
Pixel = [R, G, B, λx, λy]
Test λ in {0.1, 0.5, 1.0}.

---

## 6. Dataset Requirements
Candidate dataset: Crop/Weed Field Image Dataset (CWFID) (Haug & Ostermann, 2014/2015).
At least 20 outdoor/agricultural images.

---

## 7. Performance Measurements
A. Clustering Compactness J = Σ ||xi - μci||^2 and J_pixel = J / N.
B. Execution Time (seconds).
C. Reproducibility (controlled seed, criteria, attempts).
