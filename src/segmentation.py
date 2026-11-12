"""MCE 401 — Computer Vision and Image Processing
Task 1: K-means Image Segmentation

Core segmentation module providing:
- Image loading and consistent resizing
- Feature space extraction (RGB, CIELAB, RGB + XY)
- K-means clustering with controlled random seeding
- Segmented image reconstruction (mean cluster color coloring)
"""

from pathlib import Path
import time
import cv2
import numpy as np


def load_image(image_path: Path, max_side: int = 320):
    """Load an image from disk and resize it consistently while preserving aspect ratio.
    
    Args:
        image_path: Path to the image file.
        max_side: Maximum allowed dimension (width or height). If 0 or None, no resizing is applied.
        
    Returns:
        tuple: (original_bgr, resized_bgr, original_width, original_height)
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    orig_h, orig_w = image.shape[:2]
    
    if max_side is not None and max_side > 0 and max(orig_h, orig_w) > max_side:
        scale = max_side / float(max(orig_h, orig_w))
        new_w = max(1, int(round(orig_w * scale)))
        new_h = max(1, int(round(orig_h * scale)))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        resized = image.copy()
        
    return image, resized, orig_w, orig_h


def build_features(image_bgr: np.ndarray, feature_space: str = "RGB", lam: float = 0.0) -> np.ndarray:
    """Construct normalized feature vectors for K-means clustering.
    
    Args:
        image_bgr: Resized OpenCV image in BGR uint8 format.
        feature_space: One of 'RGB', 'Lab', or 'RGBXY'.
        lam: Spatial weighting factor (lambda) used only when feature_space == 'RGBXY'.
        
    Returns:
        np.ndarray: C-contiguous 2D float32 array of shape (num_pixels, num_features).
    """
    h, w = image_bgr.shape[:2]
    
    if feature_space == "RGB":
        # Convert BGR -> RGB and normalize to [0.0, 1.0]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        features = rgb
        
    elif feature_space == "Lab":
        # Convert BGR uint8 to float32 in [0.0, 1.0]
        bgr_float = image_bgr.astype(np.float32) / 255.0
        # Convert using cv2.COLOR_BGR2LAB to true floating-point CIELAB
        # (L* in [0, 100], a* in [-127, 127], b* in [-127, 127])
        lab = cv2.cvtColor(bgr_float, cv2.COLOR_BGR2LAB)
        # Divide all three Lab coordinates by 100 using one common scale factor
        features = lab / 100.0
        
    elif feature_space == "RGBXY":
        # Normalized RGB in [0, 1]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        # Spatial grid normalized to [0, 1]
        y_coords, x_coords = np.indices((h, w), dtype=np.float32)
        x_norm = x_coords / max(w - 1, 1)
        y_norm = y_coords / max(h - 1, 1)
        
        # Apply spatial weighting parameter lambda
        x_weighted = np.float32(lam) * x_norm
        y_weighted = np.float32(lam) * y_norm
        
        # Stack into 5-dimensional feature tensor: [R, G, B, lambda*x, lambda*y]
        features = np.dstack([rgb, x_weighted, y_weighted])
        
    else:
        raise ValueError(f"Unknown feature space: {feature_space}. Must be 'RGB', 'Lab', or 'RGBXY'.")
        
    # Reshape (H, W, D) -> (H * W, D)
    samples = features.reshape(-1, features.shape[-1]).astype(np.float32)
    return np.ascontiguousarray(samples)


def run_kmeans(samples: np.ndarray,
               k: int,
               random_seed: int = 42,
               max_iter: int = 100,
               epsilon: float = 0.001,
               attempts: int = 3):
    """Execute OpenCV K-means clustering with controlled initialisation and timing.
    
    Args:
        samples: (N, D) float32 feature array.
        k: Number of clusters.
        random_seed: Random seed for OpenCV RNG.
        max_iter: Maximum iterations before stopping.
        epsilon: Minimum centroid movement before stopping.
        attempts: Number of clustering attempts with different initial centers.
        
    Returns:
        tuple: (compactness, labels, centres, elapsed_time_seconds)
    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, epsilon)
    cv2.setRNGSeed(random_seed)
    
    start_time = time.perf_counter()
    compactness, labels, centres = cv2.kmeans(
        samples,
        k,
        None,
        criteria,
        attempts,
        cv2.KMEANS_PP_CENTERS
    )
    elapsed_time = time.perf_counter() - start_time
    
    return float(compactness), labels, centres, elapsed_time


def reconstruct_segmentation(original_resized_bgr: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Reconstruct a segmented BGR image by coloring each cluster with its mean BGR color.
    
    Args:
        original_resized_bgr: The resized BGR image before clustering.
        labels: 1D array of cluster labels assigned to each pixel.
        k: Total number of clusters.
        
    Returns:
        np.ndarray: Reconstructed BGR uint8 image.
    """
    h, w = original_resized_bgr.shape[:2]
    labels_2d = labels.reshape(h, w)
    output = np.zeros_like(original_resized_bgr)
    
    for cluster_id in range(k):
        mask = (labels_2d == cluster_id)
        if np.any(mask):
            mean_colour = original_resized_bgr[mask].mean(axis=0)
            mean_colour = np.clip(mean_colour, 0, 255).astype(np.uint8)
            output[mask] = mean_colour
            
    return output
