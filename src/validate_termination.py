"""MCE 401 — Computer Vision and Image Processing
Task B: Termination Criteria & Convergence Sensitivity Validation

This script tests the convergence sensitivity of OpenCV K-means clustering
on agricultural image '001_image.png' under varying epsilon thresholds:
[0.2, 0.01, 0.001, 0.0001].

Experimental Parameters (Controlled):
- Image: data/images/001_image.png (resized to max_side=320)
- Feature Space: Normalized RGB [0.0, 1.0] (3-dimensional)
- K: 3 clusters
- Random Seed: 42
- Max Iterations: 100
- Attempts: 3
- Initialization: cv2.KMEANS_PP_CENTERS
"""

from pathlib import Path
import time
import cv2
import numpy as np
import pandas as pd
from segmentation import load_image, build_features


def run_kmeans_with_epsilon(samples: np.ndarray,
                           k: int,
                           epsilon: float,
                           max_iter: int = 100,
                           attempts: int = 3,
                           random_seed: int = 42):
    """Run K-means with specified epsilon and record convergence metrics."""
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
    
    return float(compactness), labels.flatten(), centres, elapsed_time


def determine_actual_iterations(samples: np.ndarray,
                               k: int,
                               epsilon: float,
                               max_iter: int = 100,
                               attempts: int = 3,
                               random_seed: int = 42):
    """Determine the exact number of iterations by checking termination per step.
    
    OpenCV kmeans with TERM_CRITERIA_EPS stops when max centroid displacement < epsilon.
    We test max_iter from 1 upwards with EPS to see when the result stops changing.
    """
    # Baseline result with full convergence for this epsilon
    target_comp, _, _, _ = run_kmeans_with_epsilon(
        samples, k, epsilon=epsilon, max_iter=max_iter, attempts=attempts, random_seed=random_seed
    )
    
    for it in range(1, max_iter + 1):
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, it, epsilon)
        cv2.setRNGSeed(random_seed)
        comp, _, _ = cv2.kmeans(samples, k, None, crit, attempts, cv2.KMEANS_PP_CENTERS)
        if abs(comp - target_comp) < 1e-6:
            return it
    return max_iter


def main():
    project_dir = Path(__file__).resolve().parent.parent
    image_path = project_dir / "data" / "images" / "001_image.png"
    
    print("=" * 78)
    print("MCE 401: K-MEANS CONVERGENCE SENSITIVITY TEST (EPSILON VARIATION)")
    print("=" * 78)
    print(f"Target Image: {image_path.name}")
    
    # 1. Load and prepare image features
    _, resized_bgr, orig_w, orig_h = load_image(image_path, max_side=320)
    h, w = resized_bgr.shape[:2]
    n_pixels = h * w
    print(f"Original Resolution: {orig_w} x {orig_h}")
    print(f"Processed Resolution: {w} x {h} ({n_pixels:,} pixels)")
    
    samples = build_features(resized_bgr, feature_space="RGB")
    print(f"Feature Vector: Normalized RGB [0.0, 1.0], Shape: {samples.shape}")
    
    # 2. Test epsilon values
    epsilon_values = [0.2, 0.01, 0.001, 0.0001]
    k = 3
    random_seed = 42
    max_iter = 100
    attempts = 3
    
    results = []
    
    print("\n" + "-" * 78)
    print(f"{'Epsilon':<10} | {'Compactness (J)':<16} | {'J_pixel':<12} | {'Time (s)':<10} | {'Iter Est':<8}")
    print("-" * 78)
    
    for eps in epsilon_values:
        # Run multiple warm-up/timing trials to get stable timing
        times = []
        for _ in range(5):
            comp, labels, centres, elap = run_kmeans_with_epsilon(
                samples, k, epsilon=eps, max_iter=max_iter, attempts=attempts, random_seed=random_seed
            )
            times.append(elap)
        median_time = float(np.median(times))
        
        # Final evaluation run
        comp, labels, centres, _ = run_kmeans_with_epsilon(
            samples, k, epsilon=eps, max_iter=max_iter, attempts=attempts, random_seed=random_seed
        )
        j_pixel = comp / n_pixels
        
        # Estimate iterations to convergence
        est_iter = determine_actual_iterations(
            samples, k, epsilon=eps, max_iter=max_iter, attempts=attempts, random_seed=random_seed
        )
        
        # Sort clusters by green intensity (Centroid G channel) for consistent ordering
        # centres are in RGB [0, 1]
        cluster_info = []
        for cid in range(k):
            count = int(np.sum(labels == cid))
            pct = (count / n_pixels) * 100.0
            c_rgb = centres[cid] * 255.0
            cluster_info.append({
                'id': cid,
                'count': count,
                'pct': pct,
                'rgb': c_rgb
            })
            
        # Sort cluster info by G-channel to align across different runs:
        # High G = Vegetation, Mid G = Lit Soil, Low G = Shadow
        cluster_info_sorted = sorted(cluster_info, key=lambda x: x['rgb'][1], reverse=True)
        
        results.append({
            'epsilon': eps,
            'compactness_J': comp,
            'J_pixel': j_pixel,
            'time_seconds': median_time,
            'estimated_iterations': est_iter,
            'clusters': cluster_info_sorted
        })
        
        print(f"{eps:<10} | {comp:<16.4f} | {j_pixel:<12.6f} | {median_time:<10.4f} | {est_iter:<8}")
        
    print("-" * 78)
    
    # 3. Detailed Cluster Analysis
    print("\nDETAILED CLUSTER CENTROIDS AND PIXEL DISTRIBUTIONS:")
    print("=" * 78)
    for res in results:
        eps = res['epsilon']
        print(f"\n--- Epsilon: {eps} (Iter: {res['estimated_iterations']}, Time: {res['time_seconds']:.4f}s, J: {res['compactness_J']:.4f}) ---")
        for idx, c in enumerate(res['clusters']):
            role = "Vegetation" if idx == 0 else ("Illuminated Soil" if idx == 1 else "Shadow/Dark Soil")
            print(f"  Role: {role:<18} | Pixels: {c['count']:5d} ({c['pct']:5.2f}%) | "
                  f"Centroid RGB: [{c['rgb'][0]:5.1f}, {c['rgb'][1]:5.1f}, {c['rgb'][2]:5.1f}]")
                  
    # 4. Save results to CSV for documentation
    out_table_dir = project_dir / "outputs" / "tables"
    out_table_dir.mkdir(parents=True, exist_ok=True)
    csv_rows = []
    for res in results:
        row = {
            'image': '001_image.png',
            'K': k,
            'epsilon': res['epsilon'],
            'estimated_iterations': res['estimated_iterations'],
            'compactness_J': res['compactness_J'],
            'J_pixel': res['J_pixel'],
            'median_time_seconds': res['time_seconds']
        }
        for idx, c in enumerate(res['clusters']):
            row[f'cluster_{idx}_pct'] = c['pct']
            row[f'cluster_{idx}_R'] = c['rgb'][0]
            row[f'cluster_{idx}_G'] = c['rgb'][1]
            row[f'cluster_{idx}_B'] = c['rgb'][2]
        csv_rows.append(row)
        
    out_csv = out_table_dir / "termination_sensitivity_results.csv"
    pd.DataFrame(csv_rows).to_csv(out_csv, index=False)
    print(f"\nSaved sensitivity validation data to: {out_csv}")
    print("Task B execution finished successfully.")


if __name__ == "__main__":
    main()
