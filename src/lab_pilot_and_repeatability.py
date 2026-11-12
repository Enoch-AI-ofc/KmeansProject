"""MCE 401 — Computer Vision and Image Processing
Task 02, Part B: CIELAB Pilot and Repeatability Verification

This script executes:
1. New CIELAB pilot experiment on '001_image.png' (K=3, eps=0.001, seed=42, attempts=3).
2. Generates and saves segmented image: outputs/segmentations/001_image_Lab_K3.png.
3. Generates and saves comparison panel: outputs/figures/001_image_pilot_Lab_K3.png.
4. Performs a rigorous repeatability test running the exact same setup twice:
   - Checks whether compactness, cluster labels, and cluster centroids are identical.
   - Measures and explains any numerical or non-deterministic differences.
"""

from pathlib import Path
import time
import cv2
import numpy as np
from segmentation import load_image, build_features, run_kmeans, reconstruct_segmentation


def annotate_panel(img: np.ndarray, text: str) -> np.ndarray:
    """Add a clean header label banner to an image for panel visualization."""
    disp = img.copy()
    cv2.rectangle(disp, (0, 0), (disp.shape[1], 35), (0, 0, 0), -1)
    cv2.putText(disp, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return disp


def main():
    project_dir = Path(__file__).resolve().parent.parent
    image_path = project_dir / "data" / "images" / "001_image.png"
    
    seg_dir = project_dir / "outputs" / "segmentations"
    fig_dir = project_dir / "outputs" / "figures"
    seg_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 78)
    print("MCE 401: PART B — CIELAB PILOT & REPEATABILITY VERIFICATION")
    print("=" * 78)
    
    # Load and resize image
    orig_bgr, resized_bgr, orig_w, orig_h = load_image(image_path, max_side=320)
    h, w = resized_bgr.shape[:2]
    n_pixels = h * w
    print(f"Loaded: {image_path.name}")
    print(f"Original Resolution: {orig_w} x {orig_h} | Processed: {w} x {h} ({n_pixels:,} pixels)")
    
    # -------------------------------------------------------------
    # 1. CIELAB PILOT RUN (K = 3)
    # -------------------------------------------------------------
    print("\n[Step 1] Executing Floating-Point CIELAB Pilot (K=3, eps=0.001, seed=42)...")
    samples_lab = build_features(resized_bgr, feature_space="Lab")
    print(f"CIELAB Feature Array: Shape={samples_lab.shape}, dtype={samples_lab.dtype}")
    print(f"CIELAB Channels (L*/100, a*/100, b*/100):")
    print(f"  L/100 range: [{samples_lab[:, 0].min():.4f}, {samples_lab[:, 0].max():.4f}]")
    print(f"  a/100 range: [{samples_lab[:, 1].min():.4f}, {samples_lab[:, 1].max():.4f}]")
    print(f"  b/100 range: [{samples_lab[:, 2].min():.4f}, {samples_lab[:, 2].max():.4f}]")
    
    comp_lab, labels_lab, centres_lab, time_lab = run_kmeans(
        samples_lab, k=3, random_seed=42, max_iter=100, epsilon=0.001, attempts=3
    )
    j_pixel_lab = comp_lab / n_pixels
    
    print(f"\nCIELAB Pilot Results:")
    print(f"  Execution Time: {time_lab:.4f} seconds")
    print(f"  Compactness J:  {comp_lab:.4f}")
    print(f"  J per pixel:    {j_pixel_lab:.6f}")
    
    # Cluster distributions in CIELAB
    print("\nCIELAB Cluster Breakdown:")
    for cid in range(3):
        cnt = int(np.sum(labels_lab.flatten() == cid))
        pct = (cnt / n_pixels) * 100.0
        # centres are in [L/100, a/100, b/100]
        c_true_lab = centres_lab[cid] * 100.0
        print(f"  Cluster {cid}: {pct:5.2f}% of pixels | "
              f"Centroid CIE-Lab: [L*={c_true_lab[0]:5.1f}, a*={c_true_lab[1]:5.1f}, b*={c_true_lab[2]:5.1f}]")
              
    # Reconstruct and save Lab segmentation
    segmented_lab_bgr = reconstruct_segmentation(resized_bgr, labels_lab, k=3)
    out_seg_path = seg_dir / "001_image_Lab_K3.png"
    cv2.imwrite(str(out_seg_path), segmented_lab_bgr)
    print(f"Saved Lab segmented image to: {out_seg_path}")
    
    # Create comparison panel (Original vs CIELAB K=3)
    panel_orig = annotate_panel(resized_bgr, "Original (Resized)")
    panel_lab = annotate_panel(segmented_lab_bgr, "CIELAB K-means (K=3)")
    comparison_panel = cv2.hconcat([panel_orig, panel_lab])
    out_panel_path = fig_dir / "001_image_pilot_Lab_K3.png"
    cv2.imwrite(str(out_panel_path), comparison_panel)
    print(f"Saved Lab comparison panel to: {out_panel_path}")
    
    # -------------------------------------------------------------
    # 2. RIGOROUS REPEATABILITY TEST
    # -------------------------------------------------------------
    print("\n" + "=" * 78)
    print("[Step 2] Conducting Repeatability Verification (Two Consecutive Identical Runs)")
    print("=" * 78)
    
    # Run 1
    print("\nExecuting Run 1...")
    comp1, labels1, centres1, time1 = run_kmeans(
        samples_lab, k=3, random_seed=42, max_iter=100, epsilon=0.001, attempts=3
    )
    
    # Run 2
    print("Executing Run 2...")
    comp2, labels2, centres2, time2 = run_kmeans(
        samples_lab, k=3, random_seed=42, max_iter=100, epsilon=0.001, attempts=3
    )
    
    # Compare Compactness
    comp_diff = abs(comp1 - comp2)
    print(f"\nRepeatability Evaluation:")
    print(f"  Compactness Run 1: {comp1:.8f}")
    print(f"  Compactness Run 2: {comp2:.8f}")
    print(f"  Difference (|J1 - J2|): {comp_diff:.10e}")
    
    # Compare Centroids
    centre_diff = np.max(np.abs(centres1 - centres2))
    print(f"  Max Centroid Difference: {centre_diff:.10e}")
    
    # Compare Pixel Labels
    labels_equal = np.array_equal(labels1, labels2)
    matching_pixels = int(np.sum(labels1 == labels2))
    pct_match = (matching_pixels / n_pixels) * 100.0
    print(f"  Label Array Exact Match: {labels_equal} ({matching_pixels}/{n_pixels} pixels, {pct_match:.2f}%)")
    
    if labels_equal and comp_diff < 1e-7 and centre_diff < 1e-7:
        print("\n>>> CONCLUSION: REPEATABILITY CONFIRMED. Output is 100% deterministic and reproducible.")
    else:
        print("\n>>> WARNING: Non-deterministic behavior detected between identical runs.")


if __name__ == "__main__":
    main()
