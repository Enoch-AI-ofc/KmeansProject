"""MCE 401 — Computer Vision and Image Processing
Pilot Test: First RGB K-means Image Segmentation (K = 3)

This script executes the immediate development objective:
1. Load and inspect an agricultural image (001_image.png).
2. Resize consistently to max side 320.
3. Build normalized RGB feature vectors in [0, 1].
4. Run OpenCV cv2.kmeans() with K=3, K-means++ initialization, and seed=42.
5. Reconstruct the segmented image with cluster-mean BGR coloring.
6. Calculate compactness J, compactness per pixel J_pixel, and execution time.
7. Save the outputs and generate a side-by-side comparison panel.
"""

from pathlib import Path
import cv2
import numpy as np
from segmentation import load_image, build_features, run_kmeans, reconstruct_segmentation


def main():
    project_dir = Path(__file__).resolve().parent.parent
    image_path = project_dir / "data" / "images" / "001_image.png"
    
    seg_dir = project_dir / "outputs" / "segmentations"
    fig_dir = project_dir / "outputs" / "figures"
    seg_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading image from: {image_path}")
    orig_bgr, resized_bgr, orig_w, orig_h = load_image(image_path, max_side=320)
    h, w = resized_bgr.shape[:2]
    n_pixels = h * w
    
    print(f"Original Resolution: {orig_w} x {orig_h}")
    print(f"Processed Resolution: {w} x {h} (Total Pixels: {n_pixels:,})")
    
    # 1. Build RGB feature space
    print("\nExtracting normalized RGB features...")
    samples = build_features(resized_bgr, feature_space="RGB")
    print(f"Feature array shape: {samples.shape}, dtype: {samples.dtype}")
    print(f"Feature min: {samples.min():.4f}, max: {samples.max():.4f}")
    
    # 2. Run K-means with K=3
    k = 3
    print(f"\nRunning K-means with K={k}, K-means++ initialization, seed=42...")
    compactness, labels, centres, elapsed = run_kmeans(samples, k=k, random_seed=42)
    j_per_pixel = compactness / n_pixels
    
    print(f"Execution Time: {elapsed:.4f} seconds")
    print(f"Compactness (J): {compactness:.4f}")
    print(f"Compactness per pixel (J_pixel): {j_per_pixel:.6f}")
    
    # Analyze cluster centers in RGB and pixel percentages
    print("\nCluster Centroid Analysis:")
    for cluster_id in range(k):
        count = np.sum(labels.flatten() == cluster_id)
        pct = (count / n_pixels) * 100
        # centres are in RGB [0, 1]
        c_rgb = centres[cluster_id] * 255.0
        print(f"  Cluster {cluster_id}: {pct:5.2f}% of pixels | "
              f"Centroid RGB: [R={c_rgb[0]:.1f}, G={c_rgb[1]:.1f}, B={c_rgb[2]:.1f}]")
        
    # 3. Reconstruct segmented image
    segmented_bgr = reconstruct_segmentation(resized_bgr, labels, k=k)
    
    # Save individual segmented image
    out_seg_path = seg_dir / f"001_image_RGB_K{k}.png"
    cv2.imwrite(str(out_seg_path), segmented_bgr)
    print(f"\nSaved segmented image to: {out_seg_path}")
    
    # 4. Create comparison panel (Original vs Segmented)
    def annotate(img, text):
        disp = img.copy()
        cv2.rectangle(disp, (0, 0), (disp.shape[1], 35), (0, 0, 0), -1)
        cv2.putText(disp, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return disp
    
    panel_orig = annotate(resized_bgr, "Original (Resized)")
    panel_seg = annotate(segmented_bgr, f"RGB K-means (K={k})")
    comparison_panel = cv2.hconcat([panel_orig, panel_seg])
    
    out_panel_path = fig_dir / f"001_image_pilot_RGB_K{k}.png"
    cv2.imwrite(str(out_panel_path), comparison_panel)
    print(f"Saved comparison panel to: {out_panel_path}")
    print("\nPilot test completed successfully!")


if __name__ == "__main__":
    main()
