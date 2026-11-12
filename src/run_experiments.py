"""MCE 401 — Computer Vision and Image Processing
Task 1: Automated Experimental Engine for K-means Image Segmentation

This module executes the frozen experimental protocol across agricultural images:
- Feature Spaces: RGB (5 configs), CIELAB (5 configs), RGB+XY (15 configs) -> 25 configs/image
- Hyperparameters: K in [2, 3, 4, 5, 8], seed=42, eps=0.001, max_iter=100, attempts=3
- Cluster Interpretation: Mask-independent Excess Green rule (ExG = 2G - R - B > 0 -> Vegetation)
- Evaluation: Ground-truth mask comparison (INTER_NEAREST downsampling, TP/FP/TN/FN, Accuracy, Precision, Recall, F1, IoU)
- Reliability: Incremental checkpointing, exact discrete label saving (.npy), and failure recovery.
"""

from pathlib import Path
import argparse
import time
import cv2
import numpy as np
import pandas as pd
from segmentation import load_image, build_features, run_kmeans, reconstruct_segmentation


def annotate_image(img: np.ndarray, text: str) -> np.ndarray:
    """Overlay a header banner with clear title on an image."""
    disp = img.copy()
    cv2.rectangle(disp, (0, 0), (disp.shape[1], 32), (0, 0, 0), -1)
    cv2.putText(disp, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return disp


def calculate_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray):
    """Compute binary classification metrics between predicted and reference masks.
    
    Convention for undefined zero denominators:
    - If TP + FP == 0 (no positive predictions), precision is set to 0.0.
    - If TP + FN == 0 (no ground-truth positives), recall is set to 0.0.
    - If precision + recall == 0, F1 is set to 0.0.
    - If TP + FP + FN == 0 (empty foreground in both), IoU is set to 1.0 if perfectly matching, else 0.0.
    """
    tp = int(np.sum(pred_mask & gt_mask))
    fp = int(np.sum(pred_mask & ~gt_mask))
    tn = int(np.sum(~pred_mask & ~gt_mask))
    fn = int(np.sum(~pred_mask & gt_mask))
    total = tp + fp + tn + fn
    
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    union = tp + fp + fn
    if union > 0:
        iou = tp / union
    else:
        iou = 1.0 if (tp == 0 and fp == 0 and fn == 0) else 0.0
        
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou
    }


def parse_args():
    parser = argparse.ArgumentParser(description="MCE 401 K-means Experimental Engine")
    parser.add_argument("--dry-run", action="store_true", default=False,
                        help="Run dry-run on 029_image.png and 050_image.png (50 configurations total)")
    parser.add_argument("--images", nargs="+", default=None,
                        help="Specific image filenames to process")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Target output directory (defaults to outputs/dry_run or outputs)")
    parser.add_argument("--max-side", type=int, default=320,
                        help="Maximum image side length (default: 320)")
    parser.add_argument("--epsilon", type=float, default=0.001,
                        help="K-means stopping centroid movement tolerance")
    parser.add_argument("--max-iter", type=int, default=100,
                        help="Maximum K-means iterations")
    parser.add_argument("--attempts", type=int, default=3,
                        help="Number of K-means attempts")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    return parser.parse_args()


def main():
    args = parse_args()
    project_dir = Path(__file__).resolve().parent.parent
    
    # Configure output directory
    if args.output_dir is not None:
        output_dir = args.output_dir
    elif args.dry_run:
        output_dir = project_dir / "outputs" / "dry_run"
    else:
        output_dir = project_dir / "outputs" / "full_experiment"
        
    metrics_csv_path = output_dir / "metrics.csv"
    cluster_csv_path = output_dir / "cluster_summary.csv"
    manifest_csv_path = output_dir / "run_manifest.csv"
    labels_dir = output_dir / "labels"
    seg_dir = output_dir / "segmentations"
    panel_dir = output_dir / "panels"
    plot_dir = output_dir / "plots"
    
    for d in [output_dir, labels_dir, seg_dir, panel_dir, plot_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    # Determine image list
    if args.images is not None:
        target_images = args.images
    elif args.dry_run:
        target_images = ["029_image.png", "050_image.png"]
    else:
        selection_file = project_dir / "data" / "selected_images.txt"
        with open(selection_file, "r", encoding="utf-8") as f:
            target_images = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
            
    print("=" * 80)
    print("MCE 401: K-MEANS EXPERIMENTAL RUNNER")
    print(f"Mode: {'DRY-RUN (50 configs)' if args.dry_run else f'FULL ({len(target_images)} images, {len(target_images)*25} configs)'}")
    print(f"Target Images: {target_images}")
    print(f"Output Directory: {output_dir}")
    print(f"Parameters: max_side={args.max_side}, epsilon={args.epsilon}, max_iter={args.max_iter}, attempts={args.attempts}, seed={args.seed}")
    print("=" * 80)
    
    # Define experimental grid (25 configurations per image)
    k_values = [2, 3, 4, 5, 8]
    lambdas = [0.1, 0.5, 1.0]
    
    configurations = []
    # 1. RGB (5 configs)
    for k in k_values:
        configurations.append(("RGB", k, None))
    # 2. CIELAB (5 configs)
    for k in k_values:
        configurations.append(("Lab", k, None))
    # 3. RGB+XY (15 configs)
    for lam in lambdas:
        for k in k_values:
            configurations.append(("RGBXY", k, lam))
            
    print(f"Total configurations per image: {len(configurations)}")
    
    # Load checkpoint if exists
    existing_runs = set()
    metrics_records = []
    if metrics_csv_path.exists():
        try:
            prev_df = pd.read_csv(metrics_csv_path)
            for _, r in prev_df.iterrows():
                lam_val = float(r["lambda"]) if pd.notna(r["lambda"]) else None
                existing_runs.add((str(r["image"]), str(r["feature_space"]), int(r["K"]), lam_val))
            metrics_records = prev_df.to_dict("records")
            print(f"Found existing checkpoint with {len(existing_runs)} completed configurations.")
        except Exception as e:
            print(f"Warning: Could not read existing metrics checkpoint ({e}). Starting fresh.")
            
    cluster_records = []
    if cluster_csv_path.exists():
        try:
            cluster_records = pd.read_csv(cluster_csv_path).to_dict("records")
        except Exception:
            cluster_records = []
            
    manifest_records = []
    
    # Store segmentation images in memory for panel generation
    # Key: (image_name, feature_space, k, lam) -> BGR image
    segmented_cache = {}
    resized_images_cache = {}
    gt_masks_cache = {}
    
    total_start_time = time.perf_counter()
    completed_count = 0
    skipped_count = 0
    
    for img_idx, img_name in enumerate(target_images, start=1):
        stem = img_name.split("_")[0]
        img_path = project_dir / "data" / "images" / img_name
        raw_img_path = project_dir / "dataset-1.0" / "images" / img_name
        mask_path = project_dir / "dataset-1.0" / "masks" / f"{stem}_mask.png"
        
        # Verify file presence
        active_img_path = img_path if img_path.exists() else raw_img_path
        if not active_img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_name}")
        if not mask_path.exists():
            raise FileNotFoundError(f"Reference mask not found: {mask_path}")
            
        # Load and resize image
        orig_bgr, resized_bgr, orig_w, orig_h = load_image(active_img_path, max_side=args.max_side)
        h, w = resized_bgr.shape[:2]
        n_pixels = h * w
        resized_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
        resized_images_cache[img_name] = resized_bgr
        
        # Load and resize ground-truth mask with INTER_NEAREST
        raw_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if raw_mask is None:
            raise ValueError(f"Could not read mask: {mask_path}")
        resized_mask = cv2.resize(raw_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        # Ground-truth: 0 = Vegetation, 255 = Background
        gt_veg_bool = (resized_mask == 0)
        gt_masks_cache[img_name] = (resized_mask, gt_veg_bool)
        
        gt_veg_pixels = int(np.sum(gt_veg_bool))
        gt_veg_pct = (gt_veg_pixels / n_pixels) * 100.0
        
        manifest_records.append({
            "image": img_name,
            "original_width": orig_w,
            "original_height": orig_h,
            "processed_width": w,
            "processed_height": h,
            "processed_pixels": n_pixels,
            "gt_veg_pixels": gt_veg_pixels,
            "gt_veg_pct": round(gt_veg_pct, 4)
        })
        
        print(f"\n[{img_idx}/{len(target_images)}] Processing {img_name} ({w}x{h}, {n_pixels:,} px | GT Veg: {gt_veg_pct:.2f}%)")
        
        for feature_space, k, lam in configurations:
            cfg_key = (img_name, feature_space, k, lam)
            lam_str = f"_lambda{lam}" if lam is not None else ""
            file_stem = f"{stem}_{feature_space}_K{k}{lam_str}"
            
            # Check if already completed in checkpoint
            if cfg_key in existing_runs:
                skipped_count += 1
                # If cached image file exists, load it into cache for panel plotting
                seg_file = seg_dir / f"{file_stem}.png"
                if seg_file.exists():
                    segmented_cache[cfg_key] = cv2.imread(str(seg_file))
                continue
                
            # Build features
            samples = build_features(resized_bgr, feature_space=feature_space, lam=lam if lam else 0.0)
            
            # Run K-means
            compactness, labels_1d, centres, elapsed = run_kmeans(
                samples, k=k, random_seed=args.seed, max_iter=args.max_iter,
                epsilon=args.epsilon, attempts=args.attempts
            )
            labels_2d = labels_1d.reshape(h, w)
            j_pixel = compactness / n_pixels
            
            # Reconstruct color segmented image
            seg_bgr = reconstruct_segmentation(resized_bgr, labels_1d, k=k)
            segmented_cache[cfg_key] = seg_bgr
            
            # Save discrete labels (.npy) and visualization (.png)
            np.save(str(labels_dir / f"{file_stem}_labels.npy"), labels_2d)
            cv2.imwrite(str(seg_dir / f"{file_stem}.png"), seg_bgr)
            
            # Mask-Independent Cluster Evaluation via Excess Green (ExG)
            pred_veg_mask = np.zeros((h, w), dtype=bool)
            veg_cluster_count = 0
            veg_pixel_count = 0
            
            for cid in range(k):
                c_mask = (labels_2d == cid)
                c_count = int(np.sum(c_mask))
                c_pct = (c_count / n_pixels) * 100.0
                
                if c_count > 0:
                    c_rgb_pixels = resized_rgb[c_mask]
                    mean_r = float(c_rgb_pixels[:, 0].mean())
                    mean_g = float(c_rgb_pixels[:, 1].mean())
                    mean_b = float(c_rgb_pixels[:, 2].mean())
                    exg = float(2.0 * mean_g - mean_r - mean_b)
                else:
                    mean_r, mean_g, mean_b, exg = 0.0, 0.0, 0.0, -999.0
                    
                is_veg = bool(exg > 0.0)
                if is_veg:
                    pred_veg_mask |= c_mask
                    veg_cluster_count += 1
                    veg_pixel_count += c_count
                    
                cluster_records.append({
                    "image": img_name,
                    "feature_space": feature_space,
                    "K": k,
                    "lambda": lam if lam is not None else np.nan,
                    "cluster_id": cid,
                    "pixel_count": c_count,
                    "pixel_percentage": round(c_pct, 4),
                    "mean_R": round(mean_r, 2),
                    "mean_G": round(mean_g, 2),
                    "mean_B": round(mean_b, 2),
                    "mean_ExG": round(exg, 2),
                    "semantic_class": "vegetation" if is_veg else "background"
                })
                
            # Post-hoc Benchmark Metrics against Reference Mask
            perf = calculate_metrics(pred_veg_mask, gt_veg_bool)
            pred_veg_pct = (veg_pixel_count / n_pixels) * 100.0
            
            record = {
                "image": img_name,
                "feature_space": feature_space,
                "K": k,
                "lambda": lam if lam is not None else np.nan,
                "compactness_J": compactness,
                "J_per_pixel": j_pixel,
                "time_seconds": elapsed,
                "width": w,
                "height": h,
                "total_pixels": n_pixels,
                "veg_clusters": veg_cluster_count,
                "predicted_veg_pixels": veg_pixel_count,
                "predicted_veg_pct": round(pred_veg_pct, 4),
                "gt_veg_pixels": gt_veg_pixels,
                "gt_veg_pct": round(gt_veg_pct, 4),
                "tp": perf["tp"],
                "fp": perf["fp"],
                "tn": perf["tn"],
                "fn": perf["fn"],
                "pixel_accuracy": round(perf["accuracy"], 5),
                "precision": round(perf["precision"], 5),
                "recall": round(perf["recall"], 5),
                "f1_score": round(perf["f1"], 5),
                "iou": round(perf["iou"], 5)
            }
            metrics_records.append(record)
            existing_runs.add(cfg_key)
            completed_count += 1
            
            # Incremental save checkpoint
            pd.DataFrame(metrics_records).to_csv(metrics_csv_path, index=False)
            pd.DataFrame(cluster_records).to_csv(cluster_csv_path, index=False)
            
            lam_disp = f"lam={lam}" if lam is not None else "     "
            print(f"  [{feature_space:<5} K={k} {lam_disp}] J={compactness:8.2f} | Time={elapsed:.4f}s | "
                  f"ExG>0 Veg={pred_veg_pct:5.2f}% | IoU={perf['iou']:.4f} | F1={perf['f1']:.4f}")
                  
    total_elapsed = time.perf_counter() - total_start_time
    print(f"\nClustering phase completed: {completed_count} newly executed, {skipped_count} loaded from checkpoint. Total runtime: {total_elapsed:.2f}s")
    
    # Save final run manifest
    pd.DataFrame(manifest_records).to_csv(manifest_csv_path, index=False)
    print(f"Saved run manifest: {manifest_csv_path}")
    
    # -------------------------------------------------------------
    # GENERATE 5 REPRESENTATIVE COMPARISON PANELS (Part F)
    # -------------------------------------------------------------
    print("\nGenerating 5 comparison panels for dry-run verification...")
    
    # Helper to retrieve reconstructed BGR or load from disk
    def get_seg_img(img_name, space, k, lam=None):
        stem = img_name.split("_")[0]
        key = (img_name, space, k, lam)
        if key in segmented_cache:
            return segmented_cache[key]
        lam_str = f"_lambda{lam}" if lam is not None else ""
        p = seg_dir / f"{stem}_{space}_K{k}{lam_str}.png"
        if p.exists():
            return cv2.imread(str(p))
        return np.zeros((239, 320, 3), dtype=np.uint8)

    # Panel 1: 029_image.png (Dense Veg) — RGB K in [2, 3, 4, 5, 8]
    p1_imgs = [annotate_image(resized_images_cache["029_image.png"], "029 Original")]
    for k in [2, 3, 4, 5, 8]:
        p1_imgs.append(annotate_image(get_seg_img("029_image.png", "RGB", k), f"RGB K={k}"))
    panel1 = cv2.hconcat(p1_imgs)
    cv2.imwrite(str(panel_dir / "panel1_029_RGB_K_sweep.jpg"), panel1)
    
    # Panel 2: 029_image.png (Dense Veg) — CIELAB K in [2, 3, 4, 5, 8]
    p2_imgs = [annotate_image(resized_images_cache["029_image.png"], "029 Original")]
    for k in [2, 3, 4, 5, 8]:
        p2_imgs.append(annotate_image(get_seg_img("029_image.png", "Lab", k), f"CIELAB K={k}"))
    panel2 = cv2.hconcat(p2_imgs)
    cv2.imwrite(str(panel_dir / "panel2_029_Lab_K_sweep.jpg"), panel2)
    
    # Panel 3: 029_image.png — Spatial Weight Sweep at K=4 (Original, RGB K=4, RGBXY lam=0.1, 0.5, 1.0)
    p3_imgs = [
        annotate_image(resized_images_cache["029_image.png"], "029 Original"),
        annotate_image(get_seg_img("029_image.png", "RGB", 4), "RGB (lam=0, K=4)"),
        annotate_image(get_seg_img("029_image.png", "RGBXY", 4, 0.1), "RGBXY lam=0.1, K=4"),
        annotate_image(get_seg_img("029_image.png", "RGBXY", 4, 0.5), "RGBXY lam=0.5, K=4"),
        annotate_image(get_seg_img("029_image.png", "RGBXY", 4, 1.0), "RGBXY lam=1.0, K=4")
    ]
    panel3 = cv2.hconcat(p3_imgs)
    cv2.imwrite(str(panel_dir / "panel3_029_Spatial_lambda_sweep_K4.jpg"), panel3)
    
    # Panel 4: 050_image.png (Sparse Veg) — RGB K in [2, 3, 4, 5, 8]
    p4_imgs = [annotate_image(resized_images_cache["050_image.png"], "050 Original")]
    for k in [2, 3, 4, 5, 8]:
        p4_imgs.append(annotate_image(get_seg_img("050_image.png", "RGB", k), f"RGB K={k}"))
    panel4 = cv2.hconcat(p4_imgs)
    cv2.imwrite(str(panel_dir / "panel4_050_RGB_K_sweep.jpg"), panel4)
    
    # Panel 5: 050_image.png (Sparse Veg) — Cross-Feature Comparison at K=3
    # Build Ground Truth color mask
    raw_gt, gt_bool = gt_masks_cache["050_image.png"]
    gt_disp = np.zeros_like(resized_images_cache["050_image.png"])
    gt_disp[gt_bool] = [0, 220, 0]      # Green for Veg (0)
    gt_disp[~gt_bool] = [40, 50, 70]   # Soil tone for Background (255)
    
    p5_imgs = [
        annotate_image(resized_images_cache["050_image.png"], "050 Original"),
        annotate_image(gt_disp, "GT Mask (Veg: 1.6%)"),
        annotate_image(get_seg_img("050_image.png", "RGB", 3), "RGB K=3"),
        annotate_image(get_seg_img("050_image.png", "Lab", 3), "CIELAB K=3"),
        annotate_image(get_seg_img("050_image.png", "RGBXY", 3, 0.5), "RGBXY lam=0.5, K=3")
    ]
    panel5 = cv2.hconcat(p5_imgs)
    cv2.imwrite(str(panel_dir / "panel5_050_Cross_Feature_Comparison_K3.jpg"), panel5)
    
    print("All 5 comparison panels successfully generated and saved.")
    print("=" * 80)
    print("DRY-RUN EXECUTION COMPLETE.")
    print(f"Metrics CSV: {metrics_csv_path}")
    print(f"Cluster Summary CSV: {cluster_csv_path}")
    print(f"Panels Directory: {panel_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
