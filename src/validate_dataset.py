"""MCE 401 — Computer Vision and Image Processing
Task 02, Part C: Dataset Validation and Quality Control

This script validates all 20 images listed in data/selected_images.txt:
1. Confirms existence of original images, vegetation masks, and annotations.
2. Checks dimension compatibility (image vs mask).
3. Records unique mask values and counts.
4. Confirms semantic class mapping (0 = vegetation, 255 = soil/background).
5. Exports data/dataset_manifest.csv.
6. Generates visual quality-control panel (outputs/figures/dataset_qc_panel.png).
"""

from pathlib import Path
import cv2
import numpy as np
import pandas as pd


def main():
    project_dir = Path(__file__).resolve().parent.parent
    selection_file = project_dir / "data" / "selected_images.txt"
    img_dir = project_dir / "data" / "images"
    raw_img_dir = project_dir / "dataset-1.0" / "images"
    mask_dir = project_dir / "dataset-1.0" / "masks"
    annot_dir = project_dir / "dataset-1.0" / "annotations"
    
    fig_dir = project_dir / "outputs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    with open(selection_file, "r", encoding="utf-8") as f:
        selected_filenames = [
            line.strip() for line in f if line.strip() and not line.strip().startswith("#")
        ]
        
    print(f"Loaded {len(selected_filenames)} filenames from {selection_file}")
    
    manifest_rows = []
    qc_image_pairs = []
    
    # 5 representative images for visual QC panel
    rep_images = ["001_image.png", "007_image.png", "018_image.png", "020_image.png", "029_image.png"]
    
    for idx, fname in enumerate(selected_filenames, start=1):
        stem = fname.split("_")[0]
        img_path = img_dir / fname
        raw_img_path = raw_img_dir / fname
        mask_path = mask_dir / f"{stem}_mask.png"
        annot_png_path = annot_dir / f"{stem}_annotation.png"
        annot_yaml_path = annot_dir / f"{stem}_annotation.yaml"
        
        # Verify existence
        img_exists = img_path.exists() and raw_img_path.exists()
        mask_exists = mask_path.exists()
        annot_png_exists = annot_png_path.exists()
        annot_yaml_exists = annot_yaml_path.exists()
        
        if not img_exists:
            print(f"ERROR: Image missing: {fname}")
        if not mask_exists:
            print(f"ERROR: Mask missing for {fname}: {mask_path}")
            
        img = cv2.imread(str(img_path)) if img_exists else None
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_exists else None
        
        orig_h, orig_w = img.shape[:2] if img is not None else (None, None)
        mask_h, mask_w = mask.shape[:2] if mask is not None else (None, None)
        dim_match = (orig_w == mask_w) and (orig_h == mask_h)
        
        total_pixels = orig_w * orig_h if (orig_w and orig_h) else 0
        
        # Mask pixel distribution
        if mask is not None:
            vals, counts = np.unique(mask, return_counts=True)
            counts_dict = dict(zip(vals, counts))
            veg_count = int(counts_dict.get(0, 0))       # 0 is vegetation
            bg_count = int(counts_dict.get(255, 0))     # 255 is soil/background
            veg_pct = (veg_count / total_pixels) * 100.0
            bg_pct = (bg_count / total_pixels) * 100.0
            unique_vals_str = str(sorted(list(counts_dict.keys())))
        else:
            veg_count, bg_count, veg_pct, bg_pct = 0, 0, 0.0, 0.0
            unique_vals_str = "None"
            
        # Category classification
        if veg_pct >= 12.0:
            category = "Dense Vegetation"
        elif veg_pct >= 5.0:
            category = "Moderate Vegetation"
        else:
            category = "Sparse / Soil-Dominant"
            
        manifest_rows.append({
            "index": idx,
            "filename": fname,
            "original_width": orig_w,
            "original_height": orig_h,
            "channels": 3,
            "mask_width": mask_w,
            "mask_height": mask_h,
            "dimensions_compatible": dim_match,
            "mask_unique_values": unique_vals_str,
            "vegetation_pixel_count": veg_count,
            "vegetation_percentage": round(veg_pct, 4),
            "background_pixel_count": bg_count,
            "background_percentage": round(bg_pct, 4),
            "annotation_png_exists": annot_png_exists,
            "annotation_yaml_exists": annot_yaml_exists,
            "agronomic_category": category
        })
        
        # Collect QC pairs
        if fname in rep_images and img is not None and mask is not None:
            # Resize to max_side 320 for QC display
            scale = 320.0 / float(max(orig_h, orig_w))
            new_w = max(1, int(round(orig_w * scale)))
            new_h = max(1, int(round(orig_h * scale)))
            img_disp = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            mask_disp = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            
            # Create color mask display: Green for vegetation (0), Brown/Dark for soil (255)
            mask_rgb = np.zeros_like(img_disp)
            mask_rgb[mask_disp == 0] = [0, 210, 0]     # Bright Green for vegetation
            mask_rgb[mask_disp == 255] = [40, 50, 70]  # Soil tone for background
            
            qc_image_pairs.append((fname, img_disp, mask_rgb, veg_pct))
            
    # Save manifest CSV
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_csv_path = project_dir / "data" / "dataset_manifest.csv"
    manifest_df.to_csv(manifest_csv_path, index=False)
    print(f"\nSaved dataset manifest: {manifest_csv_path}")
    
    # Generate 5-Image Visual QC Panel
    # Each row: [Original Image | Vegetation Mask (Green=Veg, Brown=Soil)]
    row_panels = []
    for fname, img_disp, mask_disp, v_pct in qc_image_pairs:
        # Add labels
        disp_img = img_disp.copy()
        cv2.rectangle(disp_img, (0, 0), (disp_img.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(disp_img, f"{fname} (Original)", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    
        disp_mask = mask_disp.copy()
        cv2.rectangle(disp_mask, (0, 0), (disp_mask.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(disp_mask, f"Mask (Veg: {v_pct:.1f}% | Green=0, Soil=255)", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                    
        pair = cv2.hconcat([disp_img, disp_mask])
        row_panels.append(pair)
        
    qc_grid = cv2.vconcat(row_panels)
    qc_out_path = fig_dir / "dataset_qc_panel.png"
    cv2.imwrite(str(qc_out_path), qc_grid)
    print(f"Saved visual QC panel: {qc_out_path}")
    print("Dataset validation script finished successfully.")


if __name__ == "__main__":
    main()
