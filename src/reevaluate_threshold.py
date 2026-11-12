
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parent.parent

RESULTS = PROJECT / "outputs" / "full_experiment"
LABELS = RESULTS / "labels"
MASKS = PROJECT / "dataset-1.0" / "masks"

OUTPUT = RESULTS / "re_evaluation_exg35"

THRESHOLD = 35.0

TEST_IMAGES = [
    "015_image.png",
    "051_image.png",
    "007_image.png",
    "031_image.png",
    "023_image.png",
]


def calculate_metrics(prediction, ground_truth):
    tp = int(np.sum(prediction & ground_truth))
    fp = int(np.sum(prediction & ~ground_truth))
    tn = int(np.sum(~prediction & ~ground_truth))
    fn = int(np.sum(~prediction & ground_truth))

    total = tp + fp + tn + fn

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    accuracy = (tp + tn) / total if total else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "iou": iou,
        "pixel_accuracy": accuracy,
    }


def main():

    OUTPUT.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(RESULTS / "metrics.csv")
    clusters = pd.read_csv(RESULTS / "cluster_summary.csv")

    print("Original configurations:", len(metrics))

    ground_truth_cache = {}
    records = []

    for index, row in metrics.iterrows():

        image_name = row["image"]
        stem = image_name.split("_")[0]

        feature_space = row["feature_space"]
        k = int(row["K"])

        lam = (
            None if pd.isna(row["lambda"])
            else float(row["lambda"])
        )

        # Locate the saved K-means label array
        if feature_space == "RGBXY":
            label_name = (
                f"{stem}_RGBXY_K{k}_lambda{lam:.1f}_labels.npy"
            )
        else:
            label_name = f"{stem}_{feature_space}_K{k}_labels.npy"

        label_path = LABELS / label_name

        if not label_path.exists():
            raise FileNotFoundError(label_path)

        labels = np.load(label_path)

        # Load reference mask once per image
        if image_name not in ground_truth_cache:

            mask_path = MASKS / f"{stem}_mask.png"

            mask = cv2.imread(
                str(mask_path),
                cv2.IMREAD_GRAYSCALE
            )

            if mask is None:
                raise FileNotFoundError(mask_path)

            mask = cv2.resize(
                mask,
                (labels.shape[1], labels.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

            ground_truth_cache[image_name] = (mask == 0)

        ground_truth = ground_truth_cache[image_name]

        # Find the cluster records for this configuration
        cluster_rows = clusters[
            (clusters["image"] == image_name)
            & (clusters["feature_space"] == feature_space)
            & (clusters["K"] == k)
        ]

        if lam is None:
            cluster_rows = cluster_rows[
                cluster_rows["lambda"].isna()
            ]
        else:
            cluster_rows = cluster_rows[
                np.isclose(
                    cluster_rows["lambda"].astype(float),
                    lam
                )
            ]

        if len(cluster_rows) != k:
            raise ValueError(
                f"Expected {k} clusters for {label_name}, "
                f"found {len(cluster_rows)}"
            )

        # Apply the calibrated ExG threshold
        prediction = np.zeros(labels.shape, dtype=bool)

        for _, cluster in cluster_rows.iterrows():

            cid = int(cluster["cluster_id"])
            exg = float(cluster["mean_ExG"])

            if exg > THRESHOLD:
                prediction[labels == cid] = True

        result = calculate_metrics(prediction, ground_truth)

        records.append({
            "image": image_name,
            "feature_space": feature_space,
            "K": k,
            "lambda": lam,
            "threshold_exg": THRESHOLD,

            "predicted_veg_pct": (
                prediction.mean() * 100.0
            ),

            "precision": result["precision"],
            "recall": result["recall"],
            "f1_score": result["f1_score"],
            "iou": result["iou"],
            "pixel_accuracy": result["pixel_accuracy"],

            "tp": result["tp"],
            "fp": result["fp"],
            "tn": result["tn"],
            "fn": result["fn"],

            "time_seconds": row["time_seconds"],

            # Preserve original measurements for comparison
            "original_iou": row["iou"],
            "original_f1": row["f1_score"],
            "original_precision": row["precision"],
            "original_recall": row["recall"],
        })

        if (index + 1) % 100 == 0:
            print(f"Re-evaluated {index + 1}/500 configurations")

    # Save all recalculated results
    results = pd.DataFrame(records)

    results.to_csv(
        OUTPUT / "metrics_exg35_all.csv",
        index=False
    )

    # Extract the five held-out test images
    test_results = results[
        results["image"].isin(TEST_IMAGES)
    ].copy()

    test_results.to_csv(
        OUTPUT / "metrics_exg35_test.csv",
        index=False
    )

    # Average configurations within each image first.
    # This prevents images with more RGBXY settings
    # from receiving extra weight in the final comparison.
    per_image = test_results.groupby(
        ["feature_space", "image"]
    ).agg(
        original_iou=("original_iou", "mean"),
        calibrated_iou=("iou", "mean"),
        original_f1=("original_f1", "mean"),
        calibrated_f1=("f1_score", "mean"),
        calibrated_precision=("precision", "mean"),
        calibrated_recall=("recall", "mean"),
        mean_time=("time_seconds", "mean"),
    ).reset_index()

    summary = per_image.groupby("feature_space").agg(
        test_images=("image", "nunique"),
        original_mean_iou=("original_iou", "mean"),
        calibrated_mean_iou=("calibrated_iou", "mean"),
        original_mean_f1=("original_f1", "mean"),
        calibrated_mean_f1=("calibrated_f1", "mean"),
        calibrated_precision=("calibrated_precision", "mean"),
        calibrated_recall=("calibrated_recall", "mean"),
        mean_time_seconds=("mean_time", "mean"),
    ).reset_index()

    summary.to_csv(
        OUTPUT / "summary_exg35_test.csv",
        index=False
    )

    print("\n========== RE-EVALUATION COMPLETE ==========")
    print("Configurations processed:", len(results))
    print("Held-out configurations:", len(test_results))

    print("\n========== HELD-OUT TEST SUMMARY ==========")
    print(summary.round(4).to_string(index=False))

    print("\nSaved results to:")
    print(OUTPUT)

    print("\nOriginal experiment files were not modified.")


if __name__ == "__main__":
    main()
