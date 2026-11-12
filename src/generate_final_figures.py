
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


PROJECT = Path(__file__).resolve().parent.parent

RESULTS = (
    PROJECT / "outputs" / "full_experiment"
    / "re_evaluation_exg35"
)

TEST_IMAGES = [
    "015_image.png",
    "051_image.png",
    "007_image.png",
    "031_image.png",
    "023_image.png",
]

FEATURES = ["Lab", "RGB", "RGBXY"]


def main():

    data = pd.read_csv(RESULTS / "metrics_exg35_all.csv")

    test = data[data["image"].isin(TEST_IMAGES)].copy()

    if len(test) != 125:
        raise ValueError(
            f"Expected 125 held-out records, found {len(test)}"
        )

    # Average configurations within each image first.
    per_image = test.groupby(
        ["feature_space", "image"]
    ).agg(
        original_iou=("original_iou", "mean"),
        calibrated_iou=("iou", "mean"),
        original_f1=("original_f1", "mean"),
        calibrated_f1=("f1_score", "mean"),
        precision=("precision", "mean"),
        recall=("recall", "mean"),
    ).reset_index()

    # Calculate feature-space averages across the five images.
    summary = per_image.groupby("feature_space").agg(
        test_images=("image", "nunique"),
        original_iou=("original_iou", "mean"),
        calibrated_iou=("calibrated_iou", "mean"),
        calibrated_iou_std=("calibrated_iou", "std"),
        original_f1=("original_f1", "mean"),
        calibrated_f1=("calibrated_f1", "mean"),
        precision=("precision", "mean"),
        recall=("recall", "mean"),
    ).reindex(FEATURES).reset_index()

    per_image.to_csv(
        RESULTS / "per_image_test_summary.csv",
        index=False
    )

    summary.to_csv(
        RESULTS / "feature_space_test_summary.csv",
        index=False
    )

    # Create figures folder.
    figures = RESULTS / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    # GRAPH 1: Original versus calibrated IoU.
    x = np.arange(len(FEATURES))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        x - width / 2,
        summary["original_iou"],
        width,
        label="Original ExG threshold (0)"
    )

    ax.bar(
        x + width / 2,
        summary["calibrated_iou"],
        width,
        label="Calibrated ExG threshold (35)"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(FEATURES)
    ax.set_ylabel("Mean Intersection over Union (IoU)")
    ax.set_title("Vegetation Segmentation: Before and After Calibration")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        figures / "threshold_comparison_iou.png",
        dpi=200
    )
    plt.close(fig)

    # GRAPH 2: IoU across individual held-out images.
    pivot = per_image.pivot(
        index="image",
        columns="feature_space",
        values="calibrated_iou"
    ).reindex(index=TEST_IMAGES, columns=FEATURES)

    fig, ax = plt.subplots(figsize=(11, 5))

    pivot.plot(kind="bar", ax=ax)

    ax.set_title("Calibrated IoU Across Held-Out Test Images")
    ax.set_xlabel("Agricultural image")
    ax.set_ylabel("Mean IoU")
    ax.set_ylim(0, 1)
    ax.set_xticklabels(
        [name.replace("_image.png", "") for name in TEST_IMAGES],
        rotation=0
    )
    ax.legend(title="Feature space")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        figures / "heldout_iou_by_image.png",
        dpi=200
    )
    plt.close(fig)

    # GRAPH 3: Effect of K on calibrated IoU.
    # Average RGBXY lambda settings within each image and K first.
    by_image_k = test.groupby(
        ["feature_space", "image", "K"]
    )["iou"].mean().reset_index()

    k_summary = by_image_k.groupby(
        ["feature_space", "K"]
    )["iou"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(9, 5))

    for feature in FEATURES:
        subset = k_summary[
            k_summary["feature_space"] == feature
        ]

        ax.plot(
            subset["K"],
            subset["iou"],
            marker="o",
            label=feature
        )

    ax.set_title("Effect of Cluster Count on Held-Out IoU")
    ax.set_xlabel("Number of clusters (K)")
    ax.set_ylabel("Mean IoU")
    ax.set_ylim(0, 1)
    ax.legend(title="Feature space")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        figures / "heldout_iou_by_k.png",
        dpi=200
    )
    plt.close(fig)

    print("\nFINAL FIGURES GENERATED SUCCESSFULLY")

    print("\nFEATURE SPACE SUMMARY")
    print(summary.round(4).to_string(index=False))

    print("\nSAVED FILES:")
    print(" - per_image_test_summary.csv")
    print(" - feature_space_test_summary.csv")
    print(" - figures/threshold_comparison_iou.png")
    print(" - figures/heldout_iou_by_image.png")
    print(" - figures/heldout_iou_by_k.png")


if __name__ == "__main__":
    main()
