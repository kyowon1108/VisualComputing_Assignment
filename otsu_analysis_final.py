#!/usr/bin/env python3
import cv2
import numpy as np
import json

def find_glare_roi(image, percentile=95, roi_size=96):
    """Find glare ROI automatically (top 5% brightness region)"""
    h, w = image.shape[:2]

    # Calculate brightness threshold for top 5%
    threshold = np.percentile(image, percentile)

    # Find brightest regions
    bright_mask = image > threshold

    # Find center of mass of bright regions
    coords = np.argwhere(bright_mask)
    if len(coords) > 0:
        center_y, center_x = coords.mean(axis=0).astype(int)
    else:
        # Fallback to image center if no bright regions
        center_y, center_x = h//2, w//2

    # Ensure ROI stays within bounds
    x = max(0, min(center_x - roi_size//2, w - roi_size))
    y = max(0, min(center_y - roi_size//2, h - roi_size))

    return (int(x), int(y), int(roi_size), int(roi_size))

def compute_binarization_metrics(binary_image):
    """Compute binarization quality metrics"""
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (binary_image > 0).astype(np.uint8), connectivity=8
    )

    # Skip background (label 0)
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        components = num_labels - 1
        avg_area = float(np.mean(areas)) if len(areas) > 0 else 0.0
    else:
        components = 0
        avg_area = 0.0

    # Count holes (white regions within black regions)
    inverted = cv2.bitwise_not(binary_image)
    num_holes, _, _, _ = cv2.connectedComponentsWithStats(inverted, connectivity=8)
    holes = max(0, num_holes - 1)  # Subtract background

    return {
        "components": int(components),
        "avg_area": round(avg_area, 2),
        "holes": int(holes)
    }

# Load images
original = cv2.imread('images/otsu_shadow_doc_02.jpg', cv2.IMREAD_GRAYSCALE)
global_result = cv2.imread('results/otsu/result_global.png', cv2.IMREAD_GRAYSCALE)
local_result = cv2.imread('results/otsu/result_sliding.png', cv2.IMREAD_GRAYSCALE)
improved_result = cv2.imread('results/otsu/result_improved.png', cv2.IMREAD_GRAYSCALE)

# Find glare ROI
glare_roi = find_glare_roi(original)
x, y, w, h = glare_roi

# Extract ROI regions
roi_global = global_result[y:y+h, x:x+w]
roi_local = local_result[y:y+h, x:x+w]
roi_improved = improved_result[y:y+h, x:x+w]

# Compute metrics
metrics_global = compute_binarization_metrics(roi_global)
metrics_local = compute_binarization_metrics(roi_local)
metrics_improved = compute_binarization_metrics(roi_improved)

# Generate JSON output
result = {
    "task": "otsu",
    "image": "images/otsu_shadow_doc_02.jpg",
    "params": {
        "method": "improved",
        "window": 75,
        "stride": 24,
        "preblur": 1.0,
        "morph": ["open,3", "close,3"]
    },
    "metrics": {
        "roi": {
            "name": "glare",
            "xywh": list(glare_roi)
        },
        "binarization": {
            "global": metrics_global,
            "local": metrics_local,
            "improved": metrics_improved
        }
    },
    "artifacts": [
        "results/otsu/threshold_heatmap.png",
        "results/otsu/local_hist_with_T.png",
        "results/otsu/compare_otsu_contact_sheet.png"
    ],
    "warnings": []
}

print(json.dumps(result, indent=2))