#!/usr/bin/env python3
"""Parameter ablation study for HE and Otsu"""

import cv2
import numpy as np
import pandas as pd
import json
import os
from itertools import product
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def compute_he_metrics(original, result, roi):
    """Compute HE quality metrics for ROI"""
    x, y, w, h = roi
    orig_patch = original[y:y+h, x:x+w]
    result_patch = result[y:y+h, x:x+w]

    # Convert to grayscale if needed
    if len(orig_patch.shape) == 3:
        orig_patch = cv2.cvtColor(orig_patch, cv2.COLOR_BGR2GRAY)
    if len(result_patch.shape) == 3:
        result_patch = cv2.cvtColor(result_patch, cv2.COLOR_BGR2GRAY)

    # RMS contrast
    mean_val = np.mean(result_patch)
    rms_contrast = np.sqrt(np.mean((result_patch - mean_val) ** 2))

    # Edge strength (Sobel)
    sobelx = cv2.Sobel(result_patch, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(result_patch, cv2.CV_64F, 0, 1, ksize=3)
    edge_strength = np.mean(np.sqrt(sobelx**2 + sobely**2))

    return rms_contrast, edge_strength

def run_he_ablation():
    """Run HE parameter ablation study"""
    print("Starting HE ablation study...")

    # Parameters to test
    tiles = [(8, 8), (16, 16)]
    clips = [2.0, 2.5, 3.0]

    # Load image
    img = cv2.imread('images/he_dark_indoor.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find dark ROI (lowest brightness 96x96 patch)
    h, w = gray.shape
    min_brightness = float('inf')
    best_roi = None

    for y in range(0, h - 96 + 1, 48):
        for x in range(0, w - 96 + 1, 48):
            patch = gray[y:y+96, x:x+96]
            brightness = np.mean(patch)
            if brightness < min_brightness:
                min_brightness = brightness
                best_roi = (x, y, 96, 96)

    results = []

    for tile, clip in product(tiles, clips):
        try:
            # Apply CLAHE
            clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)

            # Convert to YUV, apply CLAHE to Y channel
            yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            yuv[:,:,0] = clahe.apply(yuv[:,:,0])
            result = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

            # Compute metrics
            rms_contrast, edge_strength = compute_he_metrics(img, result, best_roi)

            # Weighted score (0.6 RMS + 0.4 edge)
            score = 0.6 * rms_contrast + 0.4 * edge_strength

            results.append({
                'tile_h': tile[0],
                'tile_w': tile[1],
                'clip': clip,
                'rms_contrast': round(rms_contrast, 2),
                'edge_strength': round(edge_strength, 2),
                'score': round(score, 2)
            })

            print(f"  Tested tile={tile}, clip={clip}, score={score:.2f}")

        except Exception as e:
            print(f"  Error with tile={tile}, clip={clip}: {e}")

    # Save results
    os.makedirs('results/ablation', exist_ok=True)

    df = pd.DataFrame(results)
    df.to_csv('results/ablation/ablation_he.csv', index=False)

    # Get top 3
    df_sorted = df.sort_values('score', ascending=False)
    top3 = []
    for _, row in df_sorted.head(3).iterrows():
        top3.append({
            'tile': [int(row['tile_h']), int(row['tile_w'])],
            'clip': float(row['clip']),
            'score': float(row['score'])
        })

    with open('results/ablation/ablation_he_top3.json', 'w') as f:
        json.dump(top3, f, indent=2)

    print(f"HE ablation complete. Top score: {df_sorted.iloc[0]['score']}")
    return top3

def compute_otsu_metrics(binary_roi):
    """Compute Otsu quality metrics"""
    # Connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (binary_roi > 127).astype(np.uint8), connectivity=8
    )
    components = max(0, num_labels - 1)  # Exclude background

    # Holes (inverted components)
    inverted = cv2.bitwise_not(binary_roi)
    num_holes, _, _, _ = cv2.connectedComponentsWithStats(
        (inverted > 127).astype(np.uint8), connectivity=8
    )
    holes = max(0, num_holes - 1)

    # Edge continuity (inverse of edge breaks)
    edges = cv2.Canny(binary_roi, 50, 150)
    edge_pixels = np.sum(edges > 0)

    # Score: lower components/holes better, higher edge continuity better
    # Normalize and combine
    score = 100.0 / (1 + components + holes) + edge_pixels / 1000.0

    return components, holes, edge_pixels, score

def run_otsu_ablation():
    """Run Otsu parameter ablation study"""
    print("\nStarting Otsu ablation study...")

    # Parameters to test
    windows = [51, 75, 101]
    strides = [16, 24, 32]
    preblurs = [0.8, 1.0, 1.2]

    # Load image
    img = cv2.imread('images/otsu_shadow_doc_02.jpg', cv2.IMREAD_GRAYSCALE)

    # Find glare ROI (top 5% brightness)
    threshold = np.percentile(img, 95)
    bright_mask = img > threshold
    coords = np.argwhere(bright_mask)

    if len(coords) > 0:
        center_y, center_x = coords.mean(axis=0).astype(int)
    else:
        center_y, center_x = img.shape[0]//2, img.shape[1]//2

    # Define 96x96 ROI
    h, w = img.shape
    x = max(0, min(center_x - 48, w - 96))
    y = max(0, min(center_y - 48, h - 96))
    glare_roi = (x, y, 96, 96)

    results = []

    for window, stride, preblur in product(windows, strides, preblurs):
        try:
            # Import Otsu function
            from src.otsu import improved_otsu

            # Apply improved Otsu
            result = improved_otsu(img, window, stride, preblur, ['open,3', 'close,3'])

            # Extract ROI
            roi_binary = result['result'][y:y+96, x:x+96]

            # Compute metrics
            components, holes, edge_pixels, score = compute_otsu_metrics(roi_binary)

            results.append({
                'window': window,
                'stride': stride,
                'preblur': preblur,
                'components': components,
                'holes': holes,
                'edge_pixels': edge_pixels,
                'score': round(score, 2)
            })

            print(f"  Tested window={window}, stride={stride}, preblur={preblur}, score={score:.2f}")

        except Exception as e:
            print(f"  Error with window={window}, stride={stride}, preblur={preblur}: {e}")

    # Save results
    df = pd.DataFrame(results)
    df.to_csv('results/ablation/ablation_otsu.csv', index=False)

    # Get top 3
    df_sorted = df.sort_values('score', ascending=False)
    top3 = []
    for _, row in df_sorted.head(3).iterrows():
        top3.append({
            'window': int(row['window']),
            'stride': int(row['stride']),
            'preblur': float(row['preblur']),
            'score': float(row['score'])
        })

    with open('results/ablation/ablation_otsu_top3.json', 'w') as f:
        json.dump(top3, f, indent=2)

    print(f"Otsu ablation complete. Top score: {df_sorted.iloc[0]['score']}")
    return top3

def main():
    # Run ablation studies
    he_top3 = run_he_ablation()
    otsu_top3 = run_otsu_ablation()

    # Generate final JSON output
    result = {
        "task": "ablation",
        "he_top3": he_top3,
        "otsu_top3": otsu_top3,
        "artifacts": [
            "results/ablation/ablation_he.csv",
            "results/ablation/ablation_he_top3.json",
            "results/ablation/ablation_otsu.csv",
            "results/ablation/ablation_otsu_top3.json"
        ]
    }

    print("\n" + json.dumps(result, indent=2))

if __name__ == '__main__':
    main()