#!/usr/bin/env python3
"""
Create Otsu ROI Comparison Image
Compares Original, Global Otsu, and Improved Otsu results with 3 ROI regions highlighted
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def create_otsu_roi_comparison(src_image_path, improved_result_path, output_path):
    """Create comparison showing Original, Global Otsu, Improved Otsu with ROI highlights"""

    # Read images
    original = cv2.imread(src_image_path, cv2.IMREAD_GRAYSCALE)
    improved = cv2.imread(improved_result_path, cv2.IMREAD_GRAYSCALE)

    if original is None or improved is None:
        raise ValueError("Could not load source images")

    # Generate Global Otsu result
    from src.otsu import global_otsu
    global_result = global_otsu(original)['result']

    # Define ROI regions (same as used in CLI script)
    rois = [
        (448, 48, 160, 144),   # ROI 1: Top-right area
        (64, 144, 256, 192),   # ROI 2: Middle-left area
        (32, 24, 128, 384)     # ROI 3: Left side area
    ]

    roi_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Red, Green, Blue
    roi_labels = ['ROI 1', 'ROI 2', 'ROI 3']

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Otsu Thresholding Comparison with ROI Analysis', fontsize=16, fontweight='bold')

    # Top row: Full images
    images = [original, global_result, improved]
    titles = ['Original', 'Global Otsu', 'Improved Otsu']
    cmaps = ['gray', 'gray', 'gray']

    for i, (img, title, cmap) in enumerate(zip(images, titles, cmaps)):
        ax = axes[0, i]
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')

        # Add ROI rectangles
        for roi_idx, (x, y, w, h) in enumerate(rois):
            color = np.array(roi_colors[roi_idx]) / 255.0  # Normalize for matplotlib
            rect = patches.Rectangle((x, y), w, h, linewidth=2,
                                   edgecolor=color, facecolor='none', alpha=0.8)
            ax.add_patch(rect)

            # Add ROI label
            ax.text(x + 5, y + 15, roi_labels[roi_idx],
                   color=color, fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

    # Bottom row: ROI details
    for roi_idx, (x, y, w, h) in enumerate(rois):
        ax = axes[1, roi_idx]

        # Extract ROI regions
        roi_original = original[y:y+h, x:x+w]
        roi_global = global_result[y:y+h, x:x+w]
        roi_improved = improved[y:y+h, x:x+w]

        # Create side-by-side comparison for this ROI
        roi_comparison = np.hstack([roi_original, roi_global, roi_improved])

        ax.imshow(roi_comparison, cmap='gray')
        ax.set_title(f'{roi_labels[roi_idx]} Comparison\n(Original | Global | Improved)',
                    fontsize=12, fontweight='bold')
        ax.axis('off')

        # Add separating lines
        h_roi = roi_original.shape[0]
        w_roi = roi_original.shape[1]
        ax.axvline(x=w_roi-0.5, color='red', linewidth=2, alpha=0.7)
        ax.axvline(x=2*w_roi-0.5, color='red', linewidth=2, alpha=0.7)

        # Add text labels
        ax.text(w_roi//2, -10, 'Original', ha='center', fontsize=10, fontweight='bold')
        ax.text(1.5*w_roi, -10, 'Global', ha='center', fontsize=10, fontweight='bold')
        ax.text(2.5*w_roi, -10, 'Improved', ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()

    # Save the comparison
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"ROI comparison saved: {output_path}")

    return output_path

def main():
    # Paths
    src_image = "images/otsu_shadow_doc_02.jpg"
    improved_result = "results/video/otsu_exact_final_frames/generated_improved.png"
    output_path = "results/video/otsu_roi_comparison.png"

    # Check if improved result exists, fallback to other locations
    if not os.path.exists(improved_result):
        # Try other possible locations
        fallback_paths = [
            "results/otsu/result_improved.png",
            "results/otsu/improved.png",
            "results/video/otsu_exact_pipeline_frames/generated_improved.png"
        ]

        for path in fallback_paths:
            if os.path.exists(path):
                improved_result = path
                print(f"Using improved result from: {improved_result}")
                break
        else:
            print("Error: No improved result found. Please run the improved Otsu first.")
            return 1

    try:
        create_otsu_roi_comparison(src_image, improved_result, output_path)
        print(f"Success! ROI comparison created at: {output_path}")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())