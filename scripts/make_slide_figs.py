#!/usr/bin/env python3
"""Lightweight slide figure generation script (English-only)"""

import os
import argparse
from pathlib import Path
import glob
import matplotlib.pyplot as plt
import cv2
import numpy as np

def find_he_results():
    """Find HE result images"""
    results = {}
    base_path = "results/he"
    if not os.path.exists(base_path):
        return results

    patterns = {
        'original': f'{base_path}/*_original.png',
        'rgb_he': f'{base_path}/*_rgb_he.png',
        'y_he': f'{base_path}/*_yuv_he.png',
        'clahe': f'{base_path}/*_clahe.png'
    }

    for key, pattern in patterns.items():
        files = glob.glob(pattern)
        if files:
            results[key] = files[0]

    return results

def find_otsu_results():
    """Find Otsu result images"""
    results = {}
    base_path = "results/otsu"
    if not os.path.exists(base_path):
        return results

    patterns = {
        'original_gray': f'{base_path}/*_original_gray.png',
        'global': f'{base_path}/*_global_otsu.png',
        'improved': f'{base_path}/*_improved_otsu.png'
    }

    for key, pattern in patterns.items():
        files = glob.glob(pattern)
        if files:
            results[key] = files[0]

    return results

def create_he_summary(he_images, output_path):
    """Create 4-up HE summary: original/rgb_he/y_he/clahe"""
    if len(he_images) < 4:
        print(f"Warning: Only {len(he_images)} HE images found, need 4 for complete summary")
        return False

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    titles = ['Original', 'RGB-HE', 'Y-HE (YUV)', 'CLAHE']
    keys = ['original', 'rgb_he', 'y_he', 'clahe']

    for i, (ax, title, key) in enumerate(zip(axes.flat, titles, keys)):
        if key in he_images:
            img = cv2.imread(he_images[key])
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img_rgb)
        ax.set_title(title)
        ax.axis('off')

    plt.suptitle('Histogram Equalization Methods Comparison')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return True

def create_otsu_summary(otsu_images, output_path):
    """Create 3-up Otsu summary: original_gray/global/improved"""
    if len(otsu_images) < 3:
        print(f"Warning: Only {len(otsu_images)} Otsu images found, need 3 for complete summary")
        return False

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ['Original Grayscale', 'Global Otsu', 'Improved Otsu']
    keys = ['original_gray', 'global', 'improved']

    for ax, title, key in zip(axes, titles, keys):
        if key in otsu_images:
            img = cv2.imread(otsu_images[key], cv2.IMREAD_GRAYSCALE)
            if img is not None:
                ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')

    plt.suptitle('Otsu Thresholding Methods Comparison')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return True

def main():
    parser = argparse.ArgumentParser(description='Generate slide summary figures')
    parser.add_argument('--force', action='store_true', help='Force regeneration even if files exist')
    args = parser.parse_args()

    # Ensure output directory
    output_dir = Path('results/slides')
    output_dir.mkdir(parents=True, exist_ok=True)

    he_summary_path = output_dir / 'he_summary.png'
    otsu_summary_path = output_dir / 'otsu_summary.png'

    # Check if files exist and skip if not forced
    if not args.force:
        if he_summary_path.exists() and otsu_summary_path.exists():
            print("Summary slides already exist. Use --force to regenerate.")
            return 0

    # Generate HE summary
    he_images = find_he_results()
    if he_images:
        if create_he_summary(he_images, he_summary_path):
            print(f"Created: {he_summary_path}")
        else:
            print("Failed to create HE summary")
    else:
        print("No HE result images found")

    # Generate Otsu summary
    otsu_images = find_otsu_results()
    if otsu_images:
        if create_otsu_summary(otsu_images, otsu_summary_path):
            print(f"Created: {otsu_summary_path}")
        else:
            print("Failed to create Otsu summary")
    else:
        print("No Otsu result images found")

    return 0

if __name__ == '__main__':
    exit(main())