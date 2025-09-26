#!/usr/bin/env python3
"""Generate HE and Otsu quality metrics"""

import os
import sys
import argparse
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import csv

try:
    from skimage.metrics import structural_similarity
    from skimage.color import rgb2lab, deltaE_cie76, deltaE_ciede2000
    import pandas as pd
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install scikit-image pandas")
    sys.exit(1)

def load_image(path, as_gray=False):
    """Load image safely"""
    if not os.path.exists(path):
        return None
    if as_gray:
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None

def compute_diff_map(original, processed):
    """Compute difference map (0-255 range)"""
    if original.shape != processed.shape:
        processed = cv2.resize(processed, (original.shape[1], original.shape[0]))

    diff = np.abs(original.astype(np.float32) - processed.astype(np.float32))
    return np.clip(diff, 0, 255).astype(np.uint8)

def compute_ssim_map(original, processed):
    """Compute SSIM map"""
    if original.shape != processed.shape:
        processed = cv2.resize(processed, (original.shape[1], original.shape[0]))

    ssim_score, ssim_map = structural_similarity(
        original, processed, data_range=255, full=True
    )
    # Convert SSIM map from [-1,1] to [0,255] for visualization
    ssim_vis = ((ssim_map + 1) * 127.5).astype(np.uint8)
    return ssim_score, ssim_vis

def compute_deltaE_map(original_rgb, processed_rgb, chroma_only=False):
    """Compute Delta E map using CIE2000"""
    if original_rgb.shape != processed_rgb.shape:
        processed_rgb = cv2.resize(processed_rgb, (original_rgb.shape[1], original_rgb.shape[0]))

    # Convert to LAB (input should be float [0,1])
    orig_lab = rgb2lab(original_rgb.astype(np.float32) / 255.0)
    proc_lab = rgb2lab(processed_rgb.astype(np.float32) / 255.0)

    if chroma_only:
        # Set L* of processed = L* of original for chroma-only comparison
        proc_lab[:,:,0] = orig_lab[:,:,0]

    # Compute Delta E 2000
    delta_e = deltaE_ciede2000(orig_lab, proc_lab)

    # Clip to reasonable visualization range (0-50)
    delta_e_vis = np.clip(delta_e, 0, 50)

    return delta_e.mean(), delta_e.max(), delta_e_vis

def save_colormap_image(data, output_path, title, cmap='viridis', vmin=None, vmax=None):
    """Save data as colormap image"""
    plt.figure(figsize=(8, 6))
    plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    plt.colorbar(shrink=0.8)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight', metadata={"Date": ""})
    plt.close()

def generate_he_metrics(force=False):
    """Generate HE quality metrics"""
    print("Generating HE metrics...")

    # Load original image
    original_path = "images/he_dark_indoor.jpg"
    original_rgb = load_image(original_path)
    if original_rgb is None:
        print(f"Warning: Original HE image not found: {original_path}")
        return False

    original_gray = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)

    # Find processed images
    he_patterns = {
        'rgb_he': 'results/he/*rgb*he*.png',
        'y_he': 'results/he/result_yuv_he.png',
        'clahe': 'results/he/*clahe*.png'
    }

    he_images = {}
    for name, pattern in he_patterns.items():
        files = glob.glob(pattern)
        if not files:
            # Try alternative patterns
            alt_patterns = {
                'rgb_he': 'results/he/*rgb*global*.png',
                'y_he': 'results/he/*yuv*global*.png',
                'clahe': 'results/he/*yuv*clahe*.png'
            }
            if name in alt_patterns:
                files = glob.glob(alt_patterns[name])

        if files:
            he_images[name] = files[0]
            print(f"Found {name}: {files[0]}")
        else:
            print(f"Warning: No images found for pattern: {pattern}")

    if not he_images:
        print("No HE processed images found")
        return False

    # Generate metrics for each method
    metrics_data = []
    output_dir = "results/he_metrics_fixed"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for name, img_path in he_images.items():
        processed_rgb = load_image(img_path)
        if processed_rgb is None:
            continue

        processed_gray = cv2.cvtColor(processed_rgb, cv2.COLOR_RGB2GRAY)

        # Compute metrics
        diff_map = compute_diff_map(original_gray, processed_gray)
        ssim_score, ssim_map = compute_ssim_map(original_gray, processed_gray)
        deltaE_mean, deltaE_max, deltaE_map = compute_deltaE_map(original_rgb, processed_rgb)
        deltaE_chroma_mean, deltaE_chroma_max, deltaE_chroma_map = compute_deltaE_map(
            original_rgb, processed_rgb, chroma_only=True
        )

        # Save individual maps
        diff_path = f"{output_dir}/diff_{name}.png"
        ssim_path = f"{output_dir}/ssim_{name}.png"
        deltaE_path = f"{output_dir}/deltaE_{name}.png"
        deltaE_chroma_path = f"{output_dir}/deltaE_chroma_{name}.png"

        if not os.path.exists(diff_path) or force:
            save_colormap_image(diff_map, diff_path, f"Difference Map - {name.upper()}",
                              cmap='hot', vmin=0, vmax=255)

        if not os.path.exists(ssim_path) or force:
            save_colormap_image(ssim_map, ssim_path, f"SSIM Map - {name.upper()}",
                              cmap='gray', vmin=0, vmax=255)

        if not os.path.exists(deltaE_path) or force:
            save_colormap_image(deltaE_map, deltaE_path, f"Delta E Map - {name.upper()}",
                              cmap='plasma', vmin=0, vmax=50)

        if not os.path.exists(deltaE_chroma_path) or force:
            save_colormap_image(deltaE_chroma_map, deltaE_chroma_path,
                              f"Delta E Chroma - {name.upper()}", cmap='plasma', vmin=0, vmax=50)

        # Collect stats
        metrics_data.append({
            'name': name,
            'deltaE_mean': deltaE_mean,
            'deltaE_max': deltaE_max,
            'deltaE_chroma_mean': deltaE_chroma_mean,
            'ssim_score': ssim_score,
            'diff_mean': diff_map.mean()
        })

    # Save stats CSV
    stats_path = f"{output_dir}/he_metrics_stats.csv"
    if not os.path.exists(stats_path) or force:
        df = pd.DataFrame(metrics_data)
        df.to_csv(stats_path, index=False)
        print(f"Saved HE metrics stats: {stats_path}")

    # Generate collages
    collage_types = ['deltaE', 'ssim', 'diff']
    for ctype in collage_types:
        collage_path = f"{output_dir}/{ctype}_collage.png"
        if not os.path.exists(collage_path) or force:
            create_he_collage(output_dir, ctype, collage_path, he_images.keys())

    print("HE metrics generation completed")
    return True

def create_he_collage(base_dir, metric_type, output_path, methods):
    """Create 3-up collage for HE metrics"""
    fig, axes = plt.subplots(1, len(methods), figsize=(15, 5))
    if len(methods) == 1:
        axes = [axes]

    for i, method in enumerate(methods):
        img_path = f"{base_dir}/{metric_type}_{method}.png"
        if os.path.exists(img_path):
            img = plt.imread(img_path)
            axes[i].imshow(img)
        axes[i].set_title(f"{method.upper()}")
        axes[i].axis('off')

    plt.suptitle(f"HE {metric_type.upper()} Comparison")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Created collage: {output_path}")

def generate_otsu_metrics(force=False):
    """Generate Otsu quality metrics"""
    print("Generating Otsu metrics...")

    # Find Otsu results
    global_path = "results/otsu/global.png"
    improved_path = "results/otsu/improved.png"

    # Try alternative names
    if not os.path.exists(global_path):
        alt_globals = glob.glob("results/otsu/*global*.png")
        global_path = alt_globals[0] if alt_globals else None

    if not os.path.exists(improved_path):
        alt_improved = glob.glob("results/otsu/*improved*.png")
        improved_path = alt_improved[0] if alt_improved else None

    if not global_path or not improved_path:
        print(f"Warning: Missing Otsu images. Global: {global_path}, Improved: {improved_path}")
        return False

    global_img = load_image(global_path, as_gray=True)
    improved_img = load_image(improved_path, as_gray=True)

    if global_img is None or improved_img is None:
        print("Failed to load Otsu images")
        return False

    # Compute connected components stats
    def analyze_components(binary_img):
        # Ensure binary (0 or 255)
        binary = (binary_img > 127).astype(np.uint8) * 255
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

        # Skip background (label 0)
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]  # Skip background
            avg_area = areas.mean()
        else:
            avg_area = 0

        # Count holes (black pixels in binary)
        holes = np.sum(binary == 0)

        return num_labels - 1, avg_area, holes  # -1 to exclude background

    global_components, global_avg_area, global_holes = analyze_components(global_img)
    improved_components, improved_avg_area, improved_holes = analyze_components(improved_img)

    # Save metrics CSV
    output_dir = "results/otsu_metrics"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    metrics_path = f"{output_dir}/metrics.csv"
    if not os.path.exists(metrics_path) or force:
        with open(metrics_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['method', 'components', 'avg_area', 'holes'])
            writer.writerow(['Global', global_components, global_avg_area, global_holes])
            writer.writerow(['Improved', improved_components, improved_avg_area, improved_holes])
        print(f"Saved Otsu metrics: {metrics_path}")

    # Generate metrics table image
    table_path = f"{output_dir}/metrics_table.png"
    if not os.path.exists(table_path) or force:
        create_otsu_table(metrics_path, table_path)

    # Generate XOR map
    xor_path = f"{output_dir}/xor_map.png"
    if not os.path.exists(xor_path) or force:
        create_xor_map(global_img, improved_img, xor_path)

    # Generate comparison panel
    panel_path = f"{output_dir}/compare_panel.png"
    if not os.path.exists(panel_path) or force:
        create_comparison_panel(global_img, improved_img, panel_path)

    print("Otsu metrics generation completed")
    return True

def create_otsu_table(csv_path, output_path):
    """Create table visualization"""
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=df.values, colLabels=df.columns,
                    cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)

    plt.title('Otsu Methods Comparison', fontsize=14, pad=20)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Created table: {output_path}")

def create_xor_map(global_img, improved_img, output_path):
    """Create XOR disagreement map"""
    if global_img.shape != improved_img.shape:
        improved_img = cv2.resize(improved_img, (global_img.shape[1], global_img.shape[0]))

    # Binarize
    global_bin = (global_img > 127).astype(np.uint8)
    improved_bin = (improved_img > 127).astype(np.uint8)

    # XOR to find disagreements
    xor_map = np.bitwise_xor(global_bin, improved_bin) * 255

    plt.figure(figsize=(10, 6))
    plt.imshow(xor_map, cmap='Reds')
    plt.title('Method Disagreement (XOR Map)')
    plt.axis('off')
    plt.colorbar(shrink=0.8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Created XOR map: {output_path}")

def create_comparison_panel(global_img, improved_img, output_path):
    """Create side-by-side comparison"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(global_img, cmap='gray')
    axes[0].set_title('Global Otsu')
    axes[0].axis('off')

    axes[1].imshow(improved_img, cmap='gray')
    axes[1].set_title('Improved Otsu')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Created comparison panel: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate quality metrics')
    parser.add_argument('mode', choices=['he', 'otsu', 'all'], help='Metrics to generate')
    parser.add_argument('--force', action='store_true', help='Overwrite existing files')
    args = parser.parse_args()

    success = True

    if args.mode in ['he', 'all']:
        success &= generate_he_metrics(args.force)

    if args.mode in ['otsu', 'all']:
        success &= generate_otsu_metrics(args.force)

    if success:
        print("Metrics generation completed successfully")
        return 0
    else:
        print("Some metrics generation failed")
        return 1

if __name__ == '__main__':
    sys.exit(main())