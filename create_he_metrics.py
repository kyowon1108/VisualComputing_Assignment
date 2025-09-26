#!/usr/bin/env python3
"""Generate comprehensive HE quality metrics"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
from skimage.metrics import structural_similarity as ssim
from colorspacious import cspace_convert

def rgb_to_lab(rgb_image):
    """Convert RGB to Lab color space"""
    # Normalize to [0,1] for colorspacious
    rgb_norm = rgb_image.astype(np.float32) / 255.0
    lab = cspace_convert(rgb_norm, "sRGB1", "CIELab")
    return lab

def delta_e_2000(lab1, lab2):
    """Calculate Delta E 2000 between two Lab images"""
    # Flatten arrays for colorspacious
    lab1_flat = lab1.reshape(-1, 3)
    lab2_flat = lab2.reshape(-1, 3)

    # Calculate Delta E for each pixel
    delta_e = np.zeros(lab1_flat.shape[0])
    for i in range(lab1_flat.shape[0]):
        try:
            delta_e[i] = cspace_convert([lab1_flat[i], lab2_flat[i]], "CIELab", "delta_E_CIE2000")
        except:
            delta_e[i] = 0  # Handle edge cases

    return delta_e.reshape(lab1.shape[:2])

def apply_he_methods(image):
    """Apply all HE methods and return results"""
    results = {}

    # RGB-HE (global HE on each channel)
    rgb_he = np.zeros_like(image)
    for i in range(3):
        channel = image[:,:,i]
        hist, bins = np.histogram(channel.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * 255 / cdf[-1]
        rgb_he[:,:,i] = np.interp(channel.flatten(), bins[:-1], cdf_normalized).reshape(channel.shape)

    results['rgb_he'] = rgb_he.astype(np.uint8)

    # Y-HE (YUV space, HE on Y channel only)
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    y_channel = yuv[:,:,0]
    hist, bins = np.histogram(y_channel.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * 255 / cdf[-1]
    yuv[:,:,0] = np.interp(y_channel.flatten(), bins[:-1], cdf_normalized).reshape(y_channel.shape)
    results['y_he'] = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    # CLAHE (YUV space, CLAHE on Y channel)
    yuv_clahe = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    yuv_clahe[:,:,0] = clahe.apply(yuv_clahe[:,:,0])
    results['clahe'] = cv2.cvtColor(yuv_clahe, cv2.COLOR_YUV2BGR)

    return results

def create_difference_maps(original_gray, results):
    """Create absolute difference maps"""
    diff_maps = {}

    for method, result in results.items():
        result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        diff = np.abs(original_gray.astype(np.float32) - result_gray.astype(np.float32))
        diff_normalized = (diff / diff.max() * 255).astype(np.uint8)
        diff_maps[method] = diff_normalized

    return diff_maps

def create_ssim_maps(original_gray, results):
    """Create SSIM scores and maps"""
    ssim_data = {}

    for method, result in results.items():
        result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

        # Global SSIM score
        ssim_score, ssim_map = ssim(original_gray, result_gray, full=True, data_range=255)

        # Normalize SSIM map to 0-255
        ssim_map_normalized = ((ssim_map + 1) / 2 * 255).astype(np.uint8)

        ssim_data[method] = {
            'score': ssim_score,
            'map': ssim_map_normalized
        }

    return ssim_data

def create_delta_e_maps(original, results):
    """Create Delta E 2000 maps"""
    delta_e_maps = {}

    # Convert original to Lab
    original_lab = rgb_to_lab(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))

    for method, result in results.items():
        result_lab = rgb_to_lab(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        delta_e_map = delta_e_2000(original_lab, result_lab)
        delta_e_maps[method] = delta_e_map

    return delta_e_maps

def calculate_roi_stats(maps, roi_coords):
    """Calculate ROI statistics for maps"""
    roi_stats = {}

    for method, map_data in maps.items():
        x, y, w, h = roi_coords
        roi_values = map_data[y:y+h, x:x+w]

        roi_stats[method] = {
            'mean': float(np.mean(roi_values)),
            'std': float(np.std(roi_values))
        }

    return roi_stats

def save_individual_plots(diff_maps, ssim_data, delta_e_maps, output_dir):
    """Save individual plots with colorbars"""

    # Difference maps
    for method, diff_map in diff_maps.items():
        plt.figure(figsize=(8, 6))
        plt.imshow(diff_map, cmap='hot', vmin=0, vmax=255)
        plt.colorbar(label='Absolute Difference')
        plt.title(f'Difference Map - {method.upper().replace("_", "-")}')
        plt.axis('off')
        plt.savefig(f'{output_dir}/diff_{method}.png', dpi=150, bbox_inches='tight')
        plt.close()

    # SSIM maps
    for method, ssim_info in ssim_data.items():
        plt.figure(figsize=(8, 6))
        plt.imshow(ssim_info['map'], cmap='viridis', vmin=0, vmax=255)
        plt.colorbar(label='SSIM Index')
        plt.title(f'SSIM Map - {method.upper().replace("_", "-")} (Score: {ssim_info["score"]:.3f})')
        plt.axis('off')
        plt.savefig(f'{output_dir}/ssim_{method}.png', dpi=150, bbox_inches='tight')
        plt.close()

    # Delta E maps
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    methods = list(delta_e_maps.keys())

    for i, method in enumerate(methods):
        delta_e_map = delta_e_maps[method]
        im = axes[i].imshow(delta_e_map, cmap='plasma', vmin=0, vmax=50)
        axes[i].set_title(f'{method.upper().replace("_", "-")}')
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], label='ΔE 2000')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/deltaE_maps.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_combined_visualization(original, results, diff_maps, ssim_data, delta_e_maps, output_dir):
    """Create combined visualization table"""

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # Row 1: Original images
    methods = ['original'] + list(results.keys())
    images = [original] + list(results.values())
    method_names = ['Original', 'RGB-HE', 'Y-HE', 'CLAHE']

    for i, (method, img, name) in enumerate(zip(methods, images, method_names)):
        axes[0, i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0, i].set_title(name, fontsize=14, fontweight='bold')
        axes[0, i].axis('off')

    # Row 2: Metrics maps
    result_methods = list(results.keys())

    for i, method in enumerate(result_methods):
        if i == 0:
            # Difference map
            im = axes[1, i+1].imshow(diff_maps[method], cmap='hot')
            axes[1, i+1].set_title(f'Difference Map\n{method.upper().replace("_", "-")}')
            plt.colorbar(im, ax=axes[1, i+1], fraction=0.046)
        elif i == 1:
            # SSIM map
            im = axes[1, i+1].imshow(ssim_data[method]['map'], cmap='viridis')
            axes[1, i+1].set_title(f'SSIM Map\n{method.upper().replace("_", "-")} ({ssim_data[method]["score"]:.3f})')
            plt.colorbar(im, ax=axes[1, i+1], fraction=0.046)
        else:
            # Delta E map
            im = axes[1, i+1].imshow(delta_e_maps[method], cmap='plasma', vmin=0, vmax=50)
            axes[1, i+1].set_title(f'ΔE 2000 Map\n{method.upper().replace("_", "-")}')
            plt.colorbar(im, ax=axes[1, i+1], fraction=0.046)

        axes[1, i+1].axis('off')

    # Hide first subplot in second row
    axes[1, 0].axis('off')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/he_metrics_collage.png', dpi=150, bbox_inches='tight')
    plt.close()

def main():
    print("Creating HE quality metrics...")

    # Load original image
    original = cv2.imread('images/he_dark_indoor.jpg')
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    # Create output directory
    output_dir = 'results/he_metrics'
    os.makedirs(output_dir, exist_ok=True)

    # Apply HE methods
    print("Applying HE methods...")
    results = apply_he_methods(original)

    # Calculate metrics
    print("Calculating difference maps...")
    diff_maps = create_difference_maps(original_gray, results)

    print("Calculating SSIM maps...")
    ssim_data = create_ssim_maps(original_gray, results)

    print("Calculating Delta E maps...")
    delta_e_maps = create_delta_e_maps(original, results)

    # Define ROI (dark region from previous analysis)
    roi_coords = (432, 96, 96, 96)

    # Calculate ROI statistics
    delta_e_roi_stats = calculate_roi_stats(delta_e_maps, roi_coords)

    # Save individual plots
    print("Saving individual plots...")
    save_individual_plots(diff_maps, ssim_data, delta_e_maps, output_dir)

    # Create combined visualization
    print("Creating combined visualization...")
    create_combined_visualization(original, results, diff_maps, ssim_data, delta_e_maps, output_dir)

    # Create metrics table
    print("Creating metrics table...")
    metrics_data = []

    for method in results.keys():
        metrics_data.append({
            'method': method,
            'ssim_global': ssim_data[method]['score'],
            'delta_e_roi_mean': delta_e_roi_stats[method]['mean'],
            'delta_e_roi_std': delta_e_roi_stats[method]['std'],
            'diff_max': float(np.max(diff_maps[method])),
            'diff_mean': float(np.mean(diff_maps[method]))
        })

    df = pd.DataFrame(metrics_data)
    df.to_csv(f'{output_dir}/he_metrics_table.csv', index=False)

    print("Done!")

    # Generate artifacts list
    artifacts = []
    for method in results.keys():
        artifacts.append(f"results/he_metrics/diff_{method}.png")
        artifacts.append(f"results/he_metrics/ssim_{method}.png")

    artifacts.extend([
        "results/he_metrics/deltaE_maps.png",
        "results/he_metrics/he_metrics_collage.png",
        "results/he_metrics/he_metrics_table.csv"
    ])

    # Output JSON
    result = {
        "task": "he_metrics",
        "artifacts": artifacts
    }

    print(json.dumps(result))

if __name__ == '__main__':
    main()