#!/usr/bin/env python3
"""Create Otsu comparison metrics and visualization"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
from scipy import ndimage
from skimage import measure

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

    return (x, y, roi_size, roi_size)

def create_xor_map(global_result, improved_result):
    """Create XOR disagreement map between two binary results"""

    # Ensure binary (threshold at 127)
    global_binary = (global_result > 127).astype(np.uint8)
    improved_binary = (improved_result > 127).astype(np.uint8)

    # XOR operation
    xor_map = cv2.bitwise_xor(global_binary, improved_binary)

    # Calculate disagreement percentage
    total_pixels = xor_map.size
    disagreement_pixels = np.sum(xor_map > 0)
    disagreement_ratio = (disagreement_pixels / total_pixels) * 100

    return xor_map * 255, disagreement_ratio

def calculate_edge_continuity(binary_image):
    """Calculate edge continuity metrics (fewer breaks = better)"""

    # Find edges using Canny
    edges = cv2.Canny(binary_image, 50, 150)

    # Find connected components in edge map
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edges, connectivity=8)

    # Count edge breaks (number of separate edge components)
    edge_breaks = max(0, num_labels - 1)  # Exclude background

    # Calculate total edge length
    edge_pixels = np.sum(edges > 0)

    # Edge continuity score (higher = better continuity)
    if edge_breaks > 0:
        continuity_score = edge_pixels / edge_breaks
    else:
        continuity_score = edge_pixels

    return {
        'edge_breaks': edge_breaks,
        'total_edge_pixels': edge_pixels,
        'continuity_score': continuity_score
    }

def calculate_small_regions_metrics(binary_image, size_threshold=50):
    """Calculate small region metrics"""

    # Find all connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (binary_image > 127).astype(np.uint8), connectivity=8
    )

    if num_labels <= 1:  # Only background
        return {
            'small_regions_count': 0,
            'total_regions': 0,
            'avg_small_region_area': 0,
            'small_regions_ratio': 0
        }

    # Get areas (exclude background component 0)
    areas = stats[1:, cv2.CC_STAT_AREA]

    # Find small regions
    small_regions = areas[areas < size_threshold]

    return {
        'small_regions_count': len(small_regions),
        'total_regions': len(areas),
        'avg_small_region_area': float(np.mean(small_regions)) if len(small_regions) > 0 else 0,
        'small_regions_ratio': len(small_regions) / len(areas) if len(areas) > 0 else 0
    }

def create_comparison_visualization(original, global_result, improved_result, xor_map,
                                  roi_coords, global_metrics, improved_metrics,
                                  disagreement_ratio, output_dir):
    """Create comprehensive comparison visualization"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    x, y, w, h = roi_coords

    # Row 1: Full images
    axes[0, 0].imshow(global_result, cmap='gray')
    axes[0, 0].set_title('Global Otsu', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    # Draw ROI box
    rect1 = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
    axes[0, 0].add_patch(rect1)

    axes[0, 1].imshow(improved_result, cmap='gray')
    axes[0, 1].set_title('Improved Otsu', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')

    # Draw ROI box
    rect2 = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
    axes[0, 1].add_patch(rect2)

    # XOR map
    xor_colored = cv2.applyColorMap(xor_map, cv2.COLORMAP_HOT)
    axes[0, 2].imshow(cv2.cvtColor(xor_colored, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title(f'XOR Disagreement Map\n({disagreement_ratio:.2f}% different)',
                        fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')

    # Row 2: ROI comparisons (2x zoom)
    roi_global = global_result[y:y+h, x:x+w]
    roi_improved = improved_result[y:y+h, x:x+w]
    roi_xor = xor_map[y:y+h, x:x+w]

    # Resize ROIs for better visibility
    zoom_factor = 2
    roi_global_zoom = cv2.resize(roi_global, (w*zoom_factor, h*zoom_factor),
                                interpolation=cv2.INTER_NEAREST)
    roi_improved_zoom = cv2.resize(roi_improved, (w*zoom_factor, h*zoom_factor),
                                  interpolation=cv2.INTER_NEAREST)
    roi_xor_zoom = cv2.resize(roi_xor, (w*zoom_factor, h*zoom_factor),
                             interpolation=cv2.INTER_NEAREST)

    axes[1, 0].imshow(roi_global_zoom, cmap='gray')
    axes[1, 0].set_title('ROI - Global Otsu (2x)', fontsize=12)
    axes[1, 0].axis('off')

    axes[1, 1].imshow(roi_improved_zoom, cmap='gray')
    axes[1, 1].set_title('ROI - Improved Otsu (2x)', fontsize=12)
    axes[1, 1].axis('off')

    # XOR ROI with colormap
    roi_xor_colored = cv2.applyColorMap(roi_xor_zoom, cv2.COLORMAP_HOT)
    axes[1, 2].imshow(cv2.cvtColor(roi_xor_colored, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title('ROI - Disagreement (2x)', fontsize=12)
    axes[1, 2].axis('off')

    # Add metrics text
    metrics_text = f"""
Global Otsu Metrics:
• Edge breaks: {global_metrics['edge']['edge_breaks']}
• Small regions: {global_metrics['regions']['small_regions_count']}
• Avg small area: {global_metrics['regions']['avg_small_region_area']:.1f}

Improved Otsu Metrics:
• Edge breaks: {improved_metrics['edge']['edge_breaks']}
• Small regions: {improved_metrics['regions']['small_regions_count']}
• Avg small area: {improved_metrics['regions']['avg_small_region_area']:.1f}

Overall Disagreement: {disagreement_ratio:.2f}%
    """

    plt.figtext(0.02, 0.5, metrics_text, fontsize=10, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.suptitle('Otsu Methods Comparison - Document Binarization Quality',
                 fontsize=16, fontweight='bold', y=0.95)

    plt.tight_layout()
    plt.subplots_adjust(left=0.15)  # Make room for metrics text
    plt.savefig(f'{output_dir}/compare_panel.png', dpi=150, bbox_inches='tight')
    plt.close()

def main():
    print("Creating Otsu comparison metrics...")

    # Load images
    original = cv2.imread('images/otsu_shadow_doc_02.jpg', cv2.IMREAD_GRAYSCALE)
    global_result = cv2.imread('results/otsu/result_global.png', cv2.IMREAD_GRAYSCALE)
    improved_result = cv2.imread('results/otsu/result_improved.png', cv2.IMREAD_GRAYSCALE)

    if global_result is None or improved_result is None:
        print("Error: Could not load Otsu results. Please run Otsu processing first.")
        return

    # Create output directory
    output_dir = 'results/otsu_metrics'
    os.makedirs(output_dir, exist_ok=True)

    # Find glare ROI
    roi_coords = find_glare_roi(original)
    x, y, w, h = roi_coords

    print(f"Using glare ROI: ({x}, {y}, {w}, {h})")

    # Create XOR map
    print("Creating XOR disagreement map...")
    xor_map, disagreement_ratio = create_xor_map(global_result, improved_result)

    # Save XOR map
    plt.figure(figsize=(10, 8))
    xor_colored = cv2.applyColorMap(xor_map, cv2.COLORMAP_HOT)
    plt.imshow(cv2.cvtColor(xor_colored, cv2.COLOR_BGR2RGB))
    plt.title(f'XOR Disagreement Map\n{disagreement_ratio:.2f}% of pixels differ between methods')
    plt.axis('off')
    plt.colorbar(label='Disagreement (White = Different)', shrink=0.8)
    plt.savefig(f'{output_dir}/xor_map.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Extract ROIs for analysis
    roi_global = global_result[y:y+h, x:x+w]
    roi_improved = improved_result[y:y+h, x:x+w]

    # Calculate metrics for ROIs
    print("Calculating edge continuity metrics...")
    global_edge_metrics = calculate_edge_continuity(roi_global)
    improved_edge_metrics = calculate_edge_continuity(roi_improved)

    print("Calculating small regions metrics...")
    global_region_metrics = calculate_small_regions_metrics(roi_global)
    improved_region_metrics = calculate_small_regions_metrics(roi_improved)

    # Combine metrics
    global_metrics = {
        'edge': global_edge_metrics,
        'regions': global_region_metrics
    }

    improved_metrics = {
        'edge': improved_edge_metrics,
        'regions': improved_region_metrics
    }

    # Create comparison visualization
    print("Creating comparison visualization...")
    create_comparison_visualization(original, global_result, improved_result, xor_map,
                                  roi_coords, global_metrics, improved_metrics,
                                  disagreement_ratio, output_dir)

    # Create metrics CSV
    print("Creating metrics table...")
    metrics_data = [
        {
            'method': 'Global Otsu',
            'roi_x': x, 'roi_y': y, 'roi_w': w, 'roi_h': h,
            'disagreement_ratio_percent': disagreement_ratio,
            'edge_breaks': global_edge_metrics['edge_breaks'],
            'total_edge_pixels': global_edge_metrics['total_edge_pixels'],
            'continuity_score': global_edge_metrics['continuity_score'],
            'small_regions_count': global_region_metrics['small_regions_count'],
            'total_regions': global_region_metrics['total_regions'],
            'avg_small_region_area': global_region_metrics['avg_small_region_area'],
            'small_regions_ratio': global_region_metrics['small_regions_ratio']
        },
        {
            'method': 'Improved Otsu',
            'roi_x': x, 'roi_y': y, 'roi_w': w, 'roi_h': h,
            'disagreement_ratio_percent': disagreement_ratio,
            'edge_breaks': improved_edge_metrics['edge_breaks'],
            'total_edge_pixels': improved_edge_metrics['total_edge_pixels'],
            'continuity_score': improved_edge_metrics['continuity_score'],
            'small_regions_count': improved_region_metrics['small_regions_count'],
            'total_regions': improved_region_metrics['total_regions'],
            'avg_small_region_area': improved_region_metrics['avg_small_region_area'],
            'small_regions_ratio': improved_region_metrics['small_regions_ratio']
        }
    ]

    df = pd.DataFrame(metrics_data)
    df.to_csv(f'{output_dir}/metrics.csv', index=False)

    print("Done!")

    # Print summary
    print(f"\n=== Summary ===")
    print(f"Overall disagreement: {disagreement_ratio:.2f}%")
    print(f"Global Otsu - Edge breaks: {global_edge_metrics['edge_breaks']}, Small regions: {global_region_metrics['small_regions_count']}")
    print(f"Improved Otsu - Edge breaks: {improved_edge_metrics['edge_breaks']}, Small regions: {improved_region_metrics['small_regions_count']}")

    # Output JSON
    result = {
        "task": "otsu_metrics",
        "artifacts": [
            "results/otsu_metrics/xor_map.png",
            "results/otsu_metrics/compare_panel.png",
            "results/otsu_metrics/metrics.csv"
        ]
    }

    print(json.dumps(result))

if __name__ == '__main__':
    main()