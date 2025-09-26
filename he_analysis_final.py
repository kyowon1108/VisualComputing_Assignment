#!/usr/bin/env python3
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def find_roi_patches(image, patch_size=96):
    """Find ROI patches automatically"""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    best_dark = None
    best_bright = None
    best_grad = None
    min_brightness = float('inf')
    max_brightness = float('-inf')
    max_gradient = float('-inf')

    # Scan for patches
    for y in range(0, h - patch_size + 1, 32):
        for x in range(0, w - patch_size + 1, 32):
            patch = gray[y:y+patch_size, x:x+patch_size]

            mean_val = np.mean(patch)

            # Gradient
            sobelx = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
            grad_sum = np.sum(np.sqrt(sobelx**2 + sobely**2))

            if mean_val < min_brightness:
                min_brightness = mean_val
                best_dark = (x, y, patch_size, patch_size)

            if mean_val > max_brightness:
                max_brightness = mean_val
                best_bright = (x, y, patch_size, patch_size)

            if grad_sum > max_gradient:
                max_gradient = grad_sum
                best_grad = (x, y, patch_size, patch_size)

    return best_dark, best_bright, best_grad

def compute_metrics(image, roi):
    """Compute metrics for ROI"""
    x, y, w, h = roi
    patch = image[y:y+h, x:x+w]

    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if len(patch.shape) == 3 else patch

    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))
    rms = float(np.sqrt(np.mean((gray - mean_val)**2)))

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge = float(np.sum(np.sqrt(sobelx**2 + sobely**2)))

    return {
        "mean": round(mean_val, 2),
        "std": round(std_val, 2),
        "rms": round(rms, 2),
        "edge": round(edge, 2)
    }

def create_histogram_plots(images, names, output_prefix):
    """Create histogram and CDF plots"""
    # Individual histograms
    for img, name in zip(images, names):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        # Histogram
        hist, bins = np.histogram(gray.flatten(), bins=256, range=(0, 256))
        ax1.bar(bins[:-1], hist, width=1, color='blue', alpha=0.7)
        ax1.set_title(f'{name} Histogram')
        ax1.set_xlim([0, 255])
        ax1.grid(True, alpha=0.3)

        # CDF
        cdf = hist.cumsum()
        cdf_normalized = cdf / cdf[-1]
        ax2.plot(bins[:-1], cdf_normalized, color='red')
        ax2.set_title(f'{name} CDF')
        ax2.set_xlim([0, 255])
        ax2.set_ylim([0, 1])
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{output_prefix}hist_{name.lower().replace("-","_").replace(" ","_")}.png', dpi=100)
        plt.close()

    # CDF overlay
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['black', 'blue', 'green', 'red']

    for img, name, color in zip(images, names, colors):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        hist, bins = np.histogram(gray.flatten(), bins=256, range=(0, 256))
        cdf = hist.cumsum()
        cdf_normalized = cdf / cdf[-1]
        ax.plot(bins[:-1], cdf_normalized, color=color, label=name, linewidth=2)

    ax.set_title('CDF Overlay')
    ax.set_xlim([0, 255])
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_prefix}cdf_overlay.png', dpi=100)
    plt.close()

def create_contact_sheet(images, names, rois, output_path):
    """Create comparison contact sheet"""
    fig = plt.figure(figsize=(16, 8))

    # Full images row
    for idx, (img, name) in enumerate(zip(images, names)):
        ax = plt.subplot(2, 4, idx + 1)
        display_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(display_img)
        ax.set_title(name)
        ax.axis('off')

        # Draw ROI boxes on original only
        if idx == 0:
            colors_roi = ['red', 'yellow', 'cyan']
            labels = ['dark', 'highlight', 'high_gradient']
            for roi, color, label in zip(rois, colors_roi, labels):
                x, y, w, h = roi
                rect = patches.Rectangle((x, y), w, h, linewidth=2,
                                        edgecolor=color, facecolor='none')
                ax.add_patch(rect)
                ax.text(x, y-3, f'{label}({x},{y})', color=color, fontsize=8)

    # ROI rows
    roi_labels = ['Dark ROI', 'Bright ROI', 'High-Grad ROI']
    for roi_idx, (roi, label) in enumerate(zip(rois, roi_labels)):
        x, y, w, h = roi
        for img_idx, img in enumerate(zip(images)):
            ax = plt.subplot(2, 4, 5 + img_idx)
            roi_crop = img[0][y:y+h, x:x+w]
            roi_rgb = cv2.cvtColor(roi_crop, cv2.COLOR_BGR2RGB)

            # Zoom 2x
            zoomed = cv2.resize(roi_rgb, (w*2, h*2), interpolation=cv2.INTER_LINEAR)

            # Create subplot for each ROI
            if roi_idx == 0:
                ax.imshow(zoomed)
                if img_idx == 0:
                    ax.set_ylabel(label, fontsize=10)
                ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

# Main execution
original = cv2.imread('images/he_dark_indoor.jpg')
rgb_he = cv2.imread('results/he/result_rgb_global.png')
y_he = cv2.imread('results/he/result_yuv_global.png')
clahe = cv2.imread('results/he/result_yuv_clahe.png')

# Find ROIs
roi_dark, roi_bright, roi_grad = find_roi_patches(original)
rois = [roi_dark, roi_bright, roi_grad]

# Calculate metrics
metrics_roi = []
for roi, name in zip(rois, ["dark", "highlight", "high_gradient"]):
    roi_metrics = {
        "name": name,
        "xywh": list(roi),
        "orig": compute_metrics(original, roi),
        "rgb_he": compute_metrics(rgb_he, roi),
        "y_he": compute_metrics(y_he, roi),
        "clahe": compute_metrics(clahe, roi)
    }
    metrics_roi.append(roi_metrics)

# Generate plots
images = [original, rgb_he, y_he, clahe]
names = ['before', 'after_rgb_he', 'after_y_he', 'after_clahe']
create_histogram_plots(images, names, 'results/he/')

# Create contact sheet
create_contact_sheet(images, ['Original', 'RGB-HE', 'Y-HE', 'CLAHE'],
                    rois, 'results/he/compare_he_contact_sheet.png')

# JSON output
result = {
    "task": "he",
    "image": "images/he_dark_indoor.jpg",
    "params": {
        "space_default": "yuv",
        "clahe": {"tile": [8, 8], "clip": 2.5, "bins": 256}
    },
    "metrics": {
        "roi": metrics_roi
    },
    "artifacts": [
        "results/he/compare_he_contact_sheet.png",
        "results/he/hist_before.png",
        "results/he/hist_after_rgb_he.png",
        "results/he/hist_after_y_he.png",
        "results/he/hist_after_clahe.png",
        "results/he/cdf_overlay.png"
    ],
    "warnings": []
}

print(json.dumps(result, indent=2))