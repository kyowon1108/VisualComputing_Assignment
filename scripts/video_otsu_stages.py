#!/usr/bin/env python3
"""
Improved Otsu Step-by-Step Animation Generator
Creates frames, GIF, and MP4 showing each stage of the improved Otsu process.
"""

import argparse
import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from scipy import interpolate
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
import imageio.v3 as imageio

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from otsu import sliding_window_otsu, compute_otsu_threshold
    HAS_SRC_OTSU = True
except ImportError:
    HAS_SRC_OTSU = False

def add_text_overlay(image, text, position=(10, 30)):
    """Add ASCII text overlay with shadow for better visibility."""
    if len(image.shape) == 2:
        # Convert grayscale to RGB for text overlay
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = image.copy()

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2

    # Add black shadow
    cv2.putText(image_rgb, text, (position[0]+2, position[1]+2),
                font, font_scale, (0, 0, 0), thickness+1)
    # Add white text
    cv2.putText(image_rgb, text, position,
                font, font_scale, (255, 255, 255), thickness)

    return image_rgb

def sliding_window_otsu_fallback(image, window_size=75, stride=24):
    """Fallback implementation of sliding window Otsu."""
    h, w = image.shape

    # Grid generation
    y_coords = np.arange(window_size//2, h - window_size//2, stride)
    x_coords = np.arange(window_size//2, w - window_size//2, stride)

    if len(y_coords) == 0 or len(x_coords) == 0:
        # Fallback for small images
        global_thresh = threshold_otsu(image)
        return np.full_like(image, global_thresh, dtype=np.float32)

    # Threshold grid calculation
    threshold_grid = np.zeros((len(y_coords), len(x_coords)))

    for i, y in enumerate(y_coords):
        for j, x in enumerate(x_coords):
            # Extract window
            y_start = max(0, y - window_size//2)
            y_end = min(h, y + window_size//2)
            x_start = max(0, x - window_size//2)
            x_end = min(w, x + window_size//2)

            window = image[y_start:y_end, x_start:x_end]

            # Calculate threshold for this window
            if window.size > 0:
                try:
                    window_threshold = threshold_otsu(window)
                except ValueError:
                    # Handle uniform regions
                    window_threshold = np.mean(window)
            else:
                window_threshold = 127

            threshold_grid[i, j] = window_threshold

    # Bilinear interpolation to full image
    f = interpolate.RectBivariateSpline(y_coords, x_coords, threshold_grid,
                                      kx=min(3, len(y_coords)-1),
                                      ky=min(3, len(x_coords)-1))

    y_full = np.arange(h)
    x_full = np.arange(w)
    threshold_map = f(y_full, x_full)

    return threshold_map

def remove_small_components(binary_image, min_area=20):
    """Remove connected components smaller than min_area."""
    labeled = label(binary_image > 0)
    regions = regionprops(labeled)

    result = np.zeros_like(binary_image)
    for region in regions:
        if region.area >= min_area:
            coords = region.coords
            result[coords[:, 0], coords[:, 1]] = 255

    return result

def save_heatmap(threshold_map, save_path, title):
    """Save threshold map as heatmap with colorbar."""
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(threshold_map, cmap='viridis', interpolation='nearest')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Threshold Value', rotation=270, labelpad=20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

def process_otsu_stages(src_path, out_dir, window=75, stride=24, preblur=0.8,
                       morph_open_iter=1, morph_close_iter=2, min_area=20, mix_global=0.0):
    """Process all stages of improved Otsu and save frames."""

    # Read image
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Source image not found: {src_path}")

    image = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {src_path}")

    frames = []
    stage_names = [
        "Original (Grayscale)",
        "Pre-blur (Gaussian)",
        "Global Otsu (Reference)",
        "Local Otsu Threshold Heatmap",
        "Local Binarization (Raw)",
        "Morphology Opening",
        "Morphology Closing",
        "Small Component Removal",
        "Final Improved Mask",
        "XOR vs Global (Difference)"
    ]

    # Stage 0: Original
    frame = add_text_overlay(image, "S00: " + stage_names[0])
    frame_path = os.path.join(out_dir, "frame_00_original.png")
    cv2.imwrite(frame_path, frame)
    frames.append(frame)

    # Stage 1: Pre-blur
    if preblur > 0:
        blurred = cv2.GaussianBlur(image, (0, 0), preblur)
    else:
        blurred = image.copy()
    frame = add_text_overlay(blurred, "S01: " + stage_names[1])
    frame_path = os.path.join(out_dir, "frame_01_preblur.png")
    cv2.imwrite(frame_path, frame)
    frames.append(frame)

    # Stage 2: Global Otsu
    global_thresh = threshold_otsu(blurred)
    global_mask = (blurred > global_thresh).astype(np.uint8) * 255
    frame = add_text_overlay(global_mask, "S02: " + stage_names[2])
    frame_path = os.path.join(out_dir, "frame_02_global.png")
    cv2.imwrite(frame_path, frame)
    frames.append(frame)

    # Stage 3: Local Otsu threshold heatmap
    if HAS_SRC_OTSU:
        try:
            otsu_result = sliding_window_otsu(blurred, window, stride)
            threshold_map = otsu_result['threshold_map']
        except Exception:
            threshold_map = sliding_window_otsu_fallback(blurred, window, stride)
    else:
        threshold_map = sliding_window_otsu_fallback(blurred, window, stride)

    # Save heatmap
    heatmap_path = os.path.join(out_dir, "frame_03_heatmap.png")
    save_heatmap(threshold_map, heatmap_path, "S03: " + stage_names[3])

    # Load saved heatmap as frame
    heatmap_frame = cv2.imread(heatmap_path)
    frames.append(heatmap_frame)

    # Stage 4: Local binarization
    local_mask = (blurred > threshold_map).astype(np.uint8) * 255
    frame = add_text_overlay(local_mask, "S04: " + stage_names[4])
    frame_path = os.path.join(out_dir, "frame_04_local_raw.png")
    cv2.imwrite(frame_path, frame)
    frames.append(frame)

    # Stage 5: Morphology opening
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(local_mask, cv2.MORPH_OPEN, kernel, iterations=morph_open_iter)
    frame = add_text_overlay(opened, "S05: " + stage_names[5])
    frame_path = os.path.join(out_dir, "frame_05_opened.png")
    cv2.imwrite(frame_path, frame)
    frames.append(frame)

    # Stage 6: Morphology closing
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=morph_close_iter)
    frame = add_text_overlay(closed, "S06: " + stage_names[6])
    frame_path = os.path.join(out_dir, "frame_06_closed.png")
    cv2.imwrite(frame_path, frame)
    frames.append(frame)

    # Stage 7: Small component removal
    cleaned = remove_small_components(closed, min_area)
    frame = add_text_overlay(cleaned, "S07: " + stage_names[7])
    frame_path = os.path.join(out_dir, "frame_07_cleaned.png")
    cv2.imwrite(frame_path, frame)
    frames.append(frame)

    # Stage 8: Final improved mask (optional global mixing)
    if mix_global > 0 and mix_global <= 1:
        final_mask = ((1 - mix_global) * (cleaned > 0) + mix_global * (global_mask > 0)) * 255
        final_mask = final_mask.astype(np.uint8)
    else:
        final_mask = cleaned

    frame = add_text_overlay(final_mask, "S08: " + stage_names[8])
    frame_path = os.path.join(out_dir, "frame_08_final.png")
    cv2.imwrite(frame_path, frame)
    frames.append(frame)

    # Stage 9: XOR vs Global
    xor_diff = cv2.bitwise_xor(final_mask, global_mask)
    frame = add_text_overlay(xor_diff, "S09: " + stage_names[9])
    frame_path = os.path.join(out_dir, "frame_09_xor.png")
    cv2.imwrite(frame_path, frame)
    frames.append(frame)

    return frames

def create_gif_mp4(frames, gif_path, mp4_path):
    """Create GIF and MP4 from frames."""
    # Ensure all frames are same size and convert to RGB
    processed_frames = []

    # Find max dimensions
    max_h, max_w = 0, 0
    for frame in frames:
        if len(frame.shape) == 3:
            h, w = frame.shape[:2]
        else:
            h, w = frame.shape
        max_h = max(max_h, h)
        max_w = max(max_w, w)

    for frame in frames:
        if len(frame.shape) == 3:
            # BGR to RGB for imageio
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        # Resize to common size if needed
        if frame_rgb.shape[0] != max_h or frame_rgb.shape[1] != max_w:
            frame_rgb = cv2.resize(frame_rgb, (max_w, max_h))

        processed_frames.append(frame_rgb)

    # Create GIF
    try:
        imageio.imwrite(gif_path, processed_frames, loop=0, duration=0.4)
        gif_success = True
    except Exception as e:
        print(f"Warning: GIF creation failed: {e}")
        gif_success = False

    # Create MP4
    try:
        imageio.imwrite(mp4_path, processed_frames, fps=2.5)
        mp4_success = True
    except Exception as e:
        print(f"Warning: MP4 creation failed: {e}")
        mp4_success = False

    return gif_success, mp4_success

def main():
    parser = argparse.ArgumentParser(description='Generate Improved Otsu step-by-step animation')
    parser.add_argument('--src', required=True, help='Source image path')
    parser.add_argument('--out', required=True, help='Output directory for frames')
    parser.add_argument('--window', type=int, default=75, help='Window size')
    parser.add_argument('--stride', type=int, default=24, help='Stride')
    parser.add_argument('--preblur', type=float, default=0.8, help='Pre-blur sigma')
    parser.add_argument('--open', type=int, default=1, help='Opening iterations')
    parser.add_argument('--close', type=int, default=2, help='Closing iterations')
    parser.add_argument('--min-area', type=int, default=20, help='Minimum component area')
    parser.add_argument('--mix-global', type=float, default=0.0, help='Global mixing ratio')
    parser.add_argument('--gif', required=True, help='Output GIF path')
    parser.add_argument('--mp4', required=True, help='Output MP4 path')
    parser.add_argument('--force', action='store_true', help='Force overwrite')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.dirname(args.gif), exist_ok=True)
    os.makedirs(os.path.dirname(args.mp4), exist_ok=True)

    # Process stages
    frames = process_otsu_stages(
        args.src, args.out,
        window=args.window, stride=args.stride, preblur=args.preblur,
        morph_open_iter=args.open, morph_close_iter=args.close,
        min_area=args.min_area, mix_global=args.mix_global
    )

    # Create GIF and MP4
    gif_success, mp4_success = create_gif_mp4(frames, args.gif, args.mp4)

    print(f"Generated {len(frames)} frames in {args.out}")
    if gif_success:
        print(f"Created GIF: {args.gif}")
    if mp4_success:
        print(f"Created MP4: {args.mp4}")

if __name__ == "__main__":
    main()