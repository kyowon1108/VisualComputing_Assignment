#!/usr/bin/env python3
"""
Minimal 3-stage Local Otsu animation: Original → Threshold Map → Final Binary
Validates against existing improved result for accuracy.
"""

import argparse
import os
import sys
import json
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from scipy import interpolate
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.metrics import structural_similarity as ssim
import imageio.v3 as imageio

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from otsu import sliding_window_otsu, compute_otsu_threshold
    HAS_SRC_OTSU = True
except ImportError:
    HAS_SRC_OTSU = False

def add_text_overlay(image, text, position=(10, 40), font_scale=1.0):
    """Add ASCII text overlay with shadow for better visibility."""
    if len(image.shape) == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = image.copy()

    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2

    # Add black shadow
    cv2.putText(image_rgb, text, (position[0]+2, position[1]+2),
                font, font_scale, (0, 0, 0), thickness+1)
    # Add white text
    cv2.putText(image_rgb, text, position,
                font, font_scale, (255, 255, 255), thickness)

    return image_rgb

def create_canvas_frame(image, title, canvas_size=(1280, 720)):
    """Create centered frame on canvas with title."""
    canvas = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Calculate scaling and centering
    canvas_h, canvas_w = canvas_size[1] - 80, canvas_size[0] - 40  # Reserve space for title
    img_h, img_w = image.shape[:2]

    scale = min(canvas_w / img_w, canvas_h / img_h)
    new_w, new_h = int(img_w * scale), int(img_h * scale)

    resized = cv2.resize(image, (new_w, new_h))

    # Center the image
    start_x = (canvas_size[0] - new_w) // 2
    start_y = 80 + (canvas_size[1] - 80 - new_h) // 2

    canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized

    # Add title
    canvas = add_text_overlay(canvas, title, (20, 50), font_scale=1.2)

    return canvas

def create_threshold_heatmap(threshold_map, title, cmap='viridis'):
    """Create threshold heatmap frame using matplotlib."""
    fig, ax = plt.subplots(figsize=(12, 10))

    im = ax.imshow(threshold_map, cmap=cmap, interpolation='nearest')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Threshold Value', rotation=270, labelpad=20)

    plt.tight_layout()

    # Save to temp file and read back
    temp_path = "/tmp/heatmap_minimal.png"
    plt.savefig(temp_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    frame = cv2.imread(temp_path)
    os.remove(temp_path)

    return frame

def sliding_window_otsu_fallback(image, window_size=75, stride=24):
    """Fallback implementation of sliding window Otsu."""
    h, w = image.shape

    # Grid generation
    y_coords = np.arange(window_size//2, h - window_size//2 + 1, stride)
    x_coords = np.arange(window_size//2, w - window_size//2 + 1, stride)

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
            if window.size > 0 and window.max() > window.min():
                try:
                    window_threshold = threshold_otsu(window)
                except (ValueError, RuntimeWarning):
                    # Handle uniform regions
                    window_threshold = np.mean(window)
            else:
                window_threshold = np.mean(window) if window.size > 0 else 127

            threshold_grid[i, j] = window_threshold

    # Bilinear interpolation to full image
    if len(y_coords) > 1 and len(x_coords) > 1:
        f = interpolate.RectBivariateSpline(
            y_coords, x_coords, threshold_grid,
            kx=min(3, len(y_coords)-1), ky=min(3, len(x_coords)-1)
        )

        y_full = np.arange(h)
        x_full = np.arange(w)
        threshold_map = f(y_full, x_full)
    else:
        # Fallback for very small grids
        threshold_map = np.full((h, w), np.mean(threshold_grid))

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

def compute_metrics(img1, img2):
    """Compute SSIM and MSE between two images."""
    if img1.shape != img2.shape:
        return {"ssim": 0.0, "mse": float('inf')}

    # Ensure grayscale
    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if len(img2.shape) == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Normalize to [0, 1]
    img1_norm = img1.astype(np.float32) / 255.0
    img2_norm = img2.astype(np.float32) / 255.0

    # Compute SSIM
    ssim_val = ssim(img1_norm, img2_norm, data_range=1.0)

    # Compute MSE
    mse_val = np.mean((img1_norm - img2_norm) ** 2)

    return {"ssim": float(ssim_val), "mse": float(mse_val)}

def process_minimal_otsu(src_path, frames_dir, gif_path, mp4_path,
                        window=75, stride=24, preblur=0.8, morph_open=1, morph_close=2,
                        min_area=20, mix_global=0.0, cmap='viridis', fps=2.0):
    """Process minimal 3-stage Otsu animation."""

    # Ensure directories exist
    Path(frames_dir).mkdir(parents=True, exist_ok=True)
    Path(gif_path).parent.mkdir(parents=True, exist_ok=True)
    Path(mp4_path).parent.mkdir(parents=True, exist_ok=True)
    Path("results/otsu").mkdir(parents=True, exist_ok=True)

    # Read and preprocess image
    image = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {src_path}")

    h, w = image.shape

    # Pre-blur
    if preblur > 0:
        blurred = cv2.GaussianBlur(image, (0, 0), preblur)
    else:
        blurred = image.copy()

    # Stage 1: Original
    frame_00 = create_canvas_frame(image, "Original")
    frame_00_path = os.path.join(frames_dir, "frame_00_original.png")
    cv2.imwrite(frame_00_path, frame_00)

    # Stage 2: Local Otsu threshold map
    if HAS_SRC_OTSU:
        try:
            otsu_result = sliding_window_otsu(blurred, window, stride)
            threshold_map = otsu_result['threshold_map']
        except Exception:
            threshold_map = sliding_window_otsu_fallback(blurred, window, stride)
    else:
        threshold_map = sliding_window_otsu_fallback(blurred, window, stride)

    # Create heatmap frame
    frame_01 = create_threshold_heatmap(threshold_map, "Local Otsu - Threshold Map", cmap)
    frame_01_path = os.path.join(frames_dir, "frame_01_threshold_map.png")
    cv2.imwrite(frame_01_path, frame_01)

    # Stage 3: Final binarization
    # Apply threshold
    local_mask = (blurred >= threshold_map).astype(np.uint8) * 255

    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    if morph_open > 0:
        local_mask = cv2.morphologyEx(local_mask, cv2.MORPH_OPEN, kernel, iterations=morph_open)
    if morph_close > 0:
        local_mask = cv2.morphologyEx(local_mask, cv2.MORPH_CLOSE, kernel, iterations=morph_close)

    # Remove small components
    if min_area > 0:
        local_mask = remove_small_components(local_mask, min_area)

    # Optional global blending (default 0.0)
    if mix_global > 0:
        global_thresh = threshold_otsu(blurred)
        global_mask = (blurred >= global_thresh).astype(np.uint8) * 255
        local_mask = ((1 - mix_global) * (local_mask > 0) + mix_global * (global_mask > 0)) * 255
        local_mask = local_mask.astype(np.uint8)

    final_mask = local_mask

    frame_02 = create_canvas_frame(final_mask, "Final Binarization")
    frame_02_path = os.path.join(frames_dir, "frame_02_final.png")
    cv2.imwrite(frame_02_path, frame_02)

    # Save final result for auditing
    final_output_path = "results/otsu/improved_from_script.png"
    cv2.imwrite(final_output_path, final_mask)

    # Validate against reference
    final_match = False
    metrics = {"ssim": 0.0, "mse": float('inf')}
    final_ref_used = None

    ref_candidates = [
        "results/otsu/result_improved.png",
        "results/otsu/improved.png"
    ]

    for ref_path in ref_candidates:
        if os.path.exists(ref_path):
            ref_image = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
            if ref_image is not None:
                final_ref_used = ref_path
                metrics = compute_metrics(final_mask, ref_image)

                # Check if match is good enough
                if metrics["ssim"] >= 0.999 and metrics["mse"] <= 0.05:
                    final_match = True
                break

    # Create animation frames
    frames = [frame_00, frame_01, frame_02]

    # Ensure consistent frame sizes
    max_h = max(f.shape[0] for f in frames)
    max_w = max(f.shape[1] for f in frames)

    consistent_frames = []
    for frame in frames:
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if frame.shape[0] != max_h or frame.shape[1] != max_w:
            frame = cv2.resize(frame, (max_w, max_h))

        consistent_frames.append(frame)

    # Create GIF
    try:
        imageio.imwrite(gif_path, consistent_frames, loop=0, duration=1.0/fps)
        gif_success = True
    except Exception as e:
        print(f"Warning: GIF creation failed: {e}")
        gif_success = False

    # Create MP4
    try:
        imageio.imwrite(mp4_path, consistent_frames, fps=fps)
        mp4_success = True
    except Exception as e:
        print(f"Warning: MP4 creation failed: {e}")
        mp4_success = False

    return {
        "final_match": final_match,
        "metrics": metrics,
        "final_ref_used": final_ref_used,
        "final_output_path": final_output_path,
        "gif_success": gif_success,
        "mp4_success": mp4_success
    }

def main():
    parser = argparse.ArgumentParser(description='Generate minimal 3-stage Local Otsu animation')
    parser.add_argument('--src', required=True, help='Source image path')
    parser.add_argument('--out', required=True, help='Output directory for frames')
    parser.add_argument('--gif', required=True, help='Output GIF path')
    parser.add_argument('--mp4', required=True, help='Output MP4 path')
    parser.add_argument('--window', type=int, default=75, help='Window size')
    parser.add_argument('--stride', type=int, default=24, help='Stride')
    parser.add_argument('--preblur', type=float, default=0.8, help='Pre-blur sigma')
    parser.add_argument('--open', type=int, default=1, help='Opening iterations')
    parser.add_argument('--close', type=int, default=2, help='Closing iterations')
    parser.add_argument('--min-area', type=int, default=20, help='Minimum component area')
    parser.add_argument('--mix-global', type=float, default=0.0, help='Global mixing ratio')
    parser.add_argument('--cmap', default='viridis', help='Colormap for threshold heatmap')
    parser.add_argument('--fps', type=float, default=2.0, help='Animation FPS')
    parser.add_argument('--force', action='store_true', help='Force overwrite')

    args = parser.parse_args()

    result = process_minimal_otsu(
        args.src, args.out, args.gif, args.mp4,
        window=args.window, stride=args.stride, preblur=args.preblur,
        morph_open=args.open, morph_close=args.close, min_area=args.min_area,
        mix_global=args.mix_global, cmap=args.cmap, fps=args.fps
    )

    # Print only JSON result
    output = {
        "task": "otsu_local_minimal_gif",
        "frames_dir": args.out,
        "gif": args.gif,
        "mp4": args.mp4,
        "final_png": result["final_output_path"],
        "final_ref_used": result["final_ref_used"],
        "final_match": result["final_match"],
        "metrics": result["metrics"]
    }

    print(json.dumps(output))

if __name__ == "__main__":
    main()