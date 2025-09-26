#!/usr/bin/env python3
"""
Slow, didactic animation of Local Otsu sliding window sweep.
Shows progression from original to global Otsu to sliding-window processing.
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from scipy import interpolate
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage import morphology
import imageio.v3 as imageio

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from otsu import sliding_window_otsu, compute_otsu_threshold
    HAS_SRC_OTSU = True
except ImportError:
    HAS_SRC_OTSU = False

def add_text_overlay(image, text, position=(10, 30), font_scale=0.7):
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

def create_histogram_frame(image, threshold, title):
    """Create frame showing histogram with threshold line."""
    hist, bins = np.histogram(image.flatten(), bins=256, range=[0, 255])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(bins[:-1], hist, width=1, alpha=0.7, color='blue')
    ax.axvline(x=threshold, color='red', linestyle='--', linewidth=3,
               label=f'Otsu Threshold = {threshold:.1f}')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Pixel Intensity')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Save to temp file and read back
    temp_path = "/tmp/hist_temp.png"
    plt.savefig(temp_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    frame = cv2.imread(temp_path)
    os.remove(temp_path)
    return frame

def create_threshold_heatmap_frame(threshold_map, title, cmap='viridis'):
    """Create frame showing threshold heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Handle NaN values
    masked_map = np.ma.masked_invalid(threshold_map)

    im = ax.imshow(masked_map, cmap=cmap, interpolation='nearest')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Threshold Value', rotation=270, labelpad=20)

    plt.tight_layout()

    # Save to temp file and read back
    temp_path = "/tmp/heatmap_temp.png"
    plt.savefig(temp_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    frame = cv2.imread(temp_path)
    os.remove(temp_path)
    return frame

def draw_grid_overlay(image, window_size, stride):
    """Draw grid overlay showing tile boundaries."""
    h, w = image.shape[:2]
    if len(image.shape) == 2:
        overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        overlay = image.copy()

    # Draw grid
    for y in range(0, h - window_size + 1, stride):
        for x in range(0, w - window_size + 1, stride):
            cv2.rectangle(overlay, (x, y), (x + window_size, y + window_size),
                         (0, 255, 0), 1)

    return overlay

def create_composite_frame(left_image, middle_image, right_image, title, canvas_size=(1280, 720)):
    """Create composite frame with three panels."""
    canvas = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)

    # Calculate panel sizes
    panel_w = canvas_size[0] // 3
    panel_h = canvas_size[1] - 60  # Reserve space for title

    # Resize and place panels
    panels = [left_image, middle_image, right_image]
    for i, panel in enumerate(panels):
        if panel is not None:
            if len(panel.shape) == 2:
                panel = cv2.cvtColor(panel, cv2.COLOR_GRAY2RGB)

            # Resize maintaining aspect ratio
            h, w = panel.shape[:2]
            scale = min(panel_w / w, panel_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(panel, (new_w, new_h))

            # Center in panel
            start_x = i * panel_w + (panel_w - new_w) // 2
            start_y = 60 + (panel_h - new_h) // 2
            canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized

    # Add title
    canvas = add_text_overlay(canvas, title, (10, 40), font_scale=1.0)

    return canvas

def remove_small_components(binary_image, min_area):
    """Remove connected components smaller than min_area."""
    labeled = label(binary_image > 0)
    regions = regionprops(labeled)

    result = np.zeros_like(binary_image)
    for region in regions:
        if region.area >= min_area:
            coords = region.coords
            result[coords[:, 0], coords[:, 1]] = 255

    return result

def process_otsu_sweep(src_path, frames_dir, gif_path, mp4_path,
                      window=75, stride=24, preblur=0.8, morph_open=1, morph_close=2,
                      min_area=20, fps=2.0, sweep_mode="row", cmap="viridis"):
    """Process Otsu sweep animation."""

    # Ensure directories exist
    Path(frames_dir).mkdir(parents=True, exist_ok=True)
    Path(gif_path).parent.mkdir(parents=True, exist_ok=True)
    Path(mp4_path).parent.mkdir(parents=True, exist_ok=True)

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

    # Global Otsu
    global_thresh = threshold_otsu(blurred)
    global_mask = (blurred > global_thresh).astype(np.uint8) * 255

    frames = []
    frame_idx = 0

    # S00: Original
    frame = add_text_overlay(image, "S00: Original Grayscale")
    cv2.imwrite(f"{frames_dir}/frame_{frame_idx:03d}.png", frame)
    frames.append(frame)
    frame_idx += 1

    # S01: Global histogram + threshold
    hist_frame = create_histogram_frame(blurred, global_thresh,
                                       "S01: Global Histogram + Otsu Threshold")
    cv2.imwrite(f"{frames_dir}/frame_{frame_idx:03d}.png", hist_frame)
    frames.append(hist_frame)
    frame_idx += 1

    # S02: Global mask
    frame = add_text_overlay(global_mask, "S02: Global Otsu Binarization")
    cv2.imwrite(f"{frames_dir}/frame_{frame_idx:03d}.png", frame)
    frames.append(frame)
    frame_idx += 1

    # S03: Grid overlay
    grid_overlay = draw_grid_overlay(blurred, window, stride)
    grid_frame = add_text_overlay(grid_overlay, f"S03: Sliding Window Grid ({window}x{window}, stride={stride})")
    cv2.imwrite(f"{frames_dir}/frame_{frame_idx:03d}.png", grid_frame)
    frames.append(grid_frame)
    frame_idx += 1

    # Prepare for sweep
    threshold_map = np.full((h, w), np.nan, dtype=np.float32)
    partial_mask = global_mask.copy()

    # Calculate tile positions
    y_positions = list(range(0, h - window + 1, stride))
    x_positions = list(range(0, w - window + 1, stride))

    # Sweep processing
    if sweep_mode == "row":
        # Process row by row
        for row_idx, y in enumerate(y_positions):
            # Process all tiles in this row
            for x in x_positions:
                # Extract window
                patch = blurred[y:y+window, x:x+window]

                # Calculate threshold
                try:
                    tile_thresh = threshold_otsu(patch)
                except ValueError:
                    tile_thresh = np.mean(patch)

                # Store threshold in map
                center_y = y + window // 2
                center_x = x + window // 2
                if center_y < h and center_x < w:
                    threshold_map[center_y, center_x] = tile_thresh

            # After processing row, create interpolated partial map
            valid_mask = ~np.isnan(threshold_map)
            if np.any(valid_mask):
                # Simple interpolation for processed area
                valid_coords = np.where(valid_mask)
                valid_values = threshold_map[valid_mask]

                # Create partial threshold map (only for processed rows)
                y_max_processed = y + window
                partial_thresh_map = np.full_like(threshold_map, global_thresh)

                if len(valid_coords[0]) > 1:
                    # Use griddata for interpolation in processed area
                    from scipy.interpolate import griddata
                    yi, xi = np.mgrid[0:y_max_processed, 0:w]
                    try:
                        interpolated = griddata(
                            (valid_coords[0], valid_coords[1]), valid_values,
                            (yi, xi), method='linear', fill_value=global_thresh
                        )
                        partial_thresh_map[:y_max_processed, :] = interpolated
                    except:
                        # Fallback to nearest
                        partial_thresh_map[valid_coords] = valid_values

                # Apply partial thresholding
                partial_local = (blurred > partial_thresh_map).astype(np.uint8) * 255

                # Create heatmap for processed area only
                display_map = np.full_like(threshold_map, np.nan)
                display_map[:y_max_processed, :] = partial_thresh_map[:y_max_processed, :]
                heatmap_frame = create_threshold_heatmap_frame(
                    display_map, f"Threshold Map (Row {row_idx+1}/{len(y_positions)})", cmap
                )

                # Highlight current row
                row_highlight = blurred.copy()
                if len(row_highlight.shape) == 2:
                    row_highlight = cv2.cvtColor(row_highlight, cv2.COLOR_GRAY2RGB)

                # Add semi-transparent highlight bar
                overlay = row_highlight.copy()
                cv2.rectangle(overlay, (0, max(0, y-5)), (w, min(h, y+window+5)),
                             (0, 255, 255), -1)
                row_highlight = cv2.addWeighted(row_highlight, 0.7, overlay, 0.3, 0)

                # Create composite frame
                composite = create_composite_frame(
                    row_highlight, heatmap_frame, partial_local,
                    f"S04+: Local Otsu Sweep - Row {row_idx+1}/{len(y_positions)}"
                )

                cv2.imwrite(f"{frames_dir}/frame_{frame_idx:03d}.png", composite)
                frames.append(composite)
                frame_idx += 1

    # Final interpolation for complete threshold map
    valid_mask = ~np.isnan(threshold_map)
    if np.any(valid_mask):
        from scipy.interpolate import griddata
        valid_coords = np.where(valid_mask)
        valid_values = threshold_map[valid_mask]
        yi, xi = np.mgrid[0:h, 0:w]

        try:
            final_thresh_map = griddata(
                (valid_coords[0], valid_coords[1]), valid_values,
                (yi, xi), method='linear', fill_value=global_thresh
            )
        except:
            final_thresh_map = np.full((h, w), global_thresh)
            final_thresh_map[valid_coords] = valid_values
    else:
        final_thresh_map = np.full((h, w), global_thresh)

    # S90: Full threshold map
    heatmap_final = create_threshold_heatmap_frame(final_thresh_map, "S90: Final Threshold Map", cmap)
    cv2.imwrite(f"{frames_dir}/frame_{frame_idx:03d}.png", heatmap_final)
    frames.append(heatmap_final)
    frame_idx += 1

    # S91: Local raw mask
    local_raw = (blurred > final_thresh_map).astype(np.uint8) * 255
    frame = add_text_overlay(local_raw, "S91: Local Binarization (Raw)")
    cv2.imwrite(f"{frames_dir}/frame_{frame_idx:03d}.png", frame)
    frames.append(frame)
    frame_idx += 1

    # S92: Morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morph_result = cv2.morphologyEx(local_raw, cv2.MORPH_OPEN, kernel, iterations=morph_open)
    morph_result = cv2.morphologyEx(morph_result, cv2.MORPH_CLOSE, kernel, iterations=morph_close)
    frame = add_text_overlay(morph_result, f"S92: Morphology (Open:{morph_open}, Close:{morph_close})")
    cv2.imwrite(f"{frames_dir}/frame_{frame_idx:03d}.png", frame)
    frames.append(frame)
    frame_idx += 1

    # S93: Remove small components
    cleaned = remove_small_components(morph_result, min_area)
    frame = add_text_overlay(cleaned, f"S93: Remove Small Components (<{min_area}px)")
    cv2.imwrite(f"{frames_dir}/frame_{frame_idx:03d}.png", frame)
    frames.append(frame)
    frame_idx += 1

    # S94: Final improved
    frame = add_text_overlay(cleaned, "S94: Final Improved Otsu Result")
    cv2.imwrite(f"{frames_dir}/frame_{frame_idx:03d}.png", frame)
    frames.append(frame)
    frame_idx += 1

    # S95: XOR vs global
    xor_diff = cv2.bitwise_xor(cleaned, global_mask)
    frame = add_text_overlay(xor_diff, "S95: XOR vs Global (Difference)")
    cv2.imwrite(f"{frames_dir}/frame_{frame_idx:03d}.png", frame)
    frames.append(frame)
    frame_idx += 1

    # Create GIF and MP4
    rgb_frames = []
    for frame in frames:
        if len(frame.shape) == 3:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        rgb_frames.append(rgb_frame)

    # Ensure consistent frame sizes
    if rgb_frames:
        max_h = max(f.shape[0] for f in rgb_frames)
        max_w = max(f.shape[1] for f in rgb_frames)

        consistent_frames = []
        for frame in rgb_frames:
            if frame.shape[0] != max_h or frame.shape[1] != max_w:
                frame = cv2.resize(frame, (max_w, max_h))
            consistent_frames.append(frame)

        # Create GIF
        try:
            imageio.imwrite(gif_path, consistent_frames, loop=0, duration=1.0/fps)
        except Exception as e:
            print(f"Warning: GIF creation failed: {e}")

        # Create MP4
        try:
            imageio.imwrite(mp4_path, consistent_frames, fps=fps)
        except Exception as e:
            print(f"Warning: MP4 creation failed: {e}")

    return frame_idx

def main():
    parser = argparse.ArgumentParser(description='Generate slow Local Otsu sweep animation')
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
    parser.add_argument('--fps', type=float, default=2.0, help='Animation FPS')
    parser.add_argument('--sweep-mode', choices=['row', 'tile'], default='row',
                       help='Sweep mode: row or tile')
    parser.add_argument('--force', action='store_true', help='Force overwrite')

    args = parser.parse_args()

    frame_count = process_otsu_sweep(
        args.src, args.out, args.gif, args.mp4,
        window=args.window, stride=args.stride, preblur=args.preblur,
        morph_open=args.open, morph_close=args.close, min_area=args.min_area,
        fps=args.fps, sweep_mode=args.sweep_mode
    )

if __name__ == "__main__":
    main()