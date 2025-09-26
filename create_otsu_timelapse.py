#!/usr/bin/env python3
"""Create Otsu parameter sweep timelapse video"""

import cv2
import numpy as np
import os
import sys
from itertools import product
from PIL import Image
import imageio
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.otsu import global_otsu, improved_otsu

def create_threshold_heatmap_image(threshold_map):
    """Create threshold heatmap as colorized image"""
    # Normalize threshold map
    norm_map = cv2.normalize(threshold_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Apply colormap (VIRIDIS for better visibility)
    heatmap = cv2.applyColorMap(norm_map, cv2.COLORMAP_VIRIDIS)

    return heatmap

def create_frame(original, window, stride, preblur, frame_idx):
    """Create a single frame with original, global, and improved Otsu results"""

    # Apply Global Otsu
    global_result = global_otsu(original)['result']

    # Apply Improved Otsu
    improved_result = improved_otsu(original, window, stride, preblur, ['open,3', 'close,3'])
    improved_binary = improved_result['result']
    threshold_map = improved_result.get('threshold_map', None)

    # Create 3-panel image
    h, w = original.shape[:2]
    panel_width = w
    separator_width = 10
    total_width = panel_width * 3 + separator_width * 2

    combined = np.zeros((h, total_width, 3), dtype=np.uint8)

    # Convert grayscale images to BGR for display
    original_bgr = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    global_bgr = cv2.cvtColor(global_result, cv2.COLOR_GRAY2BGR)
    improved_bgr = cv2.cvtColor(improved_binary, cv2.COLOR_GRAY2BGR)

    # Place panels
    combined[:, :panel_width] = original_bgr
    combined[:, panel_width:panel_width+separator_width] = 255  # white separator
    combined[:, panel_width+separator_width:2*panel_width+separator_width] = global_bgr
    combined[:, 2*panel_width+separator_width:2*panel_width+2*separator_width] = 255
    combined[:, 2*panel_width+2*separator_width:] = improved_bgr

    # Add parameter text overlay
    text = f"window={window} stride={stride} preblur={preblur:.1f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 2

    # Get text size
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

    # Add background rectangle for text
    cv2.rectangle(combined, (10, 10), (text_size[0] + 20, text_size[1] + 30),
                 (0, 0, 0), -1)

    # Add text
    cv2.putText(combined, text, (15, 35), font, font_scale, (255, 255, 255),
               thickness, cv2.LINE_AA)

    # Add panel labels
    label_font_scale = 0.8
    label_thickness = 2

    labels = ["Original", "Global Otsu", "Improved Otsu"]
    label_positions = [
        (20, h-20),
        (panel_width + separator_width + 20, h-20),
        (2*panel_width + 2*separator_width + 20, h-20)
    ]

    for label, pos in zip(labels, label_positions):
        label_size = cv2.getTextSize(label, font, label_font_scale, label_thickness)[0]
        cv2.rectangle(combined, (pos[0]-5, pos[1]-label_size[1]-5),
                     (pos[0]+label_size[0]+5, pos[1]+5), (0, 0, 0), -1)
        cv2.putText(combined, label, pos, font, label_font_scale,
                   (255, 255, 255), label_thickness, cv2.LINE_AA)

    # Add threshold heatmap as picture-in-picture (bottom right)
    if threshold_map is not None:
        heatmap_bgr = create_threshold_heatmap_image(threshold_map)

        # Resize heatmap to minimap size
        minimap_size = min(w // 4, h // 4, 150)
        heatmap_resized = cv2.resize(heatmap_bgr, (minimap_size, minimap_size))

        # Position in bottom right with margin
        margin = 20
        y_start = h - minimap_size - margin
        x_start = total_width - minimap_size - margin

        # Add white border around minimap
        border_thickness = 3
        cv2.rectangle(combined,
                     (x_start - border_thickness, y_start - border_thickness),
                     (x_start + minimap_size + border_thickness, y_start + minimap_size + border_thickness),
                     (255, 255, 255), -1)

        # Insert heatmap
        combined[y_start:y_start+minimap_size, x_start:x_start+minimap_size] = heatmap_resized

        # Add minimap label
        minimap_label = "Threshold Map"
        label_size = cv2.getTextSize(minimap_label, font, 0.6, 1)[0]
        cv2.rectangle(combined,
                     (x_start, y_start - 25),
                     (x_start + label_size[0] + 10, y_start - 5),
                     (0, 0, 0), -1)
        cv2.putText(combined, minimap_label, (x_start + 5, y_start - 10),
                   font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return combined

def create_transition_frames(frame1, frame2, num_frames=10):
    """Create smooth transition frames between two frames"""
    frames = []
    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        blended = cv2.addWeighted(frame1, 1-alpha, frame2, alpha, 0)
        frames.append(blended)
    return frames

def main():
    print("Creating Otsu parameter sweep timelapse...")

    # Load original image
    original = cv2.imread('images/otsu_shadow_doc_02.jpg', cv2.IMREAD_GRAYSCALE)

    # Parameters to sweep
    windows = [51, 75, 101]
    strides = [16, 24, 32]
    preblurs = [0.8, 1.0, 1.2]

    # Create output directory
    os.makedirs('results/video', exist_ok=True)
    os.makedirs('results/video/otsu_frames', exist_ok=True)

    # Generate frames
    all_frames = []
    frame_paths = []
    frame_idx = 0

    print("Generating frames...")
    prev_frame = None

    total_combinations = len(windows) * len(strides) * len(preblurs)
    current_combo = 0

    for window in windows:
        for stride in strides:
            for preblur in preblurs:
                current_combo += 1
                print(f"  Creating frame {current_combo}/{total_combinations}: window={window}, stride={stride}, preblur={preblur}")

                try:
                    # Create main frame
                    frame = create_frame(original, window, stride, preblur, frame_idx)

                    # Add transition frames if not first frame
                    if prev_frame is not None:
                        transition_frames = create_transition_frames(prev_frame, frame)
                        for t_frame in transition_frames:
                            frame_path = f'results/video/otsu_frames/frame_{frame_idx:04d}.png'
                            cv2.imwrite(frame_path, t_frame)
                            frame_paths.append(frame_path)
                            all_frames.append(t_frame)
                            frame_idx += 1

                    # Hold main frame for 1.5 seconds (45 frames at 30fps)
                    for _ in range(45):
                        frame_path = f'results/video/otsu_frames/frame_{frame_idx:04d}.png'
                        cv2.imwrite(frame_path, frame)
                        frame_paths.append(frame_path)
                        all_frames.append(frame)
                        frame_idx += 1

                    prev_frame = frame

                except Exception as e:
                    print(f"    Error processing parameters: {e}")
                    continue

    print(f"Generated {len(all_frames)} frames")

    if len(all_frames) == 0:
        print("No frames generated. Exiting.")
        return

    # Create MP4 video using OpenCV
    print("Creating MP4 video...")
    h, w = all_frames[0].shape[:2]

    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('results/video/otsu_sweep.mp4', fourcc, 30.0, (w, h))

    for frame in all_frames:
        out.write(frame)

    out.release()

    # Convert to H.264 using ffmpeg if available
    try:
        import subprocess
        subprocess.run([
            'ffmpeg', '-i', 'results/video/otsu_sweep.mp4',
            '-c:v', 'libx264', '-preset', 'slow', '-crf', '22',
            '-y', 'results/video/otsu_sweep_h264.mp4'
        ], check=True, capture_output=True)

        # Replace with H.264 version
        os.rename('results/video/otsu_sweep_h264.mp4', 'results/video/otsu_sweep.mp4')
        print("Converted to H.264 codec")
    except:
        print("Using default codec (H.264 conversion requires ffmpeg)")

    # Create GIF
    print("Creating GIF...")

    # Resize frames for GIF (720px height)
    gif_frames = []
    target_height = 720
    scale = target_height / h
    new_width = int(w * scale)

    for frame in all_frames[::3]:  # Use every 3rd frame for smaller GIF
        # Convert BGR to RGB for PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        pil_img = pil_img.resize((new_width, target_height), Image.LANCZOS)
        gif_frames.append(np.array(pil_img))

    # Save GIF using imageio
    imageio.mimsave('results/video/otsu_sweep.gif', gif_frames, fps=12, loop=0)

    # Clean up frame files
    print("Cleaning up temporary frames...")
    for frame_path in frame_paths:
        try:
            os.remove(frame_path)
        except:
            pass

    try:
        os.rmdir('results/video/otsu_frames')
    except:
        pass

    print("Done!")

    # Output JSON
    result = {
        "task": "video_otsu",
        "artifacts": [
            "results/video/otsu_sweep.mp4",
            "results/video/otsu_sweep.gif"
        ]
    }

    import json
    print(json.dumps(result))

if __name__ == '__main__':
    main()