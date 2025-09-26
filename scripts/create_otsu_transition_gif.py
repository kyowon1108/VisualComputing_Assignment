#!/usr/bin/env python3
"""
Create slow transition GIF from Global Otsu to Improved Otsu
Shows smooth morphing transition with many intermediate frames
"""

import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import imageio.v3 as imageio
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def add_text_overlay(image, text, position=(10, 40), font_scale=1.2):
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
    canvas_h, canvas_w = canvas_size[1] - 100, canvas_size[0] - 40  # Reserve space for title
    img_h, img_w = image.shape[:2]

    scale = min(canvas_w / img_w, canvas_h / img_h)
    new_w, new_h = int(img_w * scale), int(img_h * scale)

    resized = cv2.resize(image, (new_w, new_h))

    # Center the image
    start_x = (canvas_size[0] - new_w) // 2
    start_y = 100 + (canvas_size[1] - 100 - new_h) // 2

    canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized

    # Add title
    canvas = add_text_overlay(canvas, title, (20, 60), font_scale=1.4)

    return canvas

def create_transition_frames(original, global_result, improved_result,
                           num_transition_frames=30, hold_frames=15):
    """Create transition frames from global to improved Otsu result."""
    frames = []

    # Phase 1: Hold on Original (5 frames)
    for i in range(5):
        frame = create_canvas_frame(original, "Original Image")
        frames.append(frame)

    # Phase 2: Hold on Global Otsu (hold_frames)
    for i in range(hold_frames):
        frame = create_canvas_frame(global_result, "Global Otsu Thresholding")
        frames.append(frame)

    # Phase 3: Smooth transition from Global to Improved
    global_float = global_result.astype(np.float32)
    improved_float = improved_result.astype(np.float32)

    for i in range(num_transition_frames):
        # Calculate blend ratio (ease-in-out curve)
        t = i / (num_transition_frames - 1)
        # Smooth transition curve (ease-in-out)
        smooth_t = 3*t*t - 2*t*t*t  # smoothstep function

        # Blend the two results
        blended = (1 - smooth_t) * global_float + smooth_t * improved_float
        blended_uint8 = np.clip(blended, 0, 255).astype(np.uint8)

        # Create progress text
        progress = int(smooth_t * 100)
        title = f"Transitioning to Improved Otsu... {progress}%"

        frame = create_canvas_frame(blended_uint8, title)
        frames.append(frame)

    # Phase 4: Hold on Improved Otsu (hold_frames)
    for i in range(hold_frames):
        frame = create_canvas_frame(improved_result, "Improved Otsu Result")
        frames.append(frame)

    # Phase 5: Show XOR difference (10 frames)
    # Calculate XOR for comparison
    global_binary = (global_result > 127).astype(np.uint8)
    improved_binary = (improved_result > 127).astype(np.uint8)
    xor_map = cv2.bitwise_xor(global_binary, improved_binary) * 255

    # Create colored XOR visualization
    xor_colored = np.zeros((xor_map.shape[0], xor_map.shape[1], 3), dtype=np.uint8)
    xor_colored[:, :, 0] = xor_map  # Red channel for differences
    xor_colored[:, :, 1] = improved_result // 2  # Green channel for structure
    xor_colored[:, :, 2] = improved_result // 2  # Blue channel for structure

    disagreement_pixels = np.sum(xor_map > 0)
    total_pixels = xor_map.size
    disagreement_ratio = (disagreement_pixels / total_pixels) * 100

    for i in range(10):
        title = f"Difference Map (XOR) - {disagreement_ratio:.1f}% disagreement"
        frame = create_canvas_frame(xor_colored, title)
        frames.append(frame)

    return frames

def create_otsu_transition_gif(src_path, output_gif, output_mp4=None,
                              num_transition_frames=30, hold_frames=15, fps=8):
    """Create transition GIF from Global to Improved Otsu."""

    # Read original image
    original = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    if original is None:
        raise ValueError(f"Could not read image: {src_path}")

    # Generate Global and Improved results
    from src.otsu import global_otsu, improved_otsu

    print("Generating Global Otsu result...")
    global_result = global_otsu(original)['result']

    print("Generating Improved Otsu result...")
    improved_result = improved_otsu(original)['result']

    print(f"Creating transition frames ({num_transition_frames} transition frames)...")
    frames = create_transition_frames(
        original, global_result, improved_result,
        num_transition_frames=num_transition_frames,
        hold_frames=hold_frames
    )

    print(f"Generated {len(frames)} total frames")

    # Ensure all frames are consistent size and convert to RGB
    consistent_frames = []
    max_h = max(f.shape[0] for f in frames)
    max_w = max(f.shape[1] for f in frames)

    for frame in frames:
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if frame.shape[0] != max_h or frame.shape[1] != max_w:
            frame = cv2.resize(frame, (max_w, max_h))

        consistent_frames.append(frame)

    # Create output directory
    os.makedirs(os.path.dirname(output_gif), exist_ok=True)

    # Create GIF
    print(f"Creating GIF with {fps} FPS...")
    try:
        # Use slower frame duration for smooth transition
        frame_duration = 1.0 / fps
        imageio.imwrite(output_gif, consistent_frames, loop=0, duration=frame_duration)
        print(f"GIF created: {output_gif}")
        gif_success = True
    except Exception as e:
        print(f"Warning: GIF creation failed: {e}")
        gif_success = False

    # Create MP4 if requested
    if output_mp4:
        print(f"Creating MP4 with {fps} FPS...")
        try:
            imageio.imwrite(output_mp4, consistent_frames, fps=fps)
            print(f"MP4 created: {output_mp4}")
            mp4_success = True
        except Exception as e:
            print(f"Warning: MP4 creation failed: {e}")
            mp4_success = False
    else:
        mp4_success = False

    return {
        "gif_success": gif_success,
        "mp4_success": mp4_success,
        "frame_count": len(frames),
        "output_gif": output_gif,
        "output_mp4": output_mp4
    }

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Create Global to Improved Otsu transition GIF')
    parser.add_argument('--src', default='images/otsu_shadow_doc_02.jpg',
                       help='Source image path')
    parser.add_argument('--gif', default='results/video/otsu_global_to_improved_transition.gif',
                       help='Output GIF path')
    parser.add_argument('--mp4', help='Output MP4 path (optional)')
    parser.add_argument('--transition-frames', type=int, default=30,
                       help='Number of transition frames (more = slower)')
    parser.add_argument('--hold-frames', type=int, default=15,
                       help='Frames to hold on each result')
    parser.add_argument('--fps', type=float, default=8.0,
                       help='Animation FPS (lower = slower)')
    parser.add_argument('--force', action='store_true', help='Force overwrite')

    args = parser.parse_args()

    try:
        result = create_otsu_transition_gif(
            args.src, args.gif, args.mp4,
            num_transition_frames=args.transition_frames,
            hold_frames=args.hold_frames,
            fps=args.fps
        )

        print(f"\nTransition animation created!")
        print(f"  Total frames: {result['frame_count']}")
        print(f"  GIF: {result['output_gif']} ({'✓' if result['gif_success'] else '✗'})")
        if args.mp4:
            print(f"  MP4: {result['output_mp4']} ({'✓' if result['mp4_success'] else '✗'})")

        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())