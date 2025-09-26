#!/usr/bin/env python3
"""Create HE parameter sweep timelapse video"""

import cv2
import numpy as np
import os
from itertools import product
from PIL import Image
import imageio

def create_frame(original, tile_size, clip_limit, frame_idx):
    """Create a single frame with original and CLAHE result side by side"""

    # Apply CLAHE in YUV space
    yuv = cv2.cvtColor(original, cv2.COLOR_BGR2YUV)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    yuv[:,:,0] = clahe.apply(yuv[:,:,0])
    result = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    # Create side-by-side image
    h, w = original.shape[:2]
    combined = np.zeros((h, w*2 + 20, 3), dtype=np.uint8)
    combined[:, :w] = original
    combined[:, w+20:] = result

    # Add separator line
    combined[:, w:w+20] = 255

    # Add text overlay
    text = f"tile={tile_size[0]}x{tile_size[1]} clip={clip_limit:.1f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness = 3

    # Get text size
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

    # Add background rectangle for text
    cv2.rectangle(combined, (10, 10), (text_size[0] + 20, text_size[1] + 30),
                 (0, 0, 0), -1)

    # Add text
    cv2.putText(combined, text, (15, 35), font, font_scale, (255, 255, 255),
               thickness, cv2.LINE_AA)

    # Add labels
    label_font_scale = 1.0
    label_thickness = 2

    # "Original" label
    cv2.rectangle(combined, (10, h-50), (150, h-10), (0, 0, 0), -1)
    cv2.putText(combined, "Original", (20, h-25), font, label_font_scale,
               (255, 255, 255), label_thickness, cv2.LINE_AA)

    # "CLAHE" label
    cv2.rectangle(combined, (w+30, h-50), (w+150, h-10), (0, 0, 0), -1)
    cv2.putText(combined, "CLAHE", (w+40, h-25), font, label_font_scale,
               (255, 255, 255), label_thickness, cv2.LINE_AA)

    return combined

def create_transition_frames(frame1, frame2, num_frames=15):
    """Create smooth transition frames between two frames"""
    frames = []
    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        blended = cv2.addWeighted(frame1, 1-alpha, frame2, alpha, 0)
        frames.append(blended)
    return frames

def main():
    print("Creating HE parameter sweep timelapse...")

    # Load original image
    original = cv2.imread('images/he_dark_indoor.jpg')

    # Parameters to sweep
    tile_sizes = [(8, 8), (16, 16)]
    clip_limits = [2.0, 2.5, 3.0]

    # Create output directory
    os.makedirs('results/video', exist_ok=True)
    os.makedirs('results/video/frames', exist_ok=True)

    # Generate frames
    all_frames = []
    frame_paths = []
    frame_idx = 0

    print("Generating frames...")
    prev_frame = None

    for tile_size, clip_limit in product(tile_sizes, clip_limits):
        print(f"  Creating frame for tile={tile_size}, clip={clip_limit}")

        # Create main frame
        frame = create_frame(original, tile_size, clip_limit, frame_idx)

        # Add transition frames if not first frame
        if prev_frame is not None:
            transition_frames = create_transition_frames(prev_frame, frame)
            for t_frame in transition_frames:
                frame_path = f'results/video/frames/frame_{frame_idx:04d}.png'
                cv2.imwrite(frame_path, t_frame)
                frame_paths.append(frame_path)
                all_frames.append(t_frame)
                frame_idx += 1

        # Hold main frame for 1 second (30 frames at 30fps)
        for _ in range(30):
            frame_path = f'results/video/frames/frame_{frame_idx:04d}.png'
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            all_frames.append(frame)
            frame_idx += 1

        prev_frame = frame

    print(f"Generated {len(all_frames)} frames")

    # Create MP4 video using OpenCV
    print("Creating MP4 video...")
    h, w = all_frames[0].shape[:2]

    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('results/video/he_sweep.mp4', fourcc, 30.0, (w, h))

    for frame in all_frames:
        out.write(frame)

    out.release()

    # Convert to H.264 using ffmpeg if available
    try:
        import subprocess
        subprocess.run([
            'ffmpeg', '-i', 'results/video/he_sweep.mp4',
            '-c:v', 'libx264', '-preset', 'slow', '-crf', '22',
            '-y', 'results/video/he_sweep_h264.mp4'
        ], check=True, capture_output=True)

        # Replace with H.264 version
        os.rename('results/video/he_sweep_h264.mp4', 'results/video/he_sweep.mp4')
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

    for frame in all_frames[::2]:  # Use every 2nd frame for smaller GIF
        # Convert BGR to RGB for PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        pil_img = pil_img.resize((new_width, target_height), Image.LANCZOS)
        gif_frames.append(np.array(pil_img))

    # Save GIF using imageio
    imageio.mimsave('results/video/he_sweep.gif', gif_frames, fps=15, loop=0)

    # Clean up frame files
    print("Cleaning up temporary frames...")
    for frame_path in frame_paths:
        try:
            os.remove(frame_path)
        except:
            pass

    try:
        os.rmdir('results/video/frames')
    except:
        pass

    print("Done!")

    # Output JSON
    result = {
        "task": "video_he",
        "artifacts": [
            "results/video/he_sweep.mp4",
            "results/video/he_sweep.gif"
        ]
    }

    import json
    print(json.dumps(result))

if __name__ == '__main__':
    main()