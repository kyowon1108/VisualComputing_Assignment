#!/usr/bin/env python3
"""Convert MP4 videos to GIF format"""

import os
import sys
import argparse
from pathlib import Path

try:
    import imageio
    import imageio_ffmpeg
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install imageio imageio-ffmpeg")
    sys.exit(1)

def mp4_to_gif(input_mp4, output_gif, fps=10, max_frames=None, force=False):
    """Convert MP4 to GIF using imageio"""
    if not os.path.exists(input_mp4):
        print(f"Error: Input MP4 not found: {input_mp4}")
        return False

    if os.path.exists(output_gif) and not force:
        print(f"GIF already exists: {output_gif}")
        return True

    try:
        print(f"Converting {input_mp4} -> {output_gif}")

        # Read MP4
        reader = imageio.get_reader(input_mp4, 'ffmpeg')

        # Collect frames
        frames = []
        frame_count = 0

        for frame in reader:
            frames.append(frame)
            frame_count += 1

            if max_frames and frame_count >= max_frames:
                break

        reader.close()

        if not frames:
            print("Error: No frames read from MP4")
            return False

        # Write GIF
        print(f"Writing {len(frames)} frames at {fps} fps")
        imageio.mimsave(output_gif, frames, fps=fps)

        # Check output size
        if os.path.exists(output_gif):
            size_mb = os.path.getsize(output_gif) / 1024 / 1024
            print(f"Created GIF: {output_gif} ({size_mb:.1f} MB)")
            return True
        else:
            print("Error: Failed to create GIF")
            return False

    except Exception as e:
        print(f"Error converting {input_mp4}: {e}")
        return False

def convert_he_video(force=False):
    """Convert HE sweep video to GIF"""
    input_mp4 = "results/video/he_sweep.mp4"
    output_gif = "results/video/he_sweep.gif"

    if not os.path.exists(input_mp4):
        print(f"HE video not found: {input_mp4}")
        return False

    return mp4_to_gif(input_mp4, output_gif, fps=8, max_frames=100, force=force)

def convert_otsu_video(force=False):
    """Convert Otsu sweep video to GIF"""
    input_mp4 = "results/video/otsu_sweep.mp4"
    output_gif = "results/video/otsu_sweep.gif"

    if not os.path.exists(input_mp4):
        print(f"Otsu video not found: {input_mp4}")
        return False

    return mp4_to_gif(input_mp4, output_gif, fps=8, max_frames=100, force=force)

def main():
    parser = argparse.ArgumentParser(description='Convert MP4 videos to GIF')
    parser.add_argument('--he', metavar='INPUT', help='Input HE MP4 file')
    parser.add_argument('--otsu', metavar='INPUT', help='Input Otsu MP4 file')
    parser.add_argument('--gif', metavar='OUTPUT', help='Output GIF file')
    parser.add_argument('--fps', type=int, default=10, help='GIF frame rate')
    parser.add_argument('--max-frames', type=int, help='Limit number of frames')
    parser.add_argument('--force', action='store_true', help='Overwrite existing GIFs')
    args = parser.parse_args()

    # Ensure video directory exists
    Path("results/video").mkdir(parents=True, exist_ok=True)

    success = True

    # Handle specific conversions
    if args.he and args.gif:
        success &= mp4_to_gif(args.he, args.gif, args.fps, args.max_frames, args.force)
    elif args.otsu and args.gif:
        success &= mp4_to_gif(args.otsu, args.gif, args.fps, args.max_frames, args.force)
    else:
        # Default: convert standard videos
        print("Converting standard video files...")
        success &= convert_he_video(args.force)
        success &= convert_otsu_video(args.force)

    if success:
        print("Video conversion completed successfully")
        return 0
    else:
        print("Some conversions failed")
        return 1

if __name__ == '__main__':
    sys.exit(main())