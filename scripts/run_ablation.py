#!/usr/bin/env python3
"""Run parameter ablation study for HE and Otsu methods"""

import os
import sys
import argparse
import subprocess
import json
import csv
import glob
from pathlib import Path
import tempfile

try:
    import cv2
    import numpy as np
    from skimage.metrics import structural_similarity
    from skimage.color import rgb2lab, deltaE_ciede2000
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install opencv-python scikit-image numpy")
    sys.exit(1)

def run_command(cmd, check=True):
    """Run shell command safely"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=check)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout, e.stderr

def compute_he_quality_score(original_path, result_path):
    """Compute quality score for HE result"""
    try:
        # Load images
        original = cv2.imread(original_path)
        result = cv2.imread(result_path)

        if original is None or result is None:
            return 0.0

        # Convert to RGB for deltaE
        orig_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        # Convert to grayscale for SSIM
        orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

        # Compute SSIM
        ssim_score = structural_similarity(orig_gray, result_gray, data_range=255)

        # Compute Delta E
        orig_lab = rgb2lab(orig_rgb.astype(np.float32) / 255.0)
        result_lab = rgb2lab(result_rgb.astype(np.float32) / 255.0)
        delta_e = deltaE_ciede2000(orig_lab, result_lab).mean()

        # Combined score: higher SSIM, lower Delta E is better
        # Normalize and combine (simple weighted sum)
        score = ssim_score * 0.7 + (50 - min(delta_e, 50)) / 50 * 0.3

        return max(0.0, min(1.0, score))

    except Exception as e:
        print(f"Error computing HE score: {e}")
        return 0.0

def compute_otsu_quality_score(result_path):
    """Compute quality score for Otsu result"""
    try:
        result = cv2.imread(result_path, cv2.IMREAD_GRAYSCALE)
        if result is None:
            return 0.0

        # Binarize
        binary = (result > 127).astype(np.uint8)

        # Connected components analysis
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

        if num_labels <= 1:  # Only background
            return 0.0

        # Get areas (skip background)
        areas = stats[1:, cv2.CC_STAT_AREA]
        avg_area = areas.mean()

        # Compute score based on component count and average area
        # Fewer components with reasonable size are better
        component_score = min(1.0, 1000.0 / max(num_labels - 1, 1))  # Prefer fewer components
        area_score = min(1.0, avg_area / 5000.0)  # Prefer larger components

        score = component_score * 0.6 + area_score * 0.4
        return max(0.0, min(1.0, score))

    except Exception as e:
        print(f"Error computing Otsu score: {e}")
        return 0.0

def run_he_ablation():
    """Run HE parameter ablation study"""
    print("Running HE ablation study...")

    # Check if run_he.py exists
    if not os.path.exists("run_he.py"):
        print("Warning: run_he.py not found, skipping HE ablation")
        return False

    # Check input image
    input_image = "images/he_dark_indoor.jpg"
    if not os.path.exists(input_image):
        print(f"Warning: Input image not found: {input_image}")
        return False

    # Parameter grid
    tile_sizes = [8, 16]
    clip_limits = [2.0, 2.5, 3.0]

    results = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for tile in tile_sizes:
            for clip in clip_limits:
                print(f"Testing HE: tile={tile}, clip={clip}")

                # Run HE
                temp_output = os.path.join(temp_dir, f"he_tile{tile}_clip{clip}")
                cmd = f"python run_he.py {input_image} --he-mode clahe --space yuv --tile {tile} {tile} --clip {clip} --save {temp_output}"

                success, stdout, stderr = run_command(cmd, check=False)
                if not success:
                    print(f"HE failed for tile={tile}, clip={clip}: {stderr}")
                    continue

                # Find result image
                result_files = glob.glob(f"{temp_output}/*clahe*.png")
                if not result_files:
                    result_files = glob.glob(f"{temp_output}/*.png")

                if not result_files:
                    print(f"No result found for tile={tile}, clip={clip}")
                    continue

                result_path = result_files[0]
                score = compute_he_quality_score(input_image, result_path)

                results.append({
                    'tile_size': tile,
                    'clip_limit': clip,
                    'quality_score': score
                })

                print(f"  Score: {score:.3f}")

    if not results:
        print("No HE results generated")
        return False

    # Save results
    output_dir = "results/ablation"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    csv_path = f"{output_dir}/ablation_he.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['tile_size', 'clip_limit', 'quality_score'])
        writer.writeheader()
        writer.writerows(results)

    # Get top 3 results
    top3 = sorted(results, key=lambda x: x['quality_score'], reverse=True)[:3]
    top3_path = f"{output_dir}/ablation_he_top3.json"
    with open(top3_path, 'w') as f:
        json.dump(top3, f, indent=2)

    print(f"HE ablation completed. Results: {csv_path}, Top3: {top3_path}")
    return True

def run_otsu_ablation():
    """Run Otsu parameter ablation study"""
    print("Running Otsu ablation study...")

    # Check if run_otsu.py exists
    if not os.path.exists("run_otsu.py"):
        print("Warning: run_otsu.py not found, skipping Otsu ablation")
        return False

    # Check input image
    input_image = "images/otsu_shadow_doc_02.jpg"
    if not os.path.exists(input_image):
        print(f"Warning: Input image not found: {input_image}")
        return False

    # Parameter grid
    window_sizes = [51, 75, 101]
    strides = [16, 24, 32]
    preblur_values = [0.5, 1.0, 1.5]

    results = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for window in window_sizes:
            for stride in strides:
                for preblur in preblur_values:
                    print(f"Testing Otsu: window={window}, stride={stride}, preblur={preblur}")

                    # Run Otsu
                    temp_output = os.path.join(temp_dir, f"otsu_w{window}_s{stride}_b{preblur}")
                    cmd = f"python run_otsu.py {input_image} --method improved --window {window} --stride {stride} --preblur {preblur} --save {temp_output}"

                    success, stdout, stderr = run_command(cmd, check=False)
                    if not success:
                        print(f"Otsu failed for window={window}, stride={stride}, preblur={preblur}: {stderr}")
                        continue

                    # Find result image
                    result_files = glob.glob(f"{temp_output}/*improved*.png")
                    if not result_files:
                        result_files = glob.glob(f"{temp_output}/*.png")

                    if not result_files:
                        print(f"No result found for window={window}, stride={stride}, preblur={preblur}")
                        continue

                    result_path = result_files[0]
                    score = compute_otsu_quality_score(result_path)

                    results.append({
                        'window_size': window,
                        'stride': stride,
                        'preblur': preblur,
                        'quality_score': score
                    })

                    print(f"  Score: {score:.3f}")

    if not results:
        print("No Otsu results generated")
        return False

    # Save results
    output_dir = "results/ablation"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    csv_path = f"{output_dir}/ablation_otsu.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['window_size', 'stride', 'preblur', 'quality_score'])
        writer.writeheader()
        writer.writerows(results)

    # Get top 3 results
    top3 = sorted(results, key=lambda x: x['quality_score'], reverse=True)[:3]
    top3_path = f"{output_dir}/ablation_otsu_top3.json"
    with open(top3_path, 'w') as f:
        json.dump(top3, f, indent=2)

    print(f"Otsu ablation completed. Results: {csv_path}, Top3: {top3_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Run ablation studies')
    parser.add_argument('mode', choices=['he', 'otsu', 'all'], help='Ablation to run')
    args = parser.parse_args()

    success = True

    if args.mode in ['he', 'all']:
        success &= run_he_ablation()

    if args.mode in ['otsu', 'all']:
        success &= run_otsu_ablation()

    if success:
        print("Ablation study completed successfully")
        return 0
    else:
        print("Some ablation studies failed")
        return 1

if __name__ == '__main__':
    sys.exit(main())