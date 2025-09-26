#!/usr/bin/env python3
"""Rebuild missing artifacts according to plan"""

import os
import sys
import json
import glob
import subprocess
from pathlib import Path

import cv2
import numpy as np
import imageio
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from skimage import color
from skimage.metrics import structural_similarity as ssim
import pandas as pd

def ensure_deps():
    """Ensure dependencies are installed"""
    try:
        import cv2, numpy, skimage, PIL, matplotlib, imageio, reportlab
        return True
    except ImportError:
        return False

def make_dirs():
    """Create necessary directories"""
    dirs = ['results/he_metrics', 'results/otsu_metrics', 'results/video']
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

def convert_mp4_to_gif(mp4_path, gif_path, height=480):
    """Convert MP4 to GIF using imageio"""
    if os.path.exists(gif_path):
        return False

    if not os.path.exists(mp4_path):
        return False

    try:
        reader = imageio.get_reader(mp4_path)
        fps = reader.get_meta_data()['fps']

        frames = []
        for i, frame in enumerate(reader):
            if i % 3 == 0:  # Skip frames for smaller size
                h, w = frame.shape[:2]
                scale = height / h
                new_w = int(w * scale)
                frame_resized = cv2.resize(frame, (new_w, height))
                frames.append(frame_resized)

        reader.close()

        imageio.mimsave(gif_path, frames, fps=fps//3, loop=0)
        return True
    except Exception as e:
        print(f"Error converting {mp4_path}: {e}")
        return False

def find_he_results():
    """Find existing HE result images"""
    results = {}

    # Look for RGB-HE result
    rgb_files = glob.glob("results/he/*rgb*.png") + glob.glob("results/he/*rgb*.jpg")
    if rgb_files:
        results['rgb_he'] = rgb_files[0]

    # Look for Y-HE result
    y_files = glob.glob("results/he/*yuv*.png") + glob.glob("results/he/*y_he*.png")
    if y_files:
        results['y_he'] = y_files[0]

    # Look for CLAHE result
    clahe_files = glob.glob("results/he/*clahe*.png") + glob.glob("results/he/*CLAHE*.png")
    if clahe_files:
        results['clahe'] = clahe_files[0]

    return results

def generate_he_result(original, method):
    """Generate HE result if missing"""
    if method == 'rgb_he':
        # RGB-HE: equalize each channel
        result = np.zeros_like(original)
        for i in range(3):
            result[:,:,i] = cv2.equalizeHist(original[:,:,i])
        return result

    elif method == 'y_he':
        # Y-HE: YUV space, equalize Y channel
        yuv = cv2.cvtColor(original, cv2.COLOR_BGR2YUV)
        yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    elif method == 'clahe':
        # CLAHE: YUV space, CLAHE on Y channel
        yuv = cv2.cvtColor(original, cv2.COLOR_BGR2YUV)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        yuv[:,:,0] = clahe.apply(yuv[:,:,0])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

def create_he_metrics():
    """Create missing HE metrics"""
    original_path = "images/he_dark_indoor.jpg"
    if not os.path.exists(original_path):
        return {'rgb_diff': False, 'rgb_ssim': False, 'rgb_deltaE': False, 'collage': False}

    original = cv2.imread(original_path)
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    # Find existing results
    he_results = find_he_results()

    # Generate missing results
    for method in ['rgb_he', 'y_he', 'clahe']:
        if method not in he_results:
            result = generate_he_result(original, method)
            save_path = f"results/he/{method}_generated.png"
            cv2.imwrite(save_path, result)
            he_results[method] = save_path

    created = {'rgb_diff': False, 'rgb_ssim': False, 'rgb_deltaE': False, 'collage': False}

    # Create missing diff/ssim/deltaE maps for RGB-HE
    if 'rgb_he' in he_results:
        rgb_he_img = cv2.imread(he_results['rgb_he'])
        # Resize to match original if needed
        if rgb_he_img.shape[:2] != original.shape[:2]:
            rgb_he_img = cv2.resize(rgb_he_img, (original.shape[1], original.shape[0]))
        rgb_he_gray = cv2.cvtColor(rgb_he_img, cv2.COLOR_BGR2GRAY)
        rgb_he_rgb = cv2.cvtColor(rgb_he_img, cv2.COLOR_BGR2RGB)

        # Difference map
        diff_path = "results/he_metrics/diff_rgb_he.png"
        if not os.path.exists(diff_path):
            diff = np.abs(original_gray.astype(float) - rgb_he_gray.astype(float))
            plt.figure(figsize=(8, 6))
            im = plt.imshow(diff, cmap='hot', vmin=0, vmax=255)
            plt.colorbar(im, label='Absolute Difference')
            plt.title('RGB-HE Difference Map')
            plt.axis('off')
            plt.savefig(diff_path, dpi=150, bbox_inches='tight')
            plt.close()
            created['rgb_diff'] = True

        # SSIM map
        ssim_path = "results/he_metrics/ssim_rgb_he.png"
        if not os.path.exists(ssim_path):
            ssim_score, ssim_map = ssim(original_gray, rgb_he_gray, full=True, data_range=255)
            ssim_map_norm = (ssim_map + 1) / 2  # Normalize to [0,1]
            plt.figure(figsize=(8, 6))
            im = plt.imshow(ssim_map_norm, cmap='viridis', vmin=0, vmax=1)
            plt.colorbar(im, label='SSIM Index')
            plt.title(f'RGB-HE SSIM Map (Score: {ssim_score:.3f})')
            plt.axis('off')
            plt.savefig(ssim_path, dpi=150, bbox_inches='tight')
            plt.close()
            created['rgb_ssim'] = True

        # Delta E map
        deltaE_path = "results/he_metrics/deltaE_rgb_he.png"
        if not os.path.exists(deltaE_path):
            try:
                # Convert to LAB
                orig_lab = color.rgb2lab(original_rgb.astype(float) / 255.0)
                he_lab = color.rgb2lab(rgb_he_rgb.astype(float) / 255.0)

                # Calculate Delta E (simplified)
                delta_e = np.sqrt(np.sum((orig_lab - he_lab) ** 2, axis=2))
                delta_e = np.clip(delta_e, 0, 50)  # Clip to 0-50 range

                plt.figure(figsize=(8, 6))
                im = plt.imshow(delta_e, cmap='plasma', vmin=0, vmax=50)
                plt.colorbar(im, label='ΔE 2000')
                plt.title('RGB-HE Delta E Map')
                plt.axis('off')
                plt.savefig(deltaE_path, dpi=150, bbox_inches='tight')
                plt.close()
                created['rgb_deltaE'] = True
            except Exception:
                pass

    # Create collage
    collage_path = "results/he_metrics/he_metrics_collage.png"
    if not os.path.exists(collage_path):
        try:
            fig, axes = plt.subplots(3, 3, figsize=(15, 12))

            methods = ['rgb_he', 'y_he', 'clahe']
            method_names = ['RGB-HE', 'Y-HE', 'CLAHE']

            for i, (method, name) in enumerate(zip(methods, method_names)):
                if method in he_results:
                    method_img = cv2.imread(he_results[method])
                    # Resize to match original if needed
                    if method_img.shape[:2] != original.shape[:2]:
                        method_img = cv2.resize(method_img, (original.shape[1], original.shape[0]))
                    method_gray = cv2.cvtColor(method_img, cv2.COLOR_BGR2GRAY)
                    method_rgb = cv2.cvtColor(method_img, cv2.COLOR_BGR2RGB)

                    # Difference
                    diff = np.abs(original_gray.astype(float) - method_gray.astype(float))
                    axes[i, 0].imshow(diff, cmap='hot', vmin=0, vmax=255)
                    axes[i, 0].set_title(f'{name} - Difference')
                    axes[i, 0].axis('off')

                    # SSIM
                    try:
                        _, ssim_map = ssim(original_gray, method_gray, full=True, data_range=255)
                        ssim_map_norm = (ssim_map + 1) / 2
                        axes[i, 1].imshow(ssim_map_norm, cmap='viridis', vmin=0, vmax=1)
                        axes[i, 1].set_title(f'{name} - SSIM')
                        axes[i, 1].axis('off')
                    except:
                        axes[i, 1].axis('off')

                    # Delta E
                    try:
                        orig_lab = color.rgb2lab(original_rgb.astype(float) / 255.0)
                        method_lab = color.rgb2lab(method_rgb.astype(float) / 255.0)
                        delta_e = np.sqrt(np.sum((orig_lab - method_lab) ** 2, axis=2))
                        delta_e = np.clip(delta_e, 0, 50)
                        axes[i, 2].imshow(delta_e, cmap='plasma', vmin=0, vmax=50)
                        axes[i, 2].set_title(f'{name} - ΔE')
                        axes[i, 2].axis('off')
                    except:
                        axes[i, 2].axis('off')

            plt.tight_layout()
            plt.savefig(collage_path, dpi=150, bbox_inches='tight')
            plt.close()
            created['collage'] = True
        except Exception:
            pass

    return created

def find_otsu_results():
    """Find existing Otsu results"""
    results = {}

    # Look for global result
    global_files = glob.glob("results/otsu/*global*.png")
    if global_files:
        results['global'] = global_files[0]

    # Look for improved result
    improved_files = glob.glob("results/otsu/*improved*.png")
    if improved_files:
        results['improved'] = improved_files[0]

    return results

def generate_otsu_result(original_gray, method):
    """Generate Otsu result if missing"""
    if method == 'global':
        _, result = cv2.threshold(original_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return result

    elif method == 'improved':
        # Simple local Otsu approximation
        h, w = original_gray.shape
        result = np.zeros_like(original_gray)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(original_gray, (5, 5), 1.0)

        window = 75
        stride = 24

        for y in range(0, h, stride):
            for x in range(0, w, stride):
                y_end = min(y + window, h)
                x_end = min(x + window, w)

                patch = blurred[y:y_end, x:x_end]
                if patch.size > 0:
                    _, patch_thresh = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    result[y:y_end, x:x_end] = patch_thresh

        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)

        return result

def create_otsu_metrics():
    """Create missing Otsu metrics"""
    original_path = "images/otsu_shadow_doc_02.jpg"
    if not os.path.exists(original_path):
        return {'compare_panel': False, 'xor_map': False, 'metrics_csv': False, 'metrics_table': False}

    original = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)

    # Find existing results
    otsu_results = find_otsu_results()

    # Generate missing results
    for method in ['global', 'improved']:
        if method not in otsu_results:
            result = generate_otsu_result(original, method)
            save_path = f"results/otsu/{method}_generated.png"
            cv2.imwrite(save_path, result)
            otsu_results[method] = save_path

    created = {'compare_panel': False, 'xor_map': False, 'metrics_csv': False, 'metrics_table': False}

    if 'global' in otsu_results and 'improved' in otsu_results:
        global_img = cv2.imread(otsu_results['global'], cv2.IMREAD_GRAYSCALE)
        improved_img = cv2.imread(otsu_results['improved'], cv2.IMREAD_GRAYSCALE)

        # Compare panel
        panel_path = "results/otsu_metrics/compare_panel.png"
        if not os.path.exists(panel_path):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(original, cmap='gray')
            axes[0].set_title('Original')
            axes[0].axis('off')

            axes[1].imshow(global_img, cmap='gray')
            axes[1].set_title('Global Otsu')
            axes[1].axis('off')

            axes[2].imshow(improved_img, cmap='gray')
            axes[2].set_title('Improved Otsu')
            axes[2].axis('off')

            plt.tight_layout()
            plt.savefig(panel_path, dpi=150, bbox_inches='tight')
            plt.close()
            created['compare_panel'] = True

        # XOR map
        xor_path = "results/otsu_metrics/xor_map.png"
        if not os.path.exists(xor_path):
            global_binary = (global_img > 127).astype(np.uint8)
            improved_binary = (improved_img > 127).astype(np.uint8)
            xor_map = cv2.bitwise_xor(global_binary, improved_binary) * 255

            plt.figure(figsize=(10, 8))
            plt.imshow(xor_map, cmap='hot')
            plt.colorbar(label='Disagreement')
            plt.title('XOR Disagreement Map')
            plt.axis('off')
            plt.savefig(xor_path, dpi=150, bbox_inches='tight')
            plt.close()
            created['xor_map'] = True

        # Metrics CSV
        csv_path = "results/otsu_metrics/metrics.csv"
        if not os.path.exists(csv_path):
            try:
                metrics_data = []

                for method_name, img in [('Global', global_img), ('Improved', improved_img)]:
                    binary = (img > 127).astype(np.uint8)

                    # Connected components
                    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
                    components = max(0, num_labels - 1)  # Exclude background

                    if components > 0:
                        areas = stats[1:, cv2.CC_STAT_AREA]  # Exclude background
                        avg_area = float(np.mean(areas))
                    else:
                        avg_area = 0.0

                    # Simple hole count (inverted components)
                    inverted = cv2.bitwise_not(binary)
                    num_holes, _, _, _ = cv2.connectedComponentsWithStats(inverted, connectivity=8)
                    holes = max(0, num_holes - 1)

                    metrics_data.append({
                        'method': method_name,
                        'components': components,
                        'avg_area': avg_area,
                        'holes': holes
                    })

                df = pd.DataFrame(metrics_data)
                df.to_csv(csv_path, index=False)
                created['metrics_csv'] = True
            except Exception:
                pass

        # Metrics table PNG
        table_path = "results/otsu_metrics/metrics_table.png"
        if not os.path.exists(table_path) and os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)

                fig, ax = plt.subplots(figsize=(8, 4))
                ax.axis('tight')
                ax.axis('off')

                table = ax.table(cellText=df.values, colLabels=df.columns,
                                cellLoc='center', loc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(12)
                table.scale(1.2, 2)

                plt.title('Otsu Methods Comparison', fontsize=14, pad=20)
                plt.savefig(table_path, dpi=150, bbox_inches='tight')
                plt.close()
                created['metrics_table'] = True
            except Exception:
                pass

    return created

def main():
    # 1. Ensure dependencies
    if not ensure_deps():
        return {"task": "rebuild_missing", "error": "Missing dependencies"}

    # 2. Make directories
    make_dirs()

    notes = []

    # 3. Convert MP4s to GIFs
    he_gif_created = convert_mp4_to_gif("results/video/he_sweep.mp4", "results/video/he_sweep.gif")
    otsu_gif_created = convert_mp4_to_gif("results/video/otsu_sweep.mp4", "results/video/otsu_sweep.gif")

    if he_gif_created:
        notes.append("generated: he_sweep.gif")
    elif os.path.exists("results/video/he_sweep.gif"):
        notes.append("reused: he_sweep.gif")
    else:
        notes.append("skipped: he_sweep.gif (source missing)")

    if otsu_gif_created:
        notes.append("generated: otsu_sweep.gif")
    elif os.path.exists("results/video/otsu_sweep.gif"):
        notes.append("reused: otsu_sweep.gif")
    else:
        notes.append("skipped: otsu_sweep.gif (source missing)")

    # 4. HE metrics
    he_metrics = create_he_metrics()
    for key, created in he_metrics.items():
        if created:
            notes.append(f"generated: {key}")

    # 5. Otsu metrics
    otsu_metrics = create_otsu_metrics()
    for key, created in otsu_metrics.items():
        if created:
            notes.append(f"generated: {key}")

    # Final JSON
    result = {
        "task": "rebuild_missing",
        "videos": {
            "he_gif": he_gif_created or os.path.exists("results/video/he_sweep.gif"),
            "otsu_gif": otsu_gif_created or os.path.exists("results/video/otsu_sweep.gif")
        },
        "he_metrics": {
            "rgb_diff": he_metrics.get('rgb_diff', False) or os.path.exists("results/he_metrics/diff_rgb_he.png"),
            "rgb_ssim": he_metrics.get('rgb_ssim', False) or os.path.exists("results/he_metrics/ssim_rgb_he.png"),
            "rgb_deltaE": he_metrics.get('rgb_deltaE', False) or os.path.exists("results/he_metrics/deltaE_rgb_he.png"),
            "collage": he_metrics.get('collage', False) or os.path.exists("results/he_metrics/he_metrics_collage.png")
        },
        "otsu_metrics": {
            "compare_panel": otsu_metrics.get('compare_panel', False) or os.path.exists("results/otsu_metrics/compare_panel.png"),
            "xor_map": otsu_metrics.get('xor_map', False) or os.path.exists("results/otsu_metrics/xor_map.png"),
            "metrics_csv": otsu_metrics.get('metrics_csv', False) or os.path.exists("results/otsu_metrics/metrics.csv"),
            "metrics_table": otsu_metrics.get('metrics_table', False) or os.path.exists("results/otsu_metrics/metrics_table.png")
        },
        "notes": notes
    }

    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()