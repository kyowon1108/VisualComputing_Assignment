#!/usr/bin/env python3
"""
HE 결과 분석 및 JSON 요약 생성
Analyze HE results and generate JSON summary
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import ndimage

def find_roi_patches(image, patch_size=(96, 96)):
    """
    이미지에서 자동으로 ROI 패치를 찾습니다.
    Find ROI patches automatically from image.

    Returns:
        dict: ROI 정보 (darkest, brightest, highest_gradient)
    """
    h, w = image.shape[:2]
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    patch_h, patch_w = patch_size

    # 가능한 모든 패치 위치 계산
    patches = []
    for y in range(0, h - patch_h, patch_h // 2):
        for x in range(0, w - patch_w, patch_w // 2):
            if y + patch_h <= h and x + patch_w <= w:
                patch = gray[y:y+patch_h, x:x+patch_w]

                # 패치 특성 계산
                mean_brightness = np.mean(patch)

                # Sobel 그래디언트
                sobel_x = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
                sobel_y = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
                gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
                mean_gradient = np.mean(gradient_magnitude)

                patches.append({
                    'x': x, 'y': y,
                    'w': patch_w, 'h': patch_h,
                    'brightness': mean_brightness,
                    'gradient': mean_gradient
                })

    # 특성별로 정렬하여 ROI 선택
    rois = {}

    # 가장 어두운 패치
    patches_sorted = sorted(patches, key=lambda p: p['brightness'])
    darkest = patches_sorted[0]
    rois['darkest'] = {'x': darkest['x'], 'y': darkest['y'],
                       'w': darkest['w'], 'h': darkest['h'],
                       'brightness': darkest['brightness']}

    # 가장 밝은 패치
    brightest = patches_sorted[-1]
    rois['brightest'] = {'x': brightest['x'], 'y': brightest['y'],
                         'w': brightest['w'], 'h': brightest['h'],
                         'brightness': brightest['brightness']}

    # 그래디언트가 가장 높은 패치
    patches_sorted = sorted(patches, key=lambda p: p['gradient'], reverse=True)
    highest_grad = patches_sorted[0]
    rois['highest_gradient'] = {'x': highest_grad['x'], 'y': highest_grad['y'],
                                'w': highest_grad['w'], 'h': highest_grad['h'],
                                'gradient': highest_grad['gradient']}

    return rois

def compute_roi_metrics(image, roi):
    """
    ROI 영역의 정량적 지표를 계산합니다.
    Compute quantitative metrics for ROI region.
    """
    x, y, w, h = roi['x'], roi['y'], roi['w'], roi['h']
    roi_img = image[y:y+h, x:x+w]

    if len(roi_img.shape) == 3:
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi_img.copy()

    # 기본 통계
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))

    # RMS 대비
    rms_contrast = float(np.sqrt(np.mean((gray - mean_val) ** 2)))

    # Sobel 에지 강도
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_strength = float(np.mean(np.sqrt(sobel_x**2 + sobel_y**2)))

    return {
        'mean': round(mean_val, 2),
        'std': round(std_val, 2),
        'rms_contrast': round(rms_contrast, 2),
        'edge_strength': round(edge_strength, 2)
    }

def create_comparison_figure(original_img, results, rois, output_path):
    """
    비교 figure를 생성합니다.
    Create comparison figure.
    """
    fig = plt.figure(figsize=(20, 16))

    # 원본 이미지 및 ROI 표시
    ax1 = plt.subplot(4, 4, 1)
    if len(original_img.shape) == 3:
        plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(original_img, cmap='gray')

    # ROI 박스 그리기
    colors = {'darkest': 'blue', 'brightest': 'yellow', 'highest_gradient': 'red'}
    for roi_name, roi in rois.items():
        rect = plt.Rectangle((roi['x'], roi['y']), roi['w'], roi['h'],
                            fill=False, edgecolor=colors[roi_name], linewidth=2)
        ax1.add_patch(rect)
        ax1.text(roi['x'], roi['y']-5, roi_name, color=colors[roi_name], fontsize=8)

    plt.title('Original with ROIs')
    plt.axis('off')

    # 각 방법의 결과
    method_names = ['RGB-HE', 'Y-HE', 'CLAHE']
    for idx, (method, result) in enumerate(results.items()):
        ax = plt.subplot(4, 4, idx + 2)
        if len(result.shape) == 3:
            plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(result, cmap='gray')
        plt.title(method_names[idx])
        plt.axis('off')

    # ROI 확대 비교 (200%)
    roi_names_list = list(rois.keys())
    for roi_idx, roi_name in enumerate(roi_names_list):
        roi = rois[roi_name]
        x, y, w, h = roi['x'], roi['y'], roi['w'], roi['h']

        # 원본 ROI
        ax = plt.subplot(4, 4, 5 + roi_idx * 4)
        roi_crop = original_img[y:y+h, x:x+w]
        zoom_factor = 2
        if len(roi_crop.shape) == 3:
            roi_zoom = cv2.resize(roi_crop, None, fx=zoom_factor, fy=zoom_factor,
                                 interpolation=cv2.INTER_LINEAR)
            plt.imshow(cv2.cvtColor(roi_zoom, cv2.COLOR_BGR2RGB))
        else:
            roi_zoom = cv2.resize(roi_crop, None, fx=zoom_factor, fy=zoom_factor,
                                 interpolation=cv2.INTER_LINEAR)
            plt.imshow(roi_zoom, cmap='gray')
        plt.ylabel(f'{roi_name}\\n(200% zoom)', fontsize=10)
        plt.title('Original' if roi_idx == 0 else '', fontsize=10)
        plt.axis('off')

        # 각 방법별 ROI 확대
        for method_idx, (method, result) in enumerate(results.items()):
            ax = plt.subplot(4, 4, 6 + roi_idx * 4 + method_idx)
            roi_crop = result[y:y+h, x:x+w]

            # 200% 확대
            if len(roi_crop.shape) == 3:
                roi_zoom = cv2.resize(roi_crop, None, fx=zoom_factor, fy=zoom_factor,
                                     interpolation=cv2.INTER_LINEAR)
                plt.imshow(cv2.cvtColor(roi_zoom, cv2.COLOR_BGR2RGB))
            else:
                roi_zoom = cv2.resize(roi_crop, None, fx=zoom_factor, fy=zoom_factor,
                                     interpolation=cv2.INTER_LINEAR)
                plt.imshow(roi_zoom, cmap='gray')

            plt.title(method_names[method_idx] if roi_idx == 0 else '', fontsize=10)
            plt.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def create_histogram_cdf_plots(original_img, results, output_path):
    """
    히스토그램과 CDF 비교 플롯을 생성합니다.
    Create histogram and CDF comparison plots.
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # 그레이스케일 변환
    if len(original_img.shape) == 3:
        gray_orig = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    else:
        gray_orig = original_img.copy()

    # 히스토그램 계산
    methods = ['Original', 'RGB-HE', 'Y-HE', 'CLAHE']
    images = [gray_orig] + [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
                            for img in results.values()]

    for idx, (method, img) in enumerate(zip(methods, images)):
        # 히스토그램
        hist, bins = np.histogram(img.flatten(), bins=256, range=(0, 256))
        axes[0, idx].plot(bins[:-1], hist, color='blue', alpha=0.7)
        axes[0, idx].set_title(f'{method} Histogram')
        axes[0, idx].set_xlim([0, 255])
        axes[0, idx].grid(True, alpha=0.3)

        # CDF
        cdf = hist.cumsum()
        cdf_normalized = cdf / cdf[-1]
        axes[1, idx].plot(bins[:-1], cdf_normalized, color='red', alpha=0.7)
        axes[1, idx].set_title(f'{method} CDF')
        axes[1, idx].set_xlim([0, 255])
        axes[1, idx].set_ylim([0, 1])
        axes[1, idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    # 원본 이미지 로드
    original_path = 'images/he_dark_indoor.jpg'
    original_img = cv2.imread(original_path)

    # 처리된 결과 이미지 로드
    results = {
        'rgb_he': cv2.imread('results/he_rgb/result_rgb_global.png'),
        'y_he': cv2.imread('results/he_yuv/result_yuv_global.png'),
        'clahe': cv2.imread('results/he_clahe/result_yuv_clahe.png')
    }

    # ROI 자동 선택
    rois = find_roi_patches(original_img)

    # 각 방법별 ROI 지표 계산
    metrics = {}

    # 원본 지표
    metrics['original'] = {}
    for roi_name, roi in rois.items():
        metrics['original'][roi_name] = compute_roi_metrics(original_img, roi)

    # 각 방법별 지표
    for method_key, method_img in results.items():
        metrics[method_key] = {}
        for roi_name, roi in rois.items():
            metrics[method_key][roi_name] = compute_roi_metrics(method_img, roi)

    # 비교 figure 생성
    os.makedirs('results/he_analysis', exist_ok=True)
    create_comparison_figure(original_img, results, rois,
                            'results/he_analysis/comparison_sheet.png')

    # 히스토그램/CDF 플롯 생성
    create_histogram_cdf_plots(original_img, results,
                              'results/he_analysis/histogram_cdf.png')

    # JSON 요약 생성
    summary = {
        "input": original_path,
        "methods": {
            "RGB-HE": {
                "command": "python run_he.py images/he_dark_indoor.jpg --space rgb --he-mode global",
                "output": "results/he_rgb/result_rgb_global.png"
            },
            "Y-HE": {
                "command": "python run_he.py images/he_dark_indoor.jpg --space yuv --he-mode global",
                "output": "results/he_yuv/result_yuv_global.png"
            },
            "CLAHE": {
                "command": "python run_he.py images/he_dark_indoor.jpg --space yuv --he-mode clahe --tile 8 8 --clip 2.5",
                "output": "results/he_clahe/result_yuv_clahe.png"
            }
        },
        "roi_selection": {
            "darkest": {"coords": rois['darkest'], "criteria": "Lowest mean brightness"},
            "brightest": {"coords": rois['brightest'], "criteria": "Highest mean brightness"},
            "highest_gradient": {"coords": rois['highest_gradient'], "criteria": "Highest edge gradient"}
        },
        "metrics": metrics,
        "improvements": {
            "RGB-HE": {
                "darkest": {
                    "brightness_increase": round(metrics['rgb_he']['darkest']['mean'] - metrics['original']['darkest']['mean'], 2),
                    "contrast_change": round(metrics['rgb_he']['darkest']['rms_contrast'] - metrics['original']['darkest']['rms_contrast'], 2)
                },
                "overall": "Increases brightness but may affect color balance"
            },
            "Y-HE": {
                "darkest": {
                    "brightness_increase": round(metrics['y_he']['darkest']['mean'] - metrics['original']['darkest']['mean'], 2),
                    "contrast_change": round(metrics['y_he']['darkest']['rms_contrast'] - metrics['original']['darkest']['rms_contrast'], 2)
                },
                "overall": "Preserves color while improving luminance"
            },
            "CLAHE": {
                "darkest": {
                    "brightness_increase": round(metrics['clahe']['darkest']['mean'] - metrics['original']['darkest']['mean'], 2),
                    "contrast_change": round(metrics['clahe']['darkest']['rms_contrast'] - metrics['original']['darkest']['rms_contrast'], 2)
                },
                "overall": "Local adaptive enhancement with controlled contrast"
            }
        },
        "visualizations": {
            "comparison_sheet": "results/he_analysis/comparison_sheet.png",
            "histogram_cdf": "results/he_analysis/histogram_cdf.png"
        }
    }

    # JSON 저장
    with open('results/he_analysis/summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Analysis complete!")
    print(f"Summary saved to: results/he_analysis/summary.json")
    print(f"Comparison sheet: results/he_analysis/comparison_sheet.png")
    print(f"Histogram/CDF plots: results/he_analysis/histogram_cdf.png")

    # 간단한 요약 출력
    print("\n=== Summary ===")
    print(f"ROI Selection:")
    for roi_name, roi in rois.items():
        print(f"  {roi_name}: x={roi['x']}, y={roi['y']}, w={roi['w']}, h={roi['h']}")

    print(f"\nBest improvements in darkest region:")
    for method in ['RGB-HE', 'Y-HE', 'CLAHE']:
        improvement = summary['improvements'][method]['darkest']
        print(f"  {method}: brightness +{improvement['brightness_increase']}, contrast {improvement['contrast_change']:+.2f}")

if __name__ == '__main__':
    main()