#!/usr/bin/env python3
"""
Enhanced HE Command Line Interface
개선된 히스토그램 평활화 명령행 인터페이스

Usage:
    python run_he.py images/he_dark_indoor.jpg --space yuv --he-mode clahe --tile 8 8 --clip 2.5 --show-plots --save results/he/
"""

import argparse
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.he import he_luma_bgr, extract_roi_metrics, calculate_rms_contrast, calculate_edge_strength

def parse_roi_string(roi_string: str):
    """ROI 문자열을 파싱합니다. 형식: 'x,y,w,h;x,y,w,h;...'"""
    if not roi_string:
        return []

    rois = []
    for roi_part in roi_string.split(';'):
        if roi_part.strip():
            try:
                x, y, w, h = map(int, roi_part.split(','))
                rois.append((x, y, w, h))
            except ValueError:
                print(f"Warning: Invalid ROI format '{roi_part}', skipping...")

    return rois

def create_comparison_contact_sheet(original_img, results, rois, save_path, dpi=300):
    """비교 콘택트 시트를 생성합니다."""
    n_methods = len(results)
    n_rois = len(rois)

    # 그리드 크기 결정: 풀샷 1행 + ROI별 1행씩
    fig_height = 4 * (1 + n_rois)  # 각 행당 4인치
    fig_width = 4 * n_methods  # 각 열당 4인치

    fig, axes = plt.subplots(1 + n_rois, n_methods, figsize=(fig_width, fig_height), dpi=dpi)

    if n_methods == 1:
        axes = axes.reshape(-1, 1)

    # 첫 번째 행: 풀샷 비교
    method_names = list(results.keys())

    for col, method_name in enumerate(method_names):
        result_img = results[method_name]["img"]
        if len(result_img.shape) == 3:
            result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

        axes[0, col].imshow(result_img if col > 0 else cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        axes[0, col].set_title(f"{method_name}", fontsize=12, fontweight='bold')
        axes[0, col].axis('off')

        # 원본에만 ROI 박스 표시
        if col == 0:
            for i, (x, y, w, h) in enumerate(rois):
                rect = plt.Rectangle((x, y), w, h, linewidth=3, edgecolor='red', facecolor='none')
                axes[0, col].add_patch(rect)
                axes[0, col].text(x, y-5, f'ROI{i+1}', color='red', fontweight='bold', fontsize=10)

    # ROI별 확대 비교
    for roi_idx, (x, y, w, h) in enumerate(rois):
        row = roi_idx + 1

        for col, method_name in enumerate(method_names):
            if col == 0:  # 원본
                roi_img = original_img[y:y+h, x:x+w]
                if len(roi_img.shape) == 3:
                    roi_img = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)
                title = f"Original ROI{roi_idx+1}"
            else:  # 처리된 이미지
                result_img = results[method_name]["img"]
                roi_img = result_img[y:y+h, x:x+w]
                if len(roi_img.shape) == 3:
                    roi_img = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)
                title = f"{method_name} ROI{roi_idx+1}"

            # 200% 확대를 위해 interpolation 사용
            roi_img_resized = cv2.resize(roi_img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

            axes[row, col].imshow(roi_img_resized)
            axes[row, col].set_title(title, fontsize=10)
            axes[row, col].axis('off')

    plt.suptitle('HE Methods Comparison with ROI Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"Comparison contact sheet saved: {save_path}")

def create_histogram_plots(result, save_dir):
    """히스토그램과 CDF 플롯을 생성합니다."""
    # 히스토그램 before/after
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Original histogram
    ax1.bar(range(256), result["hist"]["original"], alpha=0.7, color='gray', edgecolor='black')
    ax1.set_title('Original Histogram', fontweight='bold')
    ax1.set_xlabel('Pixel Value')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)

    # Equalized histogram
    ax2.bar(range(256), result["hist"]["equalized"], alpha=0.7, color='blue', edgecolor='black')
    ax2.set_title('Equalized Histogram', fontweight='bold')
    ax2.set_xlabel('Pixel Value')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'hist_before_after.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # CDF overlay plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.plot(result["cdf"]["original"], label='Original CDF', linewidth=2, color='gray')
    ax.plot(result["cdf"]["equalized"], label='Equalized CDF', linewidth=2, color='blue')

    # 몇 개 픽셀값의 매핑 예시 화살표
    sample_values = [50, 100, 150, 200]
    for val in sample_values:
        if val < len(result["cdf"]["original"]):
            orig_cdf = result["cdf"]["original"][val]
            new_val = int(orig_cdf * 255)
            if new_val < len(result["cdf"]["equalized"]):
                ax.annotate('', xy=(new_val, result["cdf"]["equalized"][new_val]),
                           xytext=(val, orig_cdf),
                           arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

    ax.set_title('CDF Overlay with Mapping Examples', fontweight='bold')
    ax.set_xlabel('Pixel Value')
    ax.set_ylabel('Cumulative Probability')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cdf_overlay.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Histogram plots saved in: {save_dir}")

def main():
    parser = argparse.ArgumentParser(description='Enhanced Histogram Equalization Tool')
    parser.add_argument('input', help='Input image path')
    parser.add_argument('--he-mode','--algorithm', dest='he_mode',
                       choices=['he','ahe','clahe'], default='clahe')
    parser.add_argument('--space','--method', dest='space',
                       choices=['rgb','yuv','lab','hsv'], default='yuv')
    parser.add_argument('--tile', nargs=2, type=int, default=[8, 8],
                       help='AHE/CLAHE tile grid size (width height)')
    parser.add_argument('--clip', type=float, default=2.5,
                       help='CLAHE clip limit (2.0-3.0 recommended)')
    parser.add_argument('--bins', type=int, default=256,
                       help='Number of histogram bins')
    parser.add_argument('--show-plots', action='store_true',
                       help='Generate and save histogram/CDF plots')
    parser.add_argument('--roi', type=str, default='',
                       help='ROI coordinates as "x,y,w,h;x,y,w,h;..." format')
    parser.add_argument('--save', type=str, default='results/he/',
                       help='Output directory for saving results')
    # run_he.py (argparse 일부)
    

    args = parser.parse_args()

    # 출력 디렉토리 생성
    save_dir = Path(args.save)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 이미지 로드
    if not os.path.exists(args.input):
        print(f"Error: Input image '{args.input}' not found")
        return 1

    original_img = cv2.imread(args.input)
    if original_img is None:
        print(f"Error: Cannot load image '{args.input}'")
        return 1

    print(f"Processing image: {args.input}")
    print(f"Image size: {original_img.shape}")
    print(f"Color space: {args.space}")
    print(f"HE mode: {args.he_mode}")
    print(f"Tile size: {args.tile}")
    if args.he_mode == 'clahe':
        print(f"Clip limit: {args.clip}")

    # ROI 파싱
    rois = parse_roi_string(args.roi)
    if rois:
        print(f"ROIs: {rois}")

        # 기본 ROI 제안 (HE용)
        h, w = original_img.shape[:2]
        default_rois = [
            (int(w*0.1), int(h*0.1), int(w*0.3), int(h*0.2)),  # 키보드 하우징 상단(암부)
            (int(w*0.5), int(h*0.6), int(w*0.2), int(h*0.2)),  # 마우스 주변(암부+스펙큘러)
            (int(w*0.2), int(h*0.8), int(w*0.6), int(h*0.1))   # 모니터 아래 바(광원 대비)
        ]
        if not rois:
            rois = default_rois
            print(f"Using default ROIs: {rois}")

    # HE 적용
    clip_value = args.clip if args.he_mode == 'clahe' else None
    result = he_luma_bgr(
        original_img,
        space=args.space,
        mode=args.he_mode,
        tile=tuple(args.tile),
        clip=clip_value,
        bins=args.bins
    )

    # 결과 이미지 저장
    result_path = save_dir / f'result_{args.space}_{args.he_mode}.png'
    cv2.imwrite(str(result_path), result["img"])
    print(f"Result image saved: {result_path}")

    # 히스토그램/CDF 플롯 생성
    if args.show_plots:
        create_histogram_plots(result, str(save_dir))

    # ROI 분석 및 비교 시트 생성
    if rois:
        # 다양한 방법으로 처리한 결과들
        methods = {
            'Original': {'img': original_img},
            f'RGB-HE': he_luma_bgr(original_img, space='rgb', mode='global'),
            f'Y({args.space})-HE': he_luma_bgr(original_img, space=args.space, mode='global'),
            f'CLAHE': result
        }

        # 비교 콘택트 시트 생성
        contact_sheet_path = save_dir / 'compare_he_contact_sheet.png'
        create_comparison_contact_sheet(original_img, methods, rois, str(contact_sheet_path))

        # ROI별 지표 계산 및 CSV 저장
        roi_metrics = []
        for roi_idx, roi in enumerate(rois):
            for method_name, method_result in methods.items():
                if method_name == 'Original':
                    img_for_analysis = original_img
                else:
                    img_for_analysis = method_result["img"]

                metrics = extract_roi_metrics(img_for_analysis, roi)
                metrics.update({
                    'method': method_name,
                    'roi_id': roi_idx + 1,
                    'roi_coords': f"{roi[0]},{roi[1]},{roi[2]},{roi[3]}"
                })
                roi_metrics.append(metrics)

        # CSV로 저장
        df = pd.DataFrame(roi_metrics)
        csv_path = save_dir / 'roi_analysis.csv'
        df.to_csv(csv_path, index=False)
        print(f"ROI analysis saved: {csv_path}")

    print("Processing completed successfully!")
    return 0

if __name__ == '__main__':
    sys.exit(main())