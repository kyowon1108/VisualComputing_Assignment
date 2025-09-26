#!/usr/bin/env python3
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
"""
Enhanced Otsu Command Line Interface
개선된 Otsu 임계값 명령행 인터페이스

Usage:
    python run_otsu.py images/otsu_shadow_doc_02.jpg --method improved --window 75 --stride 24 --preblur 1.0 --morph open,3 --morph close,3 --show-plots --save results/otsu/
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

from src.otsu import (global_otsu, block_based_otsu, sliding_window_otsu, improved_otsu,
                     create_threshold_heatmap, create_local_histogram_with_threshold,
                     create_otsu_comparison_contact_sheet)

def parse_morph_operations(morph_list):
    """형태학적 연산 리스트를 파싱합니다."""
    if not morph_list:
        return []
    return morph_list

def main():
    parser = argparse.ArgumentParser(description='Enhanced Otsu Thresholding Tool')
    parser.add_argument('input', help='Input image path')
    parser.add_argument('--method', choices=['global', 'block', 'sliding', 'improved'],
                       default='improved', help='Otsu method to use')
    parser.add_argument('--window', type=int, default=75,
                       help='Window size for local methods (odd number recommended: 51-101)')
    parser.add_argument('--stride', type=int, default=24,
                       help='Stride for overlapping control (smaller = smoother)')
    parser.add_argument('--preblur', type=float, default=1.0,
                       help='Gaussian blur sigma for preprocessing (0.8-1.5 for glare reduction)')
    parser.add_argument('--morph', action='append', default=[],
                       help='Morphological operations (e.g., open,3 close,3). Can be used multiple times.')
    parser.add_argument('--show-plots', action='store_true',
                       help='Generate and save analysis plots')
    parser.add_argument('--roi', type=str, default='',
                       help='ROI coordinates for analysis as "x,y,w,h;x,y,w,h;..." format')
    parser.add_argument('--save', type=str, default='results/otsu/',
                       help='Output directory for saving results')

    args = parser.parse_args()

    # 기본 형태학적 연산 설정
    if not args.morph:
        args.morph = ['open,3', 'close,3']

    # 출력 디렉토리 생성
    save_dir = Path(args.save)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 이미지 로드
    if not os.path.exists(args.input):
        print(f"Error: Input image '{args.input}' not found")
        return 1

    original_img = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    if original_img is None:
        print(f"Error: Cannot load image '{args.input}'")
        return 1

    print(f"Processing image: {args.input}")
    print(f"Image size: {original_img.shape}")
    print(f"Method: {args.method}")
    print(f"Window size: {args.window}")
    print(f"Stride: {args.stride}")
    print(f"Preblur sigma: {args.preblur}")
    print(f"Morphological operations: {args.morph}")

    # ROI 파싱
    rois = []
    if args.roi:
        for roi_part in args.roi.split(';'):
            if roi_part.strip():
                try:
                    x, y, w, h = map(int, roi_part.split(','))
                    rois.append((x, y, w, h))
                except ValueError:
                    print(f"Warning: Invalid ROI format '{roi_part}', skipping...")

    # 기본 ROI 제안 (Otsu용)
    if not rois:
        h, w = original_img.shape
        rois = [
            (int(w*0.7), int(h*0.1), int(w*0.25), int(h*0.3)),  # 우상단 글레어 영역
            (int(w*0.1), int(h*0.3), int(w*0.4), int(h*0.4)),   # 좌측 균일 텍스트 영역
            (int(w*0.05), int(h*0.05), int(w*0.2), int(h*0.8))  # 제본(스파이럴) 경계
        ]
        print(f"Using default ROIs for document analysis: {rois}")

    # Otsu 방법 적용
    if args.method == 'global':
        result = global_otsu(original_img)
    elif args.method == 'block':
        result = block_based_otsu(original_img, args.window, args.stride)
    elif args.method == 'sliding':
        result = sliding_window_otsu(original_img, args.window, args.stride)
    elif args.method == 'improved':
        result = improved_otsu(original_img, args.window, args.stride, args.preblur, args.morph)

    # 결과 이미지 저장
    result_path = save_dir / f'result_{args.method}.png'
    cv2.imwrite(str(result_path), result['result'])
    print(f"Result image saved: {result_path}")

    # 분석 플롯 생성
    if args.show_plots:
        # 임계값 히트맵 (local methods only)
        if 'threshold_map' in result:
            heatmap_path = save_dir / 'threshold_heatmap.png'
            create_threshold_heatmap(result['threshold_map'], str(heatmap_path),
                                   f'{args.method.title()} Method Threshold Map')

        # 선택 윈도우의 히스토그램 (글레어 ROI)
        if rois and 'threshold_map' in result:
            # 첫 번째 ROI (글레어 영역)를 선택
            glare_roi = rois[0]
            x, y, w, h = glare_roi

            # ROI 중심의 임계값 가져오기
            roi_threshold = result['threshold_map'][y + h//2, x + w//2]

            local_hist_path = save_dir / 'local_hist_with_T.png'
            create_local_histogram_with_threshold(original_img, glare_roi, roi_threshold, str(local_hist_path))

    # 방법 비교 및 콘택트 시트 생성
    if args.show_plots:
        # Global과 Improved 비교
        comparison_results = {
            'global_otsu': global_otsu(original_img),
            'improved': result
        }

        contact_sheet_path = save_dir / 'compare_otsu_contact_sheet.png'
        create_otsu_comparison_contact_sheet(original_img, comparison_results, rois, str(contact_sheet_path))

    # ROI별 지표 계산 및 CSV 저장
    if rois:
        roi_metrics = []
        methods_to_compare = {
            'Original': original_img,
            'Global': global_otsu(original_img)['result'],
            args.method.title(): result['result']
        }

        for roi_idx, roi in enumerate(rois):
            x, y, w, h = roi
            for method_name, method_result in methods_to_compare.items():
                roi_image = method_result[y:y+h, x:x+w]

                # 지표 계산
                mean_brightness = np.mean(roi_image)
                brightness_std = np.std(roi_image)

                # 에지 강도 (Sobel)
                sobel_x = cv2.Sobel(roi_image, cv2.CV_64F, 1, 0, ksize=3)
                sobel_y = cv2.Sobel(roi_image, cv2.CV_64F, 0, 1, ksize=3)
                edge_strength = np.sum(np.sqrt(sobel_x**2 + sobel_y**2))

                metrics = {
                    'method': method_name,
                    'roi_id': roi_idx + 1,
                    'roi_coords': f"{x},{y},{w},{h}",
                    'mean_brightness': mean_brightness,
                    'brightness_std': brightness_std,
                    'edge_strength': edge_strength
                }
                roi_metrics.append(metrics)

        # CSV로 저장
        df = pd.DataFrame(roi_metrics)
        csv_path = save_dir / 'roi_analysis.csv'
        df.to_csv(csv_path, index=False)
        print(f"ROI analysis saved: {csv_path}")

    # 결과 정보 출력
    print("\n=== Processing Results ===")
    print(f"Method: {result['method']}")
    if 'threshold' in result:
        print(f"Global threshold: {result['threshold']:.1f}")
    if 'parameters' in result:
        print("Parameters:")
        for key, value in result['parameters'].items():
            print(f"  {key}: {value}")

    print("Processing completed successfully!")
    return 0

if __name__ == '__main__':
    sys.exit(main())