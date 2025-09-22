#!/usr/bin/env python3
"""
Local Otsu Thresholding 명령줄 실행 스크립트
Local Otsu Thresholding Command Line Script

Usage:
    python run_otsu.py <image_path> [method] [options]

Example:
    python run_otsu.py images/test.jpg sliding --block-size 32 --stride 16 --save results/
"""

import sys
import os
import argparse

# 현재 스크립트의 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from src.utils import load_image, save_image
from src.otsu import (global_otsu_thresholding, local_otsu_block_based,
                     local_otsu_sliding_window, compare_otsu_methods,
                     local_otsu_improved_boundary)
import cv2

def main():
    parser = argparse.ArgumentParser(description='Local Otsu Thresholding')

    parser.add_argument('image_path', help='입력 이미지 파일 경로 / Input image file path')
    parser.add_argument('--method', choices=['global', 'block', 'sliding', 'improved', 'compare'], default='compare',
                       help='처리 방법 / Processing method: global, block, sliding, improved(개선된 블록), compare(모든 방법 비교)')
    parser.add_argument('--block-size', type=int, default=32,
                       help='블록/윈도우 크기 (기본값: 32) / Block/Window size (default: 32)')
    parser.add_argument('--stride', type=int, default=16,
                       help='슬라이딩 윈도우 스트라이드 (기본값: 16) / Sliding window stride (default: 16)')
    parser.add_argument('--save', metavar='DIR',
                       help='결과 저장 디렉토리 / Result saving directory')
    parser.add_argument('--show-comparison', action='store_true',
                       help='비교 결과 시각화 표시 / Show comparison visualization')

    args = parser.parse_args()

    try:
        # 이미지 로드
        print(f"이미지 로딩 중... / Loading image: {args.image_path}")
        image = load_image(args.image_path)

        # 그레이스케일로 변환
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            print(f"그레이스케일로 변환 / Converted to grayscale: {gray_image.shape}")
        else:
            gray_image = image
            print(f"이미지 크기 / Image size: {gray_image.shape}")

        # 방법별 처리
        if args.method == 'global':
            print("Global Otsu Thresholding 실행 중...")
            result, threshold_info = global_otsu_thresholding(gray_image, show_process=True)
            results = {'global': {'result': result, 'info': threshold_info}}

        elif args.method == 'block':
            print(f"Block-based Local Otsu 실행 중 (블록 크기: {args.block_size}x{args.block_size})...")
            result, info = local_otsu_block_based(
                gray_image, block_size=(args.block_size, args.block_size), show_process=True
            )
            threshold_map = info.get('threshold_map', None)
            results = {'block': {'result': result, 'threshold_map': threshold_map, 'info': info}}

        elif args.method == 'sliding':
            print(f"Sliding Window Local Otsu 실행 중 (윈도우: {args.block_size}x{args.block_size}, 스트라이드: {args.stride})...")
            result, info = local_otsu_sliding_window(
                gray_image,
                window_size=(args.block_size, args.block_size),
                stride=args.stride,
                show_process=True
            )
            threshold_map = info.get('threshold_map', None)
            results = {'sliding': {'result': result, 'threshold_map': threshold_map, 'info': info}}

        elif args.method == 'improved':
            print(f"Improved Local Otsu 실행 중 (블록: {args.block_size}x{args.block_size}, 겹침 처리)...")
            result, info = local_otsu_improved_boundary(
                gray_image,
                block_size=(args.block_size, args.block_size),
                overlap_ratio=0.5,
                blend_method='weighted_average',
                show_process=True
            )
            threshold_map = info.get('threshold_map', None)
            results = {'improved': {'result': result, 'threshold_map': threshold_map, 'info': info}}

        elif args.method == 'compare':
            print("모든 방법 비교 실행 중... / Running comparison of all methods...")
            results = compare_otsu_methods(
                gray_image,
                show_comparison=args.show_comparison
            )

        # 결과 정보 출력
        print("\n=== 처리 결과 정보 / Processing Result Information ===")
        for method_name, method_result in results.items():
            print(f"\n{method_name.upper()} 방법:")
            if 'info' in method_result:
                info = method_result['info']
                if isinstance(info, dict):
                    for key, value in info.items():
                        print(f"  {key}: {value}")
                else:
                    print(f"  임계값 / Threshold: {info}")

        # 결과 저장
        if args.save:
            if not os.path.exists(args.save):
                os.makedirs(args.save)

            base_name = os.path.splitext(os.path.basename(args.image_path))[0]

            for method_name, method_result in results.items():
                if 'result' in method_result:
                    output_path = os.path.join(args.save, f"{base_name}_otsu_{method_name}.jpg")
                    save_image(method_result['result'], output_path)
                    print(f"결과 저장됨 / Result saved: {output_path}")

        else:
            print("\n결과 저장하려면 --save 옵션을 사용하세요 / Use --save option to save results")

        print("Local Otsu Thresholding 완료! / Local Otsu Thresholding completed!")

    except Exception as e:
        print(f"오류 발생 / Error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()