#!/usr/bin/env python3
"""
히스토그램 평활화 명령줄 실행 스크립트
Histogram Equalization Command Line Script

Usage:
    python run_he.py <image_path> [options]

Examples:
    # Global HE (기본)
    python run_he.py images/test.jpg --algorithm he --method yuv --save results/

    # Adaptive HE
    python run_he.py images/test.jpg --algorithm ahe --tile-size 16 --save results/

    # CLAHE
    python run_he.py images/test.jpg --algorithm clahe --clip-limit 3.0 --tile-size 8 --save results/
"""

import sys
import os
import argparse

# 현재 스크립트의 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from src.utils import load_image, save_image
from src.he import (histogram_equalization_color, histogram_equalization_grayscale,
                   clahe_implementation, visualize_color_he_process)

def main():
    parser = argparse.ArgumentParser(description='컬러 이미지 히스토그램 평활화 / Color Image Histogram Equalization')

    parser.add_argument('image_path', help='입력 이미지 파일 경로 / Input image file path')
    parser.add_argument('--method', choices=['yuv', 'rgb', 'gray'], default='yuv',
                       help='처리 방법: yuv(권장), rgb, gray / Processing method: yuv(recommended), rgb, gray')
    parser.add_argument('--algorithm', choices=['he', 'ahe', 'clahe'], default='he',
                       help='알고리즘: he(일반), ahe(적응적), clahe(제한적응적) / Algorithm: he(global), ahe(adaptive), clahe(contrast-limited)')
    parser.add_argument('--clip-limit', type=float, default=2.0,
                       help='CLAHE 클립 한계 (기본값: 2.0) / CLAHE clip limit (default: 2.0)')
    parser.add_argument('--tile-size', type=int, default=8,
                       help='CLAHE/AHE 타일 크기 (기본값: 8) / CLAHE/AHE tile size (default: 8)')
    parser.add_argument('--show-process', action='store_true',
                       help='중간 과정 시각화 표시 / Show intermediate process visualization')
    parser.add_argument('--save', metavar='DIR',
                       help='결과 저장 디렉토리 / Result saving directory')

    args = parser.parse_args()

    try:
        # 이미지 로드
        print(f"이미지 로딩 중... / Loading image: {args.image_path}")
        image = load_image(args.image_path)
        print(f"이미지 크기 / Image size: {image.shape}")

        # 히스토그램 평활화 실행
        algorithm_name = {
            'he': 'Global HE',
            'ahe': 'Adaptive HE',
            'clahe': 'CLAHE'
        }[args.algorithm]

        print(f"{algorithm_name} 실행 중 ({args.method} 방법)... / Running {algorithm_name} ({args.method} method)...")

        # 알고리즘별 실행
        if args.algorithm == 'he':
            # 기본 히스토그램 평활화
            result, info = histogram_equalization_color(
                image,
                method=args.method,
                show_process=args.show_process
            )

        elif args.algorithm == 'ahe':
            # AHE는 CLAHE에서 clip_limit을 매우 높게 설정
            import cv2
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()

            # AHE (clip_limit을 매우 높게 설정하여 클리핑 비활성화)
            result, info = clahe_implementation(
                gray,
                clip_limit=999.0,  # 매우 높은 값으로 클리핑 비활성화
                tile_size=(args.tile_size, args.tile_size),
                show_process=args.show_process
            )

        elif args.algorithm == 'clahe':
            # CLAHE
            import cv2
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()

            result, info = clahe_implementation(
                gray,
                clip_limit=args.clip_limit,
                tile_size=(args.tile_size, args.tile_size),
                show_process=args.show_process
            )

        # 결과 정보 출력
        print("\n=== 처리 결과 정보 / Processing Result Information ===")
        for key, value in info.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for sub_key, sub_value in value.items():
                    print(f"  {sub_key}: {sub_value}")
            else:
                print(f"{key}: {value}")

        # 결과 저장
        if args.save:
            if not os.path.exists(args.save):
                os.makedirs(args.save)

            base_name = os.path.splitext(os.path.basename(args.image_path))[0]

            if args.algorithm == 'he':
                output_path = os.path.join(args.save, f"{base_name}_he_{args.method}.jpg")
            elif args.algorithm == 'ahe':
                output_path = os.path.join(args.save, f"{base_name}_ahe_tile{args.tile_size}.jpg")
            elif args.algorithm == 'clahe':
                output_path = os.path.join(args.save, f"{base_name}_clahe_clip{args.clip_limit}_tile{args.tile_size}.jpg")

            save_image(result, output_path)
            print(f"\n결과 저장됨 / Result saved: {output_path}")
        else:
            print("\n결과 저장하려면 --save 옵션을 사용하세요 / Use --save option to save results")

        print("히스토그램 평활화 완료! / Histogram equalization completed!")

    except Exception as e:
        print(f"오류 발생 / Error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()