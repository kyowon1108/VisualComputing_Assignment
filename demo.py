#!/usr/bin/env python3
"""
비쥬얼컴퓨팅 과제1 데모 스크립트
Visual Computing Assignment 1 Demo Script

이 스크립트는 모든 기능을 한번에 테스트할 수 있는 데모를 제공합니다.
This script provides a demo to test all functions at once.
"""

import sys
import os
import numpy as np

# 현재 스크립트의 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from src.utils import load_image, save_image, create_test_image
from src.he import histogram_equalization_color, visualize_color_he_process
from src.otsu import compare_otsu_methods

def run_demo():
    """데모 실행"""
    print("=" * 80)
    print("비쥬얼컴퓨팅 과제1 - 종합 데모")
    print("Visual Computing Assignment 1 - Comprehensive Demo")
    print("=" * 80)

    # 결과 디렉토리 생성
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"결과 디렉토리 생성됨 / Results directory created: {results_dir}")

    # 테스트 이미지 생성 (실제 이미지가 없는 경우)
    test_images = []
    image_names = []

    # images 디렉토리에서 이미지 찾기
    images_dir = "images"
    if os.path.exists(images_dir):
        for filename in os.listdir(images_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                try:
                    img_path = os.path.join(images_dir, filename)
                    img = load_image(img_path)
                    test_images.append(img)
                    image_names.append(os.path.splitext(filename)[0])
                    print(f"이미지 로드됨 / Image loaded: {filename}")
                except Exception as e:
                    print(f"이미지 로드 실패 / Failed to load image {filename}: {e}")

    # 테스트 이미지가 없으면 생성
    if not test_images:
        print("실제 이미지를 찾을 수 없어 테스트 이미지를 생성합니다...")
        print("No real images found, generating test images...")

        # 다양한 테스트 패턴 생성
        test_patterns = [
            ("gradient", "그라디언트 / Gradient"),
            ("checkerboard", "체스판 / Checkerboard"),
            ("noise", "노이즈 / Noise")
        ]

        for pattern, description in test_patterns:
            print(f"테스트 이미지 생성 중: {description}")
            test_img = create_test_image(pattern=pattern, size=(512, 512))
            test_images.append(test_img)
            image_names.append(f"test_{pattern}")

    # 각 이미지에 대해 처리 실행
    for idx, (image, name) in enumerate(zip(test_images, image_names)):
        print(f"\n{'='*60}")
        print(f"이미지 {idx+1}/{len(test_images)} 처리 중: {name}")
        print(f"Processing image {idx+1}/{len(test_images)}: {name}")
        print(f"이미지 크기 / Image size: {image.shape}")
        print(f"{'='*60}")

        try:
            # 1. 히스토그램 평활화 테스트
            print("\n1. 히스토그램 평활화 테스트 / Histogram Equalization Test")
            print("-" * 50)

            he_methods = ['yuv', 'rgb']
            for method in he_methods:
                print(f"  {method.upper()} 방법으로 처리 중...")
                try:
                    result, info = histogram_equalization_color(
                        image, method=method, show_process=False
                    )

                    # 결과 저장
                    save_path = os.path.join(results_dir, f"{name}_he_{method}.jpg")
                    save_image(result, save_path)
                    print(f"    저장됨 / Saved: {save_path}")

                    # 간단한 통계 출력
                    if 'processing_time' in info:
                        print(f"    처리 시간 / Processing time: {info['processing_time']:.3f}s")

                except Exception as e:
                    print(f"    오류 / Error: {e}")

            # 2. Local Otsu Thresholding 테스트
            print("\n2. Local Otsu Thresholding 테스트 / Local Otsu Thresholding Test")
            print("-" * 50)

            try:
                # 그레이스케일로 변환
                if len(image.shape) == 3:
                    import cv2
                    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray_image = image

                print("  모든 방법 비교 실행 중... / Running comparison of all methods...")
                comparison_results = compare_otsu_methods(
                    gray_image, show_comparison=False
                )

                # 각 방법별 결과 저장
                for method_name, result_data in comparison_results.items():
                    if 'result' in result_data:
                        save_path = os.path.join(results_dir, f"{name}_otsu_{method_name}.jpg")
                        save_image(result_data['result'], save_path)
                        print(f"    {method_name} 저장됨 / {method_name} saved: {save_path}")

                        # 통계 출력
                        if 'info' in result_data:
                            info = result_data['info']
                            if isinstance(info, dict) and 'processing_time' in info:
                                print(f"      처리 시간 / Processing time: {info['processing_time']:.3f}s")

            except Exception as e:
                print(f"    Otsu 처리 오류 / Otsu processing error: {e}")

        except Exception as e:
            print(f"이미지 처리 실패 / Image processing failed: {e}")

    print(f"\n{'='*80}")
    print("데모 완료! / Demo completed!")
    print(f"모든 결과는 '{results_dir}' 디렉토리에 저장되었습니다.")
    print(f"All results are saved in '{results_dir}' directory.")
    print(f"{'='*80}")

def main():
    """메인 함수"""
    try:
        run_demo()
    except KeyboardInterrupt:
        print("\n사용자에 의해 중단됨 / Interrupted by user")
    except Exception as e:
        print(f"데모 실행 중 오류 발생 / Error during demo execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()