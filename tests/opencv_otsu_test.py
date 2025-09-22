#!/usr/bin/env python3
"""
OpenCV Otsu 비교 테스트
OpenCV Otsu Comparison Test

우리 구현과 OpenCV 구현을 비교하여 문제점 파악
Compare our implementation with OpenCV to identify issues
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 상위 디렉토리를 path에 추가하여 src 모듈 import 가능
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.otsu import global_otsu_thresholding, local_otsu_block_based
from src.utils import load_image

def test_opencv_vs_custom_otsu(image_path):
    """
    OpenCV와 우리 구현 비교
    Compare OpenCV vs our implementation
    """
    print(f"테스트 이미지: {image_path}")

    # 이미지 로드
    image = load_image(image_path)
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    print(f"이미지 크기: {gray.shape}")
    print(f"픽셀 값 범위: {gray.min()} ~ {gray.max()}")

    # 1. OpenCV Global Otsu
    ret_opencv, binary_opencv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f"OpenCV Global Otsu 임계값: {ret_opencv}")

    # 2. 우리 구현 Global Otsu
    binary_custom, info_custom = global_otsu_thresholding(gray, show_process=False)
    print(f"Custom Global Otsu 임계값: {info_custom['threshold']}")

    # 3. OpenCV Adaptive Threshold (참고용)
    adaptive_mean = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    adaptive_gaussian = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # 4. 우리 구현 Local Otsu (기본 설정 - 후처리 없음)
    binary_local_simple, info_local_simple = local_otsu_block_based(
        gray,
        block_size=(32, 32),
        adaptive_params=False,
        apply_smoothing=False,
        apply_postprocessing=False,
        show_process=False
    )

    # 5. 우리 구현 Local Otsu (Enhanced - 후처리 있음)
    binary_local_enhanced, info_local_enhanced = local_otsu_block_based(
        gray,
        adaptive_params=True,
        apply_smoothing=True,
        apply_postprocessing=True,
        show_process=False
    )

    # 결과 비교 시각화
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))

    # 원본
    axes[0, 0].imshow(gray, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # OpenCV Global Otsu
    axes[0, 1].imshow(binary_opencv, cmap='gray')
    axes[0, 1].set_title(f'OpenCV Global Otsu\n(threshold: {ret_opencv:.1f})')
    axes[0, 1].axis('off')

    # Custom Global Otsu
    axes[0, 2].imshow(binary_custom, cmap='gray')
    axes[0, 2].set_title(f'Custom Global Otsu\n(threshold: {info_custom["threshold"]})')
    axes[0, 2].axis('off')

    # OpenCV Adaptive Mean
    axes[1, 0].imshow(adaptive_mean, cmap='gray')
    axes[1, 0].set_title('OpenCV Adaptive Mean')
    axes[1, 0].axis('off')

    # OpenCV Adaptive Gaussian
    axes[1, 1].imshow(adaptive_gaussian, cmap='gray')
    axes[1, 1].set_title('OpenCV Adaptive Gaussian')
    axes[1, 1].axis('off')

    # Custom Local Otsu (Simple)
    axes[1, 2].imshow(binary_local_simple, cmap='gray')
    axes[1, 2].set_title('Custom Local Otsu (Simple)\n(No post-processing)')
    axes[1, 2].axis('off')

    # Custom Local Otsu (Enhanced)
    axes[2, 0].imshow(binary_local_enhanced, cmap='gray')
    axes[2, 0].set_title('Custom Local Otsu (Enhanced)\n(With post-processing)')
    axes[2, 0].axis('off')

    # 히스토그램
    axes[2, 1].hist(gray.flatten(), bins=256, alpha=0.7, color='blue')
    axes[2, 1].axvline(x=ret_opencv, color='red', linestyle='--', linewidth=2, label=f'OpenCV: {ret_opencv:.1f}')
    axes[2, 1].axvline(x=info_custom['threshold'], color='green', linestyle='--', linewidth=2, label=f'Custom: {info_custom["threshold"]}')
    axes[2, 1].set_title('Histogram with Thresholds')
    axes[2, 1].set_xlabel('Pixel Intensity')
    axes[2, 1].set_ylabel('Frequency')
    axes[2, 1].legend()

    # 통계 정보
    stats_text = f"""이미지 통계:
크기: {gray.shape}
평균: {gray.mean():.1f}
표준편차: {gray.std():.1f}
최소/최대: {gray.min()}/{gray.max()}

임계값 비교:
OpenCV: {ret_opencv:.1f}
Custom: {info_custom['threshold']}
차이: {abs(ret_opencv - info_custom['threshold']):.1f}

텍스트 픽셀 (어두운 부분):
비율: {(gray < ret_opencv).sum() / gray.size * 100:.1f}%
"""

    axes[2, 2].text(0.05, 0.95, stats_text, transform=axes[2, 2].transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    axes[2, 2].set_title('Statistics')
    axes[2, 2].axis('off')

    plt.tight_layout()
    plt.show()

    # 텍스트 보존도 분석
    analyze_text_preservation(gray, binary_opencv, binary_custom, binary_local_simple, binary_local_enhanced)

def analyze_text_preservation(original, opencv_result, custom_global, custom_local_simple, custom_local_enhanced):
    """
    텍스트 보존도 분석
    Analyze text preservation
    """
    print("\n=== 텍스트 보존도 분석 ===")

    # 어두운 픽셀 (텍스트) 비율 계산
    total_pixels = original.size

    # 원본에서 어두운 픽셀 (텍스트 영역 추정)
    dark_threshold = np.mean(original) - np.std(original)  # 평균 - 표준편차
    original_text_pixels = (original < dark_threshold).sum()

    print(f"원본 추정 텍스트 픽셀: {original_text_pixels} ({original_text_pixels/total_pixels*100:.1f}%)")

    # 각 결과에서 흰색 픽셀 (배경) 비율
    results = {
        'OpenCV Global': opencv_result,
        'Custom Global': custom_global,
        'Custom Local Simple': custom_local_simple,
        'Custom Local Enhanced': custom_local_enhanced
    }

    for name, result in results.items():
        white_pixels = (result > 127).sum()
        black_pixels = (result <= 127).sum()
        print(f"{name}:")
        print(f"  검은 픽셀 (텍스트): {black_pixels} ({black_pixels/total_pixels*100:.1f}%)")
        print(f"  흰 픽셀 (배경): {white_pixels} ({white_pixels/total_pixels*100:.1f}%)")

        # 텍스트 손실률 추정
        if original_text_pixels > 0:
            text_preservation = min(100, black_pixels / original_text_pixels * 100)
            print(f"  텍스트 보존률 추정: {text_preservation:.1f}%")
        print()

def test_different_postprocessing_settings(image_path):
    """
    다양한 후처리 설정 테스트
    Test different post-processing settings
    """
    print("=== 후처리 설정 테스트 ===")

    image = load_image(image_path)
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # 다양한 후처리 설정
    postprocess_configs = [
        {
            'name': 'No Post-processing',
            'params': {'apply_postprocessing': False}
        },
        {
            'name': 'Opening Only',
            'params': {
                'apply_postprocessing': True,
                'postprocess_params': {
                    'remove_small': False,
                    'apply_opening': True,
                    'apply_closing': False,
                    'kernel_size': 3
                }
            }
        },
        {
            'name': 'Closing Only',
            'params': {
                'apply_postprocessing': True,
                'postprocess_params': {
                    'remove_small': False,
                    'apply_opening': False,
                    'apply_closing': True,
                    'kernel_size': 3
                }
            }
        },
        {
            'name': 'Remove Small Only',
            'params': {
                'apply_postprocessing': True,
                'postprocess_params': {
                    'remove_small': True,
                    'min_size': 50,
                    'apply_opening': False,
                    'apply_closing': False
                }
            }
        },
        {
            'name': 'All (Current)',
            'params': {
                'apply_postprocessing': True,
                'postprocess_params': {
                    'remove_small': True,
                    'min_size': 200,  # 더 큰 값으로 테스트
                    'apply_opening': True,
                    'apply_closing': True,
                    'kernel_size': 3
                }
            }
        }
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # 원본
    axes[0].imshow(gray, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')

    for i, config in enumerate(postprocess_configs):
        result, info = local_otsu_block_based(
            gray,
            block_size=(32, 32),
            adaptive_params=False,
            apply_smoothing=False,
            show_process=False,
            **config['params']
        )

        axes[i + 1].imshow(result, cmap='gray')
        axes[i + 1].set_title(config['name'])
        axes[i + 1].axis('off')

        # 텍스트 픽셀 비율 출력
        black_pixels = (result <= 127).sum()
        total_pixels = result.size
        print(f"{config['name']}: 검은 픽셀 {black_pixels/total_pixels*100:.1f}%")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 테스트 이미지 경로
    image_path = "../images/otsu_shadow_doc_01.jpg"

    if not os.path.exists(image_path):
        print(f"이미지 파일을 찾을 수 없습니다: {image_path}")
        print("다음 중 하나를 사용하세요:")
        for img in os.listdir("../images/"):
            if img.endswith(('.jpg', '.png')):
                print(f"  ../images/{img}")
        sys.exit(1)

    # 메인 비교 테스트
    test_opencv_vs_custom_otsu(image_path)

    # 후처리 설정 테스트
    test_different_postprocessing_settings(image_path)