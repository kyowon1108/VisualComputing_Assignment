#!/usr/bin/env python3
"""
HE 알고리즘들 테스트 (HE, AHE, CLAHE)
Test HE algorithms (HE, AHE, CLAHE)
"""

import sys
import os
import numpy as np
import cv2

# 상위 디렉토리의 src 모듈 import를 위한 경로 설정
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.he import histogram_equalization_color, histogram_equalization_grayscale, clahe_implementation
from src.utils import load_image

def test_all_algorithms():
    """
    모든 HE 알고리즘 테스트
    """
    print("=== HE 알고리즘 테스트 ===")

    # 테스트 이미지 로드
    image_path = "images/he_dark_indoor.jpg"
    if not os.path.exists(image_path):
        print(f"이미지를 찾을 수 없습니다: {image_path}")
        return

    image = load_image(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    print(f"이미지 크기: {image.shape}")
    print(f"그레이스케일 크기: {gray.shape}")

    # 1. Global HE (YUV)
    print("\n1. Global HE (YUV) 테스트...")
    try:
        he_result, he_info = histogram_equalization_color(image, method='yuv', show_process=False)
        print(f"✅ Global HE 성공: {he_result.shape}")
    except Exception as e:
        print(f"❌ Global HE 실패: {e}")

    # 2. Global HE (Grayscale)
    print("\n2. Global HE (Grayscale) 테스트...")
    try:
        he_gray_result, he_gray_info = histogram_equalization_grayscale(gray, show_process=False)
        print(f"✅ Global HE (Gray) 성공: {he_gray_result.shape}")
    except Exception as e:
        print(f"❌ Global HE (Gray) 실패: {e}")

    # 3. CLAHE (다양한 설정)
    clahe_configs = [
        (2.0, (8, 8), "Standard"),
        (3.0, (8, 8), "Higher Clip"),
        (2.0, (16, 16), "Larger Tiles"),
        (999.0, (8, 8), "AHE-like")
    ]

    for clip_limit, tile_size, desc in clahe_configs:
        print(f"\n3. CLAHE 테스트 ({desc}: clip={clip_limit}, tile={tile_size})...")
        try:
            clahe_result, clahe_info = clahe_implementation(
                gray,
                clip_limit=clip_limit,
                tile_size=tile_size,
                show_process=False
            )
            print(f"✅ CLAHE ({desc}) 성공: {clahe_result.shape}")
            print(f"   타일 수: {clahe_info.get('num_tiles_x', 'N/A')} x {clahe_info.get('num_tiles_y', 'N/A')}")
        except Exception as e:
            print(f"❌ CLAHE ({desc}) 실패: {e}")

    # 4. 간단한 시각적 결과 비교
    print("\n=== 결과 비교 ===")
    try:
        # 픽셀값 통계 비교
        print(f"원본 - 평균: {np.mean(gray):.2f}, 표준편차: {np.std(gray):.2f}")

        if 'he_gray_result' in locals():
            print(f"Global HE - 평균: {np.mean(he_gray_result):.2f}, 표준편차: {np.std(he_gray_result):.2f}")

        if 'clahe_result' in locals():
            print(f"CLAHE - 평균: {np.mean(clahe_result):.2f}, 표준편차: {np.std(clahe_result):.2f}")

    except Exception as e:
        print(f"통계 계산 오류: {e}")

if __name__ == "__main__":
    test_all_algorithms()