#!/usr/bin/env python3
"""
최종 해결책 테스트
Test final solution for block boundary artifacts
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.otsu import local_otsu_block_based, local_otsu_improved_boundary
from src.utils import load_image

def test_final_solution(image_path):
    """
    최종 해결책 테스트
    """
    print("=== 최종 해결책 테스트 ===")

    # 이미지 로드
    image = load_image(image_path)
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    print(f"이미지 크기: {gray.shape}")

    # 1. 기존 블록 기반 (문제가 있는 방법)
    print("기존 블록 기반 방법 실행...")
    result_original, info_original = local_otsu_block_based(
        gray,
        block_size=(32, 32),
        adaptive_params=False,
        apply_smoothing=True,
        apply_postprocessing=True,
        show_process=False
    )

    # 2. 개선된 방법
    print("개선된 방법 실행...")
    result_improved, info_improved = local_otsu_improved_boundary(
        gray,
        block_size=(32, 32),
        overlap_ratio=0.5,
        blend_method='weighted_average',
        show_process=False
    )

    # 결과 시각화
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 원본
    axes[0, 0].imshow(gray, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # 기존 방법 결과
    axes[0, 1].imshow(result_original, cmap='gray')
    black_ratio_orig = (result_original <= 127).sum() / result_original.size * 100
    axes[0, 1].set_title(f'Original Method\n({black_ratio_orig:.1f}% black)')
    axes[0, 1].axis('off')

    # 개선된 방법 결과
    axes[0, 2].imshow(result_improved, cmap='gray')
    black_ratio_imp = (result_improved <= 127).sum() / result_improved.size * 100
    axes[0, 2].set_title(f'Improved Method\n({black_ratio_imp:.1f}% black)')
    axes[0, 2].axis('off')

    # 임계값 맵들
    im1 = axes[1, 0].imshow(info_original['threshold_map'], cmap='jet', vmin=50, vmax=200)
    axes[1, 0].set_title('Original Threshold Map')
    axes[1, 0].axis('off')
    plt.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im2 = axes[1, 1].imshow(info_improved['threshold_map'], cmap='jet', vmin=50, vmax=200)
    axes[1, 1].set_title('Improved Threshold Map')
    axes[1, 1].axis('off')
    plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)

    # 차이 시각화
    diff_result = np.abs(result_original.astype(float) - result_improved.astype(float))
    im3 = axes[1, 2].imshow(diff_result, cmap='hot')
    axes[1, 2].set_title('Result Difference')
    axes[1, 2].axis('off')
    plt.colorbar(im3, ax=axes[1, 2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

    # 블록 경계 아티팩트 정량화
    print("\n=== 아티팩트 정량화 ===")

    # 블록 경계에서의 불연속성 측정
    block_size = 32
    height, width = gray.shape

    def measure_boundary_discontinuity(threshold_map):
        """블록 경계에서의 불연속성 측정"""
        vertical_diffs = []
        horizontal_diffs = []

        # 수직 경계
        for x in range(block_size, width, block_size):
            if x < width - 1:
                diff = np.abs(threshold_map[:, x] - threshold_map[:, x-1])
                vertical_diffs.extend(diff)

        # 수평 경계
        for y in range(block_size, height, block_size):
            if y < height - 1:
                diff = np.abs(threshold_map[y, :] - threshold_map[y-1, :])
                horizontal_diffs.extend(diff)

        return np.mean(vertical_diffs + horizontal_diffs)

    orig_discontinuity = measure_boundary_discontinuity(info_original['threshold_map'])
    imp_discontinuity = measure_boundary_discontinuity(info_improved['threshold_map'])

    print(f"기존 방법 블록 경계 불연속성: {orig_discontinuity:.2f}")
    print(f"개선된 방법 블록 경계 불연속성: {imp_discontinuity:.2f}")
    print(f"개선률: {(1 - imp_discontinuity/orig_discontinuity)*100:.1f}%")

    # 임계값 맵 부드러움 측정
    orig_smoothness = np.std(info_original['threshold_map'])
    imp_smoothness = np.std(info_improved['threshold_map'])

    print(f"\n임계값 맵 표준편차 (낮을수록 부드러움):")
    print(f"기존 방법: {orig_smoothness:.2f}")
    print(f"개선된 방법: {imp_smoothness:.2f}")
    print(f"부드러움 개선률: {(1 - imp_smoothness/orig_smoothness)*100:.1f}%")

    return result_original, result_improved, info_original, info_improved

def crop_analysis(image_path):
    """
    특정 영역 확대 분석
    """
    print("\n=== 문제 영역 확대 분석 ===")

    image = load_image(image_path)
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # 각 방법으로 처리
    result_orig, info_orig = local_otsu_block_based(
        gray, block_size=(32, 32), show_process=False
    )

    result_imp, info_imp = local_otsu_improved_boundary(
        gray, block_size=(32, 32), show_process=False
    )

    # 중앙 영역 크롭
    center_y, center_x = gray.shape[0] // 2, gray.shape[1] // 2
    crop_size = 128
    start_y = center_y - crop_size // 2
    end_y = start_y + crop_size
    start_x = center_x - crop_size // 2
    end_x = start_x + crop_size

    # 크롭
    gray_crop = gray[start_y:end_y, start_x:end_x]
    orig_crop = result_orig[start_y:end_y, start_x:end_x]
    imp_crop = result_imp[start_y:end_y, start_x:end_x]
    thresh_orig_crop = info_orig['threshold_map'][start_y:end_y, start_x:end_x]
    thresh_imp_crop = info_imp['threshold_map'][start_y:end_y, start_x:end_x]

    # 시각화
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(gray_crop, cmap='gray')
    axes[0, 0].set_title('Original (Cropped)')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(orig_crop, cmap='gray')
    axes[0, 1].set_title('Block-based Result')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(imp_crop, cmap='gray')
    axes[0, 2].set_title('Improved Result')
    axes[0, 2].axis('off')

    im1 = axes[1, 0].imshow(thresh_orig_crop, cmap='jet')
    axes[1, 0].set_title('Block Threshold Map')
    axes[1, 0].axis('off')
    plt.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im2 = axes[1, 1].imshow(thresh_imp_crop, cmap='jet')
    axes[1, 1].set_title('Improved Threshold Map')
    axes[1, 1].axis('off')
    plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)

    # 차이 맵
    diff_crop = np.abs(thresh_orig_crop - thresh_imp_crop)
    im3 = axes[1, 2].imshow(diff_crop, cmap='hot')
    axes[1, 2].set_title('Threshold Difference')
    axes[1, 2].axis('off')
    plt.colorbar(im3, ax=axes[1, 2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    image_path = "images/otsu_shadow_doc_01.jpg"

    # 최종 해결책 테스트
    test_final_solution(image_path)

    # 확대 분석
    crop_analysis(image_path)