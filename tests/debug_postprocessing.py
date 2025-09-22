#!/usr/bin/env python3
"""
후처리 단계별 디버깅
Step-by-step post-processing debugging
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.otsu import local_otsu_block_based, apply_morphological_postprocessing
from src.utils import load_image

def debug_postprocessing_steps(image_path):
    """
    후처리 각 단계를 시각화하여 어느 단계에서 텍스트가 사라지는지 확인
    """
    print("=== 후처리 단계별 디버깅 ===")

    # 이미지 로드
    image = load_image(image_path)
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # 1. 후처리 없는 기본 결과
    binary_base, info_base = local_otsu_block_based(
        gray,
        block_size=(32, 32),
        adaptive_params=False,
        apply_smoothing=False,
        apply_postprocessing=False,
        show_process=False
    )

    # 2. 각 후처리 단계별로 적용
    # Opening만 적용
    binary_opening = apply_morphological_postprocessing(
        binary_base,
        remove_small=False,
        apply_opening=True,
        apply_closing=False,
        kernel_size=3
    )

    # Closing만 적용
    binary_closing = apply_morphological_postprocessing(
        binary_base,
        remove_small=False,
        apply_opening=False,
        apply_closing=True,
        kernel_size=3
    )

    # Small component removal만 적용 (다양한 크기로)
    binary_small_50 = apply_morphological_postprocessing(
        binary_base,
        remove_small=True,
        min_size=50,
        apply_opening=False,
        apply_closing=False
    )

    binary_small_200 = apply_morphological_postprocessing(
        binary_base,
        remove_small=True,
        min_size=200,
        apply_opening=False,
        apply_closing=False
    )

    binary_small_500 = apply_morphological_postprocessing(
        binary_base,
        remove_small=True,
        min_size=500,
        apply_opening=False,
        apply_closing=False
    )

    # Opening + Closing
    binary_open_close = apply_morphological_postprocessing(
        binary_base,
        remove_small=False,
        apply_opening=True,
        apply_closing=True,
        kernel_size=3
    )

    # 전체 적용 (현재 기본 설정)
    binary_all = apply_morphological_postprocessing(
        binary_base,
        remove_small=True,
        min_size=max(50, (gray.shape[0] * gray.shape[1]) // 10000),
        apply_opening=True,
        apply_closing=True,
        kernel_size=3
    )

    # 시각화
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))

    images = [
        (gray, "Original"),
        (binary_base, "Base (No Post-processing)"),
        (binary_opening, "Opening Only"),
        (binary_closing, "Closing Only"),
        (binary_small_50, "Remove Small (50px)"),
        (binary_small_200, "Remove Small (200px)"),
        (binary_small_500, "Remove Small (500px)"),
        (binary_open_close, "Opening + Closing"),
        (binary_all, "All Post-processing")
    ]

    for idx, (img, title) in enumerate(images):
        row, col = idx // 3, idx % 3
        if idx == 0:
            axes[row, col].imshow(img, cmap='gray')
        else:
            axes[row, col].imshow(img, cmap='gray')
            # 검은 픽셀 비율 계산
            black_ratio = (img <= 127).sum() / img.size * 100
            title += f"\n({black_ratio:.1f}% black)"

        axes[row, col].set_title(title)
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.show()

    # 상세 분석
    analyze_text_components(binary_base, binary_all, gray.shape)

def analyze_text_components(binary_base, binary_processed, image_shape):
    """
    텍스트 구성요소 분석
    """
    print("\n=== 텍스트 구성요소 분석 ===")

    # Connected components 분석
    from scipy.ndimage import label

    # 기본 결과의 connected components
    labeled_base, num_base = label(binary_base <= 127)
    print(f"기본 결과 - 연결된 구성요소 수: {num_base}")

    # 후처리 결과의 connected components
    labeled_processed, num_processed = label(binary_processed <= 127)
    print(f"후처리 결과 - 연결된 구성요소 수: {num_processed}")

    # 구성요소 크기 분석
    base_sizes = []
    processed_sizes = []

    for i in range(1, num_base + 1):
        size = (labeled_base == i).sum()
        base_sizes.append(size)

    for i in range(1, num_processed + 1):
        size = (labeled_processed == i).sum()
        processed_sizes.append(size)

    base_sizes = np.array(base_sizes)
    processed_sizes = np.array(processed_sizes)

    print(f"\n기본 결과 구성요소 크기:")
    if len(base_sizes) > 0:
        print(f"  평균: {base_sizes.mean():.1f} 픽셀")
        print(f"  중간값: {np.median(base_sizes):.1f} 픽셀")
        print(f"  최대: {base_sizes.max()} 픽셀")
        print(f"  최소: {base_sizes.min()} 픽셀")
        print(f"  50px 이하: {(base_sizes <= 50).sum()} 개")
        print(f"  200px 이하: {(base_sizes <= 200).sum()} 개")

    print(f"\n후처리 결과 구성요소 크기:")
    if len(processed_sizes) > 0:
        print(f"  평균: {processed_sizes.mean():.1f} 픽셀")
        print(f"  중간값: {np.median(processed_sizes):.1f} 픽셀")
        print(f"  최대: {processed_sizes.max()} 픽셀")
        print(f"  최소: {processed_sizes.min()} 픽셀")

    # 손실된 구성요소 분석
    total_pixels_base = base_sizes.sum() if len(base_sizes) > 0 else 0
    total_pixels_processed = processed_sizes.sum() if len(processed_sizes) > 0 else 0

    print(f"\n픽셀 손실:")
    print(f"  기본 결과 총 검은 픽셀: {total_pixels_base}")
    print(f"  후처리 결과 총 검은 픽셀: {total_pixels_processed}")
    print(f"  손실된 픽셀: {total_pixels_base - total_pixels_processed}")
    print(f"  손실률: {(total_pixels_base - total_pixels_processed) / total_pixels_base * 100:.1f}%")

    # 현재 min_size 설정값 확인
    current_min_size = max(50, (image_shape[0] * image_shape[1]) // 10000)
    print(f"\n현재 min_size 설정: {current_min_size}")
    print(f"이 크기보다 작은 구성요소가 제거됨")

def create_better_postprocessing_config(image_path):
    """
    더 나은 후처리 설정 제안
    """
    print("\n=== 개선된 후처리 설정 테스트 ===")

    image = load_image(image_path)
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # 개선된 설정들
    configs = [
        {
            'name': 'Conservative (보수적)',
            'params': {
                'remove_small': True,
                'min_size': 10,  # 매우 작은 노이즈만 제거
                'apply_opening': True,
                'apply_closing': False,  # Closing은 텍스트를 뭉개버릴 수 있음
                'kernel_size': 2  # 더 작은 커널
            }
        },
        {
            'name': 'Text-friendly (텍스트 친화적)',
            'params': {
                'remove_small': True,
                'min_size': 20,
                'apply_opening': False,  # Opening도 텍스트 세부사항을 제거할 수 있음
                'apply_closing': False,
                'kernel_size': 3
            }
        },
        {
            'name': 'Minimal (최소한)',
            'params': {
                'remove_small': True,
                'min_size': 5,  # 아주 작은 노이즈만
                'apply_opening': False,
                'apply_closing': False,
                'kernel_size': 3
            }
        }
    ]

    # 기본 결과 (후처리 없음)
    binary_base, _ = local_otsu_block_based(
        gray,
        block_size=(32, 32),
        adaptive_params=False,
        apply_smoothing=False,
        apply_postprocessing=False,
        show_process=False
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # 기본 결과
    axes[0].imshow(binary_base, cmap='gray')
    axes[0].set_title('No Post-processing')
    axes[0].axis('off')

    # 각 개선된 설정 테스트
    for i, config in enumerate(configs):
        result = apply_morphological_postprocessing(binary_base, **config['params'])

        axes[i + 1].imshow(result, cmap='gray')
        black_ratio = (result <= 127).sum() / result.size * 100
        axes[i + 1].set_title(f"{config['name']}\n({black_ratio:.1f}% black)")
        axes[i + 1].axis('off')

    plt.tight_layout()
    plt.show()

    return configs

if __name__ == "__main__":
    image_path = "../images/otsu_shadow_doc_01.jpg"

    if not os.path.exists(image_path):
        print(f"이미지 파일을 찾을 수 없습니다: {image_path}")
        sys.exit(1)

    # 단계별 디버깅
    debug_postprocessing_steps(image_path)

    # 개선된 설정 테스트
    better_configs = create_better_postprocessing_config(image_path)

    print("\n=== 권장사항 ===")
    print("1. min_size를 현재 설정보다 훨씬 작게 (5-20 픽셀)")
    print("2. Opening/Closing 연산을 선택적으로 사용")
    print("3. 텍스트 문서의 경우 후처리를 최소화")
    print("4. 또는 후처리를 완전히 비활성화하고 다른 방법 고려")