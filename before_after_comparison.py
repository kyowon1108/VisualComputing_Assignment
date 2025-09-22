#!/usr/bin/env python3
"""
수정 전후 비교
Before/After comparison of post-processing improvements
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.otsu import local_otsu_block_based, apply_morphological_postprocessing
from src.utils import load_image

def compare_before_after(image_path):
    """
    수정 전후 비교
    """
    print("=== 후처리 설정 수정 전후 비교 ===")

    # 이미지 로드
    image = load_image(image_path)
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # 1. 후처리 없는 기본 결과
    binary_base, _ = local_otsu_block_based(
        gray,
        block_size=(32, 32),
        adaptive_params=False,
        apply_smoothing=False,
        apply_postprocessing=False,
        show_process=False
    )

    # 2. 이전 설정 (공격적 후처리)
    binary_old_settings = apply_morphological_postprocessing(
        binary_base,
        remove_small=True,
        min_size=50,  # 이전 설정
        apply_opening=True,
        apply_closing=True,
        kernel_size=3
    )

    # 3. 새로운 설정 (텍스트 친화적)
    binary_new_settings = apply_morphological_postprocessing(
        binary_base,
        remove_small=True,
        min_size=6,  # 새로운 설정 (480*640 // 50000 = 6)
        apply_opening=False,
        apply_closing=False,
        kernel_size=2
    )

    # 4. Enhanced 메서드 (자동 설정)
    binary_enhanced, info_enhanced = local_otsu_block_based(
        gray,
        adaptive_params=True,
        apply_smoothing=True,
        apply_postprocessing=True,
        show_process=False
    )

    # 결과 비교
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 원본
    axes[0, 0].imshow(gray, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # 후처리 없음
    axes[0, 1].imshow(binary_base, cmap='gray')
    black_ratio_base = (binary_base <= 127).sum() / binary_base.size * 100
    axes[0, 1].set_title(f'No Post-processing\n({black_ratio_base:.1f}% black)')
    axes[0, 1].axis('off')

    # 이전 설정 (공격적)
    axes[0, 2].imshow(binary_old_settings, cmap='gray')
    black_ratio_old = (binary_old_settings <= 127).sum() / binary_old_settings.size * 100
    axes[0, 2].set_title(f'Old Settings (Aggressive)\n({black_ratio_old:.1f}% black)')
    axes[0, 2].axis('off')

    # 새로운 설정 (보수적)
    axes[1, 0].imshow(binary_new_settings, cmap='gray')
    black_ratio_new = (binary_new_settings <= 127).sum() / binary_new_settings.size * 100
    axes[1, 0].set_title(f'New Settings (Conservative)\n({black_ratio_new:.1f}% black)')
    axes[1, 0].axis('off')

    # Enhanced 메서드
    axes[1, 1].imshow(binary_enhanced, cmap='gray')
    black_ratio_enhanced = (binary_enhanced <= 127).sum() / binary_enhanced.size * 100
    axes[1, 1].set_title(f'Enhanced Method\n({black_ratio_enhanced:.1f}% black)')
    axes[1, 1].axis('off')

    # 설정 비교 표
    settings_text = """설정 비교:

이전 설정 (문제 있음):
• min_size: 50px
• Opening: ON (텍스트 제거)
• Closing: ON (텍스트 뭉개짐)
• kernel_size: 3

새로운 설정 (개선됨):
• min_size: 6px (텍스트 보존)
• Opening: OFF
• Closing: OFF
• kernel_size: 2

결과:
• 텍스트 보존율 향상
• 블록 경계 스무딩 유지
• 노이즈 제거 최소화"""

    axes[1, 2].text(0.05, 0.95, settings_text, transform=axes[1, 2].transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    axes[1, 2].set_title('Settings Comparison')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.show()

    # 구성요소 분석
    analyze_components_comparison(binary_base, binary_old_settings, binary_new_settings, binary_enhanced)

def analyze_components_comparison(base, old, new, enhanced):
    """
    구성요소 분석 비교
    """
    from scipy.ndimage import label

    print("\n=== 연결된 구성요소 분석 ===")

    results = {
        'Base (No post-processing)': base,
        'Old Settings (Aggressive)': old,
        'New Settings (Conservative)': new,
        'Enhanced Method': enhanced
    }

    for name, result in results.items():
        labeled, num_components = label(result <= 127)

        # 구성요소 크기 계산
        sizes = []
        for i in range(1, num_components + 1):
            size = (labeled == i).sum()
            sizes.append(size)

        if sizes:
            sizes = np.array(sizes)
            print(f"\n{name}:")
            print(f"  구성요소 수: {num_components}")
            print(f"  평균 크기: {sizes.mean():.1f}px")
            print(f"  중간값: {np.median(sizes):.1f}px")
            print(f"  10px 이하: {(sizes <= 10).sum()} 개")
            print(f"  50px 이하: {(sizes <= 50).sum()} 개")
        else:
            print(f"\n{name}: 구성요소 없음")

if __name__ == "__main__":
    image_path = "images/otsu_shadow_doc_01.jpg"
    compare_before_after(image_path)