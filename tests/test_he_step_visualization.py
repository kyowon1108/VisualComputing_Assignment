#!/usr/bin/env python3
"""
히스토그램 평활화 4단계 과정 시각화 테스트
Test visualization of 4-step histogram equalization process

이 테스트는 다음 4단계를 시각화합니다:
1. 원본 RGB 이미지
2. YUV 변환 후 Y 채널 (휘도)
3. Y 채널에 히스토그램 평활화 적용
4. 최종 RGB 결과 이미지

This test visualizes the following 4 steps:
1. Original RGB image
2. Y channel after YUV conversion (Luminance)
3. Histogram equalization applied to Y channel
4. Final RGB result image
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

# 현재 스크립트의 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from src.utils import load_image, rgb_to_yuv, yuv_to_rgb, compute_histogram
from src.he import calculate_cdf, histogram_equalization_grayscale


def test_he_step_visualization(image_path: str, save_figure: bool = True):
    """
    히스토그램 평활화의 4단계 과정을 시각화합니다.
    Visualize the 4-step process of histogram equalization.

    Args:
        image_path (str): 입력 이미지 경로 / Input image path
        save_figure (bool): figure를 이미지로 저장할지 여부 / Whether to save figure as image
    """
    print(f"이미지 로딩 중: {image_path}")

    # Step 1: 원본 RGB 이미지 로드
    original_rgb = load_image(image_path)
    if len(original_rgb.shape) != 3:
        raise ValueError("컬러 이미지가 필요합니다 / Color image required")

    print(f"원본 이미지 크기: {original_rgb.shape}")

    # Step 2: RGB -> YUV 변환 및 Y 채널 분리
    print("RGB -> YUV 변환 중...")
    yuv_image = rgb_to_yuv(original_rgb)
    y_channel = yuv_image[:, :, 0]  # Y (휘도) 채널
    u_channel = yuv_image[:, :, 1]  # U 채널
    v_channel = yuv_image[:, :, 2]  # V 채널

    print(f"Y 채널 범위: {y_channel.min():.2f} ~ {y_channel.max():.2f}")

    # Step 3: Y 채널에 히스토그램 평활화 적용
    print("Y 채널에 히스토그램 평활화 적용 중...")
    y_equalized, he_info = histogram_equalization_grayscale(y_channel, show_process=False)

    # CDF 정보가 없으면 직접 계산
    if 'original_cdf' not in he_info:
        original_hist, _ = compute_histogram(y_channel)
        equalized_hist, _ = compute_histogram(y_equalized)
        original_cdf = calculate_cdf(original_hist)
        equalized_cdf = calculate_cdf(equalized_hist)
        he_info['original_cdf'] = original_cdf
        he_info['equalized_cdf'] = equalized_cdf

    # Step 4: YUV -> RGB 변환 (평활화된 Y 채널 사용)
    print("YUV -> RGB 변환 중...")
    yuv_equalized = np.stack([y_equalized, u_channel, v_channel], axis=2)
    final_rgb = yuv_to_rgb(yuv_equalized)

    # 4단계 시각화
    saved_path = visualize_4_steps(original_rgb, y_channel, y_equalized, final_rgb, he_info,
                                   image_path, save_figure)

    return {
        'original_rgb': original_rgb,
        'y_channel': y_channel,
        'y_equalized': y_equalized,
        'final_rgb': final_rgb,
        'he_info': he_info,
        'saved_figure_path': saved_path
    }


def visualize_4_steps(original_rgb, y_channel, y_equalized, final_rgb, he_info,
                      image_path, save_figure=True):
    """
    4단계 과정을 시각화합니다.
    Visualize the 4-step process.

    Args:
        save_figure (bool): figure를 저장할지 여부
        image_path (str): 원본 이미지 경로 (저장 파일명 생성용)

    Returns:
        str: 저장된 figure 파일 경로 (저장하지 않으면 None)
    """
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle('히스토그램 평활화 4단계 과정\nHistogram Equalization 4-Step Process',
                 fontsize=16, fontweight='bold')

    # Step 1: 원본 RGB 이미지
    axes[0, 0].imshow(original_rgb)
    axes[0, 0].set_title('Step 1: 원본 RGB 이미지\nOriginal RGB Image')
    axes[0, 0].axis('off')

    # Step 2: Y 채널 (그레이스케일)
    axes[0, 1].imshow(y_channel, cmap='gray')
    axes[0, 1].set_title('Step 2: Y 채널 (휘도)\nY Channel (Luminance)')
    axes[0, 1].axis('off')

    # Step 3: 평활화된 Y 채널
    axes[0, 2].imshow(y_equalized, cmap='gray')
    axes[0, 2].set_title('Step 3: 평활화된 Y 채널\nEqualized Y Channel')
    axes[0, 2].axis('off')

    # Step 4: 최종 RGB 결과
    axes[0, 3].imshow(final_rgb)
    axes[0, 3].set_title('Step 4: 최종 RGB 결과\nFinal RGB Result')
    axes[0, 3].axis('off')

    # 히스토그램 비교 (Y 채널)
    original_hist, _ = compute_histogram(y_channel)
    equalized_hist, _ = compute_histogram(y_equalized)

    # 원본 Y 채널 히스토그램
    axes[1, 0].bar(range(256), original_hist, alpha=0.7, color='blue', width=1.0)
    axes[1, 0].set_title('원본 Y 채널 히스토그램\nOriginal Y Channel Histogram')
    axes[1, 0].set_xlabel('픽셀 강도 / Pixel Intensity')
    axes[1, 0].set_ylabel('빈도수 / Frequency')
    axes[1, 0].grid(True, alpha=0.3)

    # 평활화된 Y 채널 히스토그램
    axes[1, 1].bar(range(256), equalized_hist, alpha=0.7, color='green', width=1.0)
    axes[1, 1].set_title('평활화된 Y 채널 히스토그램\nEqualized Y Channel Histogram')
    axes[1, 1].set_xlabel('픽셀 강도 / Pixel Intensity')
    axes[1, 1].set_ylabel('빈도수 / Frequency')
    axes[1, 1].grid(True, alpha=0.3)

    # 히스토그램 비교
    axes[1, 2].bar(range(256), original_hist, alpha=0.5, color='blue', label='원본 / Original', width=1.0)
    axes[1, 2].bar(range(256), equalized_hist, alpha=0.5, color='green', label='평활화 / Equalized', width=1.0)
    axes[1, 2].set_title('히스토그램 비교\nHistogram Comparison')
    axes[1, 2].set_xlabel('픽셀 강도 / Pixel Intensity')
    axes[1, 2].set_ylabel('빈도수 / Frequency')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    # 차이 이미지 (원본 vs 결과)
    # RGB를 그레이스케일로 변환하여 차이 계산
    original_gray = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)
    final_gray = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2GRAY)
    diff_image = np.abs(final_gray.astype(np.float32) - original_gray.astype(np.float32))

    im = axes[1, 3].imshow(diff_image, cmap='hot')
    axes[1, 3].set_title('차이 이미지\nDifference Image')
    axes[1, 3].axis('off')
    plt.colorbar(im, ax=axes[1, 3], fraction=0.046, pad=0.04)

    # CDF 비교
    original_cdf = he_info['original_cdf']
    equalized_cdf = he_info['equalized_cdf']

    axes[2, 0].plot(range(256), original_cdf, 'b-', linewidth=2, label='원본 CDF / Original CDF')
    axes[2, 0].set_title('원본 CDF\nOriginal CDF')
    axes[2, 0].set_xlabel('픽셀 강도 / Pixel Intensity')
    axes[2, 0].set_ylabel('누적 확률 / Cumulative Probability')
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].set_ylim(0, 1)

    axes[2, 1].plot(range(256), equalized_cdf, 'g-', linewidth=2, label='평활화 CDF / Equalized CDF')
    axes[2, 1].set_title('평활화된 CDF\nEqualized CDF')
    axes[2, 1].set_xlabel('픽셀 강도 / Pixel Intensity')
    axes[2, 1].set_ylabel('누적 확률 / Cumulative Probability')
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].set_ylim(0, 1)

    # CDF 비교
    axes[2, 2].plot(range(256), original_cdf, 'b-', linewidth=2, label='원본 CDF / Original CDF')
    axes[2, 2].plot(range(256), equalized_cdf, 'g-', linewidth=2, label='평활화 CDF / Equalized CDF')
    axes[2, 2].plot(range(256), np.linspace(0, 1, 256), 'r--', linewidth=2, label='이상적 CDF / Ideal CDF')
    axes[2, 2].set_title('CDF 비교\nCDF Comparison')
    axes[2, 2].set_xlabel('픽셀 강도 / Pixel Intensity')
    axes[2, 2].set_ylabel('누적 확률 / Cumulative Probability')
    axes[2, 2].legend()
    axes[2, 2].grid(True, alpha=0.3)
    axes[2, 2].set_ylim(0, 1)

    # 처리 정보 요약
    info_text = f"""처리 정보 / Processing Information:

원본 이미지 크기 / Original Image Size: {original_rgb.shape}
색공간 변환 / Color Space Conversion: RGB → YUV → RGB

Y 채널 통계 / Y Channel Statistics:
• 원본 범위 / Original Range: [{y_channel.min():.2f}, {y_channel.max():.2f}]
• 원본 평균 / Original Mean: {np.mean(y_channel):.2f}
• 평활화 후 평균 / Equalized Mean: {np.mean(y_equalized):.2f}

히스토그램 평활화 효과 / Histogram Equalization Effect:
• 동적 범위 확장 / Dynamic Range Extension
• 대비 개선 / Contrast Enhancement
• 색상 보존 (U, V 채널) / Color Preservation (U, V channels)

수학적 원리 / Mathematical Principle:
• CDF 기반 픽셀 매핑 / CDF-based Pixel Mapping
• y' = 255 × CDF(y) / y' = 255 × CDF(y)
"""

    axes[2, 3].text(0.05, 0.95, info_text, transform=axes[2, 3].transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    axes[2, 3].set_title('처리 정보 요약\nProcessing Information Summary')
    axes[2, 3].axis('off')

    plt.tight_layout()

    saved_path = None
    if save_figure:
        # 저장 경로 생성
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        save_dir = os.path.join(os.path.dirname(image_path), '..', 'results')
        os.makedirs(save_dir, exist_ok=True)

        saved_path = os.path.join(save_dir, f'{base_name}_he_4steps_analysis.png')
        plt.savefig(saved_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"📊 Figure 저장됨: {saved_path}")

    plt.show()
    return saved_path


def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(description='히스토그램 평활화 4단계 과정 시각화')
    parser.add_argument('image_path', help='입력 이미지 경로')
    parser.add_argument('--no-save', action='store_true',
                       help='figure를 저장하지 않음 (기본값: 저장함)')

    args = parser.parse_args()

    try:
        result = test_he_step_visualization(args.image_path, save_figure=not args.no_save)
        print("\n✅ 4단계 과정 시각화가 완료되었습니다!")
        print("🔍 각 단계별 결과를 확인하여 히스토그램 평활화의 원리를 이해해보세요.")

        if result['saved_figure_path']:
            print(f"💾 분석 결과가 저장되었습니다: {result['saved_figure_path']}")

        return result

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return None


if __name__ == "__main__":
    main()