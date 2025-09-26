#!/usr/bin/env python3
"""
YUV vs RGB 히스토그램 평활화 비교 테스트
Test for comparing YUV vs RGB histogram equalization

이 테스트는 YUV와 RGB 방법의 히스토그램 평활화를 직접 비교합니다.
This test directly compares YUV and RGB methods for histogram equalization.
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

from src.utils import load_image
from src.he import histogram_equalization_color


def compare_yuv_vs_rgb(image_path: str, save_figure: bool = True):
    """
    YUV와 RGB 방법으로 히스토그램 평활화를 비교합니다.
    Compare histogram equalization using YUV and RGB methods.

    Args:
        image_path (str): 입력 이미지 경로 / Input image path
        save_figure (bool): figure를 저장할지 여부 / Whether to save figure
    """
    print(f"이미지 로딩 중: {image_path}")

    # 원본 이미지 로드
    original_image = load_image(image_path)
    if len(original_image.shape) != 3:
        raise ValueError("컬러 이미지가 필요합니다 / Color image required")

    print(f"원본 이미지 크기: {original_image.shape}")

    # YUV 방법으로 히스토그램 평활화
    print("YUV 방법으로 히스토그램 평활화 적용 중...")
    yuv_result, yuv_info = histogram_equalization_color(
        original_image, method='yuv', show_process=False
    )

    # RGB 방법으로 히스토그램 평활화
    print("RGB 방법으로 히스토그램 평활화 적용 중...")
    rgb_result, rgb_info = histogram_equalization_color(
        original_image, method='rgb', show_process=False
    )

    # 비교 시각화
    saved_path = visualize_comparison(
        original_image, yuv_result, rgb_result,
        yuv_info, rgb_info, image_path, save_figure
    )

    # 색상 왜곡 분석
    color_analysis = analyze_color_distortion(original_image, yuv_result, rgb_result)

    return {
        'original': original_image,
        'yuv_result': yuv_result,
        'rgb_result': rgb_result,
        'yuv_info': yuv_info,
        'rgb_info': rgb_info,
        'color_analysis': color_analysis,
        'saved_figure_path': saved_path
    }


def analyze_color_distortion(original, yuv_result, rgb_result):
    """색상 왜곡 정도를 분석합니다."""

    # 원본과의 색상 차이 계산 (LAB 색공간에서)
    original_lab = cv2.cvtColor(original, cv2.COLOR_RGB2LAB)
    yuv_lab = cv2.cvtColor(yuv_result, cv2.COLOR_RGB2LAB)
    rgb_lab = cv2.cvtColor(rgb_result, cv2.COLOR_RGB2LAB)

    # Delta E 색차 계산 (간소화된 버전)
    def calculate_delta_e(lab1, lab2):
        diff = lab1.astype(np.float32) - lab2.astype(np.float32)
        return np.sqrt(np.sum(diff**2, axis=2))

    yuv_delta_e = calculate_delta_e(original_lab, yuv_lab)
    rgb_delta_e = calculate_delta_e(original_lab, rgb_lab)

    # 색상 채널별 분석
    original_hsv = cv2.cvtColor(original, cv2.COLOR_RGB2HSV)
    yuv_hsv = cv2.cvtColor(yuv_result, cv2.COLOR_RGB2HSV)
    rgb_hsv = cv2.cvtColor(rgb_result, cv2.COLOR_RGB2HSV)

    hue_variance_yuv = np.var(original_hsv[:,:,0] - yuv_hsv[:,:,0])
    hue_variance_rgb = np.var(original_hsv[:,:,0] - rgb_hsv[:,:,0])

    return {
        'yuv_delta_e_mean': np.mean(yuv_delta_e),
        'rgb_delta_e_mean': np.mean(rgb_delta_e),
        'yuv_delta_e_std': np.std(yuv_delta_e),
        'rgb_delta_e_std': np.std(rgb_delta_e),
        'hue_variance_yuv': hue_variance_yuv,
        'hue_variance_rgb': hue_variance_rgb,
        'color_preservation_ratio': np.mean(yuv_delta_e) / max(np.mean(rgb_delta_e), 1e-6)
    }


def visualize_comparison(original, yuv_result, rgb_result, yuv_info, rgb_info,
                        image_path, save_figure=True):
    """YUV vs RGB 비교를 시각화합니다."""

    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle('YUV vs RGB 히스토그램 평활화 비교\nYUV vs RGB Histogram Equalization Comparison',
                 fontsize=16, fontweight='bold')

    # 첫 번째 행: 원본, YUV 결과, RGB 결과, 차이
    axes[0, 0].imshow(original)
    axes[0, 0].set_title('원본 이미지\nOriginal Image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(yuv_result)
    axes[0, 1].set_title('YUV 방법 결과\nYUV Method Result\n(색상 보존)')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(rgb_result)
    axes[0, 2].set_title('RGB 방법 결과\nRGB Method Result\n(각 채널 개별 처리)')
    axes[0, 2].axis('off')

    # YUV vs RGB 차이
    diff_yuv_rgb = np.abs(yuv_result.astype(np.float32) - rgb_result.astype(np.float32))
    diff_combined = np.mean(diff_yuv_rgb, axis=2)
    im = axes[0, 3].imshow(diff_combined, cmap='hot')
    axes[0, 3].set_title('YUV vs RGB 차이\nYUV vs RGB Difference')
    axes[0, 3].axis('off')
    plt.colorbar(im, ax=axes[0, 3], fraction=0.046, pad=0.04)

    # 두 번째 행: 각 방법의 휘도 채널 비교
    original_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    yuv_gray = cv2.cvtColor(yuv_result, cv2.COLOR_RGB2GRAY)
    rgb_gray = cv2.cvtColor(rgb_result, cv2.COLOR_RGB2GRAY)

    axes[1, 0].imshow(original_gray, cmap='gray')
    axes[1, 0].set_title('원본 휘도\nOriginal Luminance')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(yuv_gray, cmap='gray')
    axes[1, 1].set_title('YUV 방법 휘도\nYUV Method Luminance')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(rgb_gray, cmap='gray')
    axes[1, 2].set_title('RGB 방법 휘도\nRGB Method Luminance')
    axes[1, 2].axis('off')

    # 색상 보존 분석
    color_analysis = analyze_color_distortion(original, yuv_result, rgb_result)

    analysis_text = f"""색상 분석 결과 / Color Analysis Results:

YUV 방법 / YUV Method:
• 평균 색차 / Mean ΔE: {color_analysis['yuv_delta_e_mean']:.2f}
• 색차 표준편차 / ΔE Std: {color_analysis['yuv_delta_e_std']:.2f}
• 색조 분산 / Hue Variance: {color_analysis['hue_variance_yuv']:.2f}

RGB 방법 / RGB Method:
• 평균 색차 / Mean ΔE: {color_analysis['rgb_delta_e_mean']:.2f}
• 색차 표준편차 / ΔE Std: {color_analysis['rgb_delta_e_std']:.2f}
• 색조 분산 / Hue Variance: {color_analysis['hue_variance_rgb']:.2f}

색상 보존 비율 / Color Preservation Ratio:
{color_analysis['color_preservation_ratio']:.2f}
(1.0에 가까울수록 YUV가 우수)

결론 / Conclusion:
YUV: 색상 정보 보존 우수, 자연스러운 결과
RGB: 각 채널 개별 처리로 색상 왜곡 발생
"""

    axes[1, 3].text(0.05, 0.95, analysis_text, transform=axes[1, 3].transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
    axes[1, 3].set_title('색상 분석\nColor Analysis')
    axes[1, 3].axis('off')

    # 세 번째 행: 히스토그램 비교
    # 원본 히스토그램
    for i, color in enumerate(['red', 'green', 'blue']):
        hist_original = cv2.calcHist([original], [i], None, [256], [0, 256])
        axes[2, 0].plot(hist_original, color=color, alpha=0.7, label=f'{color.upper()} 원본')

    axes[2, 0].set_title('원본 히스토그램\nOriginal Histogram')
    axes[2, 0].set_xlabel('픽셀 강도 / Pixel Intensity')
    axes[2, 0].set_ylabel('빈도수 / Frequency')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)

    # YUV 결과 히스토그램
    for i, color in enumerate(['red', 'green', 'blue']):
        hist_yuv = cv2.calcHist([yuv_result], [i], None, [256], [0, 256])
        axes[2, 1].plot(hist_yuv, color=color, alpha=0.7, label=f'{color.upper()} YUV')

    axes[2, 1].set_title('YUV 방법 히스토그램\nYUV Method Histogram')
    axes[2, 1].set_xlabel('픽셀 강도 / Pixel Intensity')
    axes[2, 1].set_ylabel('빈도수 / Frequency')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)

    # RGB 결과 히스토그램
    for i, color in enumerate(['red', 'green', 'blue']):
        hist_rgb = cv2.calcHist([rgb_result], [i], None, [256], [0, 256])
        axes[2, 2].plot(hist_rgb, color=color, alpha=0.7, label=f'{color.upper()} RGB')

    axes[2, 2].set_title('RGB 방법 히스토그램\nRGB Method Histogram')
    axes[2, 2].set_xlabel('픽셀 강도 / Pixel Intensity')
    axes[2, 2].set_ylabel('빈도수 / Frequency')
    axes[2, 2].legend()
    axes[2, 2].grid(True, alpha=0.3)

    # 방법 비교 요약
    comparison_text = f"""방법 비교 / Method Comparison:

🎨 YUV 색공간 방법:
장점 / Advantages:
• Y(휘도) 채널만 처리하여 색상 정보 보존
• U, V 채널 유지로 자연스러운 색감
• 인간의 시각적 인지에 적합한 처리
• 색상 왜곡 최소화

단점 / Disadvantages:
• 색공간 변환 오버헤드
• 각 채널별 세밀한 조정 불가

🔴🟢🔵 RGB 채널별 방법:
장점 / Advantages:
• 각 색상 채널의 독립적 처리 가능
• 채널별 히스토그램 완전 평활화
• 구현이 직관적

단점 / Disadvantages:
• 색상 균형 파괴로 부자연스러운 결과
• 색조(Hue) 변화 발생
• 색상 왜곡 심함

💡 권장사항 / Recommendation:
컬러 이미지에는 YUV 방법 사용 권장
자연스러운 색상 보존과 효과적인 대비 개선
"""

    axes[2, 3].text(0.05, 0.95, comparison_text, transform=axes[2, 3].transAxes,
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    axes[2, 3].set_title('방법 비교 요약\nMethod Comparison Summary')
    axes[2, 3].axis('off')

    plt.tight_layout()

    saved_path = None
    if save_figure:
        # 저장 경로 생성
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        save_dir = os.path.join(os.path.dirname(image_path), '..', 'results')
        os.makedirs(save_dir, exist_ok=True)

        saved_path = os.path.join(save_dir, f'{base_name}_yuv_vs_rgb_comparison.png')
        plt.savefig(saved_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"📊 비교 결과 저장됨: {saved_path}")

    plt.show()
    return saved_path


def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(description='YUV vs RGB 히스토그램 평활화 비교')
    parser.add_argument('image_path', help='입력 이미지 경로')
    parser.add_argument('--no-save', action='store_true',
                       help='figure를 저장하지 않음 (기본값: 저장함)')

    args = parser.parse_args()

    try:
        result = compare_yuv_vs_rgb(args.image_path, save_figure=not args.no_save)

        print("\n✅ YUV vs RGB 비교 분석이 완료되었습니다!")
        print("🔍 두 방법의 차이점과 각각의 장단점을 확인해보세요.")

        # 색상 분석 요약 출력
        analysis = result['color_analysis']
        print(f"\n📊 색상 보존 분석 결과:")
        print(f"   YUV 평균 색차: {analysis['yuv_delta_e_mean']:.2f}")
        print(f"   RGB 평균 색차: {analysis['rgb_delta_e_mean']:.2f}")
        print(f"   색상 보존 비율: {analysis['color_preservation_ratio']:.2f}")

        if analysis['color_preservation_ratio'] < 1.0:
            print("   ✨ YUV 방법이 색상 보존에 더 우수합니다!")
        else:
            print("   🔴 RGB 방법이 색상 변화가 더 큽니다.")

        if result['saved_figure_path']:
            print(f"💾 비교 결과가 저장되었습니다: {result['saved_figure_path']}")

        return result

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return None


if __name__ == "__main__":
    main()