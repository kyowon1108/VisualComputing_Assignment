#!/usr/bin/env python3
"""
히스토그램 평활화 방법들 비교 스크립트
Compare different histogram equalization methods

원본, HE, AHE, CLAHE의 결과를 한 화면에서 비교합니다.
Compare results of Original, HE, AHE, CLAHE in one screen.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from src.utils import load_image
from src.he import histogram_equalization_color, histogram_equalization_color_clahe, compute_histogram

def compare_he_methods(image_path: str, method: str = 'yuv', tile_size: int = 8, clip_limit: float = 2.0, save_dir: str = None):
    """
    HE 방법들을 비교합니다.
    Compare HE methods.

    Args:
        image_path (str): 입력 이미지 경로 / Input image path
        method (str): 처리 방법 ('yuv' 또는 'rgb') / Processing method
        tile_size (int): AHE/CLAHE 타일 크기 / Tile size for AHE/CLAHE
        clip_limit (float): CLAHE 클립 한계 / CLAHE clip limit
        save_dir (str): 저장 디렉토리 / Save directory
    """
    # 이미지 로드
    print(f"이미지 로딩 중... / Loading image: {image_path}")
    original = load_image(image_path)
    print(f"이미지 크기 / Image size: {original.shape}")

    # 각 방법으로 처리
    print("HE 처리 중... / Processing HE...")
    he_result, he_info = histogram_equalization_color(original, method=method, show_process=False)

    print("AHE 처리 중... / Processing AHE...")
    ahe_result, ahe_info = histogram_equalization_color_clahe(
        original, method=method, clip_limit=999.0, tile_size=(tile_size, tile_size),
        show_process=False, use_opencv=True, algorithm_name='AHE'
    )

    print("CLAHE 처리 중... / Processing CLAHE...")
    clahe_result, clahe_info = histogram_equalization_color_clahe(
        original, method=method, clip_limit=clip_limit, tile_size=(tile_size, tile_size),
        show_process=False, use_opencv=True, algorithm_name='CLAHE'
    )

    # 비교 시각화
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # 상단: 이미지들
    images = [original, he_result, ahe_result, clahe_result]
    titles = ['Original', f'HE ({method.upper()})', f'AHE ({method.upper()})', f'CLAHE ({method.upper()})']

    for i, (img, title) in enumerate(zip(images, titles)):
        axes[0, i].imshow(img)
        axes[0, i].set_title(title, fontsize=12, fontweight='bold')
        axes[0, i].axis('off')

    # 하단: 히스토그램들
    if method == 'yuv':
        # YUV의 경우 Y 채널 히스토그램
        from src.utils import rgb_to_yuv
        y_channels = []
        for img in images:
            if len(img.shape) == 3:
                yuv = rgb_to_yuv(img)
                y_channels.append(yuv[:, :, 0])
            else:
                y_channels.append(img)

        for i, (y_channel, title) in enumerate(zip(y_channels, titles)):
            hist, _ = compute_histogram(y_channel)
            axes[1, i].bar(range(256), hist, alpha=0.7, color=['blue', 'green', 'orange', 'red'][i])
            axes[1, i].set_title(f'{title}\nY Channel Histogram', fontsize=10)
            axes[1, i].set_xlabel('Pixel Intensity')
            axes[1, i].set_ylabel('Frequency')
            axes[1, i].set_xlim(0, 255)

    elif method == 'rgb':
        # RGB의 경우 전체 밝기 히스토그램 (그레이스케일 변환)
        gray_images = []
        for img in images:
            if len(img.shape) == 3:
                gray = np.dot(img, [0.299, 0.587, 0.114]).astype(np.uint8)
                gray_images.append(gray)
            else:
                gray_images.append(img)

        for i, (gray_img, title) in enumerate(zip(gray_images, titles)):
            hist, _ = compute_histogram(gray_img)
            axes[1, i].bar(range(256), hist, alpha=0.7, color=['blue', 'green', 'orange', 'red'][i])
            axes[1, i].set_title(f'{title}\nGrayscale Histogram', fontsize=10)
            axes[1, i].set_xlabel('Pixel Intensity')
            axes[1, i].set_ylabel('Frequency')
            axes[1, i].set_xlim(0, 255)

    # 파라미터 정보 텍스트 추가
    param_text = f"비교 파라미터 / Comparison Parameters:\n"
    param_text += f"• 처리 방법 / Method: {method.upper()}\n"
    param_text += f"• AHE/CLAHE 타일 크기 / Tile Size: {tile_size}×{tile_size}\n"
    param_text += f"• CLAHE 클립 한계 / Clip Limit: {clip_limit}\n"
    param_text += f"• AHE 클립 한계 / AHE Clip Limit: 999.0 (거의 무제한)\n\n"
    param_text += f"특징 비교 / Feature Comparison:\n"
    param_text += f"• HE: 전역 대비 개선, 간단함\n"
    param_text += f"• AHE: 지역적 강한 대비, 세부 강조\n"
    param_text += f"• CLAHE: 균형잡힌 대비, 노이즈 억제"

    fig.text(0.02, 0.02, param_text, fontsize=9, verticalalignment='bottom',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

    plt.suptitle(f'히스토그램 평활화 방법 비교 / Histogram Equalization Methods Comparison\n({method.upper()} 방법)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.15)

    # 저장
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        filename = f"he_comparison_{method}_tile{tile_size}_clip{clip_limit}.png"
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"비교 결과 저장됨 / Comparison saved: {save_path}")

    plt.show()

    # 결과 정보 출력
    print("\n=== 처리 결과 요약 / Processing Results Summary ===")
    print(f"원본 이미지 크기 / Original image size: {original.shape}")
    print(f"처리 방법 / Processing method: {method.upper()}")
    print(f"HE: 전역 히스토그램 평활화 / Global histogram equalization")
    print(f"AHE: 적응적 HE (클립={999.0}) / Adaptive HE (clip={999.0})")
    print(f"CLAHE: 제한적 적응적 HE (클립={clip_limit}) / Contrast Limited Adaptive HE (clip={clip_limit})")

def main():
    parser = argparse.ArgumentParser(description='히스토그램 평활화 방법들 비교 / Compare histogram equalization methods')
    parser.add_argument('image_path', help='입력 이미지 파일 경로 / Input image file path')
    parser.add_argument('--method', choices=['yuv', 'rgb'], default='yuv',
                       help='처리 방법: yuv(권장), rgb / Processing method: yuv(recommended), rgb')
    parser.add_argument('--tile-size', type=int, default=8,
                       help='AHE/CLAHE 타일 크기 (기본값: 8) / AHE/CLAHE tile size (default: 8)')
    parser.add_argument('--clip-limit', type=float, default=2.0,
                       help='CLAHE 클립 한계 (기본값: 2.0) / CLAHE clip limit (default: 2.0)')
    parser.add_argument('--save', metavar='DIR',
                       help='결과 저장 디렉토리 / Result saving directory')

    args = parser.parse_args()

    try:
        compare_he_methods(args.image_path, args.method, args.tile_size, args.clip_limit, args.save)
        print("\n비교 완료! / Comparison completed!")
    except Exception as e:
        print(f"오류 발생 / Error occurred: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())