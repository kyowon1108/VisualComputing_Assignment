#!/usr/bin/env python3
"""
새로운 컬러스페이스 지원 테스트 스크립트
Test script for new colorspace implementations

YCbCr과 LAB 컬러스페이스 지원이 제대로 구현되었는지 확인합니다.
Verify that YCbCr and LAB colorspace support is properly implemented.
"""

import sys
import os

# 프로젝트 루트 디렉토리를 Python path에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from src.he import histogram_equalization_color
from src.utils import load_image, display_images
from src.utils import rgb_to_ycbcr, ycbcr_to_rgb, rgb_to_lab, lab_to_rgb

def test_colorspace_conversions():
    """컬러스페이스 변환 함수들이 정상적으로 작동하는지 테스트"""
    print("=== 컬러스페이스 변환 테스트 / Colorspace Conversion Test ===")

    # 테스트 이미지 생성
    test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

    # YCbCr 변환 테스트
    try:
        ycbcr = rgb_to_ycbcr(test_image)
        rgb_back = ycbcr_to_rgb(ycbcr)
        print("✓ YCbCr 변환 성공 / YCbCr conversion successful")
        print(f"  - 원본 형태: {test_image.shape}, 범위: {test_image.min()}-{test_image.max()}")
        print(f"  - YCbCr 형태: {ycbcr.shape}, 범위: {ycbcr.min()}-{ycbcr.max()}")
        print(f"  - 복원 형태: {rgb_back.shape}, 범위: {rgb_back.min()}-{rgb_back.max()}")
    except Exception as e:
        print(f"✗ YCbCr 변환 실패: {e}")
        return False

    # LAB 변환 테스트
    try:
        lab = rgb_to_lab(test_image)
        rgb_back = lab_to_rgb(lab)
        print("✓ LAB 변환 성공 / LAB conversion successful")
        print(f"  - 원본 형태: {test_image.shape}, 범위: {test_image.min()}-{test_image.max()}")
        print(f"  - LAB 형태: {lab.shape}, 범위: {lab.min()}-{lab.max()}")
        print(f"  - 복원 형태: {rgb_back.shape}, 범위: {rgb_back.min()}-{rgb_back.max()}")
    except Exception as e:
        print(f"✗ LAB 변환 실패: {e}")
        return False

    return True

def test_histogram_equalization_methods():
    """새로운 히스토그램 평활화 방법들이 정상적으로 작동하는지 테스트"""
    print("\n=== 히스토그램 평활화 방법 테스트 / Histogram Equalization Methods Test ===")

    # 테스트 이미지 생성 (어두운 이미지)
    test_image = np.random.randint(0, 128, (100, 100, 3), dtype=np.uint8)

    methods = ['yuv', 'rgb', 'ycbcr', 'lab']
    results = {}

    for method in methods:
        try:
            equalized, process_info = histogram_equalization_color(test_image, method=method, show_process=False)
            results[method] = {
                'image': equalized,
                'info': process_info,
                'success': True
            }
            print(f"✓ {method.upper()} 방법 성공 / {method.upper()} method successful")
            print(f"  - 처리된 채널: {process_info['processed_channel']}")
            print(f"  - 결과 범위: {equalized.min()}-{equalized.max()}")
        except Exception as e:
            print(f"✗ {method.upper()} 방법 실패: {e}")
            results[method] = {'success': False, 'error': str(e)}

    return results

def compare_colorspace_methods(image_path: str = None):
    """실제 이미지로 컬러스페이스 방법들을 비교"""
    print(f"\n=== 컬러스페이스 방법 비교 / Colorspace Methods Comparison ===")

    # 이미지 로드
    if image_path and os.path.exists(image_path):
        try:
            image = load_image(image_path, color_mode='color')
            print(f"이미지 로드 성공: {image_path}")
            print(f"이미지 크기: {image.shape}")
        except Exception as e:
            print(f"이미지 로드 실패: {e}")
            return
    else:
        # 테스트 이미지 생성
        image = create_test_image_for_comparison()
        print("테스트 이미지 생성됨")

    methods = ['yuv', 'rgb', 'ycbcr', 'lab']
    images = [image]  # 원본 이미지
    titles = ['Original']

    # 각 방법으로 히스토그램 평활화 수행
    for method in methods:
        try:
            equalized, _ = histogram_equalization_color(image, method=method, show_process=False)
            images.append(equalized)
            titles.append(f'{method.upper()} HE')
            print(f"✓ {method.upper()} 처리 완료")
        except Exception as e:
            print(f"✗ {method.upper()} 처리 실패: {e}")
            # 실패한 경우 원본 이미지로 대체
            images.append(image)
            titles.append(f'{method.upper()} (Failed)')

    # 결과 시각화
    plt.figure(figsize=(20, 4))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), i+1)
        plt.imshow(img)
        plt.title(title, fontsize=12)
        plt.axis('off')

    plt.suptitle('Colorspace Methods Comparison for Histogram Equalization', fontsize=16)
    plt.tight_layout()

    # 결과 저장
    save_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'colorspace_comparison.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"비교 결과 저장됨: {save_path}")

    plt.show()

def create_test_image_for_comparison():
    """비교용 테스트 이미지 생성"""
    # 그라디언트와 패턴이 있는 테스트 이미지
    height, width = 200, 300
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # 수평 그라디언트 (어두운 영역에서 밝은 영역으로)
    for i in range(width):
        intensity = int(i / width * 200 + 30)  # 30-230 범위
        image[:, i, :] = intensity

    # 일부 영역에 패턴 추가
    for i in range(0, height, 20):
        for j in range(0, width, 20):
            if (i // 20 + j // 20) % 2 == 0:
                image[i:i+10, j:j+10, :] = 100  # 어두운 사각형

    return image

def analyze_colorspace_properties():
    """각 컬러스페이스의 특성을 분석"""
    print("\n=== 컬러스페이스 특성 분석 / Colorspace Properties Analysis ===")

    # RGB 테스트 이미지 생성
    test_image = create_test_image_for_comparison()

    # 각 컬러스페이스로 변환하여 채널별 분포 분석
    conversions = {
        'YUV': (None, None),  # 기존 함수 사용
        'YCbCr': (rgb_to_ycbcr, ycbcr_to_rgb),
        'LAB': (rgb_to_lab, lab_to_rgb)
    }

    plt.figure(figsize=(15, 10))

    for idx, (space_name, (to_func, from_func)) in enumerate(conversions.items()):
        if space_name == 'YUV':
            # 기존 YUV 함수 사용
            from src.utils import rgb_to_yuv
            converted = rgb_to_yuv(test_image)
        else:
            converted = to_func(test_image)

        # 각 채널의 히스토그램 분석
        for ch in range(3):
            plt.subplot(3, 3, idx*3 + ch + 1)
            channel_data = converted[:, :, ch].flatten()
            plt.hist(channel_data, bins=50, alpha=0.7, density=True)

            channel_names = {
                'YUV': ['Y', 'U', 'V'],
                'YCbCr': ['Y', 'Cb', 'Cr'],
                'LAB': ['L', 'A', 'B']
            }

            plt.title(f'{space_name} - {channel_names[space_name][ch]} Channel')
            plt.xlabel('Pixel Value')
            plt.ylabel('Density')
            plt.grid(True, alpha=0.3)

    plt.suptitle('Channel Distribution Analysis for Different Colorspaces', fontsize=16)
    plt.tight_layout()

    # 분석 결과 저장
    save_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'colorspace_analysis.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"분석 결과 저장됨: {save_path}")

    plt.show()

def main():
    """메인 테스트 함수"""
    print("YCbCr 및 LAB 컬러스페이스 확장 테스트")
    print("=" * 60)

    # 1. 컬러스페이스 변환 테스트
    if not test_colorspace_conversions():
        print("기본 변환 테스트 실패, 종료합니다.")
        return

    # 2. 히스토그램 평활화 방법 테스트
    results = test_histogram_equalization_methods()

    # 성공한 방법들 확인
    successful_methods = [method for method, result in results.items() if result.get('success', False)]
    print(f"\n성공한 방법들: {successful_methods}")

    if len(successful_methods) >= 4:
        print("✓ 모든 컬러스페이스 방법이 정상적으로 작동합니다!")

        # 3. 실제 비교 테스트
        print("\n실제 이미지로 비교 테스트를 수행합니다...")
        compare_colorspace_methods()

        # 4. 컬러스페이스 특성 분석
        print("컬러스페이스 특성을 분석합니다...")
        analyze_colorspace_properties()

    else:
        print(f"일부 방법이 실패했습니다. 실패한 방법들:")
        for method, result in results.items():
            if not result.get('success', False):
                print(f"  - {method}: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()