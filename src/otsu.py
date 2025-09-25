"""
Enhanced Otsu Thresholding Module
개선된 Otsu 임계값 처리 모듈

이 모듈은 다양한 Otsu 임계값 방법을 구현합니다.
This module implements various Otsu thresholding methods.

주요 기능 / Key Features:
1. Global Otsu thresholding / 전역 Otsu 임계값
2. Block-based local Otsu / 블록 기반 지역 Otsu
3. Sliding window Otsu / 슬라이딩 윈도우 Otsu
4. Improved method with preprocessing and postprocessing / 전처리 및 후처리가 포함된 개선된 방법
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Dict, Any, List
from scipy import ndimage
from skimage import morphology, filters
import matplotlib.pyplot as plt

def compute_otsu_threshold(histogram: np.ndarray) -> float:
    """
    히스토그램으로부터 Otsu 임계값을 계산합니다.
    Compute Otsu threshold from histogram.

    Args:
        histogram: 히스토그램 배열

    Returns:
        최적 임계값
    """
    total_pixels = np.sum(histogram)
    if total_pixels == 0:
        return 127.0

    # 가중 평균 계산
    total_mean = np.sum(np.arange(256) * histogram) / total_pixels

    max_variance = 0.0
    optimal_threshold = 0

    w0 = 0.0  # 배경 픽셀 비율
    sum0 = 0.0  # 배경 픽셀 가중 합

    for t in range(256):
        w0 += histogram[t] / total_pixels
        if w0 == 0:
            continue

        w1 = 1.0 - w0  # 전경 픽셀 비율
        if w1 == 0:
            break

        sum0 += t * histogram[t] / total_pixels

        mean0 = sum0 / w0  # 배경 평균
        mean1 = (total_mean - sum0) / w1  # 전경 평균

        # 클래스 간 분산 계산
        between_variance = w0 * w1 * (mean0 - mean1) ** 2

        if between_variance > max_variance:
            max_variance = between_variance
            optimal_threshold = t

    return float(optimal_threshold)

def global_otsu(image: np.ndarray) -> Dict[str, Any]:
    """
    전역 Otsu 임계값을 적용합니다.
    Apply global Otsu thresholding.

    Args:
        image: 입력 그레이스케일 이미지

    Returns:
        결과 딕셔너리 (result, threshold, histogram 등)
    """
    if len(image.shape) != 2:
        raise ValueError("입력 이미지는 그레이스케일이어야 합니다")

    # 히스토그램 계산
    histogram, _ = np.histogram(image.flatten(), bins=256, range=[0, 256])

    # Otsu 임계값 계산
    threshold = compute_otsu_threshold(histogram)

    # 이진화 적용
    result = (image > threshold).astype(np.uint8) * 255

    return {
        'result': result,
        'threshold': threshold,
        'histogram': histogram,
        'method': 'global_otsu'
    }

def block_based_otsu(image: np.ndarray, window_size: int = 75, stride: int = 24) -> Dict[str, Any]:
    """
    블록 기반 지역 Otsu를 적용합니다.
    Apply block-based local Otsu thresholding.

    Args:
        image: 입력 그레이스케일 이미지
        window_size: 블록 크기
        stride: 스트라이드 (겹침 제어)

    Returns:
        결과 딕셔너리
    """
    h, w = image.shape
    threshold_map = np.zeros_like(image, dtype=np.float32)
    weight_map = np.zeros_like(image, dtype=np.float32)

    # 블록별 임계값 계산
    thresholds = []
    positions = []

    for y in range(0, h - window_size + 1, stride):
        for x in range(0, w - window_size + 1, stride):
            # 블록 추출
            block = image[y:y+window_size, x:x+window_size]

            # 블록의 히스토그램 계산
            block_hist, _ = np.histogram(block.flatten(), bins=256, range=[0, 256])

            # 블록별 Otsu 임계값
            block_threshold = compute_otsu_threshold(block_hist)

            thresholds.append(block_threshold)
            positions.append((x + window_size//2, y + window_size//2))

            # 가중치를 이용한 임계값 할당
            y_end = min(y + window_size, h)
            x_end = min(x + window_size, w)

            threshold_map[y:y_end, x:x_end] += block_threshold
            weight_map[y:y_end, x:x_end] += 1.0

    # 가중 평균으로 최종 임계값 맵 생성
    threshold_map = np.divide(threshold_map, weight_map,
                             out=np.full_like(threshold_map, 127.0),
                             where=(weight_map != 0))

    # 이진화 적용
    result = (image > threshold_map).astype(np.uint8) * 255

    return {
        'result': result,
        'threshold_map': threshold_map,
        'thresholds': thresholds,
        'positions': positions,
        'method': 'block_based'
    }

def sliding_window_otsu(image: np.ndarray, window_size: int = 75, stride: int = 24) -> Dict[str, Any]:
    """
    슬라이딩 윈도우 Otsu를 적용합니다.
    Apply sliding window Otsu thresholding.

    Args:
        image: 입력 그레이스케일 이미지
        window_size: 윈도우 크기
        stride: 스트라이드

    Returns:
        결과 딕셔너리
    """
    h, w = image.shape

    # 그리드 생성
    y_coords = np.arange(window_size//2, h - window_size//2, stride)
    x_coords = np.arange(window_size//2, w - window_size//2, stride)

    # 임계값 그리드 계산
    threshold_grid = np.zeros((len(y_coords), len(x_coords)))

    for i, y in enumerate(y_coords):
        for j, x in enumerate(x_coords):
            # 윈도우 추출
            y_start = max(0, y - window_size//2)
            y_end = min(h, y + window_size//2)
            x_start = max(0, x - window_size//2)
            x_end = min(w, x + window_size//2)

            window = image[y_start:y_end, x_start:x_end]

            # 윈도우의 히스토그램과 임계값 계산
            window_hist, _ = np.histogram(window.flatten(), bins=256, range=[0, 256])
            window_threshold = compute_otsu_threshold(window_hist)

            threshold_grid[i, j] = window_threshold

    # 양선형 보간으로 전체 이미지에 임계값 할당
    from scipy import interpolate

    # 보간 함수 생성
    f = interpolate.RectBivariateSpline(y_coords, x_coords, threshold_grid)

    # 모든 픽셀에 대한 임계값 계산
    y_full = np.arange(h)
    x_full = np.arange(w)
    threshold_map = f(y_full, x_full)

    # 이진화 적용
    result = (image > threshold_map).astype(np.uint8) * 255

    return {
        'result': result,
        'threshold_map': threshold_map,
        'threshold_grid': threshold_grid,
        'grid_coords': (y_coords, x_coords),
        'method': 'sliding_window'
    }

def apply_preprocessing(image: np.ndarray, preblur: float = 1.0) -> np.ndarray:
    """전처리를 적용합니다 (가우시안 블러)."""
    if preblur > 0:
        return cv2.GaussianBlur(image, (0, 0), preblur)
    return image.copy()

def apply_morphological_operations(binary_image: np.ndarray, operations: List[str]) -> np.ndarray:
    """형태학적 연산을 적용합니다."""
    result = binary_image.copy()

    for op in operations:
        if ',' in op:
            op_name, kernel_size = op.split(',')
            kernel_size = int(kernel_size)
        else:
            op_name = op
            kernel_size = 3

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        if op_name.lower() == 'open':
            result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
        elif op_name.lower() == 'close':
            result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
        elif op_name.lower() == 'dilate':
            result = cv2.dilate(result, kernel)
        elif op_name.lower() == 'erode':
            result = cv2.erode(result, kernel)

    return result

def improved_otsu(image: np.ndarray,
                 window_size: int = 75,
                 stride: int = 24,
                 preblur: float = 1.0,
                 morph_ops: List[str] = ['open,3', 'close,3']) -> Dict[str, Any]:
    """
    개선된 Otsu 방법을 적용합니다.
    Apply improved Otsu method with preprocessing and postprocessing.

    Args:
        image: 입력 그레이스케일 이미지
        window_size: 윈도우 크기
        stride: 스트라이드
        preblur: 전처리 가우시안 시그마
        morph_ops: 후처리 형태학적 연산 리스트

    Returns:
        결과 딕셔너리
    """
    # 전처리
    preprocessed = apply_preprocessing(image, preblur)

    # 슬라이딩 윈도우 Otsu 적용
    otsu_result = sliding_window_otsu(preprocessed, window_size, stride)

    # 후처리 (형태학적 연산)
    postprocessed = apply_morphological_operations(otsu_result['result'], morph_ops)

    return {
        'result': postprocessed,
        'preprocessed': preprocessed,
        'before_postprocess': otsu_result['result'],
        'threshold_map': otsu_result['threshold_map'],
        'threshold_grid': otsu_result['threshold_grid'],
        'grid_coords': otsu_result['grid_coords'],
        'method': 'improved',
        'parameters': {
            'window_size': window_size,
            'stride': stride,
            'preblur': preblur,
            'morph_ops': morph_ops
        }
    }

def create_threshold_heatmap(threshold_map: np.ndarray, save_path: str, title: str = "Threshold Heatmap"):
    """임계값 히트맵을 생성하고 저장합니다."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    im = ax.imshow(threshold_map, cmap='viridis', interpolation='nearest')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')

    # 컬러바 추가
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Threshold Value', rotation=270, labelpad=20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Threshold heatmap saved: {save_path}")

def create_local_histogram_with_threshold(image: np.ndarray, roi: Tuple[int, int, int, int],
                                        threshold: float, save_path: str):
    """선택 ROI의 히스토그램과 임계값을 시각화합니다."""
    x, y, w, h = roi
    roi_image = image[y:y+h, x:x+w]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ROI 이미지 표시
    ax1.imshow(roi_image, cmap='gray')
    ax1.set_title(f'Selected ROI ({x},{y},{w},{h})', fontweight='bold')
    ax1.axis('off')

    # ROI 히스토그램과 임계값
    hist, bins = np.histogram(roi_image.flatten(), bins=256, range=[0, 256])
    ax2.bar(range(256), hist, alpha=0.7, color='gray', edgecolor='black')
    ax2.axvline(x=threshold, color='red', linestyle='--', linewidth=3, label=f'Threshold: {threshold:.1f}')
    ax2.set_title('ROI Histogram with Selected Threshold', fontweight='bold')
    ax2.set_xlabel('Pixel Value')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Local histogram with threshold saved: {save_path}")

def create_otsu_comparison_contact_sheet(original_img: np.ndarray, results: Dict[str, Any],
                                       rois: List[Tuple[int, int, int, int]], save_path: str, dpi=300):
    """Otsu 방법들의 비교 콘택트 시트를 생성합니다."""
    methods = ['Original', 'Global', 'Improved']
    n_methods = len(methods)
    n_rois = len(rois)

    # 그리드 크기 결정: 풀샷 1행 + ROI별 1행씩
    fig_height = 4 * (1 + n_rois)
    fig_width = 4 * n_methods

    fig, axes = plt.subplots(1 + n_rois, n_methods, figsize=(fig_width, fig_height), dpi=dpi)

    if n_methods == 1:
        axes = axes.reshape(-1, 1)
    if axes.ndim == 1:
        axes = axes.reshape(1, -1)

    # 첫 번째 행: 풀샷 비교
    method_results = {
        'Original': original_img,
        'Global': results.get('global_otsu', {}).get('result', original_img),
        'Improved': results.get('improved', {}).get('result', original_img)
    }

    for col, method_name in enumerate(methods):
        result_img = method_results[method_name]

        axes[0, col].imshow(result_img, cmap='gray')
        axes[0, col].set_title(f"{method_name}", fontsize=12, fontweight='bold')
        axes[0, col].axis('off')

        # 원본에만 ROI 박스 표시
        if col == 0:
            for i, (x, y, w, h) in enumerate(rois):
                rect = plt.Rectangle((x, y), w, h, linewidth=3, edgecolor='red', facecolor='none')
                axes[0, col].add_patch(rect)
                axes[0, col].text(x, y-5, f'ROI{i+1}', color='red', fontweight='bold', fontsize=10)

    # ROI별 확대 비교
    for roi_idx, (x, y, w, h) in enumerate(rois):
        row = roi_idx + 1

        for col, method_name in enumerate(methods):
            result_img = method_results[method_name]
            roi_img = result_img[y:y+h, x:x+w]

            # 200% 확대
            roi_img_resized = cv2.resize(roi_img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)

            axes[row, col].imshow(roi_img_resized, cmap='gray')
            axes[row, col].set_title(f"{method_name} ROI{roi_idx+1}", fontsize=10)
            axes[row, col].axis('off')

    plt.suptitle('Otsu Methods Comparison with ROI Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"Otsu comparison contact sheet saved: {save_path}")

# 기존 함수들과의 호환성을 위한 래퍼
def compare_otsu_methods(image: np.ndarray) -> Dict[str, Any]:
    """여러 Otsu 방법을 비교합니다."""
    results = {}

    # Global Otsu
    results['global_otsu'] = global_otsu(image)

    # Block-based Otsu
    results['block_based'] = block_based_otsu(image)

    # Sliding window Otsu
    results['sliding_window'] = sliding_window_otsu(image)

    # Improved Otsu
    results['improved'] = improved_otsu(image)

    return results