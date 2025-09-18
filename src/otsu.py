"""
Local Otsu Thresholding 모듈
Local Otsu Thresholding Module

이 모듈은 Local Otsu Thresholding을 직접 구현합니다.
This module directly implements Local Otsu Thresholding.

주요 기능 / Key Features:
1. 직접 구현된 Otsu 임계값 계산 / Direct implementation of Otsu threshold calculation
2. 블록 기반 Local Otsu Thresholding / Block-based Local Otsu Thresholding
3. 슬라이딩 윈도우 기반 Local Otsu Thresholding / Sliding window-based Local Otsu Thresholding
4. Inter-class variance 최대화와 within-class variance 최소화의 수학적 관계 설명
5. 단계별 중간 과정 시각화 / Step-by-step intermediate process visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from .utils import compute_histogram, validate_image_input, display_images
from typing import Tuple, Optional, Union

def calculate_otsu_threshold(histogram: np.ndarray, show_process: bool = False) -> Tuple[int, dict]:
    """
    Otsu 방법을 사용하여 최적의 임계값을 계산합니다.
    Calculate optimal threshold using Otsu's method.

    Otsu 방법의 수학적 원리:
    1. Inter-class variance (클래스 간 분산) 최대화
    2. Within-class variance (클래스 내 분산) 최소화
    3. 수학적 관계: σ²(total) = σ²(within) + σ²(between)

    Mathematical principle of Otsu's method:
    1. Maximize inter-class variance
    2. Minimize within-class variance
    3. Mathematical relationship: σ²(total) = σ²(within) + σ²(between)

    σ²(between) = w₀ * w₁ * (μ₀ - μ₁)²
    여기서:
    - w₀, w₁: 각 클래스의 확률 (픽셀 비율)
    - μ₀, μ₁: 각 클래스의 평균값

    Args:
        histogram (np.ndarray): 입력 히스토그램 / Input histogram
        show_process (bool): 중간 과정 표시 여부 / Whether to show intermediate process

    Returns:
        Tuple[int, dict]: (최적 임계값, 계산 정보) / (optimal threshold, calculation info)
    """
    if np.sum(histogram) == 0:
        return 0, {'message': 'Empty histogram'}

    # 총 픽셀 수 / Total number of pixels
    total_pixels = np.sum(histogram)

    # 전체 평균 계산 / Calculate overall mean
    pixel_values = np.arange(256)
    overall_mean = np.sum(pixel_values * histogram) / total_pixels

    # 각 가능한 임계값에 대해 inter-class variance 계산
    # Calculate inter-class variance for each possible threshold
    max_variance = 0
    optimal_threshold = 0
    variance_history = []

    for threshold in range(256):
        # 클래스 0 (0 ~ threshold)
        w0 = np.sum(histogram[:threshold + 1]) / total_pixels
        if w0 == 0:
            variance_history.append(0)
            continue

        # 클래스 1 (threshold+1 ~ 255)
        w1 = np.sum(histogram[threshold + 1:]) / total_pixels
        if w1 == 0:
            variance_history.append(0)
            continue

        # 각 클래스의 평균 계산 / Calculate mean of each class
        if w0 > 0:
            mean0 = np.sum(pixel_values[:threshold + 1] * histogram[:threshold + 1]) / np.sum(histogram[:threshold + 1])
        else:
            mean0 = 0

        if w1 > 0:
            mean1 = np.sum(pixel_values[threshold + 1:] * histogram[threshold + 1:]) / np.sum(histogram[threshold + 1:])
        else:
            mean1 = 0

        # Inter-class variance 계산 / Calculate inter-class variance
        # σ²(between) = w₀ * w₁ * (μ₀ - μ₁)²
        inter_class_variance = w0 * w1 * (mean0 - mean1) ** 2

        variance_history.append(inter_class_variance)

        # 최대 inter-class variance를 갖는 임계값 찾기
        # Find threshold with maximum inter-class variance
        if inter_class_variance > max_variance:
            max_variance = inter_class_variance
            optimal_threshold = threshold

    # 계산 정보 저장 / Store calculation information
    calculation_info = {
        'optimal_threshold': optimal_threshold,
        'max_inter_class_variance': max_variance,
        'variance_history': variance_history,
        'total_pixels': total_pixels,
        'overall_mean': overall_mean
    }

    # 중간 과정 시각화 / Visualize intermediate process
    if show_process:
        visualize_otsu_calculation(histogram, calculation_info)

    return optimal_threshold, calculation_info

def apply_threshold(image: np.ndarray, threshold: int, max_value: int = 255) -> np.ndarray:
    """
    임계값을 적용하여 이진 이미지를 생성합니다.
    Apply threshold to create binary image.

    Args:
        image (np.ndarray): 입력 이미지 / Input image
        threshold (int): 임계값 / Threshold value
        max_value (int): 최대값 (일반적으로 255) / Maximum value (typically 255)

    Returns:
        np.ndarray: 이진 이미지 / Binary image
    """
    binary_image = np.where(image > threshold, max_value, 0).astype(np.uint8)
    return binary_image

def global_otsu_thresholding(image: np.ndarray, show_process: bool = True) -> Tuple[np.ndarray, dict]:
    """
    전역 Otsu Thresholding을 수행합니다.
    Perform global Otsu thresholding.

    Args:
        image (np.ndarray): 입력 그레이스케일 이미지 / Input grayscale image
        show_process (bool): 중간 과정 표시 여부 / Whether to show intermediate process

    Returns:
        Tuple[np.ndarray, dict]: (이진 이미지, 처리 정보) / (binary image, processing info)
    """
    validate_image_input(image)

    if len(image.shape) != 2:
        raise ValueError("그레이스케일 이미지가 필요합니다 / Grayscale image required")

    # 히스토그램 계산 / Calculate histogram
    histogram, _ = compute_histogram(image)

    # Otsu 임계값 계산 / Calculate Otsu threshold
    optimal_threshold, calc_info = calculate_otsu_threshold(histogram, show_process=False)

    # 임계값 적용 / Apply threshold
    binary_image = apply_threshold(image, optimal_threshold)

    # 처리 정보 저장 / Store processing information
    process_info = {
        'method': 'global_otsu',
        'threshold': optimal_threshold,
        'histogram': histogram,
        'calculation_info': calc_info
    }

    # 중간 과정 시각화 / Visualize intermediate process
    if show_process:
        visualize_global_otsu_process(image, binary_image, process_info)

    return binary_image, process_info

def local_otsu_block_based(image: np.ndarray, block_size: Tuple[int, int] = (32, 32), show_process: bool = True) -> Tuple[np.ndarray, dict]:
    """
    블록 기반 Local Otsu Thresholding을 수행합니다.
    Perform block-based Local Otsu Thresholding.

    블록 기반 방법의 특징:
    - 이미지를 균등한 크기의 블록으로 분할
    - 각 블록마다 독립적으로 Otsu 임계값 계산
    - 계산 효율성이 높음
    - 블록 경계에서 불연속성 발생 가능

    Characteristics of block-based method:
    - Divide image into equal-sized blocks
    - Calculate Otsu threshold independently for each block
    - High computational efficiency
    - Possible discontinuity at block boundaries

    Args:
        image (np.ndarray): 입력 그레이스케일 이미지 / Input grayscale image
        block_size (Tuple[int, int]): 블록 크기 (높이, 너비) / Block size (height, width)
        show_process (bool): 중간 과정 표시 여부 / Whether to show intermediate process

    Returns:
        Tuple[np.ndarray, dict]: (이진 이미지, 처리 정보) / (binary image, processing info)
    """
    validate_image_input(image)

    if len(image.shape) != 2:
        raise ValueError("그레이스케일 이미지가 필요합니다 / Grayscale image required")

    height, width = image.shape
    block_h, block_w = block_size

    # 결과 이미지 초기화 / Initialize result image
    binary_image = np.zeros_like(image)
    threshold_map = np.zeros_like(image, dtype=np.float32)

    # 블록 정보 저장 / Store block information
    block_info = []

    # 각 블록에 대해 Otsu 임계값 계산 및 적용
    # Calculate and apply Otsu threshold for each block
    for i in range(0, height, block_h):
        for j in range(0, width, block_w):
            # 현재 블록 추출 / Extract current block
            end_i = min(i + block_h, height)
            end_j = min(j + block_w, width)
            block = image[i:end_i, j:end_j]

            # 블록의 히스토그램 계산 / Calculate block histogram
            block_hist, _ = compute_histogram(block)

            # 블록에 대한 Otsu 임계값 계산 / Calculate Otsu threshold for block
            if np.sum(block_hist) > 0:
                block_threshold, block_calc_info = calculate_otsu_threshold(block_hist, show_process=False)
            else:
                block_threshold = 127  # 기본값 / Default value
                block_calc_info = {'message': 'Empty block'}

            # 임계값 적용 / Apply threshold
            block_binary = apply_threshold(block, block_threshold)
            binary_image[i:end_i, j:end_j] = block_binary

            # 임계값 맵 업데이트 / Update threshold map
            threshold_map[i:end_i, j:end_j] = block_threshold

            # 블록 정보 저장 (일부만) / Store block information (partial)
            if len(block_info) < 9:  # 최대 9개 블록 정보만 저장 / Store max 9 blocks info
                block_info.append({
                    'position': (i, j),
                    'size': (end_i - i, end_j - j),
                    'threshold': block_threshold,
                    'histogram': block_hist,
                    'calc_info': block_calc_info
                })

    # 처리 정보 저장 / Store processing information
    process_info = {
        'method': 'local_otsu_block_based',
        'block_size': block_size,
        'threshold_map': threshold_map,
        'block_info': block_info,
        'num_blocks': len(block_info)
    }

    # 중간 과정 시각화 / Visualize intermediate process
    if show_process:
        visualize_local_otsu_process(image, binary_image, process_info)

    return binary_image, process_info

def local_otsu_sliding_window(image: np.ndarray, window_size: Tuple[int, int] = (32, 32), stride: int = 8, show_process: bool = True) -> Tuple[np.ndarray, dict]:
    """
    슬라이딩 윈도우 기반 Local Otsu Thresholding을 수행합니다.
    Perform sliding window-based Local Otsu Thresholding.

    슬라이딩 윈도우 방법의 특징:
    - 지정된 스트라이드로 윈도우를 이동하며 처리
    - 중앙 픽셀에 대한 임계값을 윈도우 영역에서 계산
    - 더 부드러운 결과, 하지만 계산 비용이 높음
    - 윈도우 겹침으로 인한 더 나은 연속성

    Characteristics of sliding window method:
    - Process by moving window with specified stride
    - Calculate threshold for center pixel from window area
    - Smoother results but higher computational cost
    - Better continuity due to window overlap

    Args:
        image (np.ndarray): 입력 그레이스케일 이미지 / Input grayscale image
        window_size (Tuple[int, int]): 윈도우 크기 (높이, 너비) / Window size (height, width)
        stride (int): 스트라이드 (윈도우 이동 간격) / Stride (window movement interval)
        show_process (bool): 중간 과정 표시 여부 / Whether to show intermediate process

    Returns:
        Tuple[np.ndarray, dict]: (이진 이미지, 처리 정보) / (binary image, processing info)
    """
    validate_image_input(image)

    if len(image.shape) != 2:
        raise ValueError("그레이스케일 이미지가 필요합니다 / Grayscale image required")

    height, width = image.shape
    window_h, window_w = window_size
    half_h, half_w = window_h // 2, window_w // 2

    # 결과 이미지 및 임계값 맵 초기화 / Initialize result image and threshold map
    binary_image = np.zeros_like(image)
    threshold_map = np.zeros_like(image, dtype=np.float32)
    processed_mask = np.zeros_like(image, dtype=bool)

    # 윈도우 정보 저장 / Store window information
    window_info = []

    # 슬라이딩 윈도우로 이미지 처리 / Process image with sliding window
    for i in range(half_h, height - half_h, stride):
        for j in range(half_w, width - half_w, stride):
            # 현재 윈도우 영역 정의 / Define current window area
            start_i = max(0, i - half_h)
            end_i = min(height, i + half_h + 1)
            start_j = max(0, j - half_w)
            end_j = min(width, j + half_w + 1)

            # 윈도우 영역 추출 / Extract window area
            window = image[start_i:end_i, start_j:end_j]

            # 윈도우의 히스토그램 계산 / Calculate window histogram
            window_hist, _ = compute_histogram(window)

            # 윈도우에 대한 Otsu 임계값 계산 / Calculate Otsu threshold for window
            if np.sum(window_hist) > 0:
                window_threshold, window_calc_info = calculate_otsu_threshold(window_hist, show_process=False)
            else:
                window_threshold = 127  # 기본값 / Default value
                window_calc_info = {'message': 'Empty window'}

            # 윈도우 중앙 영역의 픽셀들에 임계값 적용
            # Apply threshold to pixels in window center area
            center_start_i = max(start_i, i - stride // 2)
            center_end_i = min(end_i, i + stride // 2 + 1)
            center_start_j = max(start_j, j - stride // 2)
            center_end_j = min(end_j, j + stride // 2 + 1)

            center_region = image[center_start_i:center_end_i, center_start_j:center_end_j]
            center_binary = apply_threshold(center_region, window_threshold)

            # 결과에 반영 / Apply to result
            binary_image[center_start_i:center_end_i, center_start_j:center_end_j] = center_binary
            threshold_map[center_start_i:center_end_i, center_start_j:center_end_j] = window_threshold
            processed_mask[center_start_i:center_end_i, center_start_j:center_end_j] = True

            # 윈도우 정보 저장 (일부만) / Store window information (partial)
            if len(window_info) < 16:  # 최대 16개 윈도우 정보만 저장 / Store max 16 windows info
                window_info.append({
                    'center_position': (i, j),
                    'window_area': (start_i, end_i, start_j, end_j),
                    'threshold': window_threshold,
                    'histogram': window_hist,
                    'calc_info': window_calc_info
                })

    # 처리되지 않은 영역에 대해 전역 임계값 적용
    # Apply global threshold to unprocessed areas
    if not np.all(processed_mask):
        global_hist, _ = compute_histogram(image)
        global_threshold, _ = calculate_otsu_threshold(global_hist, show_process=False)

        unprocessed_mask = ~processed_mask
        binary_image[unprocessed_mask] = apply_threshold(image[unprocessed_mask], global_threshold)
        threshold_map[unprocessed_mask] = global_threshold

    # 처리 정보 저장 / Store processing information
    process_info = {
        'method': 'local_otsu_sliding_window',
        'window_size': window_size,
        'stride': stride,
        'threshold_map': threshold_map,
        'window_info': window_info,
        'num_windows': len(window_info),
        'processed_ratio': np.sum(processed_mask) / processed_mask.size
    }

    # 중간 과정 시각화 / Visualize intermediate process
    if show_process:
        visualize_local_otsu_process(image, binary_image, process_info)

    return binary_image, process_info

def compare_otsu_methods(image: np.ndarray, show_comparison: bool = True) -> dict:
    """
    다양한 Otsu 방법들을 비교합니다.
    Compare various Otsu methods.

    Args:
        image (np.ndarray): 입력 그레이스케일 이미지 / Input grayscale image
        show_comparison (bool): 비교 결과 표시 여부 / Whether to show comparison

    Returns:
        dict: 각 방법의 결과와 비교 정보 / Results and comparison info for each method
    """
    validate_image_input(image)

    # 각 방법 적용 / Apply each method
    global_result, global_info = global_otsu_thresholding(image, show_process=False)
    block_result, block_info = local_otsu_block_based(image, block_size=(32, 32), show_process=False)
    sliding_result, sliding_info = local_otsu_sliding_window(image, window_size=(32, 32), stride=8, show_process=False)

    # 비교 정보 생성 / Generate comparison information
    comparison_info = {
        'global_otsu': {
            'result': global_result,
            'info': global_info,
            'threshold': global_info['threshold']
        },
        'block_based': {
            'result': block_result,
            'info': block_info,
            'avg_threshold': np.mean(block_info['threshold_map'])
        },
        'sliding_window': {
            'result': sliding_result,
            'info': sliding_info,
            'avg_threshold': np.mean(sliding_info['threshold_map'])
        }
    }

    # 비교 시각화 / Visualize comparison
    if show_comparison:
        visualize_otsu_comparison(image, comparison_info)

    return comparison_info

def visualize_otsu_calculation(histogram: np.ndarray, calc_info: dict) -> None:
    """
    Otsu 계산 과정을 시각화합니다.
    Visualize Otsu calculation process.

    Args:
        histogram (np.ndarray): 입력 히스토그램 / Input histogram
        calc_info (dict): 계산 정보 / Calculation information
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 히스토그램 표시 / Display histogram
    axes[0].bar(range(256), histogram, alpha=0.7, color='blue')
    axes[0].axvline(x=calc_info['optimal_threshold'], color='red', linestyle='--', linewidth=2,
                    label=f'Optimal Threshold: {calc_info["optimal_threshold"]}')
    axes[0].set_title('Histogram')
    axes[0].set_xlabel('픽셀 강도 / Pixel Intensity')
    axes[0].set_ylabel('빈도수 / Frequency')
    axes[0].legend()

    # Inter-class variance 그래프 / Inter-class variance graph
    axes[1].plot(range(256), calc_info['variance_history'], 'g-', linewidth=2)
    axes[1].axvline(x=calc_info['optimal_threshold'], color='red', linestyle='--', linewidth=2)
    axes[1].axhline(y=calc_info['max_inter_class_variance'], color='red', linestyle=':', alpha=0.7)
    axes[1].set_title('Inter-Class Variance')
    axes[1].set_xlabel('임계값 / Threshold')
    axes[1].set_ylabel('Inter-Class Variance')
    axes[1].grid(True, alpha=0.3)

    # 수학적 원리 설명 / Mathematical principle explanation
    math_text = "Otsu 방법의 수학적 원리:\n\n"
    math_text += "1. Inter-class variance 최대화\n"
    math_text += "   σ²(between) = w₀ × w₁ × (μ₀ - μ₁)²\n\n"
    math_text += "2. Within-class variance 최소화\n"
    math_text += "   σ²(within) = w₀ × σ₀² + w₁ × σ₁²\n\n"
    math_text += "3. 수학적 관계\n"
    math_text += "   σ²(total) = σ²(within) + σ²(between)\n\n"
    math_text += f"최적 임계값: {calc_info['optimal_threshold']}\n"
    math_text += f"최대 Inter-class variance: {calc_info['max_inter_class_variance']:.4f}"

    axes[2].text(0.05, 0.95, math_text, transform=axes[2].transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    axes[2].set_title('Mathematical Principle')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

def visualize_global_otsu_process(original: np.ndarray, binary: np.ndarray, process_info: dict) -> None:
    """
    전역 Otsu 과정을 시각화합니다.
    Visualize global Otsu process.

    Args:
        original (np.ndarray): 원본 이미지 / Original image
        binary (np.ndarray): 이진 이미지 / Binary image
        process_info (dict): 처리 정보 / Processing information
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 원본 이미지 / Original image
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # 이진 이미지 / Binary image
    axes[0, 1].imshow(binary, cmap='gray')
    axes[0, 1].set_title(f'Global Otsu 결과\n(임계값: {process_info["threshold"]})')
    axes[0, 1].axis('off')

    # 히스토그램과 임계값 / Histogram and threshold
    axes[1, 0].bar(range(256), process_info['histogram'], alpha=0.7, color='blue')
    axes[1, 0].axvline(x=process_info['threshold'], color='red', linestyle='--', linewidth=2,
                      label=f'임계값: {process_info["threshold"]}')
    axes[1, 0].set_title('히스토그램\nHistogram')
    axes[1, 0].set_xlabel('픽셀 강도 / Pixel Intensity')
    axes[1, 0].set_ylabel('빈도수 / Frequency')
    axes[1, 0].legend()

    # Inter-class variance / Inter-class variance
    calc_info = process_info['calculation_info']
    axes[1, 1].plot(range(256), calc_info['variance_history'], 'g-', linewidth=2)
    axes[1, 1].axvline(x=process_info['threshold'], color='red', linestyle='--', linewidth=2)
    axes[1, 1].set_title('Inter-Class Variance')
    axes[1, 1].set_xlabel('임계값 / Threshold')
    axes[1, 1].set_ylabel('Variance')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def visualize_local_otsu_process(original: np.ndarray, binary: np.ndarray, process_info: dict) -> None:
    """
    Local Otsu 과정을 시각화합니다.
    Visualize local Otsu process.

    Args:
        original (np.ndarray): 원본 이미지 / Original image
        binary (np.ndarray): 이진 이미지 / Binary image
        process_info (dict): 처리 정보 / Processing information
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 원본 이미지 / Original image
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # 임계값 맵 / Threshold map
    threshold_map = process_info['threshold_map']
    im = axes[0, 1].imshow(threshold_map, cmap='jet')
    axes[0, 1].set_title('임계값 맵\nThreshold Map')
    axes[0, 1].axis('off')
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # 이진 결과 / Binary result
    axes[0, 2].imshow(binary, cmap='gray')
    axes[0, 2].set_title(f'{process_info["method"].replace("_", " ").title()}\n결과')
    axes[0, 2].axis('off')

    # 방법별 정보 / Method-specific information
    if 'block_size' in process_info:
        info_text = f"블록 기반 Local Otsu\n\n"
        info_text += f"블록 크기: {process_info['block_size']}\n"
        info_text += f"블록 수: {process_info['num_blocks']}\n"
        info_text += f"평균 임계값: {np.mean(threshold_map):.1f}\n"
        info_text += f"임계값 표준편차: {np.std(threshold_map):.1f}"
    elif 'window_size' in process_info:
        info_text = f"슬라이딩 윈도우 Local Otsu\n\n"
        info_text += f"윈도우 크기: {process_info['window_size']}\n"
        info_text += f"스트라이드: {process_info['stride']}\n"
        info_text += f"윈도우 수: {process_info['num_windows']}\n"
        info_text += f"처리 비율: {process_info['processed_ratio']:.1%}\n"
        info_text += f"평균 임계값: {np.mean(threshold_map):.1f}"

    axes[1, 0].text(0.05, 0.95, info_text, transform=axes[1, 0].transAxes, fontsize=11,
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    axes[1, 0].set_title('방법 정보\nMethod Information')
    axes[1, 0].axis('off')

    # 임계값 분포 히스토그램 / Threshold distribution histogram
    threshold_values = threshold_map.flatten()
    threshold_values = threshold_values[threshold_values > 0]  # 0이 아닌 값만 / Non-zero values only
    axes[1, 1].hist(threshold_values, bins=50, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 1].set_title('임계값 분포\nThreshold Distribution')
    axes[1, 1].set_xlabel('임계값 / Threshold Value')
    axes[1, 1].set_ylabel('빈도수 / Frequency')

    # 일부 블록/윈도우의 히스토그램 예시 / Example histograms of some blocks/windows
    if process_info.get('block_info') or process_info.get('window_info'):
        sample_info = process_info.get('block_info') or process_info.get('window_info')
        if sample_info:
            sample = sample_info[0]
            axes[1, 2].bar(range(256), sample['histogram'], alpha=0.7, color='purple')
            axes[1, 2].axvline(x=sample['threshold'], color='red', linestyle='--', linewidth=2)
            axes[1, 2].set_title(f'샘플 영역 히스토그램\n(임계값: {sample["threshold"]})')
            axes[1, 2].set_xlabel('픽셀 강도 / Pixel Intensity')
            axes[1, 2].set_ylabel('빈도수 / Frequency')

    plt.tight_layout()
    plt.show()

def visualize_otsu_comparison(original: np.ndarray, comparison_info: dict) -> None:
    """
    Otsu 방법들의 비교를 시각화합니다.
    Visualize comparison of Otsu methods.

    Args:
        original (np.ndarray): 원본 이미지 / Original image
        comparison_info (dict): 비교 정보 / Comparison information
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # 원본 이미지 / Original image
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # 각 방법의 결과 / Results of each method
    methods = ['global_otsu', 'block_based', 'sliding_window']
    titles = ['Global Otsu', 'Block-based Local Otsu', 'Sliding Window Local Otsu']

    for i, (method, title) in enumerate(zip(methods, titles)):
        result = comparison_info[method]['result']
        axes[0, i + 1].imshow(result, cmap='gray')

        if method == 'global_otsu':
            threshold_info = f"임계값: {comparison_info[method]['threshold']}"
        else:
            threshold_info = f"평균 임계값: {comparison_info[method]['avg_threshold']:.1f}"

        axes[0, i + 1].set_title(f'{title}\n{threshold_info}')
        axes[0, i + 1].axis('off')

    # 임계값 맵 비교 (Local 방법들만) / Threshold map comparison (Local methods only)
    axes[1, 0].text(0.1, 0.5, "비교 분석:\n\n• Global Otsu: 단일 임계값\n  - 계산 빠름\n  - 불균등 조명에 취약\n\n• Block-based: 블록별 임계값\n  - 지역적 적응\n  - 블록 경계 불연속\n\n• Sliding Window: 중첩 윈도우\n  - 부드러운 전환\n  - 계산 비용 높음",
                  transform=axes[1, 0].transAxes, fontsize=10,
                  verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
    axes[1, 0].set_title('방법 비교\nMethod Comparison')
    axes[1, 0].axis('off')

    # Block-based 임계값 맵 / Block-based threshold map
    if 'threshold_map' in comparison_info['block_based']['info']:
        threshold_map = comparison_info['block_based']['info']['threshold_map']
        im1 = axes[1, 1].imshow(threshold_map, cmap='jet')
        axes[1, 1].set_title('Block-based 임계값 맵')
        axes[1, 1].axis('off')
        plt.colorbar(im1, ax=axes[1, 1], fraction=0.046, pad=0.04)

    # Sliding window 임계값 맵 / Sliding window threshold map
    if 'threshold_map' in comparison_info['sliding_window']['info']:
        threshold_map = comparison_info['sliding_window']['info']['threshold_map']
        im2 = axes[1, 2].imshow(threshold_map, cmap='jet')
        axes[1, 2].set_title('Sliding Window 임계값 맵')
        axes[1, 2].axis('off')
        plt.colorbar(im2, ax=axes[1, 2], fraction=0.046, pad=0.04)

    # 임계값 분포 비교 / Threshold distribution comparison
    for i, method in enumerate(['block_based', 'sliding_window']):
        if 'threshold_map' in comparison_info[method]['info']:
            threshold_map = comparison_info[method]['info']['threshold_map']
            threshold_values = threshold_map.flatten()
            threshold_values = threshold_values[threshold_values > 0]
            axes[1, 3].hist(threshold_values, bins=30, alpha=0.5,
                           label=method.replace('_', ' ').title(), density=True)

    axes[1, 3].axvline(x=comparison_info['global_otsu']['threshold'],
                      color='red', linestyle='--', linewidth=2, label='Global Otsu')
    axes[1, 3].set_title('임계값 분포 비교\nThreshold Distribution Comparison')
    axes[1, 3].set_xlabel('임계값 / Threshold Value')
    axes[1, 3].set_ylabel('밀도 / Density')
    axes[1, 3].legend()

    plt.tight_layout()
    plt.show()

# OpenCV 내장 함수 사용 예시 (주석으로 표시) / Example usage of OpenCV built-in functions (commented)
def example_opencv_otsu_usage():
    """
    OpenCV를 사용한 Otsu Thresholding 예시입니다.
    Example of Otsu thresholding using OpenCV.
    """
    pass
    # Global Otsu thresholding:
    # ret, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Adaptive thresholding (mean-based):
    # thresh_mean = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
    #                                     cv2.THRESH_BINARY, 11, 2)

    # Adaptive thresholding (Gaussian-based):
    # thresh_gaussian = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                                         cv2.THRESH_BINARY, 11, 2)