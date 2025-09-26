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
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import morphology, filters
from .utils import compute_histogram, validate_image_input

def calculate_otsu_threshold(histogram: np.ndarray, show_process: bool = False) -> Tuple[float, dict]:
    """
    히스토그램으로부터 Otsu 임계값을 계산합니다.
    Calculate Otsu threshold from histogram with detailed calculation information.

    Args:
        histogram (np.ndarray): 히스토그램 배열 / Histogram array
        show_process (bool): 계산 과정 표시 여부 / Whether to show calculation process

    Returns:
        Tuple[float, dict]: (최적 임계값, 계산 정보) / (optimal threshold, calculation info)
    """
    threshold = compute_otsu_threshold(histogram)

    # 계산 정보 생성
    calc_info = {
        'optimal_threshold': threshold,
        'max_inter_class_variance': 0.0,
        'variance_history': np.zeros(256)
    }

    # Inter-class variance 히스토리 계산 (시각화용)
    total_pixels = np.sum(histogram)
    if total_pixels > 0:
        total_mean = np.sum(np.arange(256) * histogram) / total_pixels
        w0 = 0.0
        sum0 = 0.0

        for t in range(256):
            w0 += histogram[t] / total_pixels
            if w0 == 0 or w0 == 1:
                continue

            w1 = 1.0 - w0
            sum0 += t * histogram[t] / total_pixels

            mean0 = sum0 / w0 if w0 > 0 else 0
            mean1 = (total_mean - sum0) / w1 if w1 > 0 else 0

            between_variance = w0 * w1 * (mean0 - mean1) ** 2
            calc_info['variance_history'][t] = between_variance

            if between_variance > calc_info['max_inter_class_variance']:
                calc_info['max_inter_class_variance'] = between_variance

    return threshold, calc_info

def apply_threshold(image: np.ndarray, threshold: float, max_value: int = 255) -> np.ndarray:
    """
    임계값을 적용하여 이진 이미지를 생성합니다.
    Apply threshold to create binary image.

    Args:
        image (np.ndarray): 입력 이미지 / Input image
        threshold (float): 임계값 / Threshold value
        max_value (int): 최대값 / Maximum value

    Returns:
        np.ndarray: 이진 이미지 / Binary image
    """
    return np.where(image > threshold, max_value, 0).astype(np.uint8)

def calculate_adaptive_parameters(image: np.ndarray) -> Tuple[Tuple[int, int], int]:
    """
    이미지 크기에 따른 적응적 파라미터를 계산합니다.
    Calculate adaptive parameters based on image size.

    Args:
        image (np.ndarray): 입력 이미지 / Input image

    Returns:
        Tuple[Tuple[int, int], int]: (블록/윈도우 크기, 스트라이드) / (block/window size, stride)
    """
    height, width = image.shape
    image_size = height * width

    # 이미지 크기에 따른 적응적 블록 크기 계산
    if image_size < 50000:  # 작은 이미지
        block_size = (16, 16)
        stride = 4
    elif image_size < 200000:  # 중간 이미지
        block_size = (32, 32)
        stride = 8
    else:  # 큰 이미지
        block_size = (64, 64)
        stride = 16

    return block_size, stride

def apply_boundary_smoothing(threshold_map: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    임계값 맵에 경계 스무딩을 적용합니다.
    Apply boundary smoothing to threshold map.

    Args:
        threshold_map (np.ndarray): 임계값 맵 / Threshold map
        sigma (float): 가우시안 시그마 / Gaussian sigma

    Returns:
        np.ndarray: 스무딩된 임계값 맵 / Smoothed threshold map
    """
    return cv2.GaussianBlur(threshold_map.astype(np.float32), (0, 0), sigma)

def apply_morphological_postprocessing(binary_image: np.ndarray,
                                     remove_small: bool = True,
                                     min_size: int = 10,
                                     apply_opening: bool = False,
                                     apply_closing: bool = False,
                                     kernel_size: int = 3) -> np.ndarray:
    """
    형태학적 후처리를 적용합니다.
    Apply morphological post-processing.

    Args:
        binary_image (np.ndarray): 이진 이미지 / Binary image
        remove_small (bool): 작은 객체 제거 여부 / Whether to remove small objects
        min_size (int): 최소 객체 크기 / Minimum object size
        apply_opening (bool): 열림 연산 적용 여부 / Whether to apply opening
        apply_closing (bool): 닫힘 연산 적용 여부 / Whether to apply closing
        kernel_size (int): 커널 크기 / Kernel size

    Returns:
        np.ndarray: 후처리된 이진 이미지 / Post-processed binary image
    """
    result = binary_image.copy()

    # 작은 객체 제거
    if remove_small and min_size > 0:
        # 연결된 컴포넌트 분석
        num_labels, labels = cv2.connectedComponents(result)
        for label in range(1, num_labels):
            if np.sum(labels == label) < min_size:
                result[labels == label] = 0

    # 형태학적 연산
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    if apply_opening:
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)

    if apply_closing:
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)

    return result

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

def local_otsu_block_opencv(image: np.ndarray,
                           block_size: int = 32,
                           show_process: bool = True) -> Tuple[np.ndarray, dict]:
    """
    OpenCV 적응적 임계값을 사용한 블록 기반 처리 (격자 아티팩트 없음)
    Block-based processing using OpenCV adaptive thresholding (no grid artifacts)

    OpenCV의 ADAPTIVE_THRESH_MEAN과 ADAPTIVE_THRESH_GAUSSIAN을 사용하여
    블록 경계의 아티팩트 없이 지역적 임계값을 적용합니다.

    Uses OpenCV's ADAPTIVE_THRESH_MEAN and ADAPTIVE_THRESH_GAUSSIAN to apply
    local thresholding without block boundary artifacts.

    Args:
        image (np.ndarray): 입력 그레이스케일 이미지 / Input grayscale image
        block_size (int): 적응적 임계값 블록 크기 / Adaptive threshold block size
        show_process (bool): 중간 과정 표시 여부 / Whether to show intermediate process

    Returns:
        Tuple[np.ndarray, dict]: (이진 이미지, 처리 정보) / (binary image, processing info)
    """
    validate_image_input(image)

    if len(image.shape) != 2:
        raise ValueError("그레이스케일 이미지가 필요합니다 / Grayscale image required")

    # OpenCV는 홀수 블록 크기만 허용 / OpenCV only allows odd block sizes
    if block_size % 2 == 0:
        block_size += 1

    # OpenCV 적응적 임계값 처리 (가우시안 가중 평균)
    # OpenCV adaptive thresholding with Gaussian weighted mean
    binary_image = cv2.adaptiveThreshold(
        image,
        255,                                    # 최대값 / Maximum value
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,        # 가우시안 가중 평균 / Gaussian weighted mean
        cv2.THRESH_BINARY,                     # 이진화 타입 / Binary type
        block_size,                            # 블록 크기 / Block size
        2                                      # 상수 C / Constant C
    )

    # 처리 정보 저장 / Store processing information
    process_info = {
        'method': 'opencv_adaptive_block',
        'block_size': block_size,
        'adaptive_method': 'ADAPTIVE_THRESH_GAUSSIAN_C',
        'threshold_type': 'THRESH_BINARY',
        'constant_c': 2,
        'grid_artifacts': 'None (OpenCV interpolation)'
    }

    # 중간 과정 시각화 / Visualize intermediate process
    if show_process:
        visualize_opencv_adaptive_process(image, binary_image, process_info, "Block-based")

    return binary_image, process_info

def local_otsu_sliding_opencv(image: np.ndarray,
                             block_size: int = 32,
                             show_process: bool = True) -> Tuple[np.ndarray, dict]:
    """
    OpenCV 적응적 임계값을 사용한 슬라이딩 윈도우 기반 처리 (격자 아티팩트 없음)
    Sliding window-based processing using OpenCV adaptive thresholding (no grid artifacts)

    OpenCV의 적응적 임계값 처리는 내부적으로 슬라이딩 윈도우와 유사한 방식으로
    각 픽셀 주변의 지역적 정보를 사용하여 임계값을 결정합니다.

    OpenCV's adaptive thresholding internally uses a sliding window-like approach
    to determine threshold values using local information around each pixel.

    Args:
        image (np.ndarray): 입력 그레이스케일 이미지 / Input grayscale image
        block_size (int): 적응적 임계값 윈도우 크기 / Adaptive threshold window size
        show_process (bool): 중간 과정 표시 여부 / Whether to show intermediate process

    Returns:
        Tuple[np.ndarray, dict]: (이진 이미지, 처리 정보) / (binary image, processing info)
    """
    validate_image_input(image)

    if len(image.shape) != 2:
        raise ValueError("그레이스케일 이미지가 필요합니다 / Grayscale image required")

    # OpenCV는 홀수 블록 크기만 허용 / OpenCV only allows odd block sizes
    if block_size % 2 == 0:
        block_size += 1

    # OpenCV 적응적 임계값 처리 (평균 기반)
    # OpenCV adaptive thresholding with mean-based approach
    binary_image = cv2.adaptiveThreshold(
        image,
        255,                                    # 최대값 / Maximum value
        cv2.ADAPTIVE_THRESH_MEAN_C,            # 평균 기반 / Mean-based
        cv2.THRESH_BINARY,                     # 이진화 타입 / Binary type
        block_size,                            # 윈도우 크기 / Window size
        5                                      # 상수 C / Constant C
    )

    # 처리 정보 저장 / Store processing information
    process_info = {
        'method': 'opencv_adaptive_sliding',
        'window_size': block_size,
        'adaptive_method': 'ADAPTIVE_THRESH_MEAN_C',
        'threshold_type': 'THRESH_BINARY',
        'constant_c': 5,
        'grid_artifacts': 'None (OpenCV sliding window)'
    }

    # 중간 과정 시각화 / Visualize intermediate process
    if show_process:
        visualize_opencv_adaptive_process(image, binary_image, process_info, "Sliding window-based")

    return binary_image, process_info

def local_otsu_block_based(image: np.ndarray,
                          block_size: Optional[Tuple[int, int]] = None,
                          show_process: bool = True,
                          adaptive_params: bool = True,
                          apply_smoothing: bool = True,
                          smoothing_sigma: float = 1.0,
                          apply_postprocessing: bool = True,
                          postprocess_params: Optional[dict] = None,
                          improved_boundary: bool = False) -> Tuple[np.ndarray, dict]:
    """
    블록 기반 Local Otsu Thresholding을 수행합니다. (개선된 버전)
    Perform block-based Local Otsu Thresholding (Enhanced version).

    블록 기반 방법의 특징:
    - 이미지를 균등한 크기의 블록으로 분할
    - 각 블록마다 독립적으로 Otsu 임계값 계산
    - 계산 효율성이 높음
    - [NEW] 블록 경계 아티팩트 감소를 위한 가우시안 스무딩
    - [NEW] 적응적 파라미터 설정
    - [NEW] 형태학적 후처리

    Characteristics of block-based method:
    - Divide image into equal-sized blocks
    - Calculate Otsu threshold independently for each block
    - High computational efficiency
    - [NEW] Gaussian smoothing to reduce block boundary artifacts
    - [NEW] Adaptive parameter setting
    - [NEW] Morphological post-processing

    Args:
        image (np.ndarray): 입력 그레이스케일 이미지 / Input grayscale image
        block_size (Optional[Tuple[int, int]]): 블록 크기 (None이면 적응적 계산) / Block size (adaptive if None)
        show_process (bool): 중간 과정 표시 여부 / Whether to show intermediate process
        adaptive_params (bool): 적응적 파라미터 사용 여부 / Whether to use adaptive parameters
        apply_smoothing (bool): 경계 스무딩 적용 여부 / Whether to apply boundary smoothing
        smoothing_sigma (float): 가우시안 스무딩 표준편차 / Gaussian smoothing sigma
        apply_postprocessing (bool): 후처리 적용 여부 / Whether to apply post-processing
        postprocess_params (Optional[dict]): 후처리 파라미터 / Post-processing parameters

    Returns:
        Tuple[np.ndarray, dict]: (이진 이미지, 처리 정보) / (binary image, processing info)
    """
    validate_image_input(image)

    if len(image.shape) != 2:
        raise ValueError("그레이스케일 이미지가 필요합니다 / Grayscale image required")

    height, width = image.shape

    # 적응적 파라미터 설정 / Adaptive parameter setting
    if adaptive_params and block_size is None:
        block_size, _ = calculate_adaptive_parameters(image)
    elif block_size is None:
        block_size = (32, 32)  # 기본값 / Default value

    block_h, block_w = block_size

    # 후처리 파라미터 설정 / Set post-processing parameters
    if postprocess_params is None:
        postprocess_params = {
            'remove_small': True,
            'min_size': max(5, (height * width) // 50000),  # 텍스트를 위해 더 작은 값 / Smaller value for text
            'apply_opening': False,  # 텍스트 세부사항 보존 / Preserve text details
            'apply_closing': False,  # 텍스트 연결성 보존 / Preserve text connectivity
            'kernel_size': 2  # 더 작은 커널 / Smaller kernel
        }

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

    # [NEW] 경계 아티팩트 처리 / Boundary artifact processing
    original_threshold_map = threshold_map.copy()
    if apply_smoothing:
        threshold_map = apply_boundary_smoothing(threshold_map, sigma=smoothing_sigma)
        # 스무딩된 임계값으로 다시 이진화 / Re-binarize with smoothed thresholds
        binary_image = np.zeros_like(image)
        for i in range(height):
            for j in range(width):
                binary_image[i, j] = 255 if image[i, j] > threshold_map[i, j] else 0

    # [NEW] 형태학적 후처리 / Morphological post-processing
    if apply_postprocessing:
        binary_image = apply_morphological_postprocessing(binary_image, **postprocess_params)

    # 처리 정보 저장 / Store processing information
    process_info = {
        'method': 'local_otsu_block_based_enhanced',
        'block_size': block_size,
        'threshold_map': threshold_map,
        'original_threshold_map': original_threshold_map,
        'block_info': block_info,
        'num_blocks': len(block_info),
        'adaptive_params': adaptive_params,
        'apply_smoothing': apply_smoothing,
        'smoothing_sigma': smoothing_sigma,
        'apply_postprocessing': apply_postprocessing,
        'postprocess_params': postprocess_params
    }

    # 중간 과정 시각화 / Visualize intermediate process
    if show_process:
        visualize_local_otsu_process(image, binary_image, process_info)

    return binary_image, process_info

def local_otsu_sliding_window(image: np.ndarray,
                             window_size: Optional[Tuple[int, int]] = None,
                             stride: Optional[int] = None,
                             show_process: bool = True,
                             adaptive_params: bool = True,
                             apply_postprocessing: bool = True,
                             postprocess_params: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
    """
    슬라이딩 윈도우 기반 Local Otsu Thresholding을 수행합니다. (개선된 버전)
    Perform sliding window-based Local Otsu Thresholding (Enhanced version).

    슬라이딩 윈도우 방법의 특징:
    - 지정된 스트라이드로 윈도우를 이동하며 처리
    - 중앙 픽셀에 대한 임계값을 윈도우 영역에서 계산
    - 더 부드러운 결과, 하지만 계산 비용이 높음
    - 윈도우 겹침으로 인한 더 나은 연속성
    - [NEW] 적응적 파라미터 설정
    - [NEW] 형태학적 후처리

    Characteristics of sliding window method:
    - Process by moving window with specified stride
    - Calculate threshold for center pixel from window area
    - Smoother results but higher computational cost
    - Better continuity due to window overlap
    - [NEW] Adaptive parameter setting
    - [NEW] Morphological post-processing

    Args:
        image (np.ndarray): 입력 그레이스케일 이미지 / Input grayscale image
        window_size (Optional[Tuple[int, int]]): 윈도우 크기 (None이면 적응적 계산) / Window size (adaptive if None)
        stride (Optional[int]): 스트라이드 (None이면 적응적 계산) / Stride (adaptive if None)
        show_process (bool): 중간 과정 표시 여부 / Whether to show intermediate process
        adaptive_params (bool): 적응적 파라미터 사용 여부 / Whether to use adaptive parameters
        apply_postprocessing (bool): 후처리 적용 여부 / Whether to apply post-processing
        postprocess_params (Optional[dict]): 후처리 파라미터 / Post-processing parameters

    Returns:
        Tuple[np.ndarray, dict]: (이진 이미지, 처리 정보) / (binary image, processing info)
    """
    validate_image_input(image)

    if len(image.shape) != 2:
        raise ValueError("그레이스케일 이미지가 필요합니다 / Grayscale image required")

    height, width = image.shape

    # 적응적 파라미터 설정 / Adaptive parameter setting
    if adaptive_params and (window_size is None or stride is None):
        adaptive_window_size, adaptive_stride = calculate_adaptive_parameters(image)
        window_size = window_size or adaptive_window_size
        stride = stride or adaptive_stride
    else:
        window_size = window_size or (32, 32)  # 기본값 / Default value
        stride = stride or 8  # 기본값 / Default value

    window_h, window_w = window_size
    half_h, half_w = window_h // 2, window_w // 2

    # 후처리 파라미터 설정 / Set post-processing parameters
    if postprocess_params is None:
        postprocess_params = {
            'remove_small': True,
            'min_size': max(5, (height * width) // 50000),  # 텍스트를 위해 더 작은 값 / Smaller value for text
            'apply_opening': False,  # 텍스트 세부사항 보존 / Preserve text details
            'apply_closing': False,  # 텍스트 연결성 보존 / Preserve text connectivity
            'kernel_size': 2  # 더 작은 커널 / Smaller kernel
        }

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

    # [NEW] 형태학적 후처리 / Morphological post-processing
    if apply_postprocessing:
        binary_image = apply_morphological_postprocessing(binary_image, **postprocess_params)

    # 처리 정보 저장 / Store processing information
    process_info = {
        'method': 'local_otsu_sliding_window_enhanced',
        'window_size': window_size,
        'stride': stride,
        'threshold_map': threshold_map,
        'window_info': window_info,
        'num_windows': len(window_info),
        'processed_ratio': np.sum(processed_mask) / processed_mask.size,
        'adaptive_params': adaptive_params,
        'apply_postprocessing': apply_postprocessing,
        'postprocess_params': postprocess_params
    }

    # 중간 과정 시각화 / Visualize intermediate process
    if show_process:
        visualize_local_otsu_process(image, binary_image, process_info)

    return binary_image, process_info

def local_otsu_adaptive_block(image: np.ndarray,
                             block_size: int = 32,
                             show_process: bool = True) -> Tuple[np.ndarray, dict]:
    """
    직접 구현한 적응적 블록 기반 Local Otsu Thresholding
    Direct implementation of adaptive block-based Local Otsu Thresholding

    Args:
        image (np.ndarray): 입력 그레이스케일 이미지 / Input grayscale image
        block_size (int): 블록 크기 / Block size
        show_process (bool): 중간 과정 표시 여부 / Whether to show intermediate process

    Returns:
        Tuple[np.ndarray, dict]: (이진 이미지, 처리 정보) / (binary image, processing info)
    """
    return local_otsu_block_based(
        image,
        block_size=(block_size, block_size),
        show_process=show_process,
        adaptive_params=False,
        apply_smoothing=False,
        apply_postprocessing=False
    )

def local_otsu_adaptive_sliding(image: np.ndarray,
                               window_size: int = 32,
                               stride: int = 8,
                               show_process: bool = True) -> Tuple[np.ndarray, dict]:
    """
    직접 구현한 적응적 슬라이딩 윈도우 기반 Local Otsu Thresholding
    Direct implementation of adaptive sliding window-based Local Otsu Thresholding

    Args:
        image (np.ndarray): 입력 그레이스케일 이미지 / Input grayscale image
        window_size (int): 윈도우 크기 / Window size
        stride (int): 스트라이드 / Stride
        show_process (bool): 중간 과정 표시 여부 / Whether to show intermediate process

    Returns:
        Tuple[np.ndarray, dict]: (이진 이미지, 처리 정보) / (binary image, processing info)
    """
    return local_otsu_sliding_window(
        image,
        window_size=(window_size, window_size),
        stride=stride,
        show_process=show_process,
        adaptive_params=False,
        apply_postprocessing=False
    )

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

    # 직접 구현한 방법들 / Direct implementations
    block_result, block_info = local_otsu_adaptive_block(image, block_size=32, show_process=False)
    sliding_result, sliding_info = local_otsu_adaptive_sliding(image, window_size=32, stride=8, show_process=False)

    # OpenCV 기반 방법들 / OpenCV-based methods
    block_opencv_result, block_opencv_info = local_otsu_block_opencv(image, block_size=32, show_process=False)
    sliding_opencv_result, sliding_opencv_info = local_otsu_sliding_opencv(image, block_size=32, show_process=False)

    # 개선된 방법 / Improved method
    improved_result, improved_info = local_otsu_improved_boundary(image, block_size=(32, 32), show_process=False)

    # 비교 정보 생성 / Generate comparison information
    comparison_info = {
        'global_otsu': {
            'result': global_result,
            'info': global_info,
            'threshold': global_info['threshold'],
            'method': 'Global Otsu'
        },
        'adaptive_block': {
            'result': block_result,
            'info': block_info,
            'method': 'Direct Block-based'
        },
        'adaptive_sliding': {
            'result': sliding_result,
            'info': sliding_info,
            'method': 'Direct Sliding Window'
        },
        'block_opencv': {
            'result': block_opencv_result,
            'info': block_opencv_info,
            'method': 'OpenCV Adaptive Block'
        },
        'sliding_opencv': {
            'result': sliding_opencv_result,
            'info': sliding_opencv_info,
            'method': 'OpenCV Adaptive Sliding'
        },
        'improved': {
            'result': improved_result,
            'info': improved_info,
            'method': 'Improved Boundary'
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

def visualize_otsu_comparison(original: np.ndarray, comparison_info: dict,
                             save_figure: bool = False, save_path: str = None) -> None:
    """
    Otsu 방법들의 비교를 시각화합니다.
    Visualize comparison of Otsu methods.

    Args:
        original (np.ndarray): 원본 이미지 / Original image
        comparison_info (dict): 비교 정보 / Comparison information
        save_figure (bool): figure를 저장할지 여부 / Whether to save figure
        save_path (str): 저장할 파일 경로 / File path to save
    """
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))

    # 원본 이미지 / Original image
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # 각 방법의 결과 / Results of each method
    methods = ['global_otsu', 'adaptive_block', 'adaptive_sliding',
               'block_opencv', 'sliding_opencv', 'improved']
    titles = ['Global Otsu', 'Direct Block', 'Direct Sliding',
              'OpenCV Block', 'OpenCV Sliding', 'Improved Boundary']

    for i, (method, title) in enumerate(zip(methods, titles)):
        if i < 6:  # 6개 방법 표시
            row = (i + 1) // 3
            col = (i + 1) % 3

            result = comparison_info[method]['result']
            axes[row, col].imshow(result, cmap='gray')

            if method == 'global_otsu':
                threshold_info = f"임계값: {comparison_info[method]['threshold']}"
            else:
                threshold_info = comparison_info[method].get('method', method)

            axes[row, col].set_title(f'{title}\n{threshold_info}')
            axes[row, col].axis('off')

    # 임계값 맵들 표시 / Show threshold maps
    threshold_map_methods = ['adaptive_block', 'adaptive_sliding', 'improved']
    threshold_map_titles = ['Direct Block 임계값 맵', 'Direct Sliding 임계값 맵', 'Improved 임계값 맵']

    for i, (method, title) in enumerate(zip(threshold_map_methods, threshold_map_titles)):
        row, col = 2, i
        if method in comparison_info and 'threshold_map' in comparison_info[method]['info']:
            threshold_map = comparison_info[method]['info']['threshold_map']
            im = axes[row, col].imshow(threshold_map, cmap='jet')
            axes[row, col].set_title(title)
            axes[row, col].axis('off')
            plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
        else:
            axes[row, col].text(0.5, 0.5, f'{method}\n임계값 맵 없음',
                               transform=axes[row, col].transAxes, ha='center', va='center')
            axes[row, col].set_title(title)
            axes[row, col].axis('off')

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

def local_otsu_improved_boundary(image: np.ndarray,
                               block_size: Tuple[int, int] = (32, 32),
                               overlap_ratio: float = 0.5,
                               blend_method: str = 'weighted_average',
                               show_process: bool = True) -> Tuple[np.ndarray, dict]:
    """
    개선된 경계 처리를 사용한 Local Otsu Thresholding
    Local Otsu Thresholding with improved boundary processing

    이 방법은 겹치는 블록을 사용하여 블록 경계 아티팩트를 크게 감소시킵니다.
    This method uses overlapping blocks to significantly reduce block boundary artifacts.

    Args:
        image (np.ndarray): 입력 그레이스케일 이미지 / Input grayscale image
        block_size (Tuple[int, int]): 블록 크기 / Block size
        overlap_ratio (float): 겹침 비율 / Overlap ratio
        blend_method (str): 블렌딩 방법 / Blending method
        show_process (bool): 중간 과정 표시 여부 / Whether to show intermediate process

    Returns:
        Tuple[np.ndarray, dict]: (이진 이미지, 처리 정보) / (binary image, processing info)
    """
    validate_image_input(image)

    if len(image.shape) != 2:
        raise ValueError("그레이스케일 이미지가 필요합니다 / Grayscale image required")

    height, width = image.shape
    block_h, block_w = block_size

    # 겹침 계산 / Calculate overlap
    overlap_h = int(block_h * overlap_ratio)
    overlap_w = int(block_w * overlap_ratio)
    stride_h = block_h - overlap_h
    stride_w = block_w - overlap_w

    # 결과 이미지와 가중치 맵 초기화 / Initialize result image and weight map
    binary_image = np.zeros_like(image, dtype=np.float32)
    weight_map = np.zeros_like(image, dtype=np.float32)
    threshold_map = np.zeros_like(image, dtype=np.float32)

    # 블록 처리 / Process blocks
    block_info = []
    for i in range(0, height - block_h + 1, stride_h):
        for j in range(0, width - block_w + 1, stride_w):
            # 현재 블록 추출 / Extract current block
            end_i = min(i + block_h, height)
            end_j = min(j + block_w, width)
            block = image[i:end_i, j:end_j]

            # 블록 히스토그램 계산 / Calculate block histogram
            block_hist, _ = compute_histogram(block)

            # 블록 임계값 계산 / Calculate block threshold
            if np.sum(block_hist) > 0:
                block_threshold, block_calc_info = calculate_otsu_threshold(block_hist, show_process=False)
            else:
                block_threshold = 127
                block_calc_info = {'message': 'Empty block'}

            # 블록 이진화 / Binarize block
            block_binary = apply_threshold(block, block_threshold)

            # 가중치를 이용한 블렌딩 / Weighted blending
            binary_image[i:end_i, j:end_j] += block_binary.astype(np.float32)
            weight_map[i:end_i, j:end_j] += 1.0
            threshold_map[i:end_i, j:end_j] += block_threshold

            # 블록 정보 저장 (일부만) / Store block information (partial)
            if len(block_info) < 9:
                block_info.append({
                    'position': (i, j),
                    'size': (end_i - i, end_j - j),
                    'threshold': block_threshold,
                    'histogram': block_hist,
                    'calc_info': block_calc_info
                })

    # 가중 평균으로 최종 결과 계산 / Calculate final result with weighted average
    binary_image = np.divide(binary_image, weight_map, out=np.zeros_like(binary_image), where=(weight_map != 0))
    threshold_map = np.divide(threshold_map, weight_map, out=np.full_like(threshold_map, 127.0), where=(weight_map != 0))

    # 이진화 / Binarization
    binary_image = (binary_image > 127.5).astype(np.uint8) * 255

    # 처리 정보 저장 / Store processing information
    process_info = {
        'method': 'local_otsu_improved_boundary',
        'block_size': block_size,
        'overlap_ratio': overlap_ratio,
        'blend_method': blend_method,
        'threshold_map': threshold_map,
        'block_info': block_info,
        'num_blocks': len(block_info)
    }

    # 중간 과정 시각화 / Visualize intermediate process
    if show_process:
        visualize_local_otsu_process(image, binary_image, process_info)

    return binary_image, process_info

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


def visualize_opencv_adaptive_process(original: np.ndarray, binary: np.ndarray, process_info: dict, method_name: str) -> None:
    """
    OpenCV 적응적 임계값 처리 과정을 시각화합니다.
    Visualize OpenCV adaptive thresholding process.

    Args:
        original (np.ndarray): 원본 이미지 / Original image
        binary (np.ndarray): 이진화된 이미지 / Binary image
        process_info (dict): 처리 정보 / Processing information
        method_name (str): 방법 이름 / Method name
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 원본 이미지 / Original image
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # 이진화 결과 / Binary result
    axes[1].imshow(binary, cmap='gray')
    axes[1].set_title(f'OpenCV Adaptive Threshold\n({method_name})')
    axes[1].axis('off')

    # 처리 정보 표시 / Display processing information
    info_text = f"OpenCV 적응적 임계값 / OpenCV Adaptive Threshold\n\n"
    info_text += f"방법 / Method: {process_info['adaptive_method']}\n"
    info_text += f"블록/윈도우 크기 / Block/Window Size: {process_info.get('block_size', process_info.get('window_size'))}\n"
    info_text += f"상수 C / Constant C: {process_info.get('constant_c')}\n"
    info_text += f"격자 아티팩트 / Grid Artifacts: {process_info['grid_artifacts']}\n\n"
    info_text += f"장점 / Advantages:\n"
    info_text += f"- 격자 경계 아티팩트 없음 / No grid boundary artifacts\n"
    info_text += f"- 최적화된 성능 / Optimized performance\n"
    info_text += f"- 부드러운 경계 처리 / Smooth boundary handling"

    axes[2].text(0.1, 0.5, info_text, transform=axes[2].transAxes, fontsize=10,
                verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    axes[2].set_title('OpenCV 처리 정보 / OpenCV Processing Info')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()
