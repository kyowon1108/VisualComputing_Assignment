"""
컬러 이미지 히스토그램 평활화 모듈
Color Image Histogram Equalization Module

이 모듈은 컬러 이미지에 대한 히스토그램 평활화를 직접 구현합니다.
This module directly implements histogram equalization for color images.

주요 기능 / Key Features:
1. 직관적인 low-level 히스토그램 평활화 구현 / Intuitive low-level histogram equalization implementation
2. 다양한 색공간을 이용한 컬러 이미지 처리 / Color image processing using various color spaces
3. 단계별 중간 과정 시각화 / Step-by-step intermediate process visualization
4. AHE/CLAHE (Adaptive Histogram Equalization) 구현 / AHE/CLAHE implementation
"""

import numpy as np
import matplotlib.pyplot as plt
from .utils import (rgb_to_yuv, yuv_to_rgb, rgb_to_ycbcr, ycbcr_to_rgb,
                   rgb_to_lab, lab_to_rgb, rgb_to_hsv, hsv_to_rgb,
                   compute_histogram, validate_image_input)
from typing import Tuple, Optional, Dict, Any
import cv2
from skimage import filters
from scipy import ndimage

def calculate_cdf(histogram: np.ndarray) -> np.ndarray:
    """
    히스토그램으로부터 누적분포함수(CDF)를 계산합니다.
    Calculate Cumulative Distribution Function (CDF) from histogram.

    CDF의 물리적 의미:
    - 특정 픽셀값 이하의 픽셀들이 전체에서 차지하는 비율
    - 히스토그램 평활화에서 새로운 픽셀값으로의 매핑 함수 역할

    Physical meaning of CDF:
    - Proportion of pixels below a certain pixel value in the total
    - Acts as mapping function to new pixel values in histogram equalization

    Args:
        histogram (np.ndarray): 입력 히스토그램 / Input histogram

    Returns:
        np.ndarray: 정규화된 CDF 값 (0-1 범위) / Normalized CDF values (0-1 range)
    """
    # 누적 합계 계산 / Calculate cumulative sum
    cdf = np.cumsum(histogram)

    # 0으로 나누기 방지 / Prevent division by zero
    if cdf[-1] == 0:
        return cdf

    # 정규화 (0-1 범위로) / Normalize to 0-1 range
    cdf_normalized = cdf / cdf[-1]
    return cdf_normalized

def histogram_equalization(image: np.ndarray, bins: int = 256) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    단일 채널 이미지에 대한 히스토그램 평활화를 수행합니다.
    Perform histogram equalization on single channel image.

    히스토그램 평활화 원리:
    1. 입력 이미지의 히스토그램과 CDF를 계산
    2. CDF를 이용하여 픽셀값을 새로운 값으로 매핑
    3. 결과적으로 히스토그램이 균등하게 분포됨

    Histogram Equalization Principle:
    1. Calculate histogram and CDF of input image
    2. Map pixel values to new values using CDF
    3. Results in uniform distribution of histogram

    Args:
        image (np.ndarray): 입력 이미지 (2D 배열) / Input image (2D array)
        bins (int): 히스토그램 빈 개수 / Number of histogram bins

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (평활화된 이미지, 원본 히스토그램, 평활화된 히스토그램)
                                                 / (Equalized image, Original histogram, Equalized histogram)
    """
    # 입력 검증 / Input validation
    if len(image.shape) != 2:
        raise ValueError("입력 이미지는 2D 배열이어야 합니다 / Input image must be 2D array")

    # 히스토그램 계산 / Calculate histogram
    hist_original, _ = compute_histogram(image, bins)

    # CDF 계산 / Calculate CDF
    cdf = calculate_cdf(hist_original)

    # 매핑 테이블 생성 (CDF를 0-255 범위로 스케일링) / Create mapping table (scale CDF to 0-255 range)
    mapping_table = np.round(cdf * (bins - 1)).astype(np.uint8)

    # 이미지에 매핑 적용 / Apply mapping to image
    equalized_image = mapping_table[image.astype(int)]

    # 평활화된 이미지의 히스토그램 계산 / Calculate histogram of equalized image
    hist_equalized, _ = compute_histogram(equalized_image, bins)

    return equalized_image, hist_original, hist_equalized

def apply_clahe_to_channel(channel: np.ndarray, clip_limit: float = 2.0,
                          tile_grid_size: Tuple[int, int] = (8, 8),
                          bins: int = 256) -> np.ndarray:
    """
    단일 채널에 CLAHE를 적용합니다.
    Apply CLAHE to single channel.

    Args:
        channel: 입력 채널
        clip_limit: 클리핑 제한값
        tile_grid_size: 타일 그리드 크기
        bins: 히스토그램 빈 개수

    Returns:
        CLAHE가 적용된 채널
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(channel.astype(np.uint8))

def bilinear_interpolation(tile_values: np.ndarray, x: float, y: float) -> float:
    """
    양선형 보간을 수행합니다.
    Perform bilinear interpolation.
    """
    x1, y1 = int(x), int(y)
    x2, y2 = min(x1 + 1, tile_values.shape[1] - 1), min(y1 + 1, tile_values.shape[0] - 1)

    if x1 == x2 and y1 == y2:
        return tile_values[y1, x1]
    elif x1 == x2:
        return tile_values[y1, x1] * (y2 - y) + tile_values[y2, x1] * (y - y1)
    elif y1 == y2:
        return tile_values[y1, x1] * (x2 - x) + tile_values[y1, x2] * (x - x1)
    else:
        return (tile_values[y1, x1] * (x2 - x) * (y2 - y) +
                tile_values[y1, x2] * (x - x1) * (y2 - y) +
                tile_values[y2, x1] * (x2 - x) * (y - y1) +
                tile_values[y2, x2] * (x - x1) * (y - y1))

def apply_ahe_to_channel(channel: np.ndarray, tile_grid_size: Tuple[int, int] = (8, 8),
                        bins: int = 256, border: str = "reflect") -> np.ndarray:
    """
    단일 채널에 AHE (Adaptive Histogram Equalization)를 적용합니다.
    Apply AHE to single channel.

    Args:
        channel: 입력 채널
        tile_grid_size: 타일 그리드 크기
        bins: 히스토그램 빈 개수
        border: 경계 처리 방법

    Returns:
        AHE가 적용된 채널
    """
    h, w = channel.shape
    tile_h, tile_w = tile_grid_size

    # 타일 크기 계산
    step_h = h // tile_h
    step_w = w // tile_w

    # 각 타일의 CDF 계산
    tile_cdfs = np.zeros((tile_h, tile_w, bins))

    for i in range(tile_h):
        for j in range(tile_w):
            # 타일 영역 추출
            y_start = i * step_h
            y_end = min((i + 1) * step_h, h)
            x_start = j * step_w
            x_end = min((j + 1) * step_w, w)

            tile = channel[y_start:y_end, x_start:x_end]

            # 타일의 히스토그램과 CDF 계산
            tile_hist, _ = compute_histogram(tile, bins)
            tile_cdf = calculate_cdf(tile_hist)
            tile_cdfs[i, j] = tile_cdf

    # 결과 이미지 초기화
    result = np.zeros_like(channel)

    # 각 픽셀에 대해 양선형 보간으로 값 계산
    for y in range(h):
        for x in range(w):
            # 타일 좌표 계산
            tile_y = min((y / step_h), tile_h - 1)
            tile_x = min((x / step_w), tile_w - 1)

            # 픽셀값
            pixel_val = channel[y, x]

            # 4개 인접 타일의 CDF 값을 양선형 보간
            tile_values = tile_cdfs[:, :, pixel_val]
            interpolated_cdf = bilinear_interpolation(tile_values, tile_x, tile_y)

            # 결과값 계산
            result[y, x] = np.round(interpolated_cdf * (bins - 1)).astype(np.uint8)

    return result

def he_luma_bgr(img_bgr: np.ndarray,
                space: str = "yuv",
                mode: str = "global",
                tile: Tuple[int, int] = (8, 8),
                clip: Optional[float] = None,
                bins: int = 256,
                border: str = "reflect") -> Dict[str, Any]:
    """
    BGR 이미지에 휘도 기반 히스토그램 평활화를 적용합니다.
    Apply luminance-based histogram equalization to BGR image.

    Args:
        img_bgr: 입력 BGR 이미지
        space: 색공간 ("yuv"|"lab"|"rgb"|"ycbcr"|"hsv")
        mode: 모드 ("global"|"ahe"|"clahe")
        tile: AHE/CLAHE 타일 크기
        clip: CLAHE 클리핑 제한값 (None이면 AHE)
        bins: 히스토그램 빈 개수
        border: AHE 타일 경계 보간 방법

    Returns:
        결과 딕셔너리 (img: 결과이미지, hist: 히스토그램 정보, cdf: CDF 정보)
    """
    # 입력 검증
    validate_image_input(img_bgr)

    # RGB로 변환 (OpenCV는 BGR을 사용)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 색공간 변환
    if space == "yuv":
        img_converted = rgb_to_yuv(img_rgb)
        luma_channel = img_converted[:, :, 0]  # Y 채널
    elif space == "lab":
        img_converted = rgb_to_lab(img_rgb)
        luma_channel = img_converted[:, :, 0]  # L 채널
    elif space == "ycbcr":
        img_converted = rgb_to_ycbcr(img_rgb)
        luma_channel = img_converted[:, :, 0]  # Y 채널
    elif space == "hsv":
        img_converted = rgb_to_hsv(img_rgb)
        luma_channel = img_converted[:, :, 2]  # V 채널
    elif space == "rgb":
        img_converted = img_rgb.copy()
        luma_channel = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)  # 그레이스케일로 변환
    else:
        raise ValueError(f"지원되지 않는 색공간: {space}")

    # 히스토그램 평활화 적용
    if mode == "global":
        enhanced_channel, hist_orig, hist_eq = histogram_equalization(luma_channel, bins)
        cdf_orig = calculate_cdf(hist_orig)
    elif mode == "ahe":
        enhanced_channel = apply_ahe_to_channel(luma_channel, tile, bins, border)
        hist_orig, _ = compute_histogram(luma_channel, bins)
        hist_eq, _ = compute_histogram(enhanced_channel, bins)
        cdf_orig = calculate_cdf(hist_orig)
    elif mode == "clahe":
        if clip is None:
            clip = 2.0  # 기본값
        enhanced_channel = apply_clahe_to_channel(luma_channel, clip, tile, bins)
        hist_orig, _ = compute_histogram(luma_channel, bins)
        hist_eq, _ = compute_histogram(enhanced_channel, bins)
        cdf_orig = calculate_cdf(hist_orig)
    else:
        raise ValueError(f"지원되지 않는 모드: {mode}")

    # 결과 이미지 생성
    if space == "rgb":
        # RGB 모드에서는 각 채널에 동일하게 적용
        result_rgb = img_rgb.copy()
        for i in range(3):
            if mode == "global":
                result_rgb[:, :, i], _, _ = histogram_equalization(img_rgb[:, :, i], bins)
            elif mode == "ahe":
                result_rgb[:, :, i] = apply_ahe_to_channel(img_rgb[:, :, i], tile, bins, border)
            elif mode == "clahe":
                result_rgb[:, :, i] = apply_clahe_to_channel(img_rgb[:, :, i], clip, tile, bins)
    else:
        # 다른 색공간에서는 휘도 채널만 처리
        result_converted = img_converted.copy()
        if space in ["yuv", "ycbcr", "lab"]:
            result_converted[:, :, 0] = enhanced_channel
        elif space == "hsv":
            result_converted[:, :, 2] = enhanced_channel

        # RGB로 역변환
        if space == "yuv":
            result_rgb = yuv_to_rgb(result_converted)
        elif space == "lab":
            result_rgb = lab_to_rgb(result_converted)
        elif space == "ycbcr":
            result_rgb = ycbcr_to_rgb(result_converted)
        elif space == "hsv":
            result_rgb = hsv_to_rgb(result_converted)

    # BGR로 변환하여 반환
    result_bgr = cv2.cvtColor(result_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)

    # CDF 정보
    cdf_eq = calculate_cdf(hist_eq)

    return {
        "img": result_bgr,
        "hist": {
            "original": hist_orig,
            "equalized": hist_eq
        },
        "cdf": {
            "original": cdf_orig,
            "equalized": cdf_eq
        },
        "luma_channel": {
            "original": luma_channel,
            "enhanced": enhanced_channel
        }
    }

def calculate_rms_contrast(image: np.ndarray) -> float:
    """RMS 대비를 계산합니다."""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return np.sqrt(np.mean((image - np.mean(image))**2))

def calculate_edge_strength(image: np.ndarray) -> float:
    """Sobel 에지 강도 합을 계산합니다."""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    return np.sum(sobel_magnitude)

def extract_roi_metrics(image: np.ndarray, roi: Tuple[int, int, int, int]) -> Dict[str, float]:
    """ROI 영역의 지표를 계산합니다."""
    x, y, w, h = roi
    roi_image = image[y:y+h, x:x+w]

    return {
        "rms_contrast": calculate_rms_contrast(roi_image),
        "edge_strength": calculate_edge_strength(roi_image),
        "mean_brightness": np.mean(roi_image),
        "brightness_std": np.std(roi_image)
    }

# 기존 함수들과의 호환성을 위한 래퍼 함수들
def histogram_equalization_color(image: np.ndarray, method: str = 'yuv', algorithm: str = 'he',
                                clip_limit: float = 2.0, tile_size: int = 8, show_process: bool = True):
    """기존 인터페이스와의 호환성을 위한 래퍼 함수"""
    if len(image.shape) == 3 and image.shape[2] == 3:
        # RGB 이미지인 경우 BGR로 변환
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = image

    mode = "clahe" if algorithm == "clahe" else "global"
    result = he_luma_bgr(img_bgr, space=method, mode=mode,
                        tile=(tile_size, tile_size), clip=clip_limit)

    # RGB로 변환하여 반환
    result_rgb = cv2.cvtColor(result["img"], cv2.COLOR_BGR2RGB)

    return {
        'enhanced_image': result_rgb,
        'colorspace': method,
        'algorithm': algorithm,
        'processing_time': 0.0,  # 실제 측정 필요
        'quality_metrics': {
            'contrast_improvement_percent': 0.0,  # 실제 계산 필요
        }
    }