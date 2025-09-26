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

def clahe_implementation(image: np.ndarray, clip_limit: float = 2.0, tile_size: Tuple[int, int] = (8, 8), show_process: bool = True, use_interpolation: bool = False) -> Tuple[np.ndarray, dict]:
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization)를 구현합니다.
    Implement CLAHE (Contrast Limited Adaptive Histogram Equalization).

    CLAHE의 핵심 원리:
    1. 이미지를 작은 타일로 분할
    2. 각 타일에서 히스토그램 계산 및 클리핑
    3. 클리핑된 히스토그램으로 로컬 히스토그램 평활화 수행
    4. 타일 경계에서 보간을 통해 부드러운 전환 (선택적)

    Core principles of CLAHE:
    1. Divide image into small tiles
    2. Calculate and clip histogram in each tile
    3. Perform local histogram equalization with clipped histogram
    4. Smooth transition through interpolation at tile boundaries

    Clip Limit 설정 기준:
    - 일반적으로 2~4 사이 값 사용
    - 낮은 값(2): 노이즈 증폭 방지, 보수적인 개선
    - 높은 값(4): 더 강한 대비 개선, 노이즈 증폭 가능성

    Clip Limit setting criteria:
    - Generally use values between 2-4
    - Low value (2): Prevent noise amplification, conservative enhancement
    - High value (4): Stronger contrast enhancement, possible noise amplification

    Args:
        image (np.ndarray): 입력 그레이스케일 이미지 / Input grayscale image
        clip_limit (float): 클립 한계값 / Clip limit value
        tile_size (Tuple[int, int]): 타일 크기 (행, 열) / Tile size (rows, cols)
        show_process (bool): 중간 과정 표시 여부 / Whether to show intermediate process
        use_interpolation (bool): 타일 경계에서 이중선형 보간 사용 여부 / Whether to use bilinear interpolation at tile boundaries

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
    if use_interpolation:
        # 보간을 사용한 CLAHE / CLAHE with interpolation
        clahe_image = clahe_with_bilinear_interpolation(image_resized, clip_limit, tile_size)
        tile_info = []  # 보간 모드에서는 타일 정보 저장하지 않음
    else:
        # 기존 방식: 각 타일별 독립 처리 / Original method: independent tile processing
        tile_info = []

        for i in range(0, new_height, tile_h):
            for j in range(0, new_width, tile_w):
                # 현재 타일 추출 / Extract current tile
                tile = image_resized[i:i+tile_h, j:j+tile_w]

                # 타일의 히스토그램 계산 / Calculate tile histogram
                hist, _ = compute_histogram(tile)

                # 히스토그램 클리핑 / Histogram clipping
                clipped_hist = clip_histogram(hist, clip_limit, tile_h * tile_w)

                # 클리핑된 히스토그램으로 CDF 계산 / Calculate CDF with clipped histogram
                cdf = calculate_cdf(clipped_hist)

                # 타일에 히스토그램 평활화 적용 / Apply histogram equalization to tile
                tile_equalized = apply_histogram_mapping(tile, cdf)

                # 결과에 반영 / Apply to result
                clahe_image[i:i+tile_h, j:j+tile_w] = tile_equalized

                # 타일 정보 저장 / Store tile information
                tile_info.append({
                    'position': (i, j),
                    'original_hist': hist,
                    'clipped_hist': clipped_hist,
                    'cdf': cdf
                })

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
def histogram_equalization_color_clahe(image: np.ndarray, method: str = 'yuv',
                                      clip_limit: float = 2.0, tile_size: Tuple[int, int] = (8, 8),
                                      show_process: bool = True, use_interpolation: bool = False,
                                      use_opencv: bool = False, algorithm_name: str = 'CLAHE') -> Tuple[np.ndarray, dict]:
    """
    컬러 이미지에 CLAHE를 적용합니다.
    Apply CLAHE to color image.

    Args:
        image (np.ndarray): 입력 컬러 이미지 / Input color image
        method (str): 처리 방법 ('yuv' 또는 'rgb') / Processing method ('yuv' or 'rgb')
        clip_limit (float): 클립 한계값 / Clip limit value
        tile_size (Tuple[int, int]): 타일 크기 / Tile size
        show_process (bool): 중간 과정 표시 여부 / Whether to show intermediate process
        use_interpolation (bool): bilinear interpolation 사용 여부 / Whether to use bilinear interpolation
        use_opencv (bool): OpenCV CLAHE 사용 여부 / Whether to use OpenCV CLAHE
        algorithm_name (str): 알고리즘 이름 (AHE 또는 CLAHE) / Algorithm name (AHE or CLAHE)

    Returns:
        Tuple[np.ndarray, dict]: (CLAHE 적용된 이미지, 처리 정보) / (CLAHE applied image, processing info)
    """
    validate_image_input(image)

    if len(image.shape) != 3:
        raise ValueError("컬러 이미지가 필요합니다 / Color image required")

    if method == 'yuv':
        # YUV 색공간으로 변환 / Convert to YUV color space
        yuv_image = rgb_to_yuv(image)

        # Y 채널에만 CLAHE 적용 / Apply CLAHE only to Y channel
        y_channel = yuv_image[:, :, 0]

        if use_opencv:
            y_clahe = clahe_opencv_implementation(y_channel, clip_limit, tile_size)
            clahe_info = {'method': 'opencv', 'clip_limit': clip_limit, 'tile_size': tile_size}
        else:
            y_clahe, clahe_info = clahe_implementation(y_channel, clip_limit, tile_size, show_process=False, use_interpolation=use_interpolation)

        # YUV 이미지 재구성 / Reconstruct YUV image
        yuv_clahe = yuv_image.copy()
        yuv_clahe[:, :, 0] = y_clahe

        # RGB로 역변환 / Convert back to RGB
        rgb_clahe = yuv_to_rgb(yuv_clahe)

        # 컬러 이미지 전용 시각화 / Color image specific visualization
        if show_process:
            visualize_color_clahe_process(image, rgb_clahe, clahe_info, method, algorithm_name)

        # 처리 정보 업데이트 / Update processing info
        clahe_info.update({
            'color_method': method,
            'processed_channels': 'Y (Luminance) only',
            'color_preservation': 'U, V channels preserved'
        })

        return rgb_clahe, clahe_info

    elif method == 'rgb':
        # RGB 각 채널을 개별적으로 처리 / Process each RGB channel individually
        rgb_clahe = np.zeros_like(image)
        channel_info = {}

        for c, channel_name in enumerate(['R', 'G', 'B']):
            channel = image[:, :, c]
            channel_clahe, channel_clahe_info = clahe_implementation(channel, clip_limit, tile_size, show_process=False)
            rgb_clahe[:, :, c] = channel_clahe
            channel_info[f'{channel_name}_channel'] = channel_clahe_info

        # 전체 처리 정보 / Overall processing info
        combined_info = {
            'color_method': method,
            'processed_channels': 'All RGB channels separately',
            'clip_limit': clip_limit,
            'tile_size': tile_size,
            'channel_details': channel_info
        }

        # 컬러 이미지 전용 시각화 / Color image specific visualization
        if show_process:
            visualize_color_clahe_process(image, rgb_clahe, combined_info, method, algorithm_name)

        return rgb_clahe, combined_info

    else:
        raise ValueError(f"지원하지 않는 방법입니다: {method} / Unsupported method: {method}")

def clip_histogram(histogram: np.ndarray, clip_limit: float, total_pixels: int) -> np.ndarray:
    """
    히스토그램을 클리핑합니다.
    Clip histogram.

    Args:
        histogram (np.ndarray): 입력 히스토그램 / Input histogram
        clip_limit (float): 클립 한계값 / Clip limit value
        total_pixels (int): 총 픽셀 수 / Total number of pixels

    Returns:
        np.ndarray: 클리핑된 히스토그램 / Clipped histogram
    """
    # AHE의 경우 클리핑을 거의 비활성화
    if clip_limit > 100:
        return histogram.astype(np.uint32)

    # 클립 임계값 계산 / Calculate clip threshold
    clip_threshold = (total_pixels / 256) * clip_limit

    # 최소 임계값 설정 (너무 작은 값 방지)
    min_threshold = max(1, total_pixels / 1000)  # 최소 0.1% 또는 1
    clip_threshold = max(clip_threshold, min_threshold)

    # 히스토그램 클리핑 / Clip histogram
    clipped_hist = histogram.astype(np.float64)
    excess = 0

    # 클리핑 수행
    for i in range(256):
        if clipped_hist[i] > clip_threshold:
            excess += clipped_hist[i] - clip_threshold
            clipped_hist[i] = clip_threshold

    # 잘린 부분을 활성 빈(active bins)에만 재분배
    active_bins = np.sum(histogram > 0)
    if active_bins > 0 and excess > 0:
        redistribution_per_bin = excess / active_bins
        for i in range(256):
            if histogram[i] > 0:
                clipped_hist[i] += redistribution_per_bin
    elif excess > 0:
        # 활성 빈이 없으면 모든 빈에 균등 재분배
        redistribution_per_bin = excess / 256
        clipped_hist += redistribution_per_bin

    return clipped_hist.astype(np.uint32)

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
        'processing_time': 0.0,
        'quality_metrics': {
            'contrast_improvement_percent': 0.0,
        }
    }

def apply_histogram_mapping(image: np.ndarray, cdf: np.ndarray) -> np.ndarray:
    """
    CDF를 이용하여 이미지에 히스토그램 매핑을 적용합니다.
    Apply histogram mapping to image using CDF.

    Args:
        image (np.ndarray): 입력 이미지 / Input image
        cdf (np.ndarray): 누적분포함수 / Cumulative Distribution Function

    Returns:
        np.ndarray: 매핑된 이미지 / Mapped image
    """
    # 룩업 테이블 생성 / Create lookup table
    lut = np.round(255 * cdf).astype(np.uint8)

    # 룩업 테이블을 이용한 픽셀값 변환 / Transform pixel values using lookup table
    mapped_image = lut[image]

    return mapped_image

def clahe_with_bilinear_interpolation(image: np.ndarray, clip_limit: float, tile_size: Tuple[int, int]) -> np.ndarray:
    """
    Bilinear interpolation을 사용한 CLAHE 구현으로 격자 아티팩트를 감소시킵니다.
    CLAHE implementation with bilinear interpolation to reduce grid artifacts.

    이 구현은 OpenCV의 CLAHE와 유사하게 타일 경계에서 보간을 수행합니다.
    This implementation performs interpolation at tile boundaries similar to OpenCV's CLAHE.

    Args:
        image (np.ndarray): 입력 그레이스케일 이미지 / Input grayscale image
        clip_limit (float): 클립 한계값 / Clip limit value
        tile_size (Tuple[int, int]): 타일 크기 / Tile size

    Returns:
        np.ndarray: 보간된 CLAHE 결과 / Interpolated CLAHE result
    """
    height, width = image.shape
    tile_h, tile_w = tile_size

    # 타일 수 계산 / Calculate number of tiles
    tiles_y = height // tile_h
    tiles_x = width // tile_w

    # 각 타일의 변환 함수(LUT) 저장 / Store transformation function (LUT) for each tile
    tile_luts = np.zeros((tiles_y, tiles_x, 256), dtype=np.uint8)

    # 각 타일에 대해 변환 함수 계산 / Calculate transformation function for each tile
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            # 타일 영역 추출 / Extract tile region
            y_start = ty * tile_h
            y_end = y_start + tile_h
            x_start = tx * tile_w
            x_end = x_start + tile_w

            tile = image[y_start:y_end, x_start:x_end]

            # 타일 히스토그램 계산 및 클리핑 / Calculate and clip tile histogram
            hist, _ = compute_histogram(tile)
            clipped_hist = clip_histogram(hist, clip_limit, tile_h * tile_w)
            cdf = calculate_cdf(clipped_hist)

            # 변환 함수(LUT) 생성 / Create transformation function (LUT)
            tile_luts[ty, tx] = np.round(255 * cdf).astype(np.uint8)

    # 결과 이미지 초기화 / Initialize result image
    result = np.zeros_like(image)

    # 각 픽셀에 대해 bilinear interpolation 수행 / Perform bilinear interpolation for each pixel
    for y in range(height):
        for x in range(width):
            pixel_value = image[y, x]

            # 현재 픽셀이 속하는 타일의 좌표 계산 / Calculate tile coordinates for current pixel
            tile_y = min(y // tile_h, tiles_y - 1)
            tile_x = min(x // tile_w, tiles_x - 1)

            # 타일 내부 좌표 / Coordinates within tile
            local_y = y % tile_h
            local_x = x % tile_w

            # 타일 경계에서 보간 수행 / Perform interpolation at tile boundaries
            if (local_y < tile_h // 4 or local_y >= 3 * tile_h // 4 or
                local_x < tile_w // 4 or local_x >= 3 * tile_w // 4):

                # 인접한 타일들의 변환 결과 수집 / Collect transformation results from neighboring tiles
                transformed_values = []
                weights = []

                # 주변 타일들 확인 / Check surrounding tiles
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        ny, nx = tile_y + dy, tile_x + dx
                        if 0 <= ny < tiles_y and 0 <= nx < tiles_x:
                            # 거리 기반 가중치 계산 / Calculate distance-based weight
                            center_y = ny * tile_h + tile_h // 2
                            center_x = nx * tile_w + tile_w // 2
                            distance = np.sqrt((y - center_y)**2 + (x - center_x)**2)
                            weight = 1.0 / (1.0 + distance)

                            transformed_values.append(tile_luts[ny, nx, pixel_value])
                            weights.append(weight)

                # 가중평균으로 최종 값 계산 / Calculate final value using weighted average
                if weights:
                    weights = np.array(weights)
                    weights = weights / np.sum(weights)
                    result[y, x] = np.round(np.sum(np.array(transformed_values) * weights)).astype(np.uint8)
                else:
                    result[y, x] = tile_luts[tile_y, tile_x, pixel_value]
            else:
                # 타일 중앙부는 해당 타일의 변환만 사용 / Use only the tile's transformation for central area
                result[y, x] = tile_luts[tile_y, tile_x, pixel_value]

    return result

def clahe_opencv_implementation(image: np.ndarray, clip_limit: float = 2.0, tile_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    OpenCV의 CLAHE를 사용한 구현입니다.
    Implementation using OpenCV's CLAHE.

    Args:
        image (np.ndarray): 입력 그레이스케일 이미지 / Input grayscale image
        clip_limit (float): 클립 한계값 / Clip limit value
        tile_size (Tuple[int, int]): 타일 크기 / Tile size

    Returns:
        np.ndarray: OpenCV CLAHE 결과 / OpenCV CLAHE result
    """
    import cv2

    # OpenCV CLAHE 객체 생성 / Create OpenCV CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)

    # CLAHE 적용 / Apply CLAHE
    result = clahe.apply(image)

    return result

def visualize_he_process(original: np.ndarray, equalized: np.ndarray, process_info: dict) -> None:
    """
    히스토그램 평활화 과정을 시각화합니다.
    Visualize histogram equalization process.

    Args:
        original (np.ndarray): 원본 이미지 / Original image
        equalized (np.ndarray): 평활화된 이미지 / Equalized image
        process_info (dict): 처리 정보 / Processing information
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 원본 이미지 / Original image
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # 원본 히스토그램 / Original histogram
    axes[0, 1].bar(range(256), process_info['original_histogram'], alpha=0.7, color='blue')
    axes[0, 1].set_title('Original Histogram')
    axes[0, 1].set_xlabel('픽셀 강도 / Pixel Intensity')
    axes[0, 1].set_ylabel('빈도수 / Frequency')

    # CDF 그래프 / CDF graph
    axes[0, 2].plot(range(256), process_info['cdf'], 'r-', linewidth=2)
    axes[0, 2].set_title('Cumulative Distribution Function')
    axes[0, 2].set_xlabel('픽셀 강도 / Pixel Intensity')
    axes[0, 2].set_ylabel('누적 확률 / Cumulative Probability')
    axes[0, 2].grid(True, alpha=0.3)

    # 평활화된 이미지 / Equalized image
    axes[1, 0].imshow(equalized, cmap='gray')
    axes[1, 0].set_title('Equalized Image')
    axes[1, 0].axis('off')

    # 평활화된 히스토그램 / Equalized histogram
    axes[1, 1].bar(range(256), process_info['equalized_histogram'], alpha=0.7, color='green')
    axes[1, 1].set_title('Equalized Histogram')
    axes[1, 1].set_xlabel('픽셀 강도 / Pixel Intensity')
    axes[1, 1].set_ylabel('빈도수 / Frequency')

    # 픽셀 매핑 함수 / Pixel mapping function
    axes[1, 2].plot(range(256), process_info['pixel_mapping'], 'g-', linewidth=2)
    axes[1, 2].set_title('Pixel Mapping Function')
    axes[1, 2].set_xlabel('입력 픽셀값 / Input Pixel Value')
    axes[1, 2].set_ylabel('출력 픽셀값 / Output Pixel Value')
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def visualize_color_he_process(original: np.ndarray, equalized: np.ndarray, process_info: dict) -> None:
    """
    컬러 이미지 히스토그램 평활화 과정을 시각화합니다.
    Visualize color image histogram equalization process.

    Args:
        original (np.ndarray): 원본 컬러 이미지 / Original color image
        equalized (np.ndarray): 평활화된 컬러 이미지 / Equalized color image
        process_info (dict): 처리 정보 / Processing information
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 원본 이미지 / Original image
    axes[0, 0].imshow(original)
    axes[0, 0].set_title('Original Color Image')
    axes[0, 0].axis('off')

    # 원본 RGB 히스토그램 / Original RGB histogram
    colors = ['red', 'green', 'blue']
    for i, color in enumerate(colors):
        hist, _ = compute_histogram(original[:, :, i])
        axes[0, 1].plot(range(256), hist, color=color, alpha=0.7, label=f'{color.upper()}')
    axes[0, 1].set_title('Original RGB Histogram')
    axes[0, 1].set_xlabel('픽셀 강도 / Pixel Intensity')
    axes[0, 1].set_ylabel('빈도수 / Frequency')
    axes[0, 1].legend()

    # 처리 방법 정보 / Processing method info
    method_text = f"처리 방법 / Method: {process_info['method'].upper()}\n"
    method_text += f"처리 채널 / Processed Channel: {process_info['processed_channel']}\n\n"

    if process_info['method'] == 'yuv':
        method_text += "YUV 색공간 사용 이유:\n- Y채널: 휘도 정보 (인간 시각 인지와 밀접)\n- U,V채널: 색상 정보 보존\n\n"
        method_text += "Why use YUV color space:\n- Y channel: Luminance (closely related to human vision)\n- U,V channels: Preserve color information"

    axes[0, 2].text(0.1, 0.5, method_text, transform=axes[0, 2].transAxes, fontsize=10,
                    verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    axes[0, 2].set_title('Processing Method Info')
    axes[0, 2].axis('off')

    # 평활화된 이미지 / Equalized image
    axes[1, 0].imshow(equalized)
    axes[1, 0].set_title('Equalized Color Image')
    axes[1, 0].axis('off')

    # 평활화된 RGB 히스토그램 / Equalized RGB histogram
    for i, color in enumerate(colors):
        hist, _ = compute_histogram(equalized[:, :, i])
        axes[1, 1].plot(range(256), hist, color=color, alpha=0.7, label=f'{color.upper()}')
    axes[1, 1].set_title('Equalized RGB Histogram')
    axes[1, 1].set_xlabel('Pixel Intensity')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()

    # 이전/이후 비교 / Before/After comparison
    axes[1, 2].subplot = plt.subplot(2, 3, 6)
    comparison_image = np.concatenate([original, equalized], axis=1)
    axes[1, 2].imshow(comparison_image)
    axes[1, 2].set_title('Comparison (Before | After)')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.show()

def visualize_clahe_process(original: np.ndarray, clahe_result: np.ndarray, process_info: dict) -> None:
    """
    CLAHE 과정을 시각화합니다.
    Visualize CLAHE process.

    Args:
        original (np.ndarray): 원본 이미지 / Original image
        clahe_result (np.ndarray): CLAHE 결과 이미지 / CLAHE result image
        process_info (dict): 처리 정보 / Processing information
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 원본 이미지 / Original image
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # CLAHE 결과 이미지 / CLAHE result image
    axes[0, 1].imshow(clahe_result, cmap='gray')
    axes[0, 1].set_title('CLAHE Result')
    axes[0, 1].axis('off')

    # 비교 / Comparison
    comparison = np.concatenate([original, clahe_result], axis=1)
    axes[0, 2].imshow(comparison, cmap='gray')
    axes[0, 2].set_title('Comparison (Before | After)')
    axes[0, 2].axis('off')

    # CLAHE 파라미터 정보 / CLAHE parameter info
    param_text = f"CLAHE 파라미터 / Parameters:\n"
    param_text += f"- Clip Limit: {process_info['clip_limit']}\n"
    param_text += f"- Tile Size: {process_info['tile_size']}\n"
    param_text += f"- 총 타일 수 / Total Tiles: {process_info['num_tiles']}\n\n"
    param_text += f"Clip Limit 효과:\n"
    param_text += f"- 낮은 값(2): 노이즈 방지\n"
    param_text += f"- 높은 값(4): 강한 대비 개선"

    axes[1, 0].text(0.1, 0.5, param_text, transform=axes[1, 0].transAxes, fontsize=10,
                    verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    axes[1, 0].set_title('CLAHE Parameters')
    axes[1, 0].axis('off')

    # 타일별 히스토그램 예시 / Example tile histograms
    if process_info['tile_info']:
        tile_info = process_info['tile_info'][0]  # 첫 번째 타일 정보 / First tile info
        axes[1, 1].bar(range(256), tile_info['original_hist'], alpha=0.5, color='blue', label='Original')
        axes[1, 1].bar(range(256), tile_info['clipped_hist'], alpha=0.7, color='red', label='Clipped')
        axes[1, 1].set_title('Tile Histogram (Original vs Clipped)')
        axes[1, 1].set_xlabel('픽셀 강도 / Pixel Intensity')
        axes[1, 1].set_ylabel('빈도수 / Frequency')
        axes[1, 1].legend()

    # 전체 히스토그램 비교 / Overall histogram comparison
    hist_original, _ = compute_histogram(original)
    hist_clahe, _ = compute_histogram(clahe_result)
    axes[1, 2].plot(range(256), hist_original, 'b-', alpha=0.7, label='Original')
    axes[1, 2].plot(range(256), hist_clahe, 'r-', alpha=0.7, label='CLAHE')
    axes[1, 2].set_title('Overall Histogram Comparison')
    axes[1, 2].set_xlabel('픽셀 강도 / Pixel Intensity')
    axes[1, 2].set_ylabel('빈도수 / Frequency')
    axes[1, 2].legend()

    plt.tight_layout()
    plt.show()

def visualize_color_clahe_process(original: np.ndarray, clahe_result: np.ndarray, process_info: dict, method: str, algorithm_name: str = 'CLAHE') -> None:
    """
    컬러 이미지 CLAHE 처리 과정을 시각화합니다.
    Visualize color image CLAHE processing.

    Args:
        original (np.ndarray): 원본 컬러 이미지 / Original color image
        clahe_result (np.ndarray): CLAHE 적용된 이미지 / CLAHE processed image
        process_info (dict): 처리 정보 / Processing information
        method (str): 처리 방법 ('yuv' 또는 'rgb') / Processing method ('yuv' or 'rgb')
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 원본 이미지 / Original image
    axes[0, 0].imshow(original)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # 결과 이미지 / Result image
    axes[0, 1].imshow(clahe_result)
    axes[0, 1].set_title(f'{algorithm_name} Result ({method.upper()})')
    axes[0, 1].axis('off')

    # 비교 / Comparison
    comparison = np.concatenate([original, clahe_result], axis=1)
    axes[0, 2].imshow(comparison)
    axes[0, 2].set_title('Comparison (Before | After)')
    axes[0, 2].axis('off')

    # 파라미터 정보 / Parameter info
    clip_limit = process_info.get('clip_limit', 2.0)
    tile_size = process_info.get('tile_size', (8, 8))

    param_text = f"{algorithm_name} 파라미터 / Parameters:\n"
    param_text += f"- Clip Limit: {clip_limit}\n"
    param_text += f"- Tile Size: {tile_size}\n"
    param_text += f"- 처리 방법 / Method: {method.upper()}\n"

    if method == 'yuv':
        param_text += f"- 처리 채널 / Processed: Y (휘도) only\n"
        param_text += f"- 보존 채널 / Preserved: U, V (색상)\n"
    elif method == 'rgb':
        param_text += f"- 처리 채널 / Processed: R, G, B 개별 처리\n"

    param_text += f"\n{algorithm_name} 특성:\n"
    if clip_limit > 100:  # AHE case
        param_text += f"- AHE: 매우 높은 클립 값({clip_limit})\n"
        param_text += f"- 클리핑 거의 없음, 최대 대비 향상\n"
        param_text += f"- 강한 지역적 대비 개선"
    else:  # CLAHE case
        param_text += f"- CLAHE: 제한된 클립 값({clip_limit})\n"
        param_text += f"- 클리핑으로 노이즈 방지\n"
        param_text += f"- 균형잡힌 대비 개선"

    axes[1, 0].text(0.1, 0.5, param_text, transform=axes[1, 0].transAxes, fontsize=10,
                    verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    axes[1, 0].set_title(f'{algorithm_name} Parameters')
    axes[1, 0].axis('off')

    # 채널별 히스토그램 (처리된 채널) / Channel histogram (processed channel)
    if method == 'yuv':
        # YUV 모드에서는 Y 채널 히스토그램 표시
        from src.utils import rgb_to_yuv
        yuv_original = rgb_to_yuv(original)
        yuv_result = rgb_to_yuv(clahe_result)

        hist_original, _ = compute_histogram(yuv_original[:, :, 0])
        hist_result, _ = compute_histogram(yuv_result[:, :, 0])

        axes[1, 1].bar(range(256), hist_original, alpha=0.5, color='blue', label='Original Y')
        axes[1, 1].bar(range(256), hist_result, alpha=0.7, color='red', label=f'{algorithm_name} Y')
        axes[1, 1].set_title('Y Channel Histogram (Luminance)')
    elif method == 'rgb':
        # RGB 모드에서는 각 채널별 히스토그램 표시
        colors = ['red', 'green', 'blue']
        for c, color in enumerate(colors):
            hist_original, _ = compute_histogram(original[:, :, c])
            hist_result, _ = compute_histogram(clahe_result[:, :, c])

            axes[1, 1].plot(range(256), hist_original, f'{color[0]}-', alpha=0.5,
                           label=f'Original {color[0].upper()}')
            axes[1, 1].plot(range(256), hist_result, f'{color[0]}--', alpha=0.8,
                           label=f'{algorithm_name} {color[0].upper()}')

        axes[1, 1].set_title('RGB Channels Histogram')

    axes[1, 1].set_xlabel('픽셀 강도 / Pixel Intensity')
    axes[1, 1].set_ylabel('빈도수 / Frequency')
    axes[1, 1].legend()

    # 전체 이미지 그레이스케일 히스토그램 비교 / Overall grayscale histogram comparison
    import cv2
    gray_original = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    gray_result = cv2.cvtColor(clahe_result, cv2.COLOR_RGB2GRAY)

    hist_gray_original, _ = compute_histogram(gray_original)
    hist_gray_result, _ = compute_histogram(gray_result)

    axes[1, 2].plot(range(256), hist_gray_original, 'b-', alpha=0.7, label='Original')
    axes[1, 2].plot(range(256), hist_gray_result, 'r-', alpha=0.7, label=algorithm_name)
    axes[1, 2].set_title('Overall Histogram Comparison')
    axes[1, 2].set_xlabel('픽셀 강도 / Pixel Intensity')
    axes[1, 2].set_ylabel('빈도수 / Frequency')
    axes[1, 2].legend()

    plt.tight_layout()
    plt.show()

# OpenCV 내장 함수 사용 예시 (주석으로 표시) / Example usage of OpenCV built-in functions (commented)
def example_opencv_he_usage():
    """
    OpenCV를 사용한 히스토그램 평활화 예시입니다.
    Example of histogram equalization using OpenCV.
    """
    pass
    # 그레이스케일 히스토그램 평활화 / Grayscale histogram equalization:
    # equalized = cv2.equalizeHist(gray_image)

    # CLAHE 적용 / CLAHE application:
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # equalized = clahe.apply(gray_image)

    # 컬러 이미지 CLAHE (YUV 색공간 사용) / Color image CLAHE (using YUV color space):
    # yuv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2YUV)
    # yuv_image[:,:,0] = clahe.apply(yuv_image[:,:,0])
    # result = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)
    pass
