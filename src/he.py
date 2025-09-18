"""
컬러 이미지 히스토그램 평활화 모듈
Color Image Histogram Equalization Module

이 모듈은 컬러 이미지에 대한 히스토그램 평활화를 직접 구현합니다.
This module directly implements histogram equalization for color images.

주요 기능 / Key Features:
1. 직관적인 low-level 히스토그램 평활화 구현 / Intuitive low-level histogram equalization implementation
2. YUV 색공간을 이용한 컬러 이미지 처리 / Color image processing using YUV color space
3. 단계별 중간 과정 시각화 / Step-by-step intermediate process visualization
4. CLAHE (Contrast Limited Adaptive Histogram Equalization) 구현 / CLAHE implementation
"""

import numpy as np
import matplotlib.pyplot as plt
from .utils import rgb_to_yuv, yuv_to_rgb, compute_histogram, validate_image_input
from typing import Tuple, Optional

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

def histogram_equalization_grayscale(image: np.ndarray, show_process: bool = True) -> Tuple[np.ndarray, dict]:
    """
    그레이스케일 이미지에 대한 히스토그램 평활화를 수행합니다.
    Perform histogram equalization on grayscale image.

    히스토그램 평활화 수학적 원리:
    1. 히스토그램 계산: h(i) = 픽셀값 i의 빈도수
    2. CDF 계산: CDF(i) = Σ(h(0) to h(i)) / 총 픽셀 수
    3. 변환 공식: y' = Scale * CDF(x)
       여기서 Scale = 255 (8비트 이미지의 경우)

    Mathematical principle of histogram equalization:
    1. Calculate histogram: h(i) = frequency of pixel value i
    2. Calculate CDF: CDF(i) = Σ(h(0) to h(i)) / total pixels
    3. Transform formula: y' = Scale * CDF(x)
       where Scale = 255 (for 8-bit images)

    Args:
        image (np.ndarray): 입력 그레이스케일 이미지 / Input grayscale image
        show_process (bool): 중간 과정 표시 여부 / Whether to show intermediate process

    Returns:
        Tuple[np.ndarray, dict]: (평활화된 이미지, 처리 정보) / (Equalized image, processing info)
    """
    validate_image_input(image)

    if len(image.shape) != 2:
        raise ValueError("그레이스케일 이미지가 필요합니다 / Grayscale image required")

    # 1단계: 원본 히스토그램 계산 / Step 1: Calculate original histogram
    hist_original, _ = compute_histogram(image)

    # 2단계: CDF 계산 / Step 2: Calculate CDF
    cdf = calculate_cdf(hist_original)

    # 3단계: CDF를 이용한 픽셀값 변환 / Step 3: Transform pixel values using CDF
    # y' = Scale * CDF(x) 공식 적용 / Apply formula y' = Scale * CDF(x)
    equalized_image = np.zeros_like(image)
    for i in range(256):
        mask = image == i
        equalized_image[mask] = np.round(255 * cdf[i]).astype(np.uint8)

    # 4단계: 평활화된 이미지의 히스토그램 계산 / Step 4: Calculate histogram of equalized image
    hist_equalized, _ = compute_histogram(equalized_image)

    # 처리 정보 저장 / Store processing information
    process_info = {
        'original_histogram': hist_original,
        'equalized_histogram': hist_equalized,
        'cdf': cdf,
        'pixel_mapping': np.round(255 * cdf).astype(np.uint8)
    }

    # 중간 과정 시각화 / Visualize intermediate process
    if show_process:
        visualize_he_process(image, equalized_image, process_info)

    return equalized_image, process_info

def histogram_equalization_color(image: np.ndarray, method: str = 'yuv', show_process: bool = True) -> Tuple[np.ndarray, dict]:
    """
    컬러 이미지에 대한 히스토그램 평활화를 수행합니다.
    Perform histogram equalization on color image.

    YUV 색공간에서 Y(휘도) 채널만 처리하는 이론적 근거:
    - Y 채널: 이미지의 밝기 정보, 인간의 시각 인지와 밀접한 관련
    - U, V 채널: 색상 정보, 이를 그대로 유지하여 자연스러운 색감 보존
    - RGB 각 채널을 개별적으로 처리하면 색상 왜곡 발생 가능

    Theoretical basis for processing only Y(luminance) channel in YUV color space:
    - Y channel: Brightness information, closely related to human visual perception
    - U, V channels: Color information, preserving them maintains natural color tone
    - Processing each RGB channel individually may cause color distortion

    Args:
        image (np.ndarray): 입력 컬러 이미지 (RGB) / Input color image (RGB)
        method (str): 처리 방법 ('yuv', 'rgb') / Processing method
        show_process (bool): 중간 과정 표시 여부 / Whether to show intermediate process

    Returns:
        Tuple[np.ndarray, dict]: (평활화된 이미지, 처리 정보) / (Equalized image, processing info)
    """
    validate_image_input(image)

    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("3채널 컬러 이미지가 필요합니다 / 3-channel color image required")

    if method == 'yuv':
        # YUV 색공간으로 변환 / Convert to YUV color space
        yuv_image = rgb_to_yuv(image)

        # Y(휘도) 채널에만 히스토그램 평활화 적용 / Apply histogram equalization only to Y(luminance) channel
        y_channel = yuv_image[:, :, 0]
        y_equalized, process_info = histogram_equalization_grayscale(y_channel, show_process=False)

        # 평활화된 Y 채널과 원본 U, V 채널 결합 / Combine equalized Y channel with original U, V channels
        yuv_equalized = yuv_image.copy()
        yuv_equalized[:, :, 0] = y_equalized

        # RGB로 다시 변환 / Convert back to RGB
        rgb_equalized = yuv_to_rgb(yuv_equalized)

        process_info['method'] = 'yuv'
        process_info['processed_channel'] = 'Y (Luminance)'

    elif method == 'rgb':
        # RGB 각 채널에 개별적으로 히스토그램 평활화 적용 / Apply histogram equalization to each RGB channel individually
        rgb_equalized = np.zeros_like(image)
        channel_info = {}

        for i, channel_name in enumerate(['R', 'G', 'B']):
            channel_equalized, channel_process_info = histogram_equalization_grayscale(image[:, :, i], show_process=False)
            rgb_equalized[:, :, i] = channel_equalized
            channel_info[channel_name] = channel_process_info

        process_info = {
            'method': 'rgb',
            'processed_channel': 'All RGB channels',
            'channel_info': channel_info
        }

    else:
        raise ValueError(f"지원하지 않는 방법입니다 / Unsupported method: {method}")

    # 중간 과정 시각화 / Visualize intermediate process
    if show_process:
        visualize_color_he_process(image, rgb_equalized, process_info)

    return rgb_equalized, process_info

def clahe_implementation(image: np.ndarray, clip_limit: float = 2.0, tile_size: Tuple[int, int] = (8, 8), show_process: bool = True) -> Tuple[np.ndarray, dict]:
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization)를 구현합니다.
    Implement CLAHE (Contrast Limited Adaptive Histogram Equalization).

    CLAHE의 핵심 원리:
    1. 이미지를 작은 타일로 분할
    2. 각 타일에서 히스토그램 계산 및 클리핑
    3. 클리핑된 히스토그램으로 로컬 히스토그램 평활화 수행
    4. 타일 경계에서 보간을 통해 부드러운 전환

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

    Returns:
        Tuple[np.ndarray, dict]: (CLAHE 적용된 이미지, 처리 정보) / (CLAHE applied image, processing info)
    """
    validate_image_input(image)

    if len(image.shape) != 2:
        raise ValueError("그레이스케일 이미지가 필요합니다 / Grayscale image required")

    height, width = image.shape
    tile_h, tile_w = tile_size

    # 타일 크기에 맞게 이미지 크기 조정 / Adjust image size to fit tile size
    new_height = (height // tile_h) * tile_h
    new_width = (width // tile_w) * tile_w
    image_resized = image[:new_height, :new_width]

    # 결과 이미지 초기화 / Initialize result image
    clahe_image = np.zeros_like(image_resized)

    # 각 타일별 히스토그램 평활화 수행 / Perform histogram equalization for each tile
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

    # 원본 크기로 복원 / Restore to original size
    if clahe_image.shape != image.shape:
        clahe_image_full = np.zeros_like(image)
        clahe_image_full[:new_height, :new_width] = clahe_image
        clahe_image_full[new_height:, :] = image[new_height:, :]
        clahe_image_full[:, new_width:] = image[:, new_width:]
        clahe_image = clahe_image_full

    # 처리 정보 저장 / Store processing information
    process_info = {
        'clip_limit': clip_limit,
        'tile_size': tile_size,
        'num_tiles': len(tile_info),
        'tile_info': tile_info[:4]  # 처음 4개 타일 정보만 저장 / Store only first 4 tiles info
    }

    # 중간 과정 시각화 / Visualize intermediate process
    if show_process:
        visualize_clahe_process(image, clahe_image, process_info)

    return clahe_image, process_info

def clip_histogram(histogram: np.ndarray, clip_limit: float, total_pixels: int) -> np.ndarray:
    """
    히스토그램을 클리핑합니다.
    Clip histogram.

    클리핑 과정:
    1. 클립 한계값 계산: (총 픽셀 수 / 256) * clip_limit
    2. 한계값을 초과하는 부분을 잘라내고 재분배
    3. 과도한 증폭을 방지하여 노이즈 감소 효과

    Clipping process:
    1. Calculate clip threshold: (total pixels / 256) * clip_limit
    2. Cut off parts exceeding threshold and redistribute
    3. Prevent excessive amplification to reduce noise

    Args:
        histogram (np.ndarray): 원본 히스토그램 / Original histogram
        clip_limit (float): 클립 한계값 / Clip limit
        total_pixels (int): 총 픽셀 수 / Total number of pixels

    Returns:
        np.ndarray: 클리핑된 히스토그램 / Clipped histogram
    """
    # 클립 임계값 계산 / Calculate clip threshold
    clip_threshold = (total_pixels / 256) * clip_limit

    # 히스토그램 클리핑 / Clip histogram
    clipped_hist = np.copy(histogram)
    excess = 0

    # 임계값을 초과하는 부분 계산 / Calculate excess parts
    for i in range(256):
        if clipped_hist[i] > clip_threshold:
            excess += clipped_hist[i] - clip_threshold
            clipped_hist[i] = clip_threshold

    # 잘린 부분을 균등하게 재분배 / Redistribute clipped parts evenly
    redistribution = excess / 256
    clipped_hist += redistribution

    return clipped_hist

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