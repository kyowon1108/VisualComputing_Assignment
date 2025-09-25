"""
이미지 처리 공통 유틸리티 모듈
Common Image Processing Utilities Module

이 모듈은 이미지 로딩, 저장, 변환 등의 공통 기능을 제공합니다.
This module provides common functionalities for image loading, saving, and conversion.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from PIL import Image
import os
import platform
from typing import Tuple, Optional, Union

# matplotlib 한글 폰트 설정 / Setup Korean font for matplotlib
def setup_korean_font():
    """matplotlib에서 한글을 표시할 수 있도록 폰트를 설정합니다."""
    try:
        system = platform.system()
        if system == 'Darwin':  # macOS
            # macOS에서 사용 가능한 한글 폰트들
            korean_fonts = ['AppleGothic', 'Apple SD Gothic Neo', 'Nanum Gothic', 'Malgun Gothic']
        elif system == 'Windows':  # Windows
            korean_fonts = ['Malgun Gothic', 'Nanum Gothic', 'AppleGothic']
        else:  # Linux
            korean_fonts = ['Nanum Gothic', 'DejaVu Sans']

        # 사용 가능한 폰트 찾기
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        korean_font = None

        for font in korean_fonts:
            if font in available_fonts:
                korean_font = font
                break

        if korean_font:
            plt.rcParams['font.family'] = korean_font
        else:
            # 한글 폰트가 없는 경우 영어만 사용
            plt.rcParams['font.family'] = 'DejaVu Sans'

        # 마이너스 부호 깨짐 방지
        plt.rcParams['axes.unicode_minus'] = False

    except Exception:
        # 폰트 설정 실패 시 기본 설정 사용
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False

# 모듈 임포트 시 자동으로 폰트 설정
setup_korean_font()

def load_image(image_path: str, color_mode: str = 'color') -> np.ndarray:
    """
    이미지 파일을 로드합니다.
    Load an image file.

    Args:
        image_path (str): 이미지 파일 경로 / Image file path
        color_mode (str): 'color', 'gray', 'bgr' 중 선택 / Choose from 'color', 'gray', 'bgr'

    Returns:
        np.ndarray: 로드된 이미지 배열 / Loaded image array
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다 / Image file not found: {image_path}")

    if color_mode == 'gray':
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    elif color_mode == 'bgr':
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    else:  # color (RGB)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if image is None:
        raise ValueError(f"이미지를 로드할 수 없습니다 / Cannot load image: {image_path}")

    return image

def save_image(image: np.ndarray, save_path: str, color_mode: str = 'rgb') -> None:
    """
    이미지를 파일로 저장합니다.
    Save an image to file.

    Args:
        image (np.ndarray): 저장할 이미지 배열 / Image array to save
        save_path (str): 저장할 파일 경로 / File path to save
        color_mode (str): 'rgb', 'bgr', 'gray' 중 선택 / Choose from 'rgb', 'bgr', 'gray'
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if color_mode == 'gray' or len(image.shape) == 2:
        cv2.imwrite(save_path, image)
    elif color_mode == 'rgb':
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, image_bgr)
    else:  # bgr
        cv2.imwrite(save_path, image)

def rgb_to_yuv(rgb_image: np.ndarray) -> np.ndarray:
    """
    RGB 이미지를 YUV 색공간으로 변환합니다.
    Convert RGB image to YUV color space.

    컬러 히스토그램 평활화에서 Y(휘도) 채널만 처리하는 이론적 근거:
    - Y 채널은 이미지의 밝기 정보를 담고 있어 인간의 시각 인지와 밀접함
    - U, V 채널은 색상 정보로, 이를 그대로 유지하여 자연스러운 색감 보존

    Theoretical basis for processing only Y(luminance) channel in color histogram equalization:
    - Y channel contains brightness information closely related to human visual perception
    - U, V channels contain color information, preserving them maintains natural color tone

    Args:
        rgb_image (np.ndarray): RGB 이미지 / RGB image

    Returns:
        np.ndarray: YUV 이미지 / YUV image
    """
    return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2YUV)

def yuv_to_rgb(yuv_image: np.ndarray) -> np.ndarray:
    """
    YUV 이미지를 RGB 색공간으로 변환합니다.
    Convert YUV image to RGB color space.

    Args:
        yuv_image (np.ndarray): YUV 이미지 / YUV image

    Returns:
        np.ndarray: RGB 이미지 / RGB image
    """
    return cv2.cvtColor(yuv_image, cv2.COLOR_YUV2RGB)

def rgb_to_ycbcr(rgb_image: np.ndarray) -> np.ndarray:
    """
    RGB 이미지를 YCbCr 색공간으로 변환합니다.
    Convert RGB image to YCbCr color space.

    YCbCr 색공간의 이론적 근거:
    - Y 채널: 휘도 정보 (인간의 시각 인지와 밀접한 관련)
    - Cb, Cr 채널: 색차 정보 (색상 정보를 효율적으로 표현)
    - YUV와 유사하지만 디지털 비디오/이미지 처리에 최적화된 색공간

    Theoretical basis of YCbCr color space:
    - Y channel: Luminance information (closely related to human visual perception)
    - Cb, Cr channels: Chrominance information (efficiently represents color information)
    - Similar to YUV but optimized for digital video/image processing

    Args:
        rgb_image (np.ndarray): RGB 이미지 / RGB image

    Returns:
        np.ndarray: YCbCr 이미지 / YCbCr image
    """
    return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2YCrCb)

def ycbcr_to_rgb(ycbcr_image: np.ndarray) -> np.ndarray:
    """
    YCbCr 이미지를 RGB 색공간으로 변환합니다.
    Convert YCbCr image to RGB color space.

    Args:
        ycbcr_image (np.ndarray): YCbCr 이미지 / YCbCr image

    Returns:
        np.ndarray: RGB 이미지 / RGB image
    """
    return cv2.cvtColor(ycbcr_image, cv2.COLOR_YCrCb2RGB)

def rgb_to_lab(rgb_image: np.ndarray) -> np.ndarray:
    """
    RGB 이미지를 LAB 색공간으로 변환합니다.
    Convert RGB image to LAB color space.

    LAB 색공간의 이론적 근거:
    - L 채널: 명도(Lightness) 정보, 인간의 밝기 인지와 가장 밀접
    - A 채널: 녹색-적색 축의 색상 정보
    - B 채널: 파랑-노랑 축의 색상 정보
    - 색상의 균등한 분포로 CLAHE 적용에 이상적

    Theoretical basis of LAB color space:
    - L channel: Lightness information, most closely related to human brightness perception
    - A channel: Green-Red axis color information
    - B channel: Blue-Yellow axis color information
    - Ideal for CLAHE application due to uniform color distribution

    Args:
        rgb_image (np.ndarray): RGB 이미지 / RGB image

    Returns:
        np.ndarray: LAB 이미지 / LAB image
    """
    return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)

def lab_to_rgb(lab_image: np.ndarray) -> np.ndarray:
    """
    LAB 이미지를 RGB 색공간으로 변환합니다.
    Convert LAB image to RGB color space.

    Args:
        lab_image (np.ndarray): LAB 이미지 / LAB image

    Returns:
        np.ndarray: RGB 이미지 / RGB image
    """
    return cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB)

def rgb_to_hsv(rgb_image: np.ndarray) -> np.ndarray:
    """
    RGB 이미지를 HSV 색공간으로 변환합니다.
    Convert RGB image to HSV color space.

    HSV 색공간의 이론적 근거:
    - H 채널: 색상(Hue) 정보, 색상환에서의 위치
    - S 채널: 채도(Saturation) 정보, 색상의 순수함 정도
    - V 채널: 명도(Value/Brightness) 정보, 인간의 밝기 인지와 관련
    - V 채널만 처리하여 색상과 채도를 보존하면서 밝기 개선

    Theoretical basis of HSV color space:
    - H channel: Hue information, position on color wheel
    - S channel: Saturation information, purity of color
    - V channel: Value/Brightness information, related to human brightness perception
    - Processing only V channel preserves hue and saturation while improving brightness

    Args:
        rgb_image (np.ndarray): RGB 이미지 / RGB image

    Returns:
        np.ndarray: HSV 이미지 / HSV image
    """
    return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

def hsv_to_rgb(hsv_image: np.ndarray) -> np.ndarray:
    """
    HSV 이미지를 RGB 색공간으로 변환합니다.
    Convert HSV image to RGB color space.

    Args:
        hsv_image (np.ndarray): HSV 이미지 / HSV image

    Returns:
        np.ndarray: RGB 이미지 / RGB image
    """
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

def compute_histogram(image: np.ndarray, bins: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """
    이미지의 히스토그램을 계산합니다.
    Compute histogram of an image.

    Args:
        image (np.ndarray): 입력 이미지 (그레이스케일) / Input image (grayscale)
        bins (int): 히스토그램 빈 개수 / Number of histogram bins

    Returns:
        Tuple[np.ndarray, np.ndarray]: (히스토그램 값, 빈 경계) / (histogram values, bin edges)
    """
    if len(image.shape) != 2:
        raise ValueError("히스토그램 계산을 위해서는 그레이스케일 이미지가 필요합니다 / Grayscale image required for histogram computation")

    hist, bin_edges = np.histogram(image.flatten(), bins=bins, range=(0, 256))
    return hist, bin_edges

def plot_histogram(image: np.ndarray, title: str = "Histogram", color: str = 'blue') -> None:
    """
    이미지의 히스토그램을 플롯합니다.
    Plot histogram of an image.

    Args:
        image (np.ndarray): 입력 이미지 / Input image
        title (str): 플롯 제목 / Plot title
        color (str): 히스토그램 색상 / Histogram color
    """
    plt.figure(figsize=(10, 4))

    if len(image.shape) == 2:  # 그레이스케일 / Grayscale
        hist, bins = compute_histogram(image)
        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.plot(bins[:-1], hist, color=color)
        plt.title(title)
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
    else:  # 컬러 이미지 / Color image
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        colors = ['red', 'green', 'blue']
        for i, color in enumerate(colors):
            hist, bins = compute_histogram(image[:, :, i])
            plt.plot(bins[:-1], hist, color=color, alpha=0.7, label=f'{color.upper()} channel')
        plt.title(title)
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def display_images(images: list, titles: list, figsize: Tuple[int, int] = (15, 5)) -> None:
    """
    여러 이미지를 나란히 표시합니다.
    Display multiple images side by side.

    Args:
        images (list): 표시할 이미지 리스트 / List of images to display
        titles (list): 각 이미지의 제목 리스트 / List of titles for each image
        figsize (Tuple[int, int]): 그림 크기 / Figure size
    """
    n_images = len(images)
    plt.figure(figsize=figsize)

    for i, (image, title) in enumerate(zip(images, titles)):
        plt.subplot(1, n_images, i + 1)
        if len(image.shape) == 2:
            plt.imshow(image, cmap='gray')
        else:
            plt.imshow(image)
        plt.title(title)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def validate_image_input(image: np.ndarray) -> None:
    """
    이미지 입력의 유효성을 검사합니다.
    Validate image input.

    Args:
        image (np.ndarray): 검사할 이미지 / Image to validate

    Raises:
        ValueError: 유효하지 않은 이미지인 경우 / If image is invalid
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("이미지는 numpy 배열이어야 합니다 / Image must be a numpy array")

    if image.size == 0:
        raise ValueError("이미지가 비어있습니다 / Image is empty")

    if len(image.shape) not in [2, 3]:
        raise ValueError("이미지는 2D(그레이스케일) 또는 3D(컬러)여야 합니다 / Image must be 2D (grayscale) or 3D (color)")

    if len(image.shape) == 3 and image.shape[2] not in [1, 3, 4]:
        raise ValueError("컬러 이미지는 1, 3, 또는 4개의 채널을 가져야 합니다 / Color image must have 1, 3, or 4 channels")

def normalize_image(image: np.ndarray, target_range: Tuple[int, int] = (0, 255)) -> np.ndarray:
    """
    이미지 픽셀 값을 지정된 범위로 정규화합니다.
    Normalize image pixel values to specified range.

    Args:
        image (np.ndarray): 입력 이미지 / Input image
        target_range (Tuple[int, int]): 목표 범위 / Target range

    Returns:
        np.ndarray: 정규화된 이미지 / Normalized image
    """
    min_val, max_val = target_range

    # 현재 이미지의 최소/최대값 계산 / Calculate current min/max values
    img_min = np.min(image)
    img_max = np.max(image)

    if img_max - img_min == 0:
        return np.full_like(image, min_val, dtype=np.uint8)

    # 정규화 수행 / Perform normalization
    normalized = (image - img_min) / (img_max - img_min) * (max_val - min_val) + min_val

    return normalized.astype(np.uint8)

def create_test_image(size: Tuple[int, int] = (200, 200), pattern: str = 'gradient') -> np.ndarray:
    """
    테스트용 이미지를 생성합니다.
    Create a test image.

    Args:
        size (Tuple[int, int]): 이미지 크기 (높이, 너비) / Image size (height, width)
        pattern (str): 패턴 종류 ('gradient', 'checkerboard', 'noise') / Pattern type

    Returns:
        np.ndarray: 생성된 테스트 이미지 / Generated test image
    """
    height, width = size

    if pattern == 'gradient':
        # 수평 그라디언트 생성 / Create horizontal gradient
        image = np.tile(np.linspace(0, 255, width, dtype=np.uint8), (height, 1))
    elif pattern == 'checkerboard':
        # 체스판 패턴 생성 / Create checkerboard pattern
        check_size = 20
        image = np.zeros((height, width), dtype=np.uint8)
        for i in range(0, height, check_size):
            for j in range(0, width, check_size):
                if (i // check_size + j // check_size) % 2 == 0:
                    image[i:i+check_size, j:j+check_size] = 255
    elif pattern == 'noise':
        # 랜덤 노이즈 이미지 생성 / Create random noise image
        image = np.random.randint(0, 256, (height, width), dtype=np.uint8)
    else:
        raise ValueError(f"지원하지 않는 패턴입니다 / Unsupported pattern: {pattern}")

    return image

# OpenCV 내장 함수 사용 예시 (주석으로 표시) / Example usage of OpenCV built-in functions (commented)
def example_opencv_usage():
    """
    OpenCV 내장 함수 사용 예시를 보여줍니다.
    Show examples of using OpenCV built-in functions.
    """
    pass
    # 히스토그램 평활화 예시 / Histogram equalization example:
    # equalized = cv2.equalizeHist(gray_image)

    # CLAHE 적용 예시 / CLAHE application example:
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # equalized = clahe.apply(gray_image)

    # Otsu 임계값 적용 예시 / Otsu thresholding example:
    # ret, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)