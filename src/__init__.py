"""
비쥬얼컴퓨팅 과제1 소스 코드 패키지
Visual Computing Assignment 1 Source Code Package

이 패키지는 컬러 이미지 히스토그램 평활화와 Local Otsu Thresholding을 구현합니다.
This package implements color image histogram equalization and Local Otsu Thresholding.
"""

__version__ = "1.0.0"
__author__ = "Visual Computing Assignment 1"

# 주요 모듈 import / Import main modules
from .utils import *
from .he import *
from .otsu import *

__all__ = [
    # utils.py
    'load_image', 'save_image', 'rgb_to_yuv', 'yuv_to_rgb',
    'compute_histogram', 'plot_histogram', 'display_images',
    'validate_image_input', 'normalize_image', 'create_test_image',

    # he.py
    'histogram_equalization_grayscale', 'histogram_equalization_color',
    'clahe_implementation', 'calculate_cdf',

    # otsu.py
    'global_otsu_thresholding', 'local_otsu_block_based',
    'local_otsu_sliding_window', 'local_otsu_adaptive_block',
    'local_otsu_adaptive_sliding', 'local_otsu_block_opencv',
    'local_otsu_sliding_opencv', 'local_otsu_improved_boundary',
    'compare_otsu_methods', 'calculate_otsu_threshold', 'apply_threshold'
]