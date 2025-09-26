#!/usr/bin/env python3
"""
Parameter Ablation Study Script
파라미터 탐색 실험 자동화 스크립트

이 스크립트는 HE와 Local Otsu의 다양한 파라미터 조합을 자동으로 테스트하고
결과를 분석하여 보고서용 표와 그래프를 생성합니다.

Usage:
    python scripts/ablation.py
"""

import os
import sys
import json
import logging
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import product
from typing import Dict, List, Tuple, Any
import time

# 프로젝트 루트를 Python path에 추가
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from src.he import he_luma_bgr, extract_roi_metrics
from src.otsu import global_otsu, improved_otsu

def setup_logging():
    """로깅 설정"""
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/ablation.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def load_test_images() -> Dict[str, np.ndarray]:
    """테스트 이미지들을 로드합니다."""
    images = {}

    # HE 테스트 이미지
    he_path = 'images/he_dark_indoor.jpg'
    if os.path.exists(he_path):
        images['he_dark_indoor'] = cv2.imread(he_path)
        logging.info(f"HE test image loaded: {he_path}")
    else:
        logging.warning(f"HE test image not found: {he_path}")

    # Otsu 테스트 이미지
    otsu_path = 'images/otsu_shadow_doc_02.jpg'
    if os.path.exists(otsu_path):
        images['otsu_shadow_doc'] = cv2.imread(otsu_path, cv2.IMREAD_GRAYSCALE)
        logging.info(f"Otsu test image loaded: {otsu_path}")
    else:
        logging.warning(f"Otsu test image not found: {otsu_path}")

    return images

def define_he_parameter_grid() -> Dict[str, List]:
    """HE 파라미터 그리드를 정의합니다."""
    return {
        'tile_sizes': [(4, 4), (8, 8), (16, 16)],
        'clip_limits': [1.5, 2.0, 2.5, 3.0, 4.0],
        'spaces': ['yuv', 'ycbcr', 'lab'],
        'modes': ['clahe']
    }

def define_otsu_parameter_grid() -> Dict[str, List]:
    """Otsu 파라미터 그리드를 정의합니다."""
    return {
        'window_sizes': [51, 75, 101],
        'strides': [16, 24, 32],
        'preblurs': [0.0, 0.8, 1.2]
    }

def define_default_rois(image_shape: Tuple[int, int], image_type: str) -> List[Tuple[int, int, int, int]]:
    """기본 ROI들을 정의합니다."""
    h, w = image_shape[:2]

    if image_type == 'he':
        # HE용 ROI (키보드, 마우스, 모니터)
        return [
            (int(w*0.1), int(h*0.1), int(w*0.3), int(h*0.2)),  # 키보드 하우징 상단
            (int(w*0.5), int(h*0.6), int(w*0.2), int(h*0.2)),  # 마우스 주변
            (int(w*0.2), int(h*0.8), int(w*0.6), int(h*0.1))   # 모니터 아래 바
        ]
    else:  # otsu
        # Otsu용 ROI (글레어, 텍스트, 경계)
        return [
            (int(w*0.7), int(h*0.1), int(w*0.25), int(h*0.3)),  # 우상단 글레어 영역
            (int(w*0.1), int(h*0.3), int(w*0.4), int(h*0.4)),   # 좌측 균일 텍스트 영역
            (int(w*0.05), int(h*0.05), int(w*0.2), int(h*0.8))  # 제본 경계
        ]

def run_he_ablation(images: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
    """HE 파라미터 탐색을 실행합니다."""
    if 'he_dark_indoor' not in images:
        logging.error("HE test image not available, skipping HE ablation")
        return []

    image = images['he_dark_indoor']
    rois = define_default_rois(image.shape, 'he')
    param_grid = define_he_parameter_grid()

    results = []
    total_combinations = len(param_grid['tile_sizes']) * len(param_grid['clip_limits']) * len(param_grid['spaces'])
    current = 0

    logging.info(f"Starting HE ablation with {total_combinations} parameter combinations...")

    for tile_size, clip_limit, space in product(
        param_grid['tile_sizes'],
        param_grid['clip_limits'],
        param_grid['spaces']
    ):
        current += 1
        logging.info(f"HE Progress: {current}/{total_combinations} - tile={tile_size}, clip={clip_limit}, space={space}")

        try:
            start_time = time.time()

            # HE 적용
            result = he_luma_bgr(
                image,
                space=space,
                mode='clahe',
                tile=tile_size,
                clip=clip_limit,
                bins=256
            )

            processing_time = time.time() - start_time

            # 전체 이미지 지표
            result_rgb = cv2.cvtColor(result["img"], cv2.COLOR_BGR2RGB)
            original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            overall_metrics = {
                'mean_brightness': np.mean(result_rgb),
                'brightness_std': np.std(result_rgb),
                'brightness_change': np.mean(result_rgb) - np.mean(original_rgb)
            }

            # ROI별 지표
            roi_metrics = []
            for roi_idx, roi in enumerate(rois):
                metrics = extract_roi_metrics(result["img"], roi)
                metrics['roi_id'] = roi_idx + 1
                roi_metrics.append(metrics)

            # 결과 저장
            result_record = {
                'method': 'CLAHE',
                'space': space,
                'tile_size': f"{tile_size[0]}x{tile_size[1]}",
                'clip_limit': clip_limit,
                'processing_time': processing_time,
                'overall_metrics': overall_metrics,
                'roi_metrics': roi_metrics,
                'success': True
            }

            results.append(result_record)

        except Exception as e:
            logging.error(f"HE ablation failed for tile={tile_size}, clip={clip_limit}, space={space}: {e}")
            results.append({
                'method': 'CLAHE',
                'space': space,
                'tile_size': f"{tile_size[0]}x{tile_size[1]}",
                'clip_limit': clip_limit,
                'processing_time': 0,
                'overall_metrics': None,
                'roi_metrics': None,
                'success': False,
                'error': str(e)
            })

    logging.info(f"HE ablation completed: {len([r for r in results if r['success']])} successful, {len([r for r in results if not r['success']])} failed")
    return results

def run_otsu_ablation(images: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
    """Otsu 파라미터 탐색을 실행합니다."""
    if 'otsu_shadow_doc' not in images:
        logging.error("Otsu test image not available, skipping Otsu ablation")
        return []

    image = images['otsu_shadow_doc']
    rois = define_default_rois(image.shape, 'otsu')
    param_grid = define_otsu_parameter_grid()

    results = []
    total_combinations = len(param_grid['window_sizes']) * len(param_grid['strides']) * len(param_grid['preblurs'])
    current = 0

    logging.info(f"Starting Otsu ablation with {total_combinations} parameter combinations...")

    for window_size, stride, preblur in product(
        param_grid['window_sizes'],
        param_grid['strides'],
        param_grid['preblurs']
    ):
        current += 1
        logging.info(f"Otsu Progress: {current}/{total_combinations} - window={window_size}, stride={stride}, preblur={preblur}")

        try:
            start_time = time.time()

            # Improved Otsu 적용
            result = improved_otsu(
                image,
                window_size=window_size,
                stride=stride,
                preblur=preblur,
                morph_ops=['open,3', 'close,3']
            )

            processing_time = time.time() - start_time

            # 전체 이미지 지표
            result_img = result['result']
            overall_metrics = {
                'mean_brightness': np.mean(result_img),
                'brightness_std': np.std(result_img),
                'binary_ratio': np.mean(result_img > 127)  # 이진화 비율
            }

            # ROI별 지표
            roi_metrics = []
            for roi_idx, roi in enumerate(rois):
                x, y, w, h = roi
                roi_img = result_img[y:y+h, x:x+w]

                # 에지 강도 계산
                sobel_x = cv2.Sobel(roi_img, cv2.CV_64F, 1, 0, ksize=3)
                sobel_y = cv2.Sobel(roi_img, cv2.CV_64F, 0, 1, ksize=3)
                edge_strength = np.sum(np.sqrt(sobel_x**2 + sobel_y**2))

                metrics = {
                    'roi_id': roi_idx + 1,
                    'mean_brightness': np.mean(roi_img),
                    'brightness_std': np.std(roi_img),
                    'edge_strength': edge_strength,
                    'binary_ratio': np.mean(roi_img > 127)
                }
                roi_metrics.append(metrics)

            # 결과 저장
            result_record = {
                'method': 'Improved_Otsu',
                'window_size': window_size,
                'stride': stride,
                'preblur': preblur,
                'processing_time': processing_time,
                'overall_metrics': overall_metrics,
                'roi_metrics': roi_metrics,
                'success': True
            }

            results.append(result_record)

        except Exception as e:
            logging.error(f"Otsu ablation failed for window={window_size}, stride={stride}, preblur={preblur}: {e}")
            results.append({
                'method': 'Improved_Otsu',
                'window_size': window_size,
                'stride': stride,
                'preblur': preblur,
                'processing_time': 0,
                'overall_metrics': None,
                'roi_metrics': None,
                'success': False,
                'error': str(e)
            })

    logging.info(f"Otsu ablation completed: {len([r for r in results if r['success']])} successful, {len([r for r in results if not r['success']])} failed")
    return results

def create_he_results_csv(results: List[Dict[str, Any]], save_path: str):
    """HE 결과를 CSV로 저장합니다."""
    rows = []

    for result in results:
        if not result['success']:
            continue

        base_row = {
            'method': result['method'],
            'space': result['space'],
            'tile_size': result['tile_size'],
            'clip_limit': result['clip_limit'],
            'processing_time': result['processing_time']
        }

        # 전체 지표 추가
        if result['overall_metrics']:
            for key, value in result['overall_metrics'].items():
                base_row[f'overall_{key}'] = value

        # ROI별 지표 추가
        if result['roi_metrics']:
            for roi_metrics in result['roi_metrics']:
                row = base_row.copy()
                row['roi_id'] = roi_metrics['roi_id']
                for key, value in roi_metrics.items():
                    if key != 'roi_id':
                        row[f'roi_{key}'] = value
                rows.append(row)
        else:
            rows.append(base_row)

    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    logging.info(f"HE results CSV saved: {save_path}")

def create_otsu_results_csv(results: List[Dict[str, Any]], save_path: str):
    """Otsu 결과를 CSV로 저장합니다."""
    rows = []

    for result in results:
        if not result['success']:
            continue

        base_row = {
            'method': result['method'],
            'window_size': result['window_size'],
            'stride': result['stride'],
            'preblur': result['preblur'],
            'processing_time': result['processing_time']
        }

        # 전체 지표 추가
        if result['overall_metrics']:
            for key, value in result['overall_metrics'].items():
                base_row[f'overall_{key}'] = value

        # ROI별 지표 추가
        if result['roi_metrics']:
            for roi_metrics in result['roi_metrics']:
                row = base_row.copy()
                row['roi_id'] = roi_metrics['roi_id']
                for key, value in roi_metrics.items():
                    if key != 'roi_id':
                        row[f'roi_{key}'] = value
                rows.append(row)
        else:
            rows.append(base_row)

    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    logging.info(f"Otsu results CSV saved: {save_path}")

def find_best_configurations(results: List[Dict[str, Any]], method_type: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """최고 성능 설정들을 찾습니다."""
    successful_results = [r for r in results if r['success']]

    if method_type == 'he':
        # HE는 RMS contrast 기준으로 정렬
        key_func = lambda x: sum([roi['rms_contrast'] for roi in x['roi_metrics']]) if x['roi_metrics'] else 0
    else:  # otsu
        # Otsu는 edge strength 기준으로 정렬
        key_func = lambda x: sum([roi['edge_strength'] for roi in x['roi_metrics']]) if x['roi_metrics'] else 0

    sorted_results = sorted(successful_results, key=key_func, reverse=True)
    return sorted_results[:top_k]

def create_montage(images_data: List[Tuple[np.ndarray, str]], save_path: str, grid_size: Tuple[int, int] = None):
    """썸네일 몽타주를 생성합니다."""
    if not images_data:
        logging.warning("No images provided for montage")
        return

    n_images = len(images_data)
    if grid_size is None:
        cols = int(np.ceil(np.sqrt(n_images)))
        rows = int(np.ceil(n_images / cols))
    else:
        rows, cols = grid_size

    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))

    if rows == 1:
        axes = [axes] if cols == 1 else axes
    elif cols == 1:
        axes = [[ax] for ax in axes]
    else:
        axes = axes.reshape(rows, cols) if rows > 1 and cols > 1 else axes

    for i, (img, title) in enumerate(images_data):
        row = i // cols
        col = i % cols

        ax = axes[row][col] if rows > 1 else axes[col]

        if len(img.shape) == 3:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(img, cmap='gray')

        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.axis('off')

    # 빈 subplot 숨기기
    for i in range(n_images, rows * cols):
        row = i // cols
        col = i % cols
        ax = axes[row][col] if rows > 1 else axes[col]
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    logging.info(f"Montage saved: {save_path}")

def main():
    """메인 실행 함수"""
    setup_logging()
    logging.info("Starting parameter ablation study...")

    # 결과 디렉토리 생성
    results_dir = Path('results/ablation')
    results_dir.mkdir(parents=True, exist_ok=True)

    # 테스트 이미지 로드
    images = load_test_images()

    if not images:
        logging.error("No test images available, exiting...")
        return 1

    # HE 파라미터 탐색
    he_results = run_he_ablation(images)
    if he_results:
        # CSV 저장
        he_csv_path = results_dir / 'ablation_he.csv'
        create_he_results_csv(he_results, str(he_csv_path))

        # 최고 성능 설정 저장
        best_he = find_best_configurations(he_results, 'he', 3)
        he_best_path = results_dir / 'ablation_he_best.json'
        with open(he_best_path, 'w') as f:
            json.dump(best_he, f, indent=2, default=str)
        logging.info(f"Best HE configurations saved: {he_best_path}")

    # Otsu 파라미터 탐색
    otsu_results = run_otsu_ablation(images)
    if otsu_results:
        # CSV 저장
        otsu_csv_path = results_dir / 'ablation_otsu.csv'
        create_otsu_results_csv(otsu_results, str(otsu_csv_path))

        # 최고 성능 설정 저장
        best_otsu = find_best_configurations(otsu_results, 'otsu', 3)
        otsu_best_path = results_dir / 'ablation_otsu_best.json'
        with open(otsu_best_path, 'w') as f:
            json.dump(best_otsu, f, indent=2, default=str)
        logging.info(f"Best Otsu configurations saved: {otsu_best_path}")

    logging.info("Parameter ablation study completed successfully!")
    return 0

if __name__ == '__main__':
    sys.exit(main())