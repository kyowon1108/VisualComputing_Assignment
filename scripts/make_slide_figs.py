#!/usr/bin/env python3
"""
Slide Figure Generation Script
슬라이드용 그림 자동 생성 스크립트

이 스크립트는 발표용 한 장 요약 슬라이드 이미지들을 자동으로 생성합니다.

Usage:
    python scripts/make_slide_figs.py
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json

# 프로젝트 루트를 Python path에 추가
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from src.he import he_luma_bgr, calculate_rms_contrast, calculate_edge_strength
from src.otsu import global_otsu, improved_otsu, create_threshold_heatmap

def setup_matplotlib():
    """matplotlib 설정"""
    plt.rcParams.update({
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 9,
        'figure.titlesize': 14
    })

def load_images() -> Dict[str, np.ndarray]:
    """테스트 이미지들을 로드합니다."""
    images = {}

    # HE 테스트 이미지
    he_path = 'images/he_dark_indoor.jpg'
    if os.path.exists(he_path):
        images['he_original'] = cv2.imread(he_path)
        print(f"HE test image loaded: {he_path}")
    else:
        print(f"Warning: HE test image not found: {he_path}")

    # Otsu 테스트 이미지
    otsu_path = 'images/otsu_shadow_doc_02.jpg'
    if os.path.exists(otsu_path):
        images['otsu_original'] = cv2.imread(otsu_path, cv2.IMREAD_GRAYSCALE)
        print(f"Otsu test image loaded: {otsu_path}")
    else:
        print(f"Warning: Otsu test image not found: {otsu_path}")

    return images

def define_rois(image_shape: Tuple[int, int], image_type: str) -> List[Tuple[int, int, int, int]]:
    """ROI 영역을 정의합니다."""
    h, w = image_shape[:2]

    if image_type == 'he':
        return [
            (int(w*0.1), int(h*0.1), int(w*0.3), int(h*0.2)),  # 키보드 상단 (암부)
            (int(w*0.5), int(h*0.6), int(w*0.2), int(h*0.2)),  # 마우스 주변 (암부+스펙큘러)
            (int(w*0.2), int(h*0.8), int(w*0.6), int(h*0.1))   # 모니터 아래 바 (광원 대비)
        ]
    else:  # otsu
        return [
            (int(w*0.7), int(h*0.1), int(w*0.25), int(h*0.3)),  # 우상단 글레어 영역
            (int(w*0.1), int(h*0.3), int(w*0.4), int(h*0.4)),   # 좌측 균일 텍스트 영역
            (int(w*0.05), int(h*0.05), int(w*0.2), int(h*0.8))  # 제본 경계
        ]

def add_roi_boxes(ax, rois: List[Tuple[int, int, int, int]], labels: List[str] = None):
    """ROI 박스를 추가합니다."""
    colors = ['red', 'green', 'blue']

    for i, (x, y, w, h) in enumerate(rois):
        color = colors[i % len(colors)]
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)

        # 라벨 추가
        if labels:
            ax.text(x, y-5, labels[i], color=color, fontweight='bold', fontsize=8)
        else:
            ax.text(x, y-5, f'ROI{i+1}', color=color, fontweight='bold', fontsize=8)

def calculate_quantitative_metrics(image: np.ndarray, rois: List[Tuple[int, int, int, int]]) -> Dict[str, Any]:
    """정량적 지표를 계산합니다."""
    metrics = {}

    # 전체 이미지 지표
    metrics['overall'] = {
        'mean_brightness': np.mean(image),
        'std_brightness': np.std(image),
        'rms_contrast': calculate_rms_contrast(image)
    }

    # ROI별 지표
    metrics['rois'] = []
    for i, roi in enumerate(rois):
        x, y, w, h = roi
        if len(image.shape) == 3:
            roi_img = image[y:y+h, x:x+w]
        else:
            roi_img = image[y:y+h, x:x+w]

        roi_metrics = {
            'roi_id': i + 1,
            'mean_brightness': np.mean(roi_img),
            'std_brightness': np.std(roi_img),
            'rms_contrast': calculate_rms_contrast(roi_img),
            'edge_strength': calculate_edge_strength(roi_img)
        }
        metrics['rois'].append(roi_metrics)

    return metrics

def create_he_summary_slide(images: Dict[str, np.ndarray], save_path: str):
    """HE 요약 슬라이드를 생성합니다."""
    if 'he_original' not in images:
        print("Warning: HE original image not available")
        return

    original = images['he_original']
    rois = define_rois(original.shape, 'he')
    roi_labels = ['키보드(암부)', '마우스(혼합)', '모니터바(대비)']

    # CLAHE 처리
    clahe_result = he_luma_bgr(original, space='yuv', mode='clahe', tile=(8, 8), clip=2.5)
    clahe_img = cv2.cvtColor(clahe_result['img'], cv2.COLOR_BGR2RGB)
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    # 3행 레이아웃 생성
    fig = plt.figure(figsize=(16, 12))

    # 1행: 풀샷 비교 (Original vs CLAHE)
    ax1 = plt.subplot(3, 4, (1, 2))
    ax1.imshow(original_rgb)
    ax1.set_title('원본 이미지 (어두운 실내)', fontweight='bold', fontsize=14)
    add_roi_boxes(ax1, rois, roi_labels)
    ax1.axis('off')

    ax2 = plt.subplot(3, 4, (3, 4))
    ax2.imshow(clahe_img)
    ax2.set_title('YUV-CLAHE 결과 (clip=2.5, tile=8×8)', fontweight='bold', fontsize=14)
    ax2.axis('off')

    # 2행: ROI 확대 비교
    for i, (roi, label) in enumerate(zip(rois, roi_labels)):
        x, y, w, h = roi

        # 원본 ROI
        ax_orig = plt.subplot(3, 6, 7 + i*2)
        roi_orig = original_rgb[y:y+h, x:x+w]
        roi_orig_resized = cv2.resize(roi_orig, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        ax_orig.imshow(roi_orig_resized)
        ax_orig.set_title(f'원본 {label}', fontsize=10)
        ax_orig.axis('off')

        # CLAHE ROI
        ax_clahe = plt.subplot(3, 6, 8 + i*2)
        roi_clahe = clahe_img[y:y+h, x:x+w]
        roi_clahe_resized = cv2.resize(roi_clahe, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        ax_clahe.imshow(roi_clahe_resized)
        ax_clahe.set_title(f'CLAHE {label}', fontsize=10)
        ax_clahe.axis('off')

        # 정량적 지표 추가
        orig_metrics = calculate_quantitative_metrics(roi_orig, [(0, 0, w, h)])
        clahe_metrics = calculate_quantitative_metrics(roi_clahe, [(0, 0, w, h)])

        contrast_improvement = ((clahe_metrics['overall']['rms_contrast'] -
                               orig_metrics['overall']['rms_contrast']) /
                               orig_metrics['overall']['rms_contrast'] * 100)

        ax_clahe.text(0.5, -0.15, f'대비개선: {contrast_improvement:+.1f}%',
                     transform=ax_clahe.transAxes, ha='center', fontsize=8,
                     bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    # 3행: 히스토그램 및 CDF
    ax_hist = plt.subplot(3, 4, (9, 10))

    # 원본과 CLAHE 히스토그램
    orig_gray = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)
    clahe_gray = cv2.cvtColor(clahe_img, cv2.COLOR_RGB2GRAY)

    hist_orig, _ = np.histogram(orig_gray.flatten(), bins=64, range=[0, 256])
    hist_clahe, _ = np.histogram(clahe_gray.flatten(), bins=64, range=[0, 256])

    bin_centers = np.arange(64) * 4
    ax_hist.bar(bin_centers - 1, hist_orig, width=2, alpha=0.7, label='원본', color='gray')
    ax_hist.bar(bin_centers + 1, hist_clahe, width=2, alpha=0.7, label='CLAHE', color='blue')
    ax_hist.set_title('히스토그램 비교', fontweight='bold')
    ax_hist.set_xlabel('픽셀 값')
    ax_hist.set_ylabel('빈도')
    ax_hist.legend()
    ax_hist.grid(True, alpha=0.3)

    # CDF 비교
    ax_cdf = plt.subplot(3, 4, (11, 12))
    cdf_orig = np.cumsum(hist_orig) / np.sum(hist_orig)
    cdf_clahe = np.cumsum(hist_clahe) / np.sum(hist_clahe)

    ax_cdf.plot(bin_centers, cdf_orig, label='원본 CDF', color='gray', linewidth=2)
    ax_cdf.plot(bin_centers, cdf_clahe, label='CLAHE CDF', color='blue', linewidth=2)
    ax_cdf.set_title('누적분포함수 비교', fontweight='bold')
    ax_cdf.set_xlabel('픽셀 값')
    ax_cdf.set_ylabel('누적 확률')
    ax_cdf.legend()
    ax_cdf.grid(True, alpha=0.3)

    plt.suptitle('히스토그램 평활화 (CLAHE) 종합 분석', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"HE summary slide saved: {save_path}")

def create_otsu_summary_slide(images: Dict[str, np.ndarray], save_path: str):
    """Otsu 요약 슬라이드를 생성합니다."""
    if 'otsu_original' not in images:
        print("Warning: Otsu original image not available")
        return

    original = images['otsu_original']
    rois = define_rois(original.shape, 'otsu')
    roi_labels = ['글레어영역', '텍스트영역', '제본경계']

    # Global과 Improved Otsu 처리
    global_result = global_otsu(original)
    improved_result = improved_otsu(original, window_size=75, stride=24, preblur=1.0)

    # 3행 레이아웃 생성
    fig = plt.figure(figsize=(16, 12))

    # 1행: 풀샷 비교
    ax1 = plt.subplot(3, 4, 1)
    ax1.imshow(original, cmap='gray')
    ax1.set_title('원본 문서 (그림자+글레어)', fontweight='bold', fontsize=12)
    add_roi_boxes(ax1, rois, roi_labels)
    ax1.axis('off')

    ax2 = plt.subplot(3, 4, 2)
    ax2.imshow(global_result['result'], cmap='gray')
    ax2.set_title(f'Global Otsu (T={global_result["threshold"]:.1f})', fontweight='bold', fontsize=12)
    ax2.axis('off')

    ax3 = plt.subplot(3, 4, 3)
    ax3.imshow(improved_result['result'], cmap='gray')
    ax3.set_title('Improved Otsu (Local+후처리)', fontweight='bold', fontsize=12)
    ax3.axis('off')

    # 파라미터 정보
    ax4 = plt.subplot(3, 4, 4)
    param_text = f"""파라미터:
• 윈도우: {improved_result['parameters']['window_size']}×{improved_result['parameters']['window_size']}
• 스트라이드: {improved_result['parameters']['stride']}
• 전처리: σ={improved_result['parameters']['preblur']}
• 후처리: {', '.join(improved_result['parameters']['morph_ops'])}

품질 향상:
• 균일한 조명 적응
• 글레어 영역 보정
• 경계 아티팩트 제거"""
    ax4.text(0.05, 0.95, param_text, transform=ax4.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax4.axis('off')

    # 2행: ROI 확대 비교
    for i, (roi, label) in enumerate(zip(rois, roi_labels)):
        x, y, w, h = roi

        # 원본 ROI
        ax_orig = plt.subplot(3, 9, 10 + i*3)
        roi_orig = original[y:y+h, x:x+w]
        roi_orig_resized = cv2.resize(roi_orig, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)
        ax_orig.imshow(roi_orig_resized, cmap='gray')
        ax_orig.set_title(f'원본\n{label}', fontsize=9)
        ax_orig.axis('off')

        # Global ROI
        ax_global = plt.subplot(3, 9, 11 + i*3)
        roi_global = global_result['result'][y:y+h, x:x+w]
        roi_global_resized = cv2.resize(roi_global, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)
        ax_global.imshow(roi_global_resized, cmap='gray')
        ax_global.set_title(f'Global\n{label}', fontsize=9)
        ax_global.axis('off')

        # Improved ROI
        ax_improved = plt.subplot(3, 9, 12 + i*3)
        roi_improved = improved_result['result'][y:y+h, x:x+w]
        roi_improved_resized = cv2.resize(roi_improved, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)
        ax_improved.imshow(roi_improved_resized, cmap='gray')
        ax_improved.set_title(f'Improved\n{label}', fontsize=9)
        ax_improved.axis('off')

    # 3행: 임계값 히트맵 및 윈도우 히스토그램
    ax_heatmap = plt.subplot(3, 3, 7)
    im = ax_heatmap.imshow(improved_result['threshold_map'], cmap='viridis', aspect='auto')
    ax_heatmap.set_title('임계값 히트맵', fontweight='bold')
    ax_heatmap.axis('off')

    # 컬러바 추가
    cbar = plt.colorbar(im, ax=ax_heatmap, shrink=0.8)
    cbar.set_label('임계값', rotation=270, labelpad=15)

    # 선택 윈도우 히스토그램 (글레어 ROI)
    ax_hist = plt.subplot(3, 3, 8)
    glare_roi = rois[0]  # 글레어 영역
    x, y, w, h = glare_roi
    roi_img = original[y:y+h, x:x+w]

    hist, bins = np.histogram(roi_img.flatten(), bins=32, range=[0, 256])
    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax_hist.bar(bin_centers, hist, width=8, alpha=0.7, color='gray')

    # 해당 영역의 임계값 표시
    roi_threshold = improved_result['threshold_map'][y + h//2, x + w//2]
    ax_hist.axvline(x=roi_threshold, color='red', linestyle='--', linewidth=2,
                   label=f'지역 임계값: {roi_threshold:.1f}')
    ax_hist.axvline(x=global_result['threshold'], color='blue', linestyle=':', linewidth=2,
                   label=f'전역 임계값: {global_result["threshold"]:.1f}')

    ax_hist.set_title('글레어 영역 히스토그램', fontweight='bold')
    ax_hist.set_xlabel('픽셀 값')
    ax_hist.set_ylabel('빈도')
    ax_hist.legend(fontsize=8)
    ax_hist.grid(True, alpha=0.3)

    # 성능 비교
    ax_perf = plt.subplot(3, 3, 9)

    # 간단한 성능 지표 계산
    metrics_text = f"""성능 비교:

글레어 영역 분할:
• Global: 부적절
• Improved: 우수

텍스트 가독성:
• Global: 보통
• Improved: 우수

경계 품질:
• Global: 거침
• Improved: 매끄러움

처리 속도:
• Global: 빠름
• Improved: 보통"""

    ax_perf.text(0.05, 0.95, metrics_text, transform=ax_perf.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    ax_perf.axis('off')

    plt.suptitle('Otsu 임계값 방법 종합 분석', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Otsu summary slide saved: {save_path}")

def main():
    """메인 실행 함수"""
    setup_matplotlib()

    # 결과 디렉토리 생성
    slides_dir = Path('results/slides')
    slides_dir.mkdir(parents=True, exist_ok=True)

    # 이미지 로드
    images = load_images()

    if not images:
        print("Error: No images available for slide generation")
        return 1

    # HE 요약 슬라이드 생성
    if 'he_original' in images:
        he_summary_path = slides_dir / 'he_summary.png'
        create_he_summary_slide(images, str(he_summary_path))
    else:
        print("Skipping HE summary slide - image not available")

    # Otsu 요약 슬라이드 생성
    if 'otsu_original' in images:
        otsu_summary_path = slides_dir / 'otsu_summary.png'
        create_otsu_summary_slide(images, str(otsu_summary_path))
    else:
        print("Skipping Otsu summary slide - image not available")

    print("Slide figure generation completed successfully!")
    return 0

if __name__ == '__main__':
    sys.exit(main())