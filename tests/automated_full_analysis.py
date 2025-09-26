#!/usr/bin/env python3
"""
완전 자동화된 포괄적 분석 스크립트
Fully Automated Comprehensive Analysis Script

he_dark_indoor.jpg와 otsu_shadow_doc_02.jpg에 대한 모든 과정을 자동으로 수행하고
모든 figure를 저장합니다. 화면 출력 없이 백그라운드에서 실행됩니다.
"""

import sys
import os
import matplotlib
matplotlib.use('Agg')  # 화면 출력 없이 파일로만 저장
import matplotlib.pyplot as plt

# 프로젝트 루트 디렉토리를 Python path에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import cv2
from src.he import histogram_equalization_color, histogram_equalization_grayscale, clahe_implementation
from src.otsu import compare_otsu_methods, global_otsu_thresholding
from src.utils import load_image, compute_histogram
import time
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class AutomatedFullAnalyzer:
    """완전 자동화된 분석 클래스"""

    def __init__(self):
        self.he_image_path = "images/he_dark_indoor.jpg"
        self.otsu_image_path = "images/otsu_shadow_doc_02.jpg"
        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)

    def run_complete_analysis(self):
        """모든 분석을 자동으로 실행"""
        print("=== 완전 자동화된 포괄적 분석 시작 ===")

        # 1. HE 분석
        print("1. HE 분석 수행 중...")
        he_results = self.analyze_he_comprehensive()

        # 2. Otsu 분석
        print("2. Otsu 분석 수행 중...")
        otsu_results = self.analyze_otsu_comprehensive()

        # 3. 단계별 시각화
        print("3. 단계별 시각화 생성 중...")
        self.create_step_by_step_visualizations(he_results, otsu_results)

        # 4. 비교 분석
        print("4. 비교 분석 생성 중...")
        self.create_comparative_analysis(he_results, otsu_results)

        # 5. 성능 분석
        print("5. 성능 분석 생성 중...")
        self.create_performance_analysis(he_results, otsu_results)

        print("=== 모든 분석 완료 ===")
        return he_results, otsu_results

    def analyze_he_comprehensive(self):
        """포괄적 HE 분석"""
        image = load_image(self.he_image_path, color_mode='color')
        print(f"  HE 이미지 로드: {image.shape}")

        colorspaces = ['yuv', 'ycbcr', 'lab', 'hsv', 'rgb']
        algorithms = ['he', 'clahe']

        results = {'original_image': image, 'methods': {}}

        for colorspace in colorspaces:
            for algorithm in algorithms:
                if colorspace == 'rgb' and algorithm == 'clahe':
                    continue

                combo_name = f"{colorspace}_{algorithm}"
                print(f"    처리 중: {combo_name}")

                start_time = time.time()
                try:
                    enhanced, process_info = histogram_equalization_color(
                        image, method=colorspace, algorithm=algorithm,
                        clip_limit=2.0, tile_size=8, show_process=False
                    )

                    # 품질 지표 계산
                    quality_metrics = self.calculate_quality_metrics(image, enhanced)

                    results['methods'][combo_name] = {
                        'enhanced_image': enhanced,
                        'process_info': process_info,
                        'processing_time': time.time() - start_time,
                        'quality_metrics': quality_metrics,
                        'colorspace': colorspace,
                        'algorithm': algorithm
                    }

                except Exception as e:
                    print(f"      실패: {str(e)}")

        return results

    def analyze_otsu_comprehensive(self):
        """포괄적 Otsu 분석"""
        image_color = load_image(self.otsu_image_path, color_mode='color')
        image_gray = load_image(self.otsu_image_path, color_mode='gray')
        print(f"  Otsu 이미지 로드: {image_gray.shape}")

        # Otsu 방법들 실행
        otsu_results = compare_otsu_methods(image_gray, show_comparison=False)

        # 개별 방법들도 실행하여 세부 정보 수집
        global_result = global_otsu_thresholding(image_gray, show_process=False)

        results = {
            'original_color': image_color,
            'original_gray': image_gray,
            'methods': otsu_results,
            'global_detailed': global_result
        }

        return results

    def calculate_quality_metrics(self, original: np.ndarray, enhanced: np.ndarray) -> Dict:
        """이미지 품질 지표 계산"""
        orig_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        enh_gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)

        # 대비 지표
        orig_contrast = np.std(orig_gray)
        enh_contrast = np.std(enh_gray)
        contrast_improvement = (enh_contrast - orig_contrast) / orig_contrast * 100 if orig_contrast > 0 else 0

        # 히스토그램 분포
        orig_hist = cv2.calcHist([orig_gray], [0], None, [256], [0, 256])
        enh_hist = cv2.calcHist([enh_gray], [0], None, [256], [0, 256])

        # 엔트로피
        orig_entropy = self.calculate_entropy(orig_hist)
        enh_entropy = self.calculate_entropy(enh_hist)

        # 밝기 통계
        orig_brightness = np.mean(orig_gray)
        enh_brightness = np.mean(enh_gray)

        return {
            'original_contrast': float(orig_contrast),
            'enhanced_contrast': float(enh_contrast),
            'contrast_improvement_percent': float(contrast_improvement),
            'original_entropy': float(orig_entropy),
            'enhanced_entropy': float(enh_entropy),
            'original_brightness': float(orig_brightness),
            'enhanced_brightness': float(enh_brightness),
            'brightness_change': float(enh_brightness - orig_brightness)
        }

    def calculate_entropy(self, histogram):
        """히스토그램으로부터 엔트로피 계산"""
        hist_norm = histogram.flatten() / np.sum(histogram)
        hist_norm = hist_norm[hist_norm > 0]
        entropy = -np.sum(hist_norm * np.log2(hist_norm)) if len(hist_norm) > 0 else 0
        return entropy

    def create_step_by_step_visualizations(self, he_results, otsu_results):
        """단계별 시각화 생성"""
        print("  단계별 시각화 생성...")

        # 1. HE 4단계 과정 (YUV 예시)
        self.create_he_4step_process(he_results)

        # 2. Otsu 계산 과정
        self.create_otsu_calculation_process(otsu_results)

        # 3. 컬러스페이스별 채널 분석
        self.create_colorspace_channel_analysis(he_results)

    def create_he_4step_process(self, he_results):
        """HE 4단계 과정 시각화"""
        original = he_results['original_image']

        # YUV 방법으로 단계별 처리
        from src.utils import rgb_to_yuv, yuv_to_rgb

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1단계: 원본 이미지
        axes[0, 0].imshow(original)
        axes[0, 0].set_title('1단계: 원본 RGB 이미지\\nOriginal RGB Image', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')

        # 2단계: YUV로 변환, Y 채널 분리
        yuv_image = rgb_to_yuv(original)
        y_channel = yuv_image[:, :, 0]
        axes[0, 1].imshow(y_channel, cmap='gray')
        axes[0, 1].set_title('2단계: YUV 변환 후 Y 채널\\nY Channel after YUV Conversion', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')

        # 3단계: Y 채널에 HE 적용
        y_enhanced, _ = histogram_equalization_grayscale(y_channel, show_process=False)
        axes[1, 0].imshow(y_enhanced, cmap='gray')
        axes[1, 0].set_title('3단계: Y 채널에 HE 적용\\nHistogram Equalization on Y Channel', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')

        # 4단계: RGB로 역변환
        yuv_enhanced = yuv_image.copy()
        yuv_enhanced[:, :, 0] = y_enhanced
        rgb_final = yuv_to_rgb(yuv_enhanced)
        axes[1, 1].imshow(rgb_final)
        axes[1, 1].set_title('4단계: 최종 RGB 결과\\nFinal RGB Result', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')

        plt.suptitle('히스토그램 평활화 4단계 과정 (YUV 방법)\\nHistogram Equalization 4-Step Process (YUV Method)',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        save_path = os.path.join(self.results_dir, 'he_4step_process.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    저장: {save_path}")

    def create_otsu_calculation_process(self, otsu_results):
        """Otsu 계산 과정 시각화"""
        original = otsu_results['original_gray']

        # 히스토그램 계산
        hist, _ = compute_histogram(original)

        # Otsu 임계값 계산 과정 시각화
        from src.otsu import calculate_otsu_threshold
        threshold, calc_info = calculate_otsu_threshold(hist, show_process=False)

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 원본 이미지
        axes[0, 0].imshow(original, cmap='gray')
        axes[0, 0].set_title('원본 그림자 문서\\nOriginal Shadow Document', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')

        # 히스토그램
        axes[0, 1].bar(range(256), hist, alpha=0.7, color='gray')
        axes[0, 1].axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'Optimal Threshold: {threshold}')
        axes[0, 1].set_title('히스토그램과 최적 임계값\\nHistogram and Optimal Threshold', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Pixel Intensity')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 이진화 결과
        binary_result = np.where(original > threshold, 255, 0).astype(np.uint8)
        axes[0, 2].imshow(binary_result, cmap='gray')
        axes[0, 2].set_title(f'이진화 결과 (임계값: {threshold})\\nBinary Result (Threshold: {threshold})', fontsize=14, fontweight='bold')
        axes[0, 2].axis('off')

        # 클래스 간 분산 그래프
        if 'between_class_variance' in calc_info:
            variances = calc_info['between_class_variance']
            axes[1, 0].plot(range(len(variances)), variances, 'b-', linewidth=2)
            axes[1, 0].axvline(x=threshold, color='red', linestyle='--', linewidth=2)
            axes[1, 0].set_title('클래스 간 분산\\nBetween-Class Variance', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Threshold Value')
            axes[1, 0].set_ylabel('Variance')
            axes[1, 0].grid(True, alpha=0.3)

        # 클래스 확률
        axes[1, 1].bar(['Background', 'Foreground'],
                      [calc_info.get('background_prob', 0.5), calc_info.get('foreground_prob', 0.5)],
                      color=['lightblue', 'lightcoral'])
        axes[1, 1].set_title('클래스 확률\\nClass Probabilities', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('Probability')

        # 통계 정보
        stats_text = f"""Otsu 분석 결과:

최적 임계값: {threshold}
배경 평균: {calc_info.get('background_mean', 0):.1f}
전경 평균: {calc_info.get('foreground_mean', 0):.1f}
배경 확률: {calc_info.get('background_prob', 0):.3f}
전경 확률: {calc_info.get('foreground_prob', 0):.3f}
클래스간 분산: {calc_info.get('max_between_variance', 0):.1f}
"""

        axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes,
                        fontsize=12, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        axes[1, 2].set_title('통계 정보\\nStatistical Information', fontsize=14, fontweight='bold')
        axes[1, 2].axis('off')

        plt.suptitle('Otsu 임계값 계산 과정\\nOtsu Thresholding Calculation Process',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        save_path = os.path.join(self.results_dir, 'otsu_calculation_process.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    저장: {save_path}")

    def create_colorspace_channel_analysis(self, he_results):
        """컬러스페이스 채널 분석"""
        original = he_results['original_image']

        from src.utils import rgb_to_yuv, rgb_to_ycbcr, rgb_to_lab, rgb_to_hsv

        fig, axes = plt.subplots(3, 4, figsize=(20, 15))

        # 원본 RGB
        axes[0, 0].imshow(original)
        axes[0, 0].set_title('Original RGB', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')

        # YUV 분해
        yuv = rgb_to_yuv(original)
        for i, (channel, name) in enumerate(zip([yuv[:,:,0], yuv[:,:,1], yuv[:,:,2]], ['Y', 'U', 'V'])):
            axes[0, i+1].imshow(channel, cmap='gray' if i == 0 else 'RdBu_r')
            axes[0, i+1].set_title(f'YUV - {name} Channel', fontsize=12)
            axes[0, i+1].axis('off')

        # YCbCr 분해
        ycbcr = rgb_to_ycbcr(original)
        for i, (channel, name) in enumerate(zip([ycbcr[:,:,0], ycbcr[:,:,1], ycbcr[:,:,2]], ['Y', 'Cb', 'Cr'])):
            if i == 0:
                axes[1, i].imshow(channel, cmap='gray')
                axes[1, i].set_title('YCbCr - Y Channel', fontsize=12)
            else:
                axes[1, i].imshow(channel, cmap='RdBu_r')
                axes[1, i].set_title(f'YCbCr - {name} Channel', fontsize=12)
            axes[1, i].axis('off')

        # 히스토그램 비교
        axes[1, 3].hist(yuv[:,:,0].flatten(), bins=50, alpha=0.5, color='blue', label='YUV-Y', density=True)
        axes[1, 3].hist(ycbcr[:,:,0].flatten(), bins=50, alpha=0.5, color='red', label='YCbCr-Y', density=True)
        axes[1, 3].set_title('Y Channel Comparison', fontsize=12)
        axes[1, 3].legend()
        axes[1, 3].grid(True, alpha=0.3)

        # LAB 분해
        lab = rgb_to_lab(original)
        for i, (channel, name) in enumerate(zip([lab[:,:,0], lab[:,:,1], lab[:,:,2]], ['L', 'A', 'B'])):
            axes[2, i].imshow(channel, cmap='gray' if i == 0 else 'RdBu_r')
            axes[2, i].set_title(f'LAB - {name} Channel', fontsize=12)
            axes[2, i].axis('off')

        # HSV V 채널
        hsv = rgb_to_hsv(original)
        axes[2, 3].imshow(hsv[:,:,2], cmap='gray')
        axes[2, 3].set_title('HSV - V Channel', fontsize=12)
        axes[2, 3].axis('off')

        plt.suptitle('컬러스페이스별 채널 분해 분석\\nColorspace Channel Decomposition Analysis',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        save_path = os.path.join(self.results_dir, 'colorspace_channels_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    저장: {save_path}")

    def create_comparative_analysis(self, he_results, otsu_results):
        """비교 분석 생성"""
        print("  비교 분석 생성...")

        # 1. HE 전체 조합 비교
        self.create_he_comprehensive_comparison(he_results)

        # 2. Otsu 방법들 비교
        self.create_otsu_methods_comparison(otsu_results)

        # 3. HE vs Otsu 교차 비교
        self.create_he_vs_otsu_comparison(he_results, otsu_results)

    def create_he_comprehensive_comparison(self, he_results):
        """HE 포괄적 비교"""
        original = he_results['original_image']
        methods = he_results['methods']

        n_methods = len(methods)
        n_cols = 5
        n_rows = (n_methods + 1 + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(25, 5*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        # 원본
        axes[0, 0].imshow(original)
        axes[0, 0].set_title('Original\\n(Dark Indoor)', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')

        # 각 방법 결과
        for i, (method_name, result) in enumerate(methods.items()):
            row = (i + 1) // n_cols
            col = (i + 1) % n_cols

            if row < n_rows and col < n_cols:
                axes[row, col].imshow(result['enhanced_image'])

                title = f"{result['colorspace'].upper()}+{result['algorithm'].upper()}\\n"
                title += f"대비: {result['quality_metrics']['contrast_improvement_percent']:+.1f}%\\n"
                title += f"시간: {result['processing_time']:.3f}s"

                axes[row, col].set_title(title, fontsize=10)
                axes[row, col].axis('off')

        # 빈 칸 처리
        for i in range(n_methods + 1, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            if row < n_rows and col < n_cols:
                axes[row, col].axis('off')

        plt.suptitle('히스토그램 평활화 종합 비교 (어두운 실내 이미지)\\nHistogram Equalization Comprehensive Comparison (Dark Indoor Image)',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        save_path = os.path.join(self.results_dir, 'he_comprehensive_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    저장: {save_path}")

    def create_otsu_methods_comparison(self, otsu_results):
        """Otsu 방법들 비교"""
        original = otsu_results['original_gray']
        methods = otsu_results['methods']

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 원본
        axes[0, 0].imshow(original, cmap='gray')
        axes[0, 0].set_title('Original Shadow Document', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')

        # 각 방법 결과
        method_names = list(methods.keys())
        positions = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

        for i, method_name in enumerate(method_names[:5]):  # 최대 5개 방법
            if i < len(positions):
                row, col = positions[i]
                result = methods[method_name]

                if 'result' in result:
                    axes[row, col].imshow(result['result'], cmap='gray')

                    threshold = result.get('threshold', 'Adaptive')
                    if isinstance(threshold, (int, float)):
                        title = f"{method_name.replace('_', ' ').title()}\\nThreshold: {threshold:.1f}"
                    else:
                        title = f"{method_name.replace('_', ' ').title()}\\nThreshold: {threshold}"

                    axes[row, col].set_title(title, fontsize=12)
                    axes[row, col].axis('off')

        plt.suptitle('Otsu 임계값 방법들 비교 (그림자 문서)\\nOtsu Thresholding Methods Comparison (Shadow Document)',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        save_path = os.path.join(self.results_dir, 'otsu_methods_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    저장: {save_path}")

    def create_he_vs_otsu_comparison(self, he_results, otsu_results):
        """HE vs Otsu 교차 비교"""
        # 최고 성능 HE 방법 찾기
        he_methods = he_results['methods']
        best_he = max(he_methods.items(), key=lambda x: x[1]['quality_metrics']['contrast_improvement_percent'])

        # 적절한 Otsu 방법 선택
        otsu_methods = otsu_results['methods']
        global_otsu = otsu_methods.get('global_otsu', list(otsu_methods.values())[0])

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # HE 상단 행
        he_original = he_results['original_image']
        he_enhanced = best_he[1]['enhanced_image']
        he_perf = best_he[1]['quality_metrics']

        axes[0, 0].imshow(he_original)
        axes[0, 0].set_title('HE: Original\\n(Dark Indoor)', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(he_enhanced)
        axes[0, 1].set_title(f'HE: Enhanced\\n({best_he[0].replace("_", "+").upper()})', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')

        # HE 히스토그램
        he_orig_gray = cv2.cvtColor(he_original, cv2.COLOR_RGB2GRAY)
        he_enh_gray = cv2.cvtColor(he_enhanced, cv2.COLOR_RGB2GRAY)
        axes[0, 2].hist(he_orig_gray.flatten(), bins=50, alpha=0.5, color='blue', density=True, label='Original')
        axes[0, 2].hist(he_enh_gray.flatten(), bins=50, alpha=0.5, color='red', density=True, label='Enhanced')
        axes[0, 2].set_title('HE: Histogram Comparison', fontsize=14, fontweight='bold')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # Otsu 하단 행
        otsu_original = otsu_results['original_gray']
        otsu_binary = global_otsu['result'] if 'result' in global_otsu else otsu_original
        otsu_threshold = global_otsu.get('threshold', 'Auto')

        axes[1, 0].imshow(otsu_original, cmap='gray')
        axes[1, 0].set_title('Otsu: Original\\n(Shadow Document)', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(otsu_binary, cmap='gray')
        if isinstance(otsu_threshold, (int, float)):
            axes[1, 1].set_title(f'Otsu: Binary\\n(Threshold: {otsu_threshold:.1f})', fontsize=14, fontweight='bold')
        else:
            axes[1, 1].set_title('Otsu: Binary\\n(Global Method)', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')

        # Otsu 히스토그램
        axes[1, 2].hist(otsu_original.flatten(), bins=50, alpha=0.7, color='gray', density=True)
        if isinstance(otsu_threshold, (int, float)):
            axes[1, 2].axvline(x=otsu_threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {otsu_threshold:.1f}')
            axes[1, 2].legend()
        axes[1, 2].set_title('Otsu: Histogram & Threshold', fontsize=14, fontweight='bold')
        axes[1, 2].grid(True, alpha=0.3)

        plt.suptitle('HE vs Otsu 교차 비교: 서로 다른 목적과 응용\\nHE vs Otsu Cross-Comparison: Different Purposes and Applications',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        save_path = os.path.join(self.results_dir, 'he_vs_otsu_cross_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    저장: {save_path}")

    def create_performance_analysis(self, he_results, otsu_results):
        """성능 분석 생성"""
        print("  성능 분석 생성...")

        methods = he_results['methods']
        method_names = list(methods.keys())

        # 데이터 추출
        processing_times = [methods[name]['processing_time'] for name in method_names]
        contrast_improvements = [methods[name]['quality_metrics']['contrast_improvement_percent'] for name in method_names]
        brightness_changes = [methods[name]['quality_metrics']['brightness_change'] for name in method_names]

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 색상 구분
        colors = ['skyblue' if 'he' in name else 'lightcoral' for name in method_names]

        # 1. 처리 시간
        bars1 = ax1.bar(range(len(method_names)), processing_times, color=colors)
        ax1.set_title('Processing Time Comparison\\n(Dark Indoor Image)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_xticks(range(len(method_names)))
        ax1.set_xticklabels([name.replace('_', '+').upper() for name in method_names], rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)

        # 2. 대비 개선
        bars2 = ax2.bar(range(len(method_names)), contrast_improvements, color=colors)
        ax2.set_title('Contrast Improvement\\n(Dark Indoor Image)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Contrast Improvement (%)')
        ax2.set_xticks(range(len(method_names)))
        ax2.set_xticklabels([name.replace('_', '+').upper() for name in method_names], rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        # 3. 밝기 변화
        bars3 = ax3.bar(range(len(method_names)), brightness_changes, color=colors)
        ax3.set_title('Brightness Change\\n(Dark Indoor Image)', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Brightness Change')
        ax3.set_xticks(range(len(method_names)))
        ax3.set_xticklabels([name.replace('_', '+').upper() for name in method_names], rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        # 4. 시간 vs 성능 산점도
        scatter = ax4.scatter(processing_times, contrast_improvements, c=[0 if 'he' in name else 1 for name in method_names],
                             cmap='RdYlBu', s=100, alpha=0.7)
        ax4.set_title('Processing Time vs Performance\\n(Dark Indoor Image)', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Processing Time (seconds)')
        ax4.set_ylabel('Contrast Improvement (%)')
        ax4.grid(True, alpha=0.3)

        # 점에 라벨 추가
        for i, name in enumerate(method_names):
            ax4.annotate(name.replace('_', '+').upper(), (processing_times[i], contrast_improvements[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)

        # 범례
        import matplotlib.patches as patches
        he_patch = patches.Patch(color='skyblue', label='HE Algorithm')
        clahe_patch = patches.Patch(color='lightcoral', label='CLAHE Algorithm')
        fig.legend(handles=[he_patch, clahe_patch], loc='upper right')

        plt.suptitle('성능 분석: 히스토그램 평활화 방법들\\nPerformance Analysis: Histogram Equalization Methods',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        save_path = os.path.join(self.results_dir, 'performance_analysis_detailed.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    저장: {save_path}")

def main():
    """메인 실행 함수"""
    analyzer = AutomatedFullAnalyzer()
    he_results, otsu_results = analyzer.run_complete_analysis()

    print("\\n=== 생성된 분석 이미지 ===")
    results_dir = "results"
    generated_files = []
    for file in os.listdir(results_dir):
        if file.endswith('.png') and file not in ['COMPREHENSIVE_ANALYSIS_REPORT.md']:
            generated_files.append(file)

    for file in sorted(generated_files):
        print(f"  - {file}")

    return he_results, otsu_results

if __name__ == "__main__":
    main()