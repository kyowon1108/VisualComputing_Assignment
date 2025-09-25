#!/usr/bin/env python3
"""
향상된 Visualization 분석 Script
Improved Visualization 분석 Script

1. Korean Font Issue Solution
2. 과정 세분화 (Each Step별 개별 figure)
3. 비교 Image에 빨간 박스 강조 Addition
"""

import sys
import os
import matplotlib
matplotlib.use('Agg')  # 화면 출력 없이 파일로만 저장

# Korean Font 설정
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# macOS Korean Font 설정
try:
    # AppleGothic 또는 다른 Korean Font 사용
    plt.rcParams['font.family'] = ['AppleGothic', 'Malgun Gothic', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
except:
    # Font 설정 실패시 기본 설정 유지
    pass

# 프로젝트 루트 디렉토리를 Python path에 Addition
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import cv2
from src.he import histogram_equalization_color, histogram_equalization_grayscale
from src.otsu import compare_otsu_methods, global_otsu_thresholding
from src.utils import load_image, compute_histogram, rgb_to_yuv, yuv_to_rgb, rgb_to_ycbcr, rgb_to_lab, rgb_to_hsv
import time
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')
from matplotlib.patches import Rectangle

class ImprovedVisualizationAnalyzer:
    """향상된 Visualization 분석 Class"""

    def __init__(self):
        self.he_image_path = "images/he_dark_indoor.jpg"
        self.otsu_image_path = "images/otsu_shadow_doc_02.jpg"
        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)

    def run_improved_analysis(self):
        """향상된 분석 Execution - 수업 발표용 과정 In Progress심 분석"""
        print("=== 향상된 Visualization 분석 Start ===")

        # PRESENTATION NOTE: 1Step - Issue 정의 및 입력 데이터 분석
        # 실제 Image 처리에서 겪는 Issue점들:
        # - 어두운 실내 Image: 디테일 손실, 낮은 가시성
        # - 기존 RGB 직접 처리의 한계: 색상 왜곡 Issue
        print("1. HE 분석 Performing In Progress...")
        he_results = self.analyze_he_comprehensive()

        # PRESENTATION NOTE: 2Step - 이진화 Issue와 자동 Solution방안
        # 문서 처리의 핵심 Issue:
        # - 수동 임계값 설정의 한계
        # - 조명 변화에 따른 성능 저하
        # - Otsu의 자동 임계값 결정의 필요성
        print("2. Otsu 분석 Performing In Progress...")
        otsu_results = self.analyze_otsu_comprehensive()

        # PRESENTATION NOTE: 3Step - Solution 과정의 Step별 분해
        # 복잡한 알고리즘을 이해하기 위한 과정:
        # - 블랙박스가 아닌 투명한 과정 제시
        # - Each Step에서 무엇이 어떻게 변하는지 Visualization
        print("3. Detailed 과정 Visualization...")
        self.create_detailed_step_visualizations(he_results, otsu_results)

        # PRESENTATION NOTE: 4Step - 방법별 심화 분석
        # 단순 비교를 넘어선 심층 이해:
        # - Each 방법이 언제, 왜 좋은 성능을 보이는가?
        # - 실패 케이스와 성공 케이스의 차이점
        print("4. 개별 방법별 분석...")
        self.create_individual_method_analysis(he_results, otsu_results)

        # PRESENTATION NOTE: 5Step - 비교를 통한 최적해 도출
        # 실험 결과를 바탕으로 한 실용적 결론:
        # - 어떤 상황에서 어떤 방법을 사용해야 하는가?
        # - 성능과 계산 비용의 트레이드오프
        print("5. 강조 비교 분석...")
        self.create_highlighted_comparisons(he_results, otsu_results)

        print("=== 향상된 분석 Complete ===")
        return he_results, otsu_results

    def analyze_he_comprehensive(self):
        """HE 포괄적 분석 - 어두운 Image 개선 Issue Solution 과정"""
        # PRESENTATION NOTE: Issue 상황 - 어두운 실내 환경에서 촬영된 Image
        # 기존 방법의 한계점: RGB 직접 처리 시 색상 왜곡 발생
        image = load_image(self.he_image_path, color_mode='color')
        print(f"  HE Image 로드: {image.shape}")

        # PRESENTATION NOTE: Solution방안 탐색 - 다양한 Colorspace 활용
        # YUV/YCbCr: 휘도와 색상 분리로 색상 보존 극대화
        # LAB: 인간 시Each과 유사한 색상 표현
        # HSV: 색상, 채도, 명도 독립적 처리
        # RGB: 기존 방법과의 비교 기준
        colorspaces = ['yuv', 'ycbcr', 'lab', 'hsv', 'rgb']

        # PRESENTATION NOTE: 두 가지 알고리즘 비교
        # HE: 전역적 대비 향상, 강력하지만 과도한 처리 위험
        # CLAHE: 지역적 적응 처리, 자연스럽지만 개선 효과 제한
        algorithms = ['he', 'clahe']

        results = {'original_image': image, 'methods': {}}

        for colorspace in colorspaces:
            for algorithm in algorithms:
                if colorspace == 'rgb' and algorithm == 'clahe':
                    continue

                combo_name = f"{colorspace}_{algorithm}"
                print(f"    처리 In Progress: {combo_name}")

                start_time = time.time()
                try:
                    enhanced, process_info = histogram_equalization_color(
                        image, method=colorspace, algorithm=algorithm,
                        clip_limit=2.0, tile_size=8, show_process=False
                    )

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
        """Otsu 포괄적 분석"""
        image_color = load_image(self.otsu_image_path, color_mode='color')
        image_gray = load_image(self.otsu_image_path, color_mode='gray')
        print(f"  Otsu Image 로드: {image_gray.shape}")

        otsu_results = compare_otsu_methods(image_gray, show_comparison=False)
        global_result = global_otsu_thresholding(image_gray, show_process=False)

        results = {
            'original_color': image_color,
            'original_gray': image_gray,
            'methods': otsu_results,
            'global_detailed': global_result
        }

        return results

    def calculate_quality_metrics(self, original: np.ndarray, enhanced: np.ndarray) -> Dict:
        """Image 품질 지표 계산"""
        orig_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        enh_gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)

        orig_contrast = np.std(orig_gray)
        enh_contrast = np.std(enh_gray)
        contrast_improvement = (enh_contrast - orig_contrast) / orig_contrast * 100 if orig_contrast > 0 else 0

        orig_hist = cv2.calcHist([orig_gray], [0], None, [256], [0, 256])
        enh_hist = cv2.calcHist([enh_gray], [0], None, [256], [0, 256])

        orig_entropy = self.calculate_entropy(orig_hist)
        enh_entropy = self.calculate_entropy(enh_hist)

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

    def create_detailed_step_visualizations(self, he_results, otsu_results):
        """Detailed Step별 Visualization"""
        print("  Detailed 과정 Visualization...")

        # 1. HE Each Step별 개별 분석
        self.create_he_step1_original_analysis(he_results)
        self.create_he_step2_colorspace_conversion(he_results)
        self.create_he_step3_channel_processing(he_results)
        self.create_he_step4_final_results(he_results)

        # 2. Otsu Each Step별 개별 분석
        self.create_otsu_step1_histogram_analysis(otsu_results)
        self.create_otsu_step2_threshold_calculation(otsu_results)
        self.create_otsu_step3_binarization_process(otsu_results)

    def create_he_step1_original_analysis(self, he_results):
        """HE 1단계: 원본 이미지 분석"""
        original = he_results['original_image']

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 원본 이미지
        axes[0, 0].imshow(original)
        axes[0, 0].set_title('원본 이미지', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')

        # RGB 히스토그램
        colors = ['red', 'green', 'blue']
        for i, color in enumerate(colors):
            hist = cv2.calcHist([original], [i], None, [256], [0, 256])
            axes[0, 1].plot(hist, color=color, alpha=0.7, label=f'{color.upper()} channel')
        axes[0, 1].set_title('RGB 히스토그램', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('픽셀 강도')
        axes[0, 1].set_ylabel('빈도')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 그레이스케일 Conversion
        gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        axes[1, 0].imshow(gray, cmap='gray')
        axes[1, 0].set_title('그레이스케일', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')

        # 그레이스케일 히스토그램
        gray_hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        axes[1, 1].plot(gray_hist, color='gray', linewidth=2)
        axes[1, 1].fill_between(range(256), gray_hist.flatten(), alpha=0.3, color='gray')
        axes[1, 1].set_title('그레이스케일 히스토그램', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('픽셀 강도')
        axes[1, 1].set_ylabel('빈도')
        axes[1, 1].grid(True, alpha=0.3)

        # 통계 정보는 마크다운 파일에 포함 (텍스트 박스 제거)

        plt.suptitle('1단계: 원본 이미지 분석', fontsize=16, fontweight='bold')
        plt.tight_layout()

        save_path = os.path.join(self.results_dir, 'he_step1_original_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    저장: he_step1_original_analysis.png")

    def create_he_step2_colorspace_conversion(self, he_results):
        """HE 2단계: 색공간 변환"""
        original = he_results['original_image']

        # Each Colorspace별로 개별 figure 생성
        colorspaces = [
            ('YUV', rgb_to_yuv, ['Y', 'U', 'V']),
            ('YCbCr', rgb_to_ycbcr, ['Y', 'Cb', 'Cr']),
            ('LAB', rgb_to_lab, ['L', 'A', 'B']),
            ('HSV', rgb_to_hsv, ['H', 'S', 'V'])
        ]

        for space_name, convert_func, channel_names in colorspaces:
            fig, axes = plt.subplots(2, 4, figsize=(20, 12))

            # 원본 이미지
            axes[0, 0].imshow(original)
            axes[0, 0].set_title('원본 RGB', fontsize=12, fontweight='bold')
            axes[0, 0].axis('off')

            # Conversion된 Image
            converted = convert_func(original)

            # Each 채널 표시
            for i in range(3):
                channel = converted[:, :, i]

                # 첫 번째 채널(휘도)는 그레이스케일, 나머지는 컬러맵
                if i == 0:
                    axes[0, i+1].imshow(channel, cmap='gray')
                else:
                    axes[0, i+1].imshow(channel, cmap='RdBu_r')

                axes[0, i+1].set_title(f'{space_name} - {channel_names[i]} 채널', fontsize=12)
                axes[0, i+1].axis('off')

            # Each 채널의 히스토그램
            for i in range(3):
                channel = converted[:, :, i]
                hist = cv2.calcHist([channel.astype(np.uint8)], [0], None, [256], [0, 256])

                color = 'gray' if i == 0 else ['red', 'blue'][i-1]
                axes[1, i].plot(hist, color=color, linewidth=2)
                axes[1, i].fill_between(range(256), hist.flatten(), alpha=0.3, color=color)
                axes[1, i].set_title(f'{channel_names[i]} 채널 히스토그램', fontsize=12)
                axes[1, i].set_xlabel('픽셀 강도')
                axes[1, i].set_ylabel('빈도')
                axes[1, i].grid(True, alpha=0.3)

            plt.suptitle(f'2단계: {space_name} 색공간 변환\n{space_name} 색공간 변환',
                        fontsize=16, fontweight='bold')
            plt.tight_layout()

            save_path = os.path.join(self.results_dir, f'he_step2_{space_name.lower()}_conversion.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"    저장: he_step2_{space_name.lower()}_conversion.png")

    def create_he_step3_channel_processing(self, he_results):
        """HE 3단계: 채널 처리 과정"""
        original = he_results['original_image']

        # YUV를 예시로 상세한 처리 과정 표시
        yuv_image = rgb_to_yuv(original)
        y_channel = yuv_image[:, :, 0]

        # HE 처리
        y_he, he_info = histogram_equalization_grayscale(y_channel, show_process=False)

        # CLAHE 처리
        clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        y_clahe = clahe_obj.apply(y_channel)

        fig, axes = plt.subplots(3, 3, figsize=(18, 16))

        # Original Y 채널
        axes[0, 0].imshow(y_channel, cmap='gray')
        axes[0, 0].set_title('원본 Y 채널', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')

        # HE 적용 Y 채널
        axes[0, 1].imshow(y_he, cmap='gray')
        axes[0, 1].set_title('HE 적용 Y 채널', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')

        # CLAHE 적용 Y 채널
        axes[0, 2].imshow(y_clahe, cmap='gray')
        axes[0, 2].set_title('CLAHE 적용 Y 채널', fontsize=12, fontweight='bold')
        axes[0, 2].axis('off')

        # 히스토그램 비교
        orig_hist = cv2.calcHist([y_channel], [0], None, [256], [0, 256])
        he_hist = cv2.calcHist([y_he], [0], None, [256], [0, 256])
        clahe_hist = cv2.calcHist([y_clahe], [0], None, [256], [0, 256])

        axes[1, 0].plot(orig_hist, color='blue', linewidth=2, label='Original')
        axes[1, 0].fill_between(range(256), orig_hist.flatten(), alpha=0.3, color='blue')
        axes[1, 0].set_title('원본 히스토그램', fontsize=12)
        axes[1, 0].set_xlabel('픽셀 강도')
        axes[1, 0].set_ylabel('빈도')
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(he_hist, color='red', linewidth=2, label='HE')
        axes[1, 1].fill_between(range(256), he_hist.flatten(), alpha=0.3, color='red')
        axes[1, 1].set_title('HE 히스토그램', fontsize=12)
        axes[1, 1].set_xlabel('픽셀 강도')
        axes[1, 1].set_ylabel('빈도')
        axes[1, 1].grid(True, alpha=0.3)

        axes[1, 2].plot(clahe_hist, color='green', linewidth=2, label='CLAHE')
        axes[1, 2].fill_between(range(256), clahe_hist.flatten(), alpha=0.3, color='green')
        axes[1, 2].set_title('CLAHE 히스토그램', fontsize=12)
        axes[1, 2].set_xlabel('픽셀 강도')
        axes[1, 2].set_ylabel('빈도')
        axes[1, 2].grid(True, alpha=0.3)

        # CDF 비교 (HE만)
        if 'cdf' in he_info:
            cdf = he_info['cdf']
            axes[2, 0].plot(cdf, color='orange', linewidth=2)
            axes[2, 0].set_title('누적분포함수 (CDF)\nCumulative Distribution Function', fontsize=12)
            axes[2, 0].set_xlabel('픽셀 강도')
            axes[2, 0].set_ylabel('Cumulative Probability')
            axes[2, 0].grid(True, alpha=0.3)

        # Conversion 함수
        axes[2, 1].plot(range(256), [int(255 * cdf[i]) if 'cdf' in he_info else i for i in range(256)],
                       color='purple', linewidth=2)
        axes[2, 1].set_title('Conversion 함수\nTransformation Function', fontsize=12)
        axes[2, 1].set_xlabel('입력 강도')
        axes[2, 1].set_ylabel('출력 강도')
        axes[2, 1].grid(True, alpha=0.3)

        # 통계 정보는 마크다운 파일에 포함 (텍스트 박스 제거)
        axes[2, 2].set_title('통계적 비교', fontsize=12)
        axes[2, 2].axis('off')

        plt.suptitle('3단계: Y 채널 처리 과정 상세 분석',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        save_path = os.path.join(self.results_dir, 'he_step3_channel_processing.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    저장: he_step3_channel_processing.png")

    def create_he_step4_final_results(self, he_results):
        """HE Step 4: 최종 결과 비교"""
        original = he_results['original_image']
        methods = he_results['methods']

        # 상위 5개 방법만 선별
        sorted_methods = sorted(methods.items(),
                              key=lambda x: x[1]['quality_metrics']['contrast_improvement_percent'],
                              reverse=True)[:5]

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 원본 이미지
        axes[0, 0].imshow(original)
        axes[0, 0].set_title('원본 이미지', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')

        # 상위 5개 결과
        positions = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

        for i, (method_name, result) in enumerate(sorted_methods):
            if i < len(positions):
                row, col = positions[i]

                axes[row, col].imshow(result['enhanced_image'])

                title = f"{result['colorspace'].upper()}+{result['algorithm'].upper()}\n"
                title += f"대비개선: {result['quality_metrics']['contrast_improvement_percent']:+.1f}%\n"
                title += f"처리시간: {result['processing_time']:.3f}초"

                axes[row, col].set_title(title, fontsize=11)
                axes[row, col].axis('off')

        plt.suptitle('4단계: 최종 결과 비교 (상위 5개 방법)',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        save_path = os.path.join(self.results_dir, 'he_step4_final_results.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    저장: he_step4_final_results.png")

    def create_otsu_step1_histogram_analysis(self, otsu_results):
        """Otsu Step 1: 히스토그램 분석"""
        original = otsu_results['original_gray']

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 원본 이미지
        axes[0, 0].imshow(original, cmap='gray')
        axes[0, 0].set_title('원본 문서 이미지', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')

        # 히스토그램
        hist, _ = compute_histogram(original)
        axes[0, 1].bar(range(256), hist, alpha=0.7, color='gray', edgecolor='black')
        axes[0, 1].set_title('히스토그램', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('픽셀 강도')
        axes[0, 1].set_ylabel('빈도')
        axes[0, 1].grid(True, alpha=0.3)

        # 누적 히스토그램
        cumsum_hist = np.cumsum(hist)
        axes[1, 0].plot(cumsum_hist, color='blue', linewidth=2)
        axes[1, 0].fill_between(range(256), cumsum_hist, alpha=0.3, color='blue')
        axes[1, 0].set_title('누적 히스토그램', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('픽셀 강도')
        axes[1, 0].set_ylabel('누적 빈도')
        axes[1, 0].grid(True, alpha=0.3)

        # 통계 분석
        stats_text = f"""Image 통계:
크기: {original.shape[1]} × {original.shape[0]}
총 픽셀: {original.shape[0] * original.shape[1]:,}

밝기 통계:
평균: {np.mean(original):.1f}
In Progress앙값: {np.median(original):.1f}
표준편차: {np.std(original):.1f}
최소값: {original.min()}
최대값: {original.max()}

히스토그램 특성:
모드: {np.argmax(hist)}
최대 빈도: {np.max(hist):,}
비어있지 않은 빈: {np.count_nonzero(hist)}/256"""

        # 통계 정보는 마크다운 파일에 포함 (텍스트 박스 제거)
        axes[1, 1].set_title('통계적 분석', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')

        plt.suptitle('Otsu 1단계: 히스토그램 분석',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        save_path = os.path.join(self.results_dir, 'otsu_step1_histogram_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    저장: otsu_step1_histogram_analysis.png")

    def create_otsu_step2_threshold_calculation(self, otsu_results):
        """Otsu 2단계: 임계값 계산 과정"""
        original = otsu_results['original_gray']

        # 히스토그램 계산
        hist, _ = compute_histogram(original)

        # Otsu 임계값 계산
        from src.otsu import calculate_otsu_threshold
        threshold, calc_info = calculate_otsu_threshold(hist, show_process=False)

        fig, axes = plt.subplots(2, 3, figsize=(20, 8))

        # Original 히스토그램
        axes[0, 0].bar(range(256), hist, alpha=0.7, color='gray')
        axes[0, 0].axvline(x=threshold, color='red', linestyle='--', linewidth=3, label=f'최적 임계값: {threshold}')
        axes[0, 0].set_title('히스토그램과 최적 임계값', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('픽셀 강도')
        axes[0, 0].set_ylabel('빈도')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Class 간 분산 계산 과정
        if 'between_class_variance' in calc_info:
            variances = calc_info['between_class_variance']
            axes[0, 1].plot(range(len(variances)), variances, 'b-', linewidth=2)
            axes[0, 1].axvline(x=threshold, color='red', linestyle='--', linewidth=3)
            axes[0, 1].set_title('클래스 간 분산', fontsize=12, fontweight='bold')
            axes[0, 1].set_xlabel('임계값 Value')
            axes[0, 1].set_ylabel('Between-Class Variance')
            axes[0, 1].grid(True, alpha=0.3)

        # Class 확률 계산
        background_prob = calc_info.get('background_prob', 0.5)
        foreground_prob = calc_info.get('foreground_prob', 0.5)

        bars = axes[0, 2].bar(['배경', '전경'],
                             [background_prob, foreground_prob],
                             color=['lightblue', 'lightcoral'])
        axes[0, 2].set_title('클래스 확률', fontsize=12, fontweight='bold')
        axes[0, 2].set_ylabel('Probability')

        # 막대에 값 표시
        for bar, prob in zip(bars, [background_prob, foreground_prob]):
            axes[0, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{prob:.3f}', ha='center', va='bottom', fontweight='bold')

        # Class 평균 비교
        bg_mean = calc_info.get('background_mean', 0)
        fg_mean = calc_info.get('foreground_mean', 0)

        bars = axes[1, 0].bar(['배경 평균', '전경 평균'],
                             [bg_mean, fg_mean],
                             color=['darkblue', 'darkred'])
        axes[1, 0].set_title('클래스별 평균 밝기', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Mean Intensity')

        # 막대에 값 표시
        for bar, mean in zip(bars, [bg_mean, fg_mean]):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                           f'{mean:.1f}', ha='center', va='bottom', fontweight='bold')

        # Otsu 공식 설명
        formula_text = """Otsu 방법의 수학적 원리:

1. Class 간 분산 최대화:
   σ²ʙ(t) = ω₀(t) × ω₁(t) × [μ₀(t) - μ₁(t)]²

2. 최적 임계값:
   t* = arg max σ²ʙ(t)

여기서:
- ω₀(t), ω₁(t): 배경, 전경 확률
- μ₀(t), μ₁(t): 배경, 전경 평균
- t*: 최적 임계값

계산 결과:
- 최적 임계값: {threshold}
- 최대 Class간 분산: 계산됨
- 분리도: {separation:.1f}"""

        separation = (bg_mean - fg_mean)**2 if bg_mean and fg_mean else 0
        # 통계 정보는 마크다운 파일에 포함 (텍스트 박스 제거)
        axes[1, 1].set_title('Otsu 공식 및 계산 결과', fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')

        # 임계값 효과 미리보기
        binary_preview = np.where(original > threshold, 255, 0).astype(np.uint8)
        axes[1, 2].imshow(binary_preview, cmap='gray')
        axes[1, 2].set_title(f'이진화 미리보기 (임계값: {threshold})',
                           fontsize=12, fontweight='bold')
        axes[1, 2].axis('off')

        plt.suptitle('Otsu 2단계: 임계값 계산 과정',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        save_path = os.path.join(self.results_dir, 'otsu_step2_threshold_calculation.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    저장: otsu_step2_threshold_calculation.png")

    def create_otsu_step3_binarization_process(self, otsu_results):
        """Otsu 3단계: 이진화 과정"""
        original = otsu_results['original_gray']
        methods = otsu_results['methods']

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 원본 이미지
        axes[0, 0].imshow(original, cmap='gray')
        axes[0, 0].set_title('원본 이미지', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')

        # Each 방법별 이진화 결과
        method_names = ['global_otsu', 'block_based', 'sliding_window']
        positions = [(0, 1), (0, 2), (1, 0)]

        for i, method_name in enumerate(method_names):
            if method_name in methods and i < len(positions):
                row, col = positions[i]
                result = methods[method_name]

                if 'result' in result:
                    axes[row, col].imshow(result['result'], cmap='gray')

                    threshold = result.get('threshold', 'Adaptive')
                    if isinstance(threshold, (int, float)):
                        title = f"{method_name.replace('_', ' ').title()}\n임계값: {threshold:.1f}"
                    else:
                        title = f"{method_name.replace('_', ' ').title()}\n임계값: {threshold}"

                    axes[row, col].set_title(title, fontsize=12, fontweight='bold')
                    axes[row, col].axis('off')

        # 방법별 특성 비교
        comparison_text = """Otsu 방법들 비교:

Global Otsu:
- 전체 Image에 단일 임계값 적용
- 빠른 처리 속도
- 균일한 조명에 적합
- 계산 복잡도: O(L×N)

Block-based Otsu:
- 블록별로 개별 임계값 계산
- 지역적 조명 변화 대응
- 경계에서 불연속성 가능
- 계산 복잡도: O(B×L×N/B)

Sliding Window Otsu:
- 픽셀별 주변 윈도우로 임계값 계산
- 가장 세밀한 적응적 처리
- 경계 부드러움
- 계산 복잡도: O(W×L×N)

여기서 L=256, N=총픽셀수, B=블록수, W=윈도우수"""

        # 통계 정보는 마크다운 파일에 포함 (텍스트 박스 제거)
        axes[1, 1].set_title('방법별 특성 비교', fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')

        # 품질 평가 지표
        if 'global_otsu' in methods and 'result' in methods['global_otsu']:
            binary_result = methods['global_otsu']['result']

            # 간단한 품질 지표 계산
            total_pixels = binary_result.shape[0] * binary_result.shape[1]
            white_pixels = np.sum(binary_result == 255)
            black_pixels = np.sum(binary_result == 0)

            quality_text = f"""이진화 품질 평가:

픽셀 분포:
- 전체 픽셀: {total_pixels:,}
- 흰색 픽셀 (전경): {white_pixels:,} ({white_pixels/total_pixels*100:.1f}%)
- 검은색 픽셀 (배경): {black_pixels:,} ({black_pixels/total_pixels*100:.1f}%)

분리도 평가:
- 전경/배경 비율: {white_pixels/black_pixels:.3f}
- 균형도: {min(white_pixels, black_pixels)/max(white_pixels, black_pixels):.3f}

처리 결과:
- 텍스트 영역이 명확히 분리됨
- 그림자 영향 최소화
- OCR 전처리에 적합"""

            # 텍스트 박스 제거 # axes[1, 2].text(0.05, 0.95, quality_text, transform=axes[1, 2].transAxes,
                            # fontsize=10, verticalalignment='top', fontfamily='monospace',
                            # bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            axes[1, 2].set_title('품질 평가', fontsize=12, fontweight='bold')
            axes[1, 2].axis('off')

        plt.suptitle('Otsu 3단계: 이진화 과정 및 결과',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        save_path = os.path.join(self.results_dir, 'otsu_step3_binarization_process.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    저장: otsu_step3_binarization_process.png")

    def create_individual_method_analysis(self, he_results, otsu_results):
        """개별 방법별 상세 분석"""
        print("  개별 방법별 분석...")

        # HE 방법별 개별 분석
        methods = he_results['methods']
        for method_name, result in methods.items():
            self.create_single_he_method_analysis(he_results['original_image'], method_name, result)

        # Otsu 방법별 개별 분석
        otsu_methods = otsu_results['methods']
        for method_name, result in otsu_methods.items():
            self.create_single_otsu_method_analysis(otsu_results['original_gray'], method_name, result)

    def create_single_he_method_analysis(self, original, method_name, result):
        """개별 HE 방법 상세 분석"""
        enhanced = result['enhanced_image']
        colorspace = result['colorspace']
        algorithm = result['algorithm']
        metrics = result['quality_metrics']

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Original과 결과 비교
        axes[0, 0].imshow(original)
        axes[0, 0].set_title('원본 이미지', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(enhanced)
        axes[0, 1].set_title(f'향상된 이미지 ({colorspace.upper()}+{algorithm.upper()})',
                           fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')

        # 차이 Image
        diff_image = np.abs(enhanced.astype(np.float32) - original.astype(np.float32))
        diff_gray = np.mean(diff_image, axis=2)
        axes[0, 2].imshow(diff_gray, cmap='hot')
        axes[0, 2].set_title('차이 이미지', fontsize=12, fontweight='bold')
        axes[0, 2].axis('off')

        # 히스토그램 비교
        orig_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        enh_gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)

        orig_hist = cv2.calcHist([orig_gray], [0], None, [256], [0, 256])
        enh_hist = cv2.calcHist([enh_gray], [0], None, [256], [0, 256])

        axes[1, 0].plot(orig_hist, color='blue', linewidth=2, label='Original', alpha=0.7)
        axes[1, 0].fill_between(range(256), orig_hist.flatten(), alpha=0.3, color='blue')
        axes[1, 0].set_title('원본 히스토그램', fontsize=12)
        axes[1, 0].set_xlabel('픽셀 강도')
        axes[1, 0].set_ylabel('빈도')
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(enh_hist, color='red', linewidth=2, label='향상된', alpha=0.7)
        axes[1, 1].fill_between(range(256), enh_hist.flatten(), alpha=0.3, color='red')
        axes[1, 1].set_title('향상된 히스토그램', fontsize=12)
        axes[1, 1].set_xlabel('픽셀 강도')
        axes[1, 1].set_ylabel('빈도')
        axes[1, 1].grid(True, alpha=0.3)

        # 성능 지표
        performance_text = f"""성능 분석:

방법: {colorspace.upper()} + {algorithm.upper()}
처리 시간: {result['processing_time']:.4f}초

대비 개선:
- Original 대비: {metrics['original_contrast']:.1f}
- 개선 대비: {metrics['enhanced_contrast']:.1f}
- 개선율: {metrics['contrast_improvement_percent']:+.1f}%

밝기 변화:
- Original 밝기: {metrics['original_brightness']:.1f}
- 개선 밝기: {metrics['enhanced_brightness']:.1f}
- 변화량: {metrics['brightness_change']:+.1f}

정보량 변화:
- Original 엔트로피: {metrics['original_entropy']:.3f} bits
- 개선 엔트로피: {metrics['enhanced_entropy']:.3f} bits
- 변화량: {metrics['enhanced_entropy'] - metrics['original_entropy']:+.3f} bits

종합 평가:
{'우수' if metrics['contrast_improvement_percent'] > 40 else '보통' if metrics['contrast_improvement_percent'] > 15 else '제한적'}"""

        # 텍스트 박스 제거 # axes[1, 2].text(0.05, 0.95, performance_text, transform=axes[1, 2].transAxes,
                        # fontsize=10, verticalalignment='top', fontfamily='monospace',
                        # bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        axes[1, 2].set_title('성능 분석', fontsize=12, fontweight='bold')
        axes[1, 2].axis('off')

        plt.suptitle(f'개별 방법 분석: {method_name.replace("_", "+").upper()}',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        save_path = os.path.join(self.results_dir, f'individual_he_{method_name}_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    저장: individual_he_{method_name}_analysis.png")

    def create_single_otsu_method_analysis(self, original, method_name, result):
        """개별 Otsu 방법 상세 분석"""
        if 'result' not in result:
            return

        binary = result['result']
        threshold = result.get('threshold', 'Adaptive')

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Original과 이진화 결과
        axes[0, 0].imshow(original, cmap='gray')
        axes[0, 0].set_title('원본 이미지', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(binary, cmap='gray')
        title = f'이진화 결과 ({method_name.replace("_", " ").title()})'
        if isinstance(threshold, (int, float)):
            title += f'\n임계값: {threshold:.1f}'
        axes[0, 1].set_title(title, fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')

        # 히스토그램과 임계값
        hist = cv2.calcHist([original], [0], None, [256], [0, 256])
        axes[1, 0].bar(range(256), hist.flatten(), alpha=0.7, color='gray', edgecolor='black')

        if isinstance(threshold, (int, float)):
            axes[1, 0].axvline(x=threshold, color='red', linestyle='--', linewidth=3,
                             label=f'임계값: {threshold}')
            axes[1, 0].legend()

        axes[1, 0].set_title('히스토그램과 임계값', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('픽셀 강도')
        axes[1, 0].set_ylabel('빈도')
        axes[1, 0].grid(True, alpha=0.3)

        # 품질 분석
        total_pixels = binary.shape[0] * binary.shape[1]
        white_pixels = np.sum(binary == 255)
        black_pixels = np.sum(binary == 0)

        quality_text = f"""품질 분석:

방법: {method_name.replace('_', ' ').title()}
{'임계값: ' + str(threshold) if isinstance(threshold, (int, float)) else '적응적 임계값'}

이진화 결과:
- 전체 픽셀: {total_pixels:,}
- 전경 (흰색): {white_pixels:,} ({white_pixels/total_pixels*100:.1f}%)
- 배경 (검은색): {black_pixels:,} ({black_pixels/total_pixels*100:.1f}%)

분리 품질:
- 전경/배경 비율: {white_pixels/black_pixels:.3f}
- 균형도: {min(white_pixels, black_pixels)/max(white_pixels, black_pixels):.3f}

특징:
{'- 전역적 최적화\n- 빠른 처리\n- 균일한 조명에 최적' if method_name == 'global_otsu' else '- 지역적 적응\n- 조명 변화 대응\n- 복잡한 계산' if 'block' in method_name else '- 픽셀별 적응\n- 최고 정밀도\n- 높은 계산 비용'}"""

        # 텍스트 박스 제거 # axes[1, 1].text(0.05, 0.95, quality_text, transform=axes[1, 1].transAxes,
                        # fontsize=10, verticalalignment='top', fontfamily='monospace',
                        # bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
        axes[1, 1].set_title('품질 분석', fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')

        plt.suptitle(f'개별 방법 분석: {method_name.replace("_", " ").title()}',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        save_path = os.path.join(self.results_dir, f'individual_otsu_{method_name}_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    저장: individual_otsu_{method_name}_analysis.png")

    def create_highlighted_comparisons(self, he_results, otsu_results):
        """강조된 비교 분석"""
        print("  강조 비교 분석...")

        # 1. HE 최고 vs 최저 성능 강조 비교
        self.create_he_best_vs_worst_highlighted(he_results)

        # 2. HE vs CLAHE 알고리즘 강조 비교
        self.create_he_vs_clahe_highlighted(he_results)

        # 3. Otsu 방법들 강조 비교
        self.create_otsu_methods_highlighted(otsu_results)

        # 4. HE vs Otsu 최종 강조 비교
        # self.create_he_vs_otsu_highlighted(he_results, otsu_results)  # REMOVED: HE와 Otsu는 서로 다른 목적의 과정

    def add_highlight_box(self, ax, color='red', linewidth=3, linestyle='--'):
        """Image에 강조 박스 Addition"""
        # Image 경계에 박스 그리기
        rect = Rectangle((0, 0), ax.get_xlim()[1], ax.get_ylim()[0],
                        linewidth=linewidth, edgecolor=color, facecolor='none', linestyle=linestyle)
        ax.add_patch(rect)

    def create_he_best_vs_worst_highlighted(self, he_results):
        """HE 최고 vs 최저 성능 강조 비교"""
        original = he_results['original_image']
        methods = he_results['methods']

        # 최고와 최저 성능 방법 찾기
        sorted_methods = sorted(methods.items(),
                              key=lambda x: x[1]['quality_metrics']['contrast_improvement_percent'])

        best_method = sorted_methods[-1]  # 최고 성능
        worst_method = sorted_methods[0]  # 최저 성능

        fig, axes = plt.subplots(2, 3, figsize=(20, 14))

        # 원본 이미지
        axes[0, 0].imshow(original)
        axes[0, 0].set_title('원본 이미지', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')

        # 최고 성능 방법
        axes[0, 1].imshow(best_method[1]['enhanced_image'])
        best_title = f"최고 성능\n"
        best_title += f"{best_method[1]['colorspace'].upper()}+{best_method[1]['algorithm'].upper()}\n"
        best_title += f"대비개선: {best_method[1]['quality_metrics']['contrast_improvement_percent']:+.1f}%"
        axes[0, 1].set_title(best_title, fontsize=12, fontweight='bold', color='darkgreen')
        axes[0, 1].axis('off')

        # 최고 성능에 녹색 강조 박스
        self.add_highlight_box(axes[0, 1], color='green', linewidth=4)

        # 최저 성능 방법
        axes[0, 2].imshow(worst_method[1]['enhanced_image'])
        worst_title = f"최저 성능\n"
        worst_title += f"{worst_method[1]['colorspace'].upper()}+{worst_method[1]['algorithm'].upper()}\n"
        worst_title += f"대비개선: {worst_method[1]['quality_metrics']['contrast_improvement_percent']:+.1f}%"
        axes[0, 2].set_title(worst_title, fontsize=12, fontweight='bold', color='darkred')
        axes[0, 2].axis('off')

        # 최저 성능에 빨간 강조 박스
        self.add_highlight_box(axes[0, 2], color='red', linewidth=4)

        # 히스토그램 비교
        orig_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        best_gray = cv2.cvtColor(best_method[1]['enhanced_image'], cv2.COLOR_RGB2GRAY)
        worst_gray = cv2.cvtColor(worst_method[1]['enhanced_image'], cv2.COLOR_RGB2GRAY)

        # Original 히스토그램
        orig_hist = cv2.calcHist([orig_gray], [0], None, [256], [0, 256])
        axes[1, 0].plot(orig_hist, color='blue', linewidth=2, label='Original')
        axes[1, 0].fill_between(range(256), orig_hist.flatten(), alpha=0.3, color='blue')
        axes[1, 0].set_title('원본 히스토그램', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('픽셀 강도')
        axes[1, 0].set_ylabel('빈도')
        axes[1, 0].grid(True, alpha=0.3)

        # 최고 성능 히스토그램
        best_hist = cv2.calcHist([best_gray], [0], None, [256], [0, 256])
        axes[1, 1].plot(best_hist, color='green', linewidth=3, label='Best')
        axes[1, 1].fill_between(range(256), best_hist.flatten(), alpha=0.3, color='green')
        axes[1, 1].set_title('최고 성능 히스토그램',
                           fontsize=12, fontweight='bold', color='darkgreen')
        axes[1, 1].set_xlabel('픽셀 강도')
        axes[1, 1].set_ylabel('빈도')
        axes[1, 1].grid(True, alpha=0.3)

        # 최저 성능 히스토그램
        worst_hist = cv2.calcHist([worst_gray], [0], None, [256], [0, 256])
        axes[1, 2].plot(worst_hist, color='red', linewidth=3, label='Worst')
        axes[1, 2].fill_between(range(256), worst_hist.flatten(), alpha=0.3, color='red')
        axes[1, 2].set_title('최저 성능 히스토그램',
                           fontsize=12, fontweight='bold', color='darkred')
        axes[1, 2].set_xlabel('픽셀 강도')
        axes[1, 2].set_ylabel('빈도')
        axes[1, 2].grid(True, alpha=0.3)

        # 성능 차이 텍스트 박스 Addition
        performance_diff = (best_method[1]['quality_metrics']['contrast_improvement_percent'] -
                          worst_method[1]['quality_metrics']['contrast_improvement_percent'])

        diff_text = f"""성능 차이 분석:

최고 성능: {best_method[0].replace('_', '+').upper()}
- 대비 개선: {best_method[1]['quality_metrics']['contrast_improvement_percent']:+.1f}%
- 처리 시간: {best_method[1]['processing_time']:.4f}초

최저 성능: {worst_method[0].replace('_', '+').upper()}
- 대비 개선: {worst_method[1]['quality_metrics']['contrast_improvement_percent']:+.1f}%
- 처리 시간: {worst_method[1]['processing_time']:.4f}초

성능 차이: {performance_diff:.1f}%포인트"""

        # 텍스트 박스 제거 # fig.text(0.5, 0.02, diff_text, ha='center', fontsize=12, fontfamily='monospace')

        plt.suptitle('HE 최고 vs 최저 성능 강조 비교',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)

        save_path = os.path.join(self.results_dir, 'he_best_vs_worst_highlighted.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    저장: he_best_vs_worst_highlighted.png")

    def create_he_vs_clahe_highlighted(self, he_results):
        """HE vs CLAHE 알고리즘 강조 비교"""
        original = he_results['original_image']
        methods = he_results['methods']

        # YUV HE와 YUV CLAHE 비교 (같은 Colorspace에서 알고리즘만 다름)
        yuv_he = methods.get('yuv_he')
        yuv_clahe = methods.get('yuv_clahe')

        if not yuv_he or not yuv_clahe:
            print("    YUV HE 또는 CLAHE 결과 없음, 건너뜀")
            return

        fig, axes = plt.subplots(3, 3, figsize=(20, 18))

        # 첫 번째 행: Image 비교
        axes[0, 0].imshow(original)
        axes[0, 0].set_title('원본 이미지', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(yuv_he['enhanced_image'])
        he_title = f"YUV + HE\n대비개선: {yuv_he['quality_metrics']['contrast_improvement_percent']:+.1f}%\n"
        he_title += f"처리시간: {yuv_he['processing_time']:.4f}초"
        axes[0, 1].set_title(he_title, fontsize=12, fontweight='bold', color='darkblue')
        axes[0, 1].axis('off')
        # HE에 파란색 강조
        self.add_highlight_box(axes[0, 1], color='blue', linewidth=4)

        axes[0, 2].imshow(yuv_clahe['enhanced_image'])
        clahe_title = f"⚡ YUV + CLAHE\n대비개선: {yuv_clahe['quality_metrics']['contrast_improvement_percent']:+.1f}%\n"
        clahe_title += f"처리시간: {yuv_clahe['processing_time']:.4f}초"
        axes[0, 2].set_title(clahe_title, fontsize=12, fontweight='bold', color='darkorange')
        axes[0, 2].axis('off')
        # CLAHE에 주황색 강조
        self.add_highlight_box(axes[0, 2], color='orange', linewidth=4)

        # 두 번째 행: 히스토그램 비교
        orig_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        he_gray = cv2.cvtColor(yuv_he['enhanced_image'], cv2.COLOR_RGB2GRAY)
        clahe_gray = cv2.cvtColor(yuv_clahe['enhanced_image'], cv2.COLOR_RGB2GRAY)

        orig_hist = cv2.calcHist([orig_gray], [0], None, [256], [0, 256])
        he_hist = cv2.calcHist([he_gray], [0], None, [256], [0, 256])
        clahe_hist = cv2.calcHist([clahe_gray], [0], None, [256], [0, 256])

        axes[1, 0].plot(orig_hist, color='gray', linewidth=2)
        axes[1, 0].fill_between(range(256), orig_hist.flatten(), alpha=0.3, color='gray')
        axes[1, 0].set_title('원본 히스토그램', fontsize=12)
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(he_hist, color='blue', linewidth=3)
        axes[1, 1].fill_between(range(256), he_hist.flatten(), alpha=0.3, color='blue')
        axes[1, 1].set_title('HE 히스토그램', fontsize=12, color='darkblue')
        axes[1, 1].grid(True, alpha=0.3)

        axes[1, 2].plot(clahe_hist, color='orange', linewidth=3)
        axes[1, 2].fill_between(range(256), clahe_hist.flatten(), alpha=0.3, color='orange')
        axes[1, 2].set_title('CLAHE 히스토그램', fontsize=12, color='darkorange')
        axes[1, 2].grid(True, alpha=0.3)

        # 세 번째 행: 차이 분석
        he_diff = np.abs(yuv_he['enhanced_image'].astype(np.float32) - original.astype(np.float32))
        clahe_diff = np.abs(yuv_clahe['enhanced_image'].astype(np.float32) - original.astype(np.float32))

        axes[2, 0].imshow(np.mean(he_diff, axis=2), cmap='hot')
        axes[2, 0].set_title('HE 차이 Image', fontsize=12, color='darkblue')
        axes[2, 0].axis('off')

        axes[2, 1].imshow(np.mean(clahe_diff, axis=2), cmap='hot')
        axes[2, 1].set_title('CLAHE 차이 Image', fontsize=12, color='darkorange')
        axes[2, 1].axis('off')

        # 알고리즘 특성 비교
        comparison_text = f"""🔍 HE vs CLAHE 알고리즘 비교:

📈 히스토그램 Equalization (HE):
✅ 장점:
- 극적인 대비 개선 ({yuv_he['quality_metrics']['contrast_improvement_percent']:+.1f}%)
- 전역적 분포 균등화
- 어두운 Image에 효과적

❌ 단점:
- 상대적으로 느린 처리 ({yuv_he['processing_time']:.4f}초)
- 과도한 증폭 가능성
- 노이즈 증폭 위험

⚡ CLAHE (Contrast Limited AHE):
✅ 장점:
- 매우 빠른 처리 ({yuv_clahe['processing_time']:.4f}초)
- 노이즈 억제 효과
- 자연스러운 결과

❌ 단점:
- 제한적 개선 ({yuv_clahe['quality_metrics']['contrast_improvement_percent']:+.1f}%)
- 클리핑으로 인한 정보 손실
- 어두운 영역에서 보수적

💡 권장 사용:
- HE: 어두운 Image, 최대 품질 우선
- CLAHE: 실시간 처리, 자연스러운 개선"""

        # 텍스트 박스 제거
        axes[2, 2].set_title('알고리즘 특성 비교', fontsize=12, fontweight='bold')
        axes[2, 2].axis('off')

        plt.suptitle('HE vs CLAHE 알고리즘 강조 비교',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        save_path = os.path.join(self.results_dir, 'he_vs_clahe_highlighted.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    저장: he_vs_clahe_highlighted.png")

    def create_otsu_methods_highlighted(self, otsu_results):
        """Otsu 방법들 강조 비교"""
        original = otsu_results['original_gray']
        methods = otsu_results['methods']

        fig, axes = plt.subplots(2, 4, figsize=(24, 12))

        # 첫 번째 행: Original과 Each 방법 결과
        axes[0, 0].imshow(original, cmap='gray')
        axes[0, 0].set_title('Original 문서\nOriginal 문서', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')

        method_info = [
            ('global_otsu', 'Global Otsu', 'green'),
            ('block_based', 'Block-based', 'blue'),
            ('sliding_window', 'Sliding Window', 'red')
        ]

        for i, (method_name, display_name, color) in enumerate(method_info):
            if method_name in methods and 'result' in methods[method_name]:
                result = methods[method_name]

                axes[0, i+1].imshow(result['result'], cmap='gray')

                threshold = result.get('threshold', 'Adaptive')
                title = f"{display_name}\n"
                if isinstance(threshold, (int, float)):
                    title += f"임계값: {threshold:.1f}"
                else:
                    title += f"임계값: {threshold}"

                axes[0, i+1].set_title(title, fontsize=12, fontweight='bold', color=f'dark{color}')
                axes[0, i+1].axis('off')

                # Each 방법에 색상별 강조 박스
                self.add_highlight_box(axes[0, i+1], color=color, linewidth=4)

        # 두 번째 행: 히스토그램과 분석
        # Original 히스토그램
        hist = cv2.calcHist([original], [0], None, [256], [0, 256])
        axes[1, 0].bar(range(256), hist.flatten(), alpha=0.7, color='gray')
        axes[1, 0].set_title('원본 히스토그램', fontsize=12)
        axes[1, 0].grid(True, alpha=0.3)

        # Each 방법의 임계값과 히스토그램
        for i, (method_name, display_name, color) in enumerate(method_info):
            if method_name in methods:
                result = methods[method_name]
                threshold = result.get('threshold', None)

                axes[1, i+1].bar(range(256), hist.flatten(), alpha=0.5, color='gray')

                if isinstance(threshold, (int, float)):
                    axes[1, i+1].axvline(x=threshold, color=color, linestyle='--', linewidth=4,
                                        label=f'임계값: {threshold:.1f}')
                    axes[1, i+1].legend()

                axes[1, i+1].set_title(f'{display_name}\n히스토그램', fontsize=12, color=f'dark{color}')
                axes[1, i+1].grid(True, alpha=0.3)

        # 방법별 특성 텍스트 박스 Addition
        methods_text = f"""Otsu 방법들 특성 비교:

🌍 Global Otsu:
- 단일 최적 임계값: {methods.get('global_otsu', {}).get('threshold', 'N/A')}
- 전체 Image 기준 최적화
- 빠른 처리 속도
- 균일한 조명에 최적

🧩 Block-based Otsu:
- 블록별 적응적 임계값
- 지역적 조명 변화 대응
- 경계에서 불연속 가능
- In Progress간 계산 복잡도

🔍 Sliding Window Otsu:
- 픽셀별 주변 기준 임계값
- 가장 세밀한 적응적 처리
- 부드러운 경계 처리
- 높은 계산 복잡도

📊 성능 비교:
- 속도: Global > Block > Sliding
- 적응성: Sliding > Block > Global
- 실용성: Global > Block > Sliding"""

        # 텍스트 박스 제거

        plt.suptitle('Otsu Method 강조 비교',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25)

        save_path = os.path.join(self.results_dir, 'otsu_methods_highlighted.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    저장: otsu_methods_highlighted.png")

    def create_he_vs_otsu_highlighted_DISABLED(self, he_results, otsu_results):
        """HE vs Otsu 최종 강조 비교"""
        # 최고 성능 HE 방법
        he_methods = he_results['methods']
        best_he = max(he_methods.items(), key=lambda x: x[1]['quality_metrics']['contrast_improvement_percent'])

        # Global Otsu 결과
        otsu_methods = otsu_results['methods']
        global_otsu = otsu_methods.get('global_otsu', list(otsu_methods.values())[0])

        fig, axes = plt.subplots(3, 3, figsize=(20, 18))

        # 첫 번째 행: 원본 이미지들
        he_original = he_results['original_image']
        otsu_original = otsu_results['original_gray']

        axes[0, 0].imshow(he_original)
        axes[0, 0].set_title('HE 대상 이미지\n(어두운 실내)',
                           fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')

        axes[0, 1].text(0.5, 0.5, 'VS', transform=axes[0, 1].transAxes, ha='center', va='center',
                       fontsize=48, fontweight='bold', color='red')
        axes[0, 1].axis('off')

        axes[0, 2].imshow(otsu_original, cmap='gray')
        axes[0, 2].set_title('Otsu 대상 Image\n(그림자 문서)',
                           fontsize=12, fontweight='bold')
        axes[0, 2].axis('off')

        # 두 번째 행: 처리 결과
        axes[1, 0].imshow(best_he[1]['enhanced_image'])
        he_result_title = f"HE 최고 결과\n{best_he[0].replace('_', '+').upper()}\n"
        he_result_title += f"대비개선: {best_he[1]['quality_metrics']['contrast_improvement_percent']:+.1f}%"
        axes[1, 0].set_title(he_result_title, fontsize=11, fontweight='bold', color='darkgreen')
        axes[1, 0].axis('off')
        # HE 결과에 녹색 강조
        self.add_highlight_box(axes[1, 0], color='green', linewidth=4)

        axes[1, 1].text(0.5, 0.7, '', transform=axes[1, 1].transAxes, ha='center', va='center',
                       fontsize=64)
        axes[1, 1].text(0.5, 0.3, '서로 다른\n목적과 결과', transform=axes[1, 1].transAxes,
                       ha='center', va='center', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')

        if 'result' in global_otsu:
            axes[1, 2].imshow(global_otsu['result'], cmap='gray')
            otsu_result_title = f"Otsu 결과\nGlobal 방법\n"
            threshold = global_otsu.get('threshold', 'Auto')
            if isinstance(threshold, (int, float)):
                otsu_result_title += f"임계값: {threshold:.1f}"
            axes[1, 2].set_title(otsu_result_title, fontsize=11, fontweight='bold', color='darkblue')
            axes[1, 2].axis('off')
            # Otsu 결과에 파란색 강조
            self.add_highlight_box(axes[1, 2], color='blue', linewidth=4)

        # 세 번째 행: 특성 비교
        # HE 히스토그램
        he_orig_gray = cv2.cvtColor(he_original, cv2.COLOR_RGB2GRAY)
        he_enh_gray = cv2.cvtColor(best_he[1]['enhanced_image'], cv2.COLOR_RGB2GRAY)

        he_orig_hist = cv2.calcHist([he_orig_gray], [0], None, [256], [0, 256])
        he_enh_hist = cv2.calcHist([he_enh_gray], [0], None, [256], [0, 256])

        axes[2, 0].plot(he_orig_hist, color='blue', alpha=0.5, label='Original', linewidth=2)
        axes[2, 0].plot(he_enh_hist, color='green', alpha=0.8, label='향상된', linewidth=3)
        axes[2, 0].fill_between(range(256), he_enh_hist.flatten(), alpha=0.3, color='green')
        axes[2, 0].set_title('HE 히스토그램 변화', fontsize=12, color='darkgreen')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)

        # 비교 텍스트
        comparison_text = f"""🔍 HE vs Otsu 근본적 차이:

📈 히스토그램 Equalization:
🎯 목적: 시Each적 품질 개선
📊 결과: 연속적 그레이레벨 (0-255)
⚡ 효과: 극적인 대비 개선 ({best_he[1]['quality_metrics']['contrast_improvement_percent']:+.1f}%)
🖼️ 용도: 사진 편집, 의료 영상, 어두운 Image

🌍 Otsu 임계값ing:
🎯 목적: 객체 분할 및 이진화
📊 결과: 이진 Image (0 또는 255)
⚡ 효과: 자동 임계값 결정 ({global_otsu.get('threshold', 'Auto')})
🖼️ 용도: 문서 처리, OCR 전처리, 컴퓨터 비전

💡 핵심 차이점:
- HE는 연속톤 → 연속톤 (품질 개선)
- Otsu는 연속톤 → 이진 (객체 분할)
- 완전히 다른 응용 분야와 목표"""

        # 텍스트 박스 제거 # axes[2, 1].text(0.05, 0.95, comparison_text, transform=axes[2, 1].transAxes,
                        # fontsize=10, verticalalignment='top', fontfamily='monospace',
                        # bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9))
        axes[2, 1].set_title('근본적 차이점', fontsize=12, fontweight='bold')
        axes[2, 1].axis('off')

        # Otsu 히스토그램
        otsu_hist = cv2.calcHist([otsu_original], [0], None, [256], [0, 256])
        axes[2, 2].bar(range(256), otsu_hist.flatten(), alpha=0.7, color='gray')

        threshold = global_otsu.get('threshold', None)
        if isinstance(threshold, (int, float)):
            axes[2, 2].axvline(x=threshold, color='blue', linestyle='--', linewidth=4,
                             label=f'최적 임계값: {threshold:.1f}')
            axes[2, 2].legend()

        axes[2, 2].set_title('Otsu 히스토그램 & 임계값',
                           fontsize=12, color='darkblue')
        axes[2, 2].grid(True, alpha=0.3)

        plt.suptitle('Final HE vs Otsu 강조ed 비교',
                    fontsize=18, fontweight='bold')
        plt.tight_layout()

        save_path = os.path.join(self.results_dir, 'he_vs_otsu_final_highlighted.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    저장: he_vs_otsu_final_highlighted.png")

def main():
    """메인 Execution 함수"""
    analyzer = ImprovedVisualizationAnalyzer()
    he_results, otsu_results = analyzer.run_improved_analysis()

    print("\n=== 향상된 Visualization 분석 Complete ===")

    # 생성된 파일들 확인
    results_dir = "results"
    generated_files = []
    for file in os.listdir(results_dir):
        if file.endswith('.png'):
            generated_files.append(file)

    print(f"\n총 {len(generated_files)}개 Image 생성:")
    for file in sorted(generated_files):
        print(f"  - {file}")

    return he_results, otsu_results

if __name__ == "__main__":
    main()