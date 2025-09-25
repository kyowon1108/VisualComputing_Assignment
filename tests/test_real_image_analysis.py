#!/usr/bin/env python3
"""
실제 이미지를 사용한 포괄적인 분석
Real Image Comprehensive Analysis

he_dark_indoor.jpg: 히스토그램 평활화 분석
otsu_shadow_doc_02.jpg: Otsu 임계값 분석
"""

import sys
import os

# 프로젝트 루트 디렉토리를 Python path에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import cv2
from src.he import histogram_equalization_color
from src.otsu import compare_otsu_methods
from src.utils import load_image
import time
from typing import Dict, List, Tuple

class RealImageAnalyzer:
    """실제 이미지를 사용한 포괄적인 분석 클래스"""

    def __init__(self):
        self.he_image_path = "images/he_dark_indoor.jpg"
        self.otsu_image_path = "images/otsu_shadow_doc_02.jpg"
        self.results = {}

    def analyze_he_with_real_image(self):
        """실제 어두운 실내 이미지로 HE 분석"""
        print("=== HE 분석: he_dark_indoor.jpg ===")

        # 이미지 로드
        image = load_image(self.he_image_path, color_mode='color')
        print(f"이미지 크기: {image.shape}")
        print(f"밝기 범위: {image.min()}-{image.max()}")
        print(f"평균 밝기: {np.mean(image):.1f}")

        # 모든 컬러스페이스와 알고리즘 조합 테스트
        colorspaces = ['yuv', 'ycbcr', 'lab', 'hsv', 'rgb']
        algorithms = ['he', 'clahe']

        he_results = {}
        performance_data = {}

        print("\\n컬러스페이스별 HE/CLAHE 분석 시작...")

        for colorspace in colorspaces:
            for algorithm in algorithms:
                # RGB+CLAHE는 제외
                if colorspace == 'rgb' and algorithm == 'clahe':
                    continue

                try:
                    combo_name = f"{colorspace}_{algorithm}"
                    print(f"처리 중: {combo_name.upper()}")

                    start_time = time.time()
                    enhanced, process_info = histogram_equalization_color(
                        image,
                        method=colorspace,
                        algorithm=algorithm,
                        clip_limit=2.0,
                        tile_size=8,
                        show_process=False
                    )
                    end_time = time.time()

                    he_results[combo_name] = {
                        'enhanced_image': enhanced,
                        'process_info': process_info,
                        'colorspace': colorspace,
                        'algorithm': algorithm
                    }

                    # 품질 지표 계산
                    quality_metrics = self.calculate_quality_metrics(image, enhanced)
                    performance_data[combo_name] = {
                        'processing_time': end_time - start_time,
                        'quality_metrics': quality_metrics
                    }

                    print(f"  완료 (시간: {end_time - start_time:.4f}초, "
                          f"대비개선: {quality_metrics['contrast_improvement_percent']:.1f}%)")

                except Exception as e:
                    print(f"  실패: {str(e)}")
                    continue

        self.results['he'] = {
            'image': image,
            'results': he_results,
            'performance': performance_data
        }

        # 시각화 결과 저장
        self.save_he_comparison_results(image, he_results, performance_data)

        return he_results, performance_data

    def analyze_otsu_with_real_image(self):
        """실제 그림자 문서 이미지로 Otsu 분석"""
        print("\\n=== Otsu 분석: otsu_shadow_doc_02.jpg ===")

        # 이미지 로드 (그레이스케일)
        image_color = load_image(self.otsu_image_path, color_mode='color')
        image_gray = load_image(self.otsu_image_path, color_mode='gray')

        print(f"이미지 크기: {image_gray.shape}")
        print(f"밝기 범위: {image_gray.min()}-{image_gray.max()}")
        print(f"평균 밝기: {np.mean(image_gray):.1f}")

        # Otsu 방법들 비교 분석
        print("\\nOtsu 방법별 분석 시작...")

        try:
            otsu_results = compare_otsu_methods(
                image_gray,
                show_comparison=False
            )

            print("Otsu 분석 완료:")
            for method, result in otsu_results.items():
                threshold = result.get('threshold', 'N/A')
                if isinstance(threshold, (int, float)):
                    print(f"  {method}: 임계값 = {threshold:.1f}")
                else:
                    print(f"  {method}: 임계값 = {threshold}")

            self.results['otsu'] = {
                'image_color': image_color,
                'image_gray': image_gray,
                'results': otsu_results
            }

            # Otsu 시각화 결과 저장
            self.save_otsu_comparison_results(image_gray, otsu_results)

            return otsu_results

        except Exception as e:
            print(f"Otsu 분석 실패: {str(e)}")
            return {}

    def calculate_quality_metrics(self, original: np.ndarray, enhanced: np.ndarray) -> Dict:
        """이미지 품질 지표 계산"""
        # 그레이스케일로 변환하여 분석
        orig_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        enh_gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)

        # 대비 지표
        orig_contrast = np.std(orig_gray)
        enh_contrast = np.std(enh_gray)
        contrast_improvement = (enh_contrast - orig_contrast) / orig_contrast * 100

        # 히스토그램 분포
        orig_hist = cv2.calcHist([orig_gray], [0], None, [256], [0, 256])
        enh_hist = cv2.calcHist([enh_gray], [0], None, [256], [0, 256])

        # 엔트로피 (정보량)
        orig_entropy = self.calculate_entropy(orig_hist)
        enh_entropy = self.calculate_entropy(enh_hist)

        # 밝기 통계
        orig_brightness = np.mean(orig_gray)
        enh_brightness = np.mean(enh_gray)

        # 동적 범위 (다이나믹 레인지)
        orig_dynamic_range = np.max(orig_gray) - np.min(orig_gray)
        enh_dynamic_range = np.max(enh_gray) - np.min(enh_gray)

        return {
            'original_contrast': float(orig_contrast),
            'enhanced_contrast': float(enh_contrast),
            'contrast_improvement_percent': float(contrast_improvement),
            'original_entropy': float(orig_entropy),
            'enhanced_entropy': float(enh_entropy),
            'entropy_change': float(enh_entropy - orig_entropy),
            'original_brightness': float(orig_brightness),
            'enhanced_brightness': float(enh_brightness),
            'brightness_change': float(enh_brightness - orig_brightness),
            'original_dynamic_range': int(orig_dynamic_range),
            'enhanced_dynamic_range': int(enh_dynamic_range),
            'dynamic_range_improvement': int(enh_dynamic_range - orig_dynamic_range)
        }

    def calculate_entropy(self, histogram):
        """히스토그램으로부터 엔트로피 계산"""
        hist_norm = histogram.flatten() / np.sum(histogram)
        hist_norm = hist_norm[hist_norm > 0]
        entropy = -np.sum(hist_norm * np.log2(hist_norm))
        return entropy

    def save_he_comparison_results(self, original_image: np.ndarray, he_results: dict, performance_data: dict):
        """HE 비교 결과 저장"""
        print("\\nHE 비교 결과 시각화 및 저장...")

        combo_names = list(he_results.keys())
        n_results = len(combo_names)

        # 그리드 레이아웃 (원본 포함)
        n_cols = 5
        n_rows = (n_results + 1 + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(25, 5*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        # 원본 이미지
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title('Original\\n(Dark Indoor)', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')

        # 각 결과 표시
        for i, combo_name in enumerate(combo_names):
            row = (i + 1) // n_cols
            col = (i + 1) % n_cols

            if row < n_rows and col < n_cols:
                result = he_results[combo_name]
                perf = performance_data[combo_name]

                axes[row, col].imshow(result['enhanced_image'])

                # 제목에 성능 정보 포함
                title = f"{result['colorspace'].upper()}+{result['algorithm'].upper()}\\n"
                title += f"Time: {perf['processing_time']:.3f}s\\n"
                title += f"Contrast: {perf['quality_metrics']['contrast_improvement_percent']:+.1f}%\\n"
                title += f"Brightness: {perf['quality_metrics']['brightness_change']:+.0f}"

                axes[row, col].set_title(title, fontsize=10)
                axes[row, col].axis('off')

        # 빈 subplot 숨기기
        for i in range(n_results + 1, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            if row < n_rows and col < n_cols:
                axes[row, col].axis('off')

        plt.suptitle('Histogram Equalization Analysis - Dark Indoor Image\\nComparison of Colorspaces and Algorithms',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        # 저장
        save_path = os.path.join(os.path.dirname(__file__), '..', 'results',
                                'he_dark_indoor_comprehensive.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"HE 종합 분석 저장: {save_path}")

        plt.show()

        # 성능 차트도 저장
        self.create_he_performance_chart(performance_data)

    def create_he_performance_chart(self, performance_data: dict):
        """HE 성능 분석 차트 생성"""
        combo_names = list(performance_data.keys())

        # 데이터 추출
        processing_times = [performance_data[name]['processing_time'] for name in combo_names]
        contrast_improvements = [performance_data[name]['quality_metrics']['contrast_improvement_percent']
                               for name in combo_names]
        brightness_changes = [performance_data[name]['quality_metrics']['brightness_change']
                            for name in combo_names]
        dynamic_range_improvements = [performance_data[name]['quality_metrics']['dynamic_range_improvement']
                                    for name in combo_names]

        # 2x2 차트 생성
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. 처리 시간
        colors = ['skyblue' if 'he' in name else 'lightcoral' for name in combo_names]
        bars1 = ax1.bar(range(len(combo_names)), processing_times, color=colors)
        ax1.set_title('Processing Time Comparison\\n(Dark Indoor Image)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_xticks(range(len(combo_names)))
        ax1.set_xticklabels([name.replace('_', '+').upper() for name in combo_names],
                           rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)

        # 2. 대비 개선
        bars2 = ax2.bar(range(len(combo_names)), contrast_improvements, color=colors)
        ax2.set_title('Contrast Improvement\\n(Dark Indoor Image)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Contrast Improvement (%)')
        ax2.set_xticks(range(len(combo_names)))
        ax2.set_xticklabels([name.replace('_', '+').upper() for name in combo_names],
                           rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        # 3. 밝기 변화
        bars3 = ax3.bar(range(len(combo_names)), brightness_changes, color=colors)
        ax3.set_title('Brightness Change\\n(Dark Indoor Image)', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Brightness Change')
        ax3.set_xticks(range(len(combo_names)))
        ax3.set_xticklabels([name.replace('_', '+').upper() for name in combo_names],
                           rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        # 4. 동적 범위 개선
        bars4 = ax4.bar(range(len(combo_names)), dynamic_range_improvements, color=colors)
        ax4.set_title('Dynamic Range Improvement\\n(Dark Indoor Image)', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Dynamic Range Improvement')
        ax4.set_xticks(range(len(combo_names)))
        ax4.set_xticklabels([name.replace('_', '+').upper() for name in combo_names],
                           rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        # 범례
        import matplotlib.patches as patches
        he_patch = patches.Patch(color='skyblue', label='HE Algorithm')
        clahe_patch = patches.Patch(color='lightcoral', label='CLAHE Algorithm')
        fig.legend(handles=[he_patch, clahe_patch], loc='upper right')

        plt.tight_layout()

        # 저장
        save_path = os.path.join(os.path.dirname(__file__), '..', 'results',
                                'he_dark_indoor_performance.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"HE 성능 차트 저장: {save_path}")

        plt.show()

    def save_otsu_comparison_results(self, original_image: np.ndarray, otsu_results: dict):
        """Otsu 비교 결과 저장"""
        print("\\nOtsu 비교 결과 시각화 및 저장...")

        method_names = list(otsu_results.keys())
        n_methods = len(method_names)

        # 그리드 레이아웃
        n_cols = 4
        n_rows = (n_methods + 1 + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        # 원본 이미지
        axes[0, 0].imshow(original_image, cmap='gray')
        axes[0, 0].set_title('Original\\n(Shadow Document)', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')

        # 각 Otsu 결과 표시
        for i, method_name in enumerate(method_names):
            row = (i + 1) // n_cols
            col = (i + 1) % n_cols

            if row < n_rows and col < n_cols:
                result = otsu_results[method_name]

                # 이진화 이미지 표시
                if 'result' in result:
                    axes[row, col].imshow(result['result'], cmap='gray')

                    threshold = result.get('threshold', 'N/A')
                    if isinstance(threshold, (int, float)):
                        title = f"{method_name.replace('_', ' ').title()}\\nThreshold: {threshold:.1f}"
                    else:
                        title = f"{method_name.replace('_', ' ').title()}\\nThreshold: {threshold}"

                    axes[row, col].set_title(title, fontsize=10)
                    axes[row, col].axis('off')

        # 빈 subplot 숨기기
        for i in range(n_methods + 1, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            if row < n_rows and col < n_cols:
                axes[row, col].axis('off')

        plt.suptitle('Otsu Thresholding Methods Comparison - Shadow Document\\nComparison of Different Otsu Implementations',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        # 저장
        save_path = os.path.join(os.path.dirname(__file__), '..', 'results',
                                'otsu_shadow_doc_comprehensive.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Otsu 종합 분석 저장: {save_path}")

        plt.show()

    def create_cross_comparison(self):
        """HE와 Otsu 결과 교차 비교"""
        print("\\n=== HE vs Otsu 교차 비교 분석 ===")

        if 'he' not in self.results or 'otsu' not in self.results:
            print("HE 또는 Otsu 분석 결과가 없습니다. 먼저 개별 분석을 수행해주세요.")
            return

        # HE 최고 성능 조합 찾기
        he_performance = self.results['he']['performance']
        best_he_combo = max(he_performance.items(),
                           key=lambda x: x[1]['quality_metrics']['contrast_improvement_percent'])

        print(f"최고 HE 성능: {best_he_combo[0]} "
              f"(대비개선: {best_he_combo[1]['quality_metrics']['contrast_improvement_percent']:.1f}%)")

        # Otsu 최고 성능 방법 찾기 (가장 적절한 임계값을 가진 방법)
        otsu_results = self.results['otsu']['results']

        # 임계값이 중간 범위(80-180)에 있는 방법을 우선으로 선택
        suitable_methods = []
        for method, result in otsu_results.items():
            threshold = result.get('threshold', None)
            if isinstance(threshold, (int, float)) and 80 <= threshold <= 180:
                suitable_methods.append((method, threshold))

        if suitable_methods:
            best_otsu_method = max(suitable_methods, key=lambda x: abs(x[1] - 128))  # 128에 가까운 것 선택
            print(f"최적 Otsu 방법: {best_otsu_method[0]} (임계값: {best_otsu_method[1]:.1f})")
        else:
            # 적절한 임계값이 없으면 첫 번째 방법 선택
            best_otsu_method = (list(otsu_results.keys())[0], "Auto")
            print(f"선택된 Otsu 방법: {best_otsu_method[0]}")

        # 교차 비교 시각화
        self.visualize_cross_comparison(best_he_combo, best_otsu_method[0])

    def visualize_cross_comparison(self, best_he_combo: tuple, best_otsu_method: str):
        """HE와 Otsu 교차 비교 시각화"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # HE 관련 이미지들
        he_original = self.results['he']['image']
        he_enhanced = self.results['he']['results'][best_he_combo[0]]['enhanced_image']
        he_perf = best_he_combo[1]['quality_metrics']

        # Otsu 관련 이미지들
        otsu_original = self.results['otsu']['image_gray']
        otsu_binary = self.results['otsu']['results'][best_otsu_method]['result']
        otsu_threshold = self.results['otsu']['results'][best_otsu_method].get('threshold', 'Auto')

        # 상단 행: HE 결과
        axes[0, 0].imshow(he_original)
        axes[0, 0].set_title('HE: Original\\n(Dark Indoor)', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(he_enhanced)
        axes[0, 1].set_title(f'HE: Enhanced\\n({best_he_combo[0].replace("_", "+").upper()})',
                           fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')

        # HE 히스토그램
        he_orig_gray = cv2.cvtColor(he_original, cv2.COLOR_RGB2GRAY)
        he_enh_gray = cv2.cvtColor(he_enhanced, cv2.COLOR_RGB2GRAY)
        axes[0, 2].hist(he_orig_gray.flatten(), bins=50, alpha=0.5, color='blue',
                       density=True, label='Original')
        axes[0, 2].hist(he_enh_gray.flatten(), bins=50, alpha=0.5, color='red',
                       density=True, label='Enhanced')
        axes[0, 2].set_title('HE: Histogram Comparison', fontsize=14, fontweight='bold')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # 하단 행: Otsu 결과
        axes[1, 0].imshow(otsu_original, cmap='gray')
        axes[1, 0].set_title('Otsu: Original\\n(Shadow Document)', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(otsu_binary, cmap='gray')
        if isinstance(otsu_threshold, (int, float)):
            axes[1, 1].set_title(f'Otsu: Binary\\n({best_otsu_method}, T={otsu_threshold:.1f})',
                               fontsize=14, fontweight='bold')
        else:
            axes[1, 1].set_title(f'Otsu: Binary\\n({best_otsu_method})',
                               fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')

        # Otsu 히스토그램과 임계값 표시
        axes[1, 2].hist(otsu_original.flatten(), bins=50, alpha=0.7, color='gray', density=True)
        if isinstance(otsu_threshold, (int, float)):
            axes[1, 2].axvline(x=otsu_threshold, color='red', linestyle='--', linewidth=2,
                             label=f'Threshold: {otsu_threshold:.1f}')
            axes[1, 2].legend()
        axes[1, 2].set_title('Otsu: Histogram & Threshold', fontsize=14, fontweight='bold')
        axes[1, 2].grid(True, alpha=0.3)

        # 성능 정보 추가
        he_info = f"""HE Performance:
Contrast: {he_perf['contrast_improvement_percent']:+.1f}%
Brightness: {he_perf['brightness_change']:+.0f}
Dynamic Range: {he_perf['dynamic_range_improvement']:+d}
Entropy: {he_perf['entropy_change']:+.3f} bits"""

        fig.text(0.02, 0.02, he_info, fontsize=10, fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        plt.suptitle('Cross-Comparison: Histogram Equalization vs Otsu Thresholding\\n'
                    'Different Image Types and Processing Goals', fontsize=16, fontweight='bold')
        plt.tight_layout()

        # 저장
        save_path = os.path.join(os.path.dirname(__file__), '..', 'results',
                                'cross_comparison_he_vs_otsu.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"교차 비교 분석 저장: {save_path}")

        plt.show()

    def generate_comprehensive_report(self):
        """종합 분석 보고서 생성"""
        print("\\n종합 분석 보고서 생성 중...")

        # 여기서 상세한 마크다운 보고서를 생성할 수 있습니다
        # (별도 함수로 구현 예정)

        print("모든 분석이 완료되었습니다!")
        print("\\n저장된 파일들:")
        print("- he_dark_indoor_comprehensive.png: HE 종합 비교")
        print("- he_dark_indoor_performance.png: HE 성능 분석")
        print("- otsu_shadow_doc_comprehensive.png: Otsu 종합 비교")
        print("- cross_comparison_he_vs_otsu.png: HE vs Otsu 교차 비교")

def main():
    """메인 실행 함수"""
    analyzer = RealImageAnalyzer()

    print("실제 이미지를 사용한 포괄적인 분석을 시작합니다...")
    print("=" * 60)

    try:
        # 1. HE 분석 (어두운 실내 이미지)
        he_results, he_performance = analyzer.analyze_he_with_real_image()

        # 2. Otsu 분석 (그림자 문서 이미지)
        otsu_results = analyzer.analyze_otsu_with_real_image()

        # 3. 교차 비교 분석
        analyzer.create_cross_comparison()

        # 4. 종합 보고서 생성
        analyzer.generate_comprehensive_report()

    except Exception as e:
        print(f"분석 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()