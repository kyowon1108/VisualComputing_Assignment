#!/usr/bin/env python3
"""
포괄적인 컬러스페이스 및 알고리즘 비교 분석
Comprehensive Colorspace and Algorithm Comparison Analysis

YUV, YCbCr, LAB, HSV 컬러스페이스와 HE, CLAHE 알고리즘의 조합을 체계적으로 비교분석합니다.
Systematically compare and analyze combinations of YUV, YCbCr, LAB, HSV colorspaces with HE, CLAHE algorithms.
"""

import sys
import os

# 프로젝트 루트 디렉토리를 Python path에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import cv2
from src.he import histogram_equalization_color
from src.utils import load_image, display_images, create_test_image
import time
from typing import Dict, List, Tuple

class ComprehensiveAnalyzer:
    """포괄적인 컬러스페이스 및 알고리즘 분석 클래스"""

    def __init__(self):
        self.colorspaces = ['yuv', 'ycbcr', 'lab', 'hsv', 'rgb']
        self.algorithms = ['he', 'clahe']
        self.results = {}
        self.performance_data = {}

    def analyze_all_combinations(self, image_path: str = None, save_results: bool = True):
        """모든 컬러스페이스와 알고리즘 조합을 분석"""
        print("=== 포괄적인 컬러스페이스 및 알고리즘 비교 분석 ===")
        print(f"분석할 컬러스페이스: {self.colorspaces}")
        print(f"분석할 알고리즘: {self.algorithms}")

        # 테스트 이미지 준비
        if image_path and os.path.exists(image_path):
            image = load_image(image_path, color_mode='color')
            image_name = os.path.basename(image_path).split('.')[0]
            print(f"이미지 로드: {image_path}")
        else:
            image = self.create_comprehensive_test_image()
            image_name = "synthetic_test"
            print("합성 테스트 이미지 생성")

        print(f"이미지 크기: {image.shape}")

        # 모든 조합 테스트
        combinations = []
        for colorspace in self.colorspaces:
            for algorithm in self.algorithms:
                # RGB는 HE만 지원 (CLAHE는 single channel에 적용되므로 RGB 전체에는 부적합)
                if colorspace == 'rgb' and algorithm == 'clahe':
                    continue
                combinations.append((colorspace, algorithm))

        print(f"총 {len(combinations)}개 조합 테스트 시작...")

        # 결과 저장용
        results = {}
        performance_data = {}

        for i, (colorspace, algorithm) in enumerate(combinations):
            try:
                print(f"[{i+1}/{len(combinations)}] {colorspace.upper()} + {algorithm.upper()} 처리 중...")

                # 성능 측정
                start_time = time.time()

                # 알고리즘 적용
                if colorspace == 'lab' and algorithm == 'he':
                    # LAB는 CLAHE가 권장되지만 HE도 테스트
                    enhanced, process_info = histogram_equalization_color(
                        image, method=colorspace, algorithm=algorithm, show_process=False
                    )
                else:
                    # 기본 파라미터로 처리
                    enhanced, process_info = histogram_equalization_color(
                        image, method=colorspace, algorithm=algorithm,
                        clip_limit=2.0, tile_size=8, show_process=False
                    )

                end_time = time.time()
                processing_time = end_time - start_time

                # 결과 저장
                combo_name = f"{colorspace}_{algorithm}"
                results[combo_name] = {
                    'enhanced_image': enhanced,
                    'process_info': process_info,
                    'colorspace': colorspace,
                    'algorithm': algorithm
                }

                performance_data[combo_name] = {
                    'processing_time': processing_time,
                    'image_quality_metrics': self.calculate_quality_metrics(image, enhanced)
                }

                print(f"  완료 (처리시간: {processing_time:.3f}초)")

            except Exception as e:
                print(f"  실패: {str(e)}")
                continue

        self.results = results
        self.performance_data = performance_data

        if save_results:
            self.save_comparison_results(image, image_name)
            self.save_performance_analysis()
            self.generate_detailed_report(image, image_name)

        return results, performance_data

    def create_comprehensive_test_image(self):
        """포괄적인 테스트를 위한 합성 이미지 생성"""
        height, width = 300, 400
        image = np.zeros((height, width, 3), dtype=np.uint8)

        # 다양한 밝기 영역 생성
        # 1. 어두운 영역 (좌상단)
        image[:height//3, :width//3] = [30, 40, 50]

        # 2. 중간 밝기 영역 (중앙)
        image[height//3:2*height//3, width//3:2*width//3] = [120, 130, 140]

        # 3. 밝은 영역 (우하단)
        image[2*height//3:, 2*width//3:] = [200, 210, 220]

        # 4. 그라디언트 영역 (우상단)
        for i in range(height//3):
            for j in range(width//3, width):
                intensity = int((j - width//3) / (2*width//3) * 180 + 40)
                image[i, j] = [intensity, intensity + 10, intensity + 20]

        # 5. 세부 텍스처 추가 (좌하단)
        for i in range(2*height//3, height):
            for j in range(width//3):
                noise = np.random.randint(-20, 20)
                base_intensity = 80 + noise
                image[i, j] = [
                    np.clip(base_intensity, 0, 255),
                    np.clip(base_intensity + 15, 0, 255),
                    np.clip(base_intensity + 30, 0, 255)
                ]

        return image

    def calculate_quality_metrics(self, original: np.ndarray, enhanced: np.ndarray) -> Dict:
        """이미지 품질 지표 계산"""
        # 그레이스케일로 변환하여 분석
        orig_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        enh_gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)

        # 대비 개선 지표
        orig_contrast = np.std(orig_gray)
        enh_contrast = np.std(enh_gray)
        contrast_improvement = (enh_contrast - orig_contrast) / orig_contrast * 100

        # 히스토그램 분포 분석
        orig_hist = cv2.calcHist([orig_gray], [0], None, [256], [0, 256])
        enh_hist = cv2.calcHist([enh_gray], [0], None, [256], [0, 256])

        # 엔트로피 계산 (정보량 측정)
        orig_entropy = self.calculate_entropy(orig_hist)
        enh_entropy = self.calculate_entropy(enh_hist)

        # 평균 밝기
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
        # 정규화
        hist_norm = histogram.flatten() / np.sum(histogram)
        # 0인 값 제거 (log(0) 방지)
        hist_norm = hist_norm[hist_norm > 0]
        # 엔트로피 계산
        entropy = -np.sum(hist_norm * np.log2(hist_norm))
        return entropy

    def save_comparison_results(self, original_image: np.ndarray, image_name: str):
        """비교 결과를 이미지로 저장"""
        print("비교 결과 이미지 저장 중...")

        # 결과 정리
        combo_names = list(self.results.keys())
        n_results = len(combo_names)

        # 그리드 레이아웃 계산 (원본 포함)
        n_cols = 4  # 한 행에 4개씩
        n_rows = (n_results + 1 + n_cols - 1) // n_cols  # 원본 포함하여 계산

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        # 원본 이미지
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')

        # 각 결과 표시
        for i, combo_name in enumerate(combo_names):
            row = (i + 1) // n_cols
            col = (i + 1) % n_cols

            if row < n_rows and col < n_cols:
                result = self.results[combo_name]
                perf = self.performance_data[combo_name]

                axes[row, col].imshow(result['enhanced_image'])

                # 제목에 성능 정보 포함
                title = f"{result['colorspace'].upper()} + {result['algorithm'].upper()}\\n"
                title += f"Time: {perf['processing_time']:.3f}s\\n"
                title += f"Contrast: +{perf['image_quality_metrics']['contrast_improvement_percent']:.1f}%"

                axes[row, col].set_title(title, fontsize=10)
                axes[row, col].axis('off')

        # 빈 subplot 숨기기
        for i in range(n_results + 1, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            if row < n_rows and col < n_cols:
                axes[row, col].axis('off')

        plt.suptitle(f'Comprehensive Colorspace and Algorithm Comparison - {image_name}',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        # 결과 저장
        save_path = os.path.join(os.path.dirname(__file__), '..', 'results',
                                f'comprehensive_comparison_{image_name}.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"비교 결과 저장: {save_path}")

        plt.show()

    def save_performance_analysis(self):
        """성능 분석 차트 저장"""
        print("성능 분석 차트 생성 중...")

        combo_names = list(self.performance_data.keys())

        # 데이터 추출
        processing_times = [self.performance_data[name]['processing_time'] for name in combo_names]
        contrast_improvements = [self.performance_data[name]['image_quality_metrics']['contrast_improvement_percent']
                               for name in combo_names]
        entropy_changes = [
            self.performance_data[name]['image_quality_metrics']['enhanced_entropy'] -
            self.performance_data[name]['image_quality_metrics']['original_entropy']
            for name in combo_names
        ]

        # 차트 생성
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. 처리 시간 비교
        bars1 = ax1.bar(range(len(combo_names)), processing_times,
                       color=['skyblue' if 'he' in name else 'lightcoral' for name in combo_names])
        ax1.set_title('Processing Time Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_xticks(range(len(combo_names)))
        ax1.set_xticklabels([name.replace('_', '+').upper() for name in combo_names],
                           rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)

        # 값 표시
        for bar, time in zip(bars1, processing_times):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{time:.3f}', ha='center', va='bottom', fontsize=8)

        # 2. 대비 개선 비교
        bars2 = ax2.bar(range(len(combo_names)), contrast_improvements,
                       color=['skyblue' if 'he' in name else 'lightcoral' for name in combo_names])
        ax2.set_title('Contrast Improvement Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Contrast Improvement (%)')
        ax2.set_xticks(range(len(combo_names)))
        ax2.set_xticklabels([name.replace('_', '+').upper() for name in combo_names],
                           rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)

        # 값 표시
        for bar, improvement in zip(bars2, contrast_improvements):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{improvement:.1f}%', ha='center', va='bottom', fontsize=8)

        # 3. 엔트로피 변화 비교
        bars3 = ax3.bar(range(len(combo_names)), entropy_changes,
                       color=['skyblue' if 'he' in name else 'lightcoral' for name in combo_names])
        ax3.set_title('Information Content Change (Entropy)', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Entropy Change (bits)')
        ax3.set_xticks(range(len(combo_names)))
        ax3.set_xticklabels([name.replace('_', '+').upper() for name in combo_names],
                           rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        # 4. 처리 시간 vs 대비 개선 산점도
        colors = ['blue' if 'he' in name else 'red' for name in combo_names]
        scatter = ax4.scatter(processing_times, contrast_improvements, c=colors, alpha=0.7, s=100)
        ax4.set_title('Processing Time vs Contrast Improvement', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Processing Time (seconds)')
        ax4.set_ylabel('Contrast Improvement (%)')
        ax4.grid(True, alpha=0.3)

        # 범례
        import matplotlib.patches as patches
        he_patch = patches.Patch(color='blue', label='HE Algorithm')
        clahe_patch = patches.Patch(color='red', label='CLAHE Algorithm')
        ax4.legend(handles=[he_patch, clahe_patch])

        # 점에 라벨 추가
        for i, name in enumerate(combo_names):
            ax4.annotate(name.replace('_', '+').upper(),
                        (processing_times[i], contrast_improvements[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)

        plt.suptitle('Performance Analysis: Colorspace and Algorithm Comparison',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        # 저장
        save_path = os.path.join(os.path.dirname(__file__), '..', 'results',
                                'performance_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"성능 분석 저장: {save_path}")

        plt.show()

    def generate_detailed_report(self, original_image: np.ndarray, image_name: str):
        """상세한 분석 보고서 생성"""
        print("상세 분석 보고서 생성 중...")

        # 각 조합별 세부 분석
        for combo_name, result in self.results.items():
            self.create_detailed_analysis_plot(original_image, combo_name, result, image_name)

        # 전체 요약 테이블 생성
        self.create_summary_table()

    def create_detailed_analysis_plot(self, original: np.ndarray, combo_name: str,
                                    result: dict, image_name: str):
        """개별 조합에 대한 상세 분석 플롯"""
        enhanced = result['enhanced_image']
        colorspace = result['colorspace']
        algorithm = result['algorithm']

        # 4x2 서브플롯 생성
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))

        # 원본과 결과 이미지
        axes[0, 0].imshow(original)
        axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(enhanced)
        axes[0, 1].set_title(f'Enhanced ({combo_name.replace("_", "+").upper()})',
                           fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')

        # 그레이스케일 변환하여 히스토그램 분석
        orig_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        enh_gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)

        # 원본 히스토그램
        axes[0, 2].hist(orig_gray.flatten(), bins=50, alpha=0.7, color='blue', density=True)
        axes[0, 2].set_title('Original Histogram', fontsize=12)
        axes[0, 2].set_xlabel('Pixel Intensity')
        axes[0, 2].set_ylabel('Density')
        axes[0, 2].grid(True, alpha=0.3)

        # 향상된 히스토그램
        axes[0, 3].hist(enh_gray.flatten(), bins=50, alpha=0.7, color='red', density=True)
        axes[0, 3].set_title('Enhanced Histogram', fontsize=12)
        axes[0, 3].set_xlabel('Pixel Intensity')
        axes[0, 3].set_ylabel('Density')
        axes[0, 3].grid(True, alpha=0.3)

        # 차이 이미지
        diff_image = np.abs(enhanced.astype(np.float32) - original.astype(np.float32))
        axes[1, 0].imshow(diff_image.astype(np.uint8))
        axes[1, 0].set_title('Difference Image', fontsize=12)
        axes[1, 0].axis('off')

        # 대비 비교 (그레이스케일)
        axes[1, 1].imshow(np.hstack([orig_gray, enh_gray]), cmap='gray')
        axes[1, 1].set_title('Side-by-side (Gray)', fontsize=12)
        axes[1, 1].axis('off')

        # 히스토그램 중첩 비교
        axes[1, 2].hist(orig_gray.flatten(), bins=50, alpha=0.5, color='blue',
                       density=True, label='Original')
        axes[1, 2].hist(enh_gray.flatten(), bins=50, alpha=0.5, color='red',
                       density=True, label='Enhanced')
        axes[1, 2].set_title('Histogram Overlay', fontsize=12)
        axes[1, 2].set_xlabel('Pixel Intensity')
        axes[1, 2].set_ylabel('Density')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

        # 성능 지표 텍스트
        perf = self.performance_data[combo_name]
        metrics = perf['image_quality_metrics']

        metrics_text = f"""Performance Metrics:

Processing Time: {perf['processing_time']:.4f} seconds

Contrast:
  Original: {metrics['original_contrast']:.2f}
  Enhanced: {metrics['enhanced_contrast']:.2f}
  Improvement: {metrics['contrast_improvement_percent']:.1f}%

Brightness:
  Original: {metrics['original_brightness']:.1f}
  Enhanced: {metrics['enhanced_brightness']:.1f}
  Change: {metrics['brightness_change']:.1f}

Entropy (Information):
  Original: {metrics['original_entropy']:.3f} bits
  Enhanced: {metrics['enhanced_entropy']:.3f} bits

Algorithm: {algorithm.upper()}
Colorspace: {colorspace.upper()}
"""

        axes[1, 3].text(0.05, 0.95, metrics_text, transform=axes[1, 3].transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        axes[1, 3].axis('off')

        plt.suptitle(f'Detailed Analysis: {combo_name.replace("_", " + ").upper()} - {image_name}',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        # 개별 결과 저장
        save_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'detailed_analysis',
                                f'{combo_name}_{image_name}_detailed.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"상세 분석 저장: {save_path}")

        plt.close()  # 메모리 절약을 위해 닫기

    def create_summary_table(self):
        """성능 요약 테이블 생성 및 저장"""
        import pandas as pd

        # 데이터 수집
        summary_data = []
        for combo_name, perf_data in self.performance_data.items():
            metrics = perf_data['image_quality_metrics']

            colorspace, algorithm = combo_name.split('_')

            summary_data.append({
                'Colorspace': colorspace.upper(),
                'Algorithm': algorithm.upper(),
                'Processing_Time(s)': f"{perf_data['processing_time']:.4f}",
                'Contrast_Improvement(%)': f"{metrics['contrast_improvement_percent']:.1f}",
                'Brightness_Change': f"{metrics['brightness_change']:.1f}",
                'Original_Entropy': f"{metrics['original_entropy']:.3f}",
                'Enhanced_Entropy': f"{metrics['enhanced_entropy']:.3f}",
                'Entropy_Change': f"{metrics['enhanced_entropy'] - metrics['original_entropy']:.3f}"
            })

        df = pd.DataFrame(summary_data)

        # CSV 저장
        csv_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'performance_summary.csv')
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"성능 요약 테이블 저장: {csv_path}")

        # 콘솔에 출력
        print("\\n=== 성능 요약 테이블 ===")
        print(df.to_string(index=False))

def main():
    """메인 실행 함수"""
    analyzer = ComprehensiveAnalyzer()

    # 기본 이미지로 분석 (실제 이미지 경로가 있다면 사용)
    test_image_path = None
    # test_image_path = "images/your_test_image.jpg"  # 실제 경로로 변경

    print("포괄적인 컬러스페이스 및 알고리즘 비교 분석을 시작합니다...")

    try:
        results, performance_data = analyzer.analyze_all_combinations(
            image_path=test_image_path,
            save_results=True
        )

        print(f"\\n분석 완료! {len(results)}개 조합이 성공적으로 처리되었습니다.")
        print("\\n주요 결과:")

        # 최고 성능 조합 찾기
        best_contrast = max(performance_data.items(),
                          key=lambda x: x[1]['image_quality_metrics']['contrast_improvement_percent'])
        fastest = min(performance_data.items(),
                     key=lambda x: x[1]['processing_time'])

        print(f"최고 대비 개선: {best_contrast[0]} ({best_contrast[1]['image_quality_metrics']['contrast_improvement_percent']:.1f}%)")
        print(f"가장 빠른 처리: {fastest[0]} ({fastest[1]['processing_time']:.4f}초)")

        print("\\n모든 결과는 'results/' 디렉토리에 저장되었습니다.")

    except Exception as e:
        print(f"분석 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()