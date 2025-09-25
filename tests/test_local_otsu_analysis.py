#!/usr/bin/env python3
"""
Local Otsu 중간 과정 상세 분석 및 시각화 테스트
Test for detailed analysis and visualization of Local Otsu intermediate processes

이 테스트는 Local Otsu의 다음 중간 과정들을 시각화합니다:
1. 원본 이미지와 임계값 처리 결과
2. 원본 vs 개선된 임계값 맵
3. 임계값 차이 맵 및 경계 불연속성 분석
4. 임계값 분포 히스토그램
5. Edge Artifact Score 분석
6. 개선 효과 요약

This test visualizes the following intermediate processes of Local Otsu:
1. Original image and thresholding results
2. Original vs improved threshold maps
3. Threshold difference map and boundary discontinuity analysis
4. Threshold distribution histogram
5. Edge Artifact Score analysis
6. Improvement summary
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage
from skimage import filters, measure

# 현재 스크립트의 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from src.utils import load_image
from src.otsu import local_otsu_block_based, local_otsu_improved_boundary


def analyze_local_otsu_process(image_path: str, save_figure: bool = True):
    """
    Local Otsu의 중간 과정을 상세 분석합니다.
    Analyze the intermediate processes of Local Otsu in detail.

    Args:
        image_path (str): 입력 이미지 경로 / Input image path
        save_figure (bool): figure를 저장할지 여부 / Whether to save figure
    """
    print(f"이미지 로딩 중: {image_path}")

    # 그레이스케일 이미지 로드
    original_image = load_image(image_path, color_mode='gray')
    print(f"원본 이미지 크기: {original_image.shape}")

    block_size = (32, 32)

    # 원본 블록 기반 Local Otsu
    print("원본 블록 기반 Local Otsu 적용 중...")
    original_result, original_info = local_otsu_block_based(
        original_image, block_size=block_size, show_process=False
    )

    # 개선된 Local Otsu
    print("개선된 Local Otsu 적용 중...")
    improved_result, improved_info = local_otsu_improved_boundary(
        original_image, block_size=block_size, show_process=False
    )

    # 중간 과정 분석
    print("중간 과정 분석 중...")
    analysis_results = perform_detailed_analysis(
        original_image, original_result, improved_result,
        original_info, improved_info
    )

    # 시각화
    saved_path = visualize_local_otsu_analysis(
        original_image, original_result, improved_result,
        original_info, improved_info, analysis_results,
        image_path, save_figure
    )

    return {
        'original_image': original_image,
        'original_result': original_result,
        'improved_result': improved_result,
        'original_info': original_info,
        'improved_info': improved_info,
        'analysis_results': analysis_results,
        'saved_figure_path': saved_path
    }


def perform_detailed_analysis(original_image, original_result, improved_result,
                            original_info, improved_info):
    """상세 분석을 수행합니다."""

    # 임계값 맵 추출
    original_threshold_map = original_info.get('threshold_map', None)
    improved_threshold_map = improved_info.get('threshold_map', None)

    analysis = {}

    if original_threshold_map is not None and improved_threshold_map is not None:
        # 임계값 차이 맵
        threshold_diff = np.abs(improved_threshold_map - original_threshold_map)
        analysis['threshold_difference'] = threshold_diff

        # 경계 불연속성 분석
        analysis['boundary_discontinuity'] = analyze_boundary_discontinuity(
            original_threshold_map, improved_threshold_map
        )

        # 임계값 분포 분석
        analysis['threshold_distribution'] = analyze_threshold_distribution(
            original_threshold_map, improved_threshold_map
        )

        # Edge Artifact Score 계산
        analysis['edge_artifact_score'] = calculate_edge_artifact_score(
            original_result, improved_result, original_image
        )

    # 개선 효과 요약
    analysis['improvement_summary'] = calculate_improvement_metrics(
        original_image, original_result, improved_result,
        original_info, improved_info
    )

    return analysis


def analyze_boundary_discontinuity(original_map, improved_map):
    """경계 불연속성을 분석합니다."""

    # 그라디언트 계산 (경계에서의 임계값 변화)
    original_grad_x = np.abs(np.gradient(original_map, axis=1))
    original_grad_y = np.abs(np.gradient(original_map, axis=0))
    original_grad_mag = np.sqrt(original_grad_x**2 + original_grad_y**2)

    improved_grad_x = np.abs(np.gradient(improved_map, axis=1))
    improved_grad_y = np.abs(np.gradient(improved_map, axis=0))
    improved_grad_mag = np.sqrt(improved_grad_x**2 + improved_grad_y**2)

    return {
        'original_gradient_magnitude': original_grad_mag,
        'improved_gradient_magnitude': improved_grad_mag,
        'original_discontinuity_score': np.mean(original_grad_mag),
        'improved_discontinuity_score': np.mean(improved_grad_mag),
        'discontinuity_reduction': (np.mean(original_grad_mag) - np.mean(improved_grad_mag)) / np.mean(original_grad_mag) * 100
    }


def analyze_threshold_distribution(original_map, improved_map):
    """임계값 분포를 분석합니다."""

    original_values = original_map[original_map > 0]
    improved_values = improved_map[improved_map > 0]

    return {
        'original_mean': np.mean(original_values),
        'original_std': np.std(original_values),
        'original_range': (np.min(original_values), np.max(original_values)),
        'improved_mean': np.mean(improved_values),
        'improved_std': np.std(improved_values),
        'improved_range': (np.min(improved_values), np.max(improved_values)),
        'original_values': original_values,
        'improved_values': improved_values
    }


def calculate_edge_artifact_score(original_result, improved_result, original_image):
    """Edge Artifact Score를 계산합니다."""

    # 원본 이미지의 엣지 검출
    true_edges = cv2.Canny(original_image.astype(np.uint8), 50, 150)

    # 임계값 처리 결과의 엣지 검출
    original_edges = cv2.Canny(original_result.astype(np.uint8) * 255, 50, 150)
    improved_edges = cv2.Canny(improved_result.astype(np.uint8) * 255, 50, 150)

    # 인공적 엣지 (실제 엣지가 아닌 부분에서 발생하는 엣지) 계산
    original_artifacts = np.logical_and(original_edges, ~true_edges)
    improved_artifacts = np.logical_and(improved_edges, ~true_edges)

    original_score = np.sum(original_artifacts) / np.sum(original_edges > 0) * 100 if np.sum(original_edges) > 0 else 0
    improved_score = np.sum(improved_artifacts) / np.sum(improved_edges > 0) * 100 if np.sum(improved_edges) > 0 else 0

    return {
        'original_edge_map': original_edges,
        'improved_edge_map': improved_edges,
        'original_artifacts': original_artifacts,
        'improved_artifacts': improved_artifacts,
        'original_score': original_score,
        'improved_score': improved_score,
        'artifact_reduction': (original_score - improved_score) / max(original_score, 1e-6) * 100
    }


def calculate_improvement_metrics(original_image, original_result, improved_result,
                                original_info, improved_info):
    """개선 지표를 계산합니다."""

    # 대비 개선 지표
    original_contrast = np.std(original_result.astype(np.float32))
    improved_contrast = np.std(improved_result.astype(np.float32))

    # 세부사항 보존 지표 (SSIM 기반)
    from skimage.metrics import structural_similarity as ssim
    detail_preservation = ssim(original_image, improved_result.astype(np.uint8) * 255)

    # 노이즈 감소 지표
    original_noise = measure.shannon_entropy(original_result)
    improved_noise = measure.shannon_entropy(improved_result)

    return {
        'contrast_improvement': (improved_contrast - original_contrast) / max(original_contrast, 1e-6) * 100,
        'detail_preservation': detail_preservation,
        'noise_reduction': (original_noise - improved_noise) / max(original_noise, 1e-6) * 100,
        'overall_quality_score': detail_preservation * 0.6 + (improved_contrast / 255.0) * 0.4
    }


def visualize_local_otsu_analysis(original_image, original_result, improved_result,
                                original_info, improved_info, analysis_results,
                                image_path, save_figure=True):
    """Local Otsu 분석 결과를 시각화합니다."""

    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle('Local Otsu 중간 과정 상세 분석\nLocal Otsu Intermediate Process Detailed Analysis',
                 fontsize=16, fontweight='bold')

    # 첫 번째 행: 원본 이미지와 결과들
    axes[0, 0].imshow(original_image, cmap='gray')
    axes[0, 0].set_title('원본 이미지\nOriginal Image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(original_result, cmap='gray')
    axes[0, 1].set_title('원본 방법 결과\nOriginal Method Result')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(improved_result, cmap='gray')
    axes[0, 2].set_title('개선된 방법 결과\nImproved Method Result')
    axes[0, 2].axis('off')

    # 차이 이미지 (더 밝을수록 더 다름)
    diff_image = np.abs(improved_result.astype(np.float32) - original_result.astype(np.float32))
    im = axes[0, 3].imshow(diff_image, cmap='hot')
    axes[0, 3].set_title('결과 차이\nResult Difference\n(Brighter = More Different)')
    axes[0, 3].axis('off')
    plt.colorbar(im, ax=axes[0, 3], fraction=0.046, pad=0.04)

    # 두 번째 행: 임계값 맵들
    original_threshold_map = original_info.get('threshold_map')
    improved_threshold_map = improved_info.get('threshold_map')

    if original_threshold_map is not None:
        im1 = axes[1, 0].imshow(original_threshold_map, cmap='jet', vmin=0, vmax=200)
        axes[1, 0].set_title('원본 임계값 맵\nOriginal Threshold Map')
        axes[1, 0].axis('off')
        plt.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)

    if improved_threshold_map is not None:
        im2 = axes[1, 1].imshow(improved_threshold_map, cmap='jet', vmin=0, vmax=200)
        axes[1, 1].set_title('개선된 임계값 맵\nImproved Threshold Map')
        axes[1, 1].axis('off')
        plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)

    # 임계값 차이 맵
    if 'threshold_difference' in analysis_results:
        threshold_diff = analysis_results['threshold_difference']
        im3 = axes[1, 2].imshow(threshold_diff, cmap='hot')
        axes[1, 2].set_title('임계값 차이 맵\nThreshold Difference Map')
        axes[1, 2].axis('off')
        plt.colorbar(im3, ax=axes[1, 2], fraction=0.046, pad=0.04)

    # 개선 효과 요약
    if 'improvement_summary' in analysis_results:
        improvement = analysis_results['improvement_summary']
        boundary_disc = analysis_results.get('boundary_discontinuity', {})
        edge_artifact = analysis_results.get('edge_artifact_score', {})

        summary_text = f"""개선 효과 요약 / Improvement Summary:

경계 불연속성 개선:
• 기존: {boundary_disc.get('original_discontinuity_score', 0):.2f}
• 개선: {boundary_disc.get('improved_discontinuity_score', 0):.2f}
• 감소율: {boundary_disc.get('discontinuity_reduction', 0):.1f}%

Edge Artifact 점수:
• 기존: {edge_artifact.get('original_score', 0):.1f}%
• 개선: {edge_artifact.get('improved_score', 0):.1f}%
• 감소율: {edge_artifact.get('artifact_reduction', 0):.1f}%

전체 품질:
• 세부사항 보존: {improvement['detail_preservation']:.1%}
• 대비 개선: {improvement['contrast_improvement']:.1f}%
• 전체 품질 점수: {improvement['overall_quality_score']:.1%}
"""

        axes[1, 3].text(0.05, 0.95, summary_text, transform=axes[1, 3].transAxes,
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
        axes[1, 3].set_title('개선 효과 요약\nImprovement Summary')
        axes[1, 3].axis('off')

    # 세 번째 행: 분석 그래프들

    # 경계 불연속성 분포
    if 'boundary_discontinuity' in analysis_results:
        boundary_data = analysis_results['boundary_discontinuity']
        orig_hist, orig_bins = np.histogram(boundary_data['original_gradient_magnitude'].flatten(), bins=50, density=True)
        imp_hist, imp_bins = np.histogram(boundary_data['improved_gradient_magnitude'].flatten(), bins=50, density=True)

        axes[2, 0].bar(orig_bins[:-1], orig_hist, alpha=0.5, color='red', label='Original', width=np.diff(orig_bins))
        axes[2, 0].bar(imp_bins[:-1], imp_hist, alpha=0.5, color='green', label='Improved', width=np.diff(imp_bins))
        axes[2, 0].set_title('경계 불연속성 분포\nBoundary Discontinuity\nDistribution')
        axes[2, 0].set_xlabel('Gradient Magnitude')
        axes[2, 0].set_ylabel('Density')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)

    # 임계값 분포 히스토그램
    if 'threshold_distribution' in analysis_results:
        thresh_data = analysis_results['threshold_distribution']

        axes[2, 1].hist(thresh_data['original_values'], bins=30, alpha=0.5, color='red',
                       label='Original', density=True)
        axes[2, 1].hist(thresh_data['improved_values'], bins=30, alpha=0.5, color='green',
                       label='Improved', density=True)
        axes[2, 1].set_title('임계값 분포\nThreshold Value\nDistribution')
        axes[2, 1].set_xlabel('Threshold Value')
        axes[2, 1].set_ylabel('Density')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)

    # Edge Artifact Score 비교
    if 'edge_artifact_score' in analysis_results:
        edge_data = analysis_results['edge_artifact_score']
        methods = ['Original', 'Improved']
        scores = [edge_data['original_score'], edge_data['improved_score']]
        colors = ['red', 'green']

        bars = axes[2, 2].bar(methods, scores, color=colors, alpha=0.7)
        axes[2, 2].set_title('Edge Artifact 점수\nEdge Artifact Score\n(Lower = Better)')
        axes[2, 2].set_ylabel('Artifact Score (%)')
        axes[2, 2].grid(True, alpha=0.3)

        # 값 표시
        for bar, score in zip(bars, scores):
            axes[2, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                           f'{score:.1f}%', ha='center', va='bottom')

    # 방법 비교
    comparison_text = f"""방법 비교 / Method Comparison:

🔴 기존 블록 기반 방법:
• 각 블록에서 독립적으로 Otsu 적용
• 블록 경계에서 임계값 불연속
• 격자 아티팩트 발생
• 처리 속도 빠름

🟢 개선된 방법:
• 겹치는 블록으로 부드러운 전환
• 가중 평균을 통한 임계값 블렌딩
• 경계 불연속성 {boundary_disc.get('discontinuity_reduction', 0):.1f}% 감소
• Edge 아티팩트 {edge_artifact.get('artifact_reduction', 0):.1f}% 감소

💡 핵심 개선사항:
• 50% 겹치는 블록 구조
• 거리 기반 가중 블렌딩
• 형태학적 후처리 최적화
• 문서 이미지에 특화된 파라미터
"""

    axes[2, 3].text(0.05, 0.95, comparison_text, transform=axes[2, 3].transAxes,
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    axes[2, 3].set_title('방법 비교\nMethod Comparison')
    axes[2, 3].axis('off')

    plt.tight_layout()

    saved_path = None
    if save_figure:
        # 저장 경로 생성
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        save_dir = os.path.join(os.path.dirname(image_path), '..', 'results')
        os.makedirs(save_dir, exist_ok=True)

        saved_path = os.path.join(save_dir, f'{base_name}_local_otsu_analysis.png')
        plt.savefig(saved_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"📊 Local Otsu 분석 결과 저장됨: {saved_path}")

    plt.show()
    return saved_path


def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(description='Local Otsu 중간 과정 상세 분석')
    parser.add_argument('image_path', help='입력 이미지 경로')
    parser.add_argument('--no-save', action='store_true',
                       help='figure를 저장하지 않음 (기본값: 저장함)')

    args = parser.parse_args()

    try:
        result = analyze_local_otsu_process(args.image_path, save_figure=not args.no_save)

        print("\n✅ Local Otsu 중간 과정 분석이 완료되었습니다!")
        print("🔍 원본 방법과 개선된 방법의 상세 비교를 확인해보세요.")

        # 주요 개선 지표 출력
        analysis = result['analysis_results']
        if 'boundary_discontinuity' in analysis:
            boundary = analysis['boundary_discontinuity']
            print(f"\n📊 주요 개선 지표:")
            print(f"   경계 불연속성: {boundary['discontinuity_reduction']:.1f}% 감소")

        if 'edge_artifact_score' in analysis:
            edge = analysis['edge_artifact_score']
            print(f"   Edge 아티팩트: {edge['artifact_reduction']:.1f}% 감소")

        if 'improvement_summary' in analysis:
            improvement = analysis['improvement_summary']
            print(f"   전체 품질 점수: {improvement['overall_quality_score']:.1%}")

        if result['saved_figure_path']:
            print(f"💾 분석 결과가 저장되었습니다: {result['saved_figure_path']}")

        return result

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()