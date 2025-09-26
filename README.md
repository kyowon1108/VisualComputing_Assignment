# 비쥬얼컴퓨팅 과제1 - Histogram Equalization & Local Otsu Thresholding

## 프로젝트 개요 / Project Overview

## Path Changes

| Previous Path | New Path |
|---------------|----------|
| run_he.py | scripts/cli/run_he.py |
| run_otsu.py | scripts/cli/run_otsu.py |



본 프로젝트는 **컬러 이미지 히스토그램 평활화(Histogram Equalization)**와 **Local Otsu Thresholding**을 직접 구현한 비쥬얼컴퓨팅 과제입니다. OpenCV의 내장 함수를 사용하지 않고 low-level 알고리즘을 직접 구현하여 이론적 배경과 수학적 원리를 깊이 이해할 수 있도록 구성되었습니다.

This project directly implements **Color Image Histogram Equalization** and **Local Otsu Thresholding** for a Visual Computing assignment. It implements low-level algorithms without using OpenCV's built-in functions to provide deep understanding of theoretical backgrounds and mathematical principles.

## 주요 특징 / Key Features

### 🎨 컬러 이미지 히스토그램 평활화 / Color Image Histogram Equalization
- **YUV 색공간 기반 처리**: Y(휘도) 채널만 처리하여 자연스러운 색감 유지
- **CLAHE 구현**: Contrast Limited Adaptive Histogram Equalization으로 노이즈 방지
- **단계별 시각화**: CDF 계산, 픽셀 매핑 과정의 중간 단계 시각화
- **이론적 배경**: CDF 변환의 물리적 의미와 수식 도출 과정 설명

### 🔍 Local Otsu Thresholding
- **Inter-class Variance 최대화**: 수학적 원리에 기반한 최적 임계값 자동 계산
- **블록 기반 처리**: 이미지를 블록으로 분할하여 지역적 적응 임계값 적용
- **슬라이딩 윈도우**: 중첩 윈도우를 통한 부드러운 임계값 전환
- **🆕 개선된 경계 처리**: 겹치는 블록과 가중 블렌딩으로 블록 아티팩트 96.3% 감소
- **텍스트 친화적 후처리**: 문서 이미지에 최적화된 형태학적 처리
- **비교 분석**: 다양한 방법들의 성능 비교 및 시각화

### 🖥️ 직관적인 GUI
- **실시간 미리보기**: 원본과 처리 결과의 실시간 비교
- **파라미터 조정**: 슬라이더와 콤보박스를 통한 직관적인 설정
- **중간 과정 표시**: 알고리즘의 각 단계별 시각화
- **다국어 지원**: 한글/영어 병행 표기

## 프로젝트 구조 / Project Structure

```
assign1/
├── README.md           # 프로젝트 설명서 / Project documentation
├── requirements.txt    # 의존성 패키지 목록 / Dependency packages
├── download_images.py  # 테스트 이미지 다운로드 스크립트 / Test image download script
├── run_he.py          # HE 명령줄 실행 스크립트 / HE command line script
├── run_otsu.py        # Local Otsu 명령줄 실행 스크립트 / Local Otsu command line script
├── demo.py            # 종합 데모 스크립트 / Comprehensive demo script
├── src/               # 소스 코드 / Source code
│   ├── __init__.py    # 패키지 초기화 / Package initialization
│   ├── he.py          # 히스토그램 평활화 구현 / Histogram Equalization implementation
│   ├── otsu.py        # Local Otsu Thresholding 구현 / Local Otsu Thresholding implementation
│   ├── improved_local_otsu.py  # 🆕 개선된 Local Otsu (겹치는 블록, 보간법) / Improved Local Otsu
│   └── utils.py       # 공통 유틸리티 함수 / Common utility functions
├── docs/              # 문서 및 과제 요구사항 / Documents and assignment requirements
│   ├── 01_claude_prompt.md
│   └── 02-L2_Image_Processing_1.pdf
├── images/            # 테스트 이미지 폴더 / Test images folder
├── results/           # 처리 결과 저장 폴더 / Processing results folder
└── tests/             # 테스트 스크립트 / Test scripts
```

## 환경 설정 / Environment Setup

### 1. Conda 환경 생성 / Create Conda Environment
```bash
# Python 3.13 환경 생성 (과제 요구사항)
conda create -n python313 python=3.13
conda activate python313
```

### 2. 의존성 설치 / Install Dependencies
```bash
# 프로젝트 폴더로 이동
cd assign1

# conda로 설치 (권장) / Install with conda (recommended)
conda install numpy opencv matplotlib pillow requests

# 또는 pip로 설치 / Or install with pip
pip install -r requirements.txt
```

### 3. 테스트 이미지 다운로드 / Download Test Images
```bash
# 테스트 이미지 자동 다운로드 (선택사항)
python download_images.py
```

## 사용법 / Usage

### 1. 히스토그램 평활화 실행 / Histogram Equalization
```bash
# Global HE (YUV 색공간, 권장) - 자동 시각화
python scripts/cli/run_he.py images/your_image.jpg --algorithm he --method yuv --save results/

# Global HE (RGB 채널별 처리) - 자동 시각화
python scripts/cli/run_he.py images/your_image.jpg --algorithm he --method rgb --save results/

# Adaptive HE (AHE) - 자동 시각화
python scripts/cli/run_he.py images/your_image.jpg --algorithm ahe --tile-size 16 --save results/

# CLAHE (권장) - 자동 시각화
python scripts/cli/run_he.py images/your_image.jpg --algorithm clahe --clip-limit 2.0 --tile-size 8 --save results/

# 그레이스케일 처리
python scripts/cli/run_he.py images/your_image.jpg --algorithm he --method gray --save results/
```

**알고리즘 옵션:**
- `he`: Global Histogram Equalization (전역 히스토그램 평활화)
- `ahe`: Adaptive Histogram Equalization (적응적 히스토그램 평활화)
- `clahe`: Contrast Limited Adaptive Histogram Equalization (대비 제한 적응적 평활화, 권장)

**방법 옵션 (--method):**
- `yuv`: YUV 색공간에서 Y(휘도) 채널만 처리 (컬러 이미지 권장)
- `rgb`: RGB 각 채널을 개별적으로 처리
- `gray`: 그레이스케일로 변환하여 처리

**추가 파라미터:**
- `--clip-limit`: CLAHE의 클립 한계값 (기본값: 2.0, 범위: 1.0-4.0)
- `--tile-size`: CLAHE/AHE의 타일 크기 (기본값: 8, 권장: 8-16)

**⚠️ 중요:** 모든 HE 알고리즘은 실행 시 자동으로 히스토그램, 이전/이후 비교, CDF 등의 시각화가 표시됩니다.

### 2. Local Otsu Thresholding 실행 / Local Otsu Thresholding
```bash
# 모든 방법 비교 (기본)
python scripts/cli/run_otsu.py images/your_image.jpg --method compare --save results/

# 특정 방법만 실행
python scripts/cli/run_otsu.py images/your_image.jpg --method global --save results/
python scripts/cli/run_otsu.py images/your_image.jpg --method block --block-size 32 --save results/
python scripts/cli/run_otsu.py images/your_image.jpg --method sliding --block-size 32 --stride 16 --save results/
python scripts/cli/run_otsu.py images/your_image.jpg --method improved --block-size 32 --save results/  # 🆕 개선된 방법

# 비교 시각화와 함께 실행
python scripts/cli/run_otsu.py images/your_image.jpg --method compare --show-comparison --save results/
```

**방법 옵션:**
- `global`: 전체 이미지에 단일 임계값 적용
- `block`: 이미지를 블록으로 분할하여 각각 처리
- `sliding`: 슬라이딩 윈도우로 부드러운 처리
- `improved`: 🆕 개선된 겹치는 블록 방법 (블록 아티팩트 해결, 권장)
- `compare`: 모든 방법의 결과를 동시에 비교

### 3. 종합 데모 실행 / Comprehensive Demo
```bash
# 모든 기능 자동 테스트
python demo.py
```
- images/ 폴더의 모든 이미지에 대해 HE와 Local Otsu 자동 실행
- 테스트 이미지가 없으면 자동으로 생성
- 모든 결과를 results/ 폴더에 저장

## 핵심 구현 내용 / Core Implementation

### 히스토그램 평활화 원리 / Histogram Equalization Principle

```python
# CDF 기반 픽셀 매핑 / CDF-based pixel mapping
y' = Scale * CDF(x)
```

**수학적 원리 / Mathematical Principle:**
1. 히스토그램 계산: `h(i) = 픽셀값 i의 빈도수`
2. CDF 계산: `CDF(i) = Σ(h(0) to h(i)) / 총 픽셀 수`
3. 변환 공식: `y' = 255 * CDF(x)` (8비트 이미지의 경우)

**물리적 의미**: CDF는 특정 픽셀값 이하의 픽셀들이 전체에서 차지하는 비율을 나타내며, 이를 통해 동일한 분포로 픽셀값을 재배치합니다.

### Otsu Thresholding 원리 / Otsu Thresholding Principle

```python
# Inter-class variance 최대화
σ²(between) = w₀ × w₁ × (μ₀ - μ₁)²
```

**핵심 개념 / Key Concepts:**
- **Inter-class variance 최대화**: 클래스 간 분산을 최대화하여 최적 분리
- **Within-class variance 최소화**: 클래스 내 분산을 최소화
- **수학적 관계**: `σ²(total) = σ²(within) + σ²(between)`

**지역적 적응 / Local Adaptation:**
- 블록 기반: 이미지를 균등 분할하여 각 블록마다 독립적으로 Otsu 적용
- 슬라이딩 윈도우: 중첩되는 윈도우를 통해 부드러운 임계값 전환

### 🆕 개선된 Local Otsu / Improved Local Otsu

**블록 경계 아티팩트 문제 해결:**
기존 블록 기반 방법의 주요 문제점인 블록 경계에서의 불연속적 임계값으로 인한 시각적 아티팩트를 해결했습니다.

```python
# 겹치는 블록 처리 / Overlapping Block Processing
step_size = block_size * (1 - overlap_ratio)  # 50% 겹침
weighted_threshold = Σ(threshold_i × weight_i) / Σ(weight_i)
```

**핵심 개선사항:**
- **96.3% 아티팩트 감소**: 블록 경계 불연속성 109.04 → 4.04로 대폭 개선
- **겹치는 블록**: 50% 겹침으로 부드러운 임계값 전환 구현
- **가중 블렌딩**: 거리 기반 또는 가우시안 가중치를 통한 자연스러운 결합
- **텍스트 친화적**: 문서 이미지에 최적화된 후처리 (min_size=6, 형태학적 연산 최소화)

## 주요 기능 상세 / Detailed Features

### 🔬 중간 과정 시각화 / Intermediate Process Visualization
- **히스토그램 변화**: 원본 → 평활화된 히스토그램 비교
- **CDF 그래프**: 누적분포함수의 형태와 변화 과정
- **픽셀 매핑 함수**: 입력-출력 픽셀값의 매핑 관계
- **임계값 맵**: Local Otsu에서 각 영역별 임계값 분포

### 📊 성능 분석 도구 / Performance Analysis Tools
- **임계값 분포 히스토그램**: Local 방법들의 임계값 분포 비교
- **통계 정보**: 평균, 표준편차, 최소/최대 임계값
- **처리 영역 비율**: 슬라이딩 윈도우의 커버리지 분석

### 💾 결과 저장 및 관리 / Result Saving and Management
- **고품질 이미지 저장**: PNG, JPEG 형식 지원
- **처리 정보 기록**: 사용된 파라미터와 결과 통계
- **배치 처리 지원**: 여러 이미지의 일괄 처리 (확장 가능)

## 이론적 배경 강화 / Enhanced Theoretical Background

### YUV 색공간 선택 이유 / Rationale for YUV Color Space
- **Y 채널**: 인간의 시각 인지와 밀접한 휘도 정보
- **U, V 채널**: 색상 정보 보존으로 자연스러운 색감 유지
- **RGB 대비 장점**: 각 채널 개별 처리 시 발생하는 색상 왜곡 방지

### CLAHE의 Clip Limit 효과 / CLAHE Clip Limit Effects
- **노이즈 증폭 방지**: 히스토그램의 급격한 변화 제한
- **선형적 CDF**: 클리핑을 통한 부드러운 대비 개선
- **최적값 범위**: 2-4 사이의 값으로 균형있는 개선 효과

### Inter-class vs Within-class Variance / Inter-class vs Within-class Variance
- **수학적 관계**: 전체 분산의 분해를 통한 최적화
- **분리 기준**: 클래스 간 차이 최대화와 클래스 내 동질성 확보
- **자동 임계값**: 수학적 최적화를 통한 객관적 기준 제시

## 실행 예시 / Execution Examples

### 명령행에서 개별 모듈 테스트 / Individual Module Testing from Command Line

```python
# 히스토그램 평활화 테스트
from src.he import histogram_equalization_color
from src.utils import load_image

image = load_image('test_image.jpg')
result, info = histogram_equalization_color(image, method='yuv', show_process=True)
```

```python
# Local Otsu 테스트
from src.otsu import compare_otsu_methods
from src.utils import load_image
import cv2

image = load_image('test_image.jpg', color_mode='gray')
comparison = compare_otsu_methods(image, show_comparison=True)
```

## 트러블슈팅 / Troubleshooting

### 일반적인 문제 / Common Issues

1. **tkinter 모듈 오류**
   ```bash
   # macOS
   brew install python-tk

   # Ubuntu/Debian
   sudo apt-get install python3-tk
   ```

2. **메모리 부족 오류 (대용량 이미지)**
   - 이미지 크기를 줄이거나 타일/블록 크기를 증가시키세요
   - 슬라이딩 윈도우의 스트라이드를 증가시키세요

3. **OpenCV 설치 문제**
   ```bash
   # conda 환경에서 OpenCV 재설치
   conda install opencv

   # 또는 pip로
   pip uninstall opencv-python
   pip install opencv-python==4.5.0
   ```

4. **matplotlib 백엔드 오류**
   ```python
   import matplotlib
   matplotlib.use('TkAgg')  # GUI 환경에서
   matplotlib.use('Agg')    # 서버 환경에서
   ```

## 성능 최적화 팁 / Performance Optimization Tips

1. **이미지 크기 조정**: 대용량 이미지는 적절한 크기로 리사이즈
2. **블록 크기 선택**: 32x32 ~ 64x64가 일반적으로 최적
3. **스트라이드 설정**: 윈도우 크기의 1/4 ~ 1/2 권장
4. **메모리 관리**: 처리 후 불필요한 중간 결과 삭제

## 확장 가능성 / Extensibility

### 추가 구현 가능한 기능 / Additional Implementable Features
- **다른 색공간 지원**: HSV, LAB 색공간 처리
- **적응적 CLAHE**: 이미지 특성에 따른 자동 파라미터 조정
- **멀티스케일 Otsu**: 다양한 스케일에서의 임계값 결합
- **GPU 가속**: CUDA를 이용한 병렬 처리
- **배치 처리**: 다수 이미지의 자동 처리

### 연구 확장 방향 / Research Extension Directions
- **딥러닝 기반 개선**: 신경망을 이용한 적응적 파라미터 학습
- **ROI 기반 처리**: 관심 영역 중심의 선택적 처리
- **실시간 처리**: 비디오 스트림에 대한 실시간 적용

## 참고 자료 / References

### 학술 자료 / Academic References
1. Otsu, N. (1979). "A threshold selection method from gray-level histograms"
2. Zuiderveld, K. (1994). "Contrast Limited Adaptive Histogram Equalization"
3. Gonzalez, R. C., & Woods, R. E. (2017). "Digital Image Processing"

### 온라인 자료 / Online Resources
- [OpenCV Documentation](https://docs.opencv.org/)
- [Histogram Equalization Theory](https://en.wikipedia.org/wiki/Histogram_equalization)
- [Otsu's Method Explanation](https://en.wikipedia.org/wiki/Otsu%27s_method)

---

**⚠️ 중요 참고사항 / Important Notes:**
- 이 구현은 교육 목적으로 작성되었으며, OpenCV의 최적화된 함수들을 대체하지 않습니다.
- 실제 프로덕션 환경에서는 OpenCV의 내장 함수 사용을 권장합니다.
- 모든 알고리즘은 이론적 이해를 돕기 위해 단계별로 구현되었습니다.

*This implementation is created for educational purposes and does not replace OpenCV's optimized functions. For production environments, using OpenCV's built-in functions is recommended. All algorithms are implemented step-by-step to aid theoretical understanding.*