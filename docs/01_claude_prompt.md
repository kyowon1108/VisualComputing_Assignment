# Claude Prompt - 비쥬얼컴퓨팅 과제1 (최종 업데이트)

## 프로젝트 개요
비쥬얼컴퓨팅 과제 목적의 Python 프로젝트를 생성하고 싶습니다.
과제를 제출한 사람들 중 **Top 1**이 될만한 학부생의 구조로 작성해 주세요.
아래 요구사항을 만족하는 코드를 단계별로 작성해 주세요. assign1 폴더에 전체적으로 구현을 해주셔야 합니다.

## 요구사항 원본
```
* 주제1: 컬러 이미지 HE (Histogram Equalization)
* 주제2: Local Otsu Thresholding
* 동작 코드 구현 '확인' 및 중간 과정 리포트 (Presentation Slide 형식)
* 테스트 이미지의 경우 직접 촬영한 것 사용
  * 640x480 사이즈 이미지 비율 왜곡 없도록
  * 주제 내용을 극대화 할 수 있는 촬영 영상 사용 (촬영 조건 및 해당 조건 선정 이유 모두 리포트에 기술해야 함)

** 제출물: 상세 작성 리포트 (PDF 제출)
** 채점 가산점: 프로세싱 중간 과정 및 상태를 출력하여 확인하고 수업 내용과 최대한 매핑하여 보고
** Top 3 선정 (가산점 +3점), Top 1 선정 (가산점 +5점 및 수업 시간 발표)

** 리포트에 영상이 없거나 중간 프로세싱에 대한 plot 및 영상을 통한 설명이 안 되어 있는 경우, 0점 처리
** 단순 GPT 복사로 추정(또는 그 이하 수준)될 경우, 0점 처리
```

## 환경 설정
- conda로 python 3.13버전 등록 (이름은 python313)

## 주요 요구사항 (최종 버전)

### 핵심 구현 요구사항
1. **컬러 이미지 Histogram Equalization**
   - YUV 색공간 기반 처리 (Y 채널만 처리, U/V 보존)
   - Global HE, Adaptive HE (AHE), CLAHE 구현
   - 🆕 **Bilinear Interpolation**: 격자 아티팩트 감소 기법
   - 🆕 **OpenCV 연동**: 성능 비교를 위한 OpenCV CLAHE 옵션
   - CDF 변환의 물리적 의미와 수식 도출 과정 설명

2. **Local Otsu Thresholding**
   - Inter-class variance 최대화 원리 구현
   - 블록 기반, 슬라이딩 윈도우 방법
   - 🆕 **개선된 Local Otsu**: 겹치는 블록과 가중 블렌딩으로 96.3% 아티팩트 감소
   - 텍스트 친화적 후처리 최적화

### 기술적 제약사항
- OpenCV의 built-in equalizeHist, threshold 함수 사용 금지
- 최대한 과정 자체를 코드로 구현해 히스토그램 출력 등으로 진행
- OpenCV 사용 예시는 주석으로 추가
- numpy, matplotlib 등 기본 라이브러리만 사용

### 문서화 요구사항
- 코드와 설명은 한글, 영어 둘 다 포함
- 깃허브 튜토리얼로 올릴 수 있게 파일 구조/프로젝트 구조도와 각 파일 역할까지 명확하게 소개
- 예시 이미지 테스트를 위해 Unsplash, OpenCV 샘플 등 공개 이미지 다운로드 가능하도록 안내
- README.md 예시(프로젝트 개요, 설치법, 사용 예시, 기능 나열, 예시 이미지/스크린샷 등)와 함께 제공

## 프로젝트 구조 (최종 버전)
```
assign1/
├── README.md           # 프로젝트 설명서 (최신화됨)
├── requirements.txt    # 의존성 패키지 목록
├── run_he.py          # HE 명령줄 실행 스크립트 (보간/OpenCV 옵션 추가)
├── run_otsu.py        # Local Otsu 명령줄 실행 스크립트
├── demo.py            # 종합 데모 스크립트
├── download_images.py # 테스트 이미지 다운로드
├── src/               # 소스 코드 패키지
│   ├── __init__.py    # 패키지 초기화
│   ├── he.py          # HE 구현 (보간, OpenCV 추가)
│   ├── otsu.py        # Local Otsu Thresholding 구현
│   ├── improved_local_otsu.py  # 🆕 개선된 Local Otsu
│   └── utils.py       # 공통 유틸리티 함수
├── docs/              # 문서 및 과제 요구사항
│   ├── 01_Claude_Prompt.md (이 파일)
│   ├── 02_L2_Image_Processing_1.pdf
│   ├── 03_How_to_Sensation.md
│   ├── 04_Principle_and_Implementation.md
│   ├── 05_Experimental_Design_and_Analysis.md
│   └── 06_Python_Execution_Guide.md  # 🆕 상세 실행 가이드
├── images/            # 테스트 이미지 폴더
├── results/           # 처리 결과 저장 폴더
└── tests/             # 테스트 스크립트
    ├── test_he_algorithms.py
    ├── test_boundary_improvement.py
    └── opencv_otsu_test.py
```

## 실행 방법 (최종 버전)

<<<<<<< Updated upstream
2. **명령줄 실행**:
   - HE: `python scripts/cli/run_he.py <image_path> --method yuv --show-process --save results/`
   - Otsu: `python scripts/cli/run_otsu.py <image_path> --method compare --save results/`
   - 종합 데모: `python demo.py` (모든 기능 자동 테스트)
=======
### 1. 히스토그램 평활화
```bash
# 기본 CLAHE (격자 아티팩트 있음)
python run_he.py images/image.jpg --algorithm clahe --method yuv --save results/
>>>>>>> Stashed changes

# 🆕 Bilinear Interpolation CLAHE (격자 아티팩트 감소)
python run_he.py images/image.jpg --algorithm clahe --interpolation --save results/

# 🆕 OpenCV CLAHE (최고 성능)
python run_he.py images/image.jpg --algorithm clahe --opencv --save results/

# Global HE
python run_he.py images/image.jpg --algorithm he --method yuv --save results/

# Adaptive HE
python run_he.py images/image.jpg --algorithm ahe --tile-size 16 --save results/
```

### 2. Local Otsu Thresholding
```bash
# 모든 방법 비교
python run_otsu.py images/image.jpg --method compare --save results/

# 🆕 개선된 방법 (블록 아티팩트 96.3% 감소)
python run_otsu.py images/image.jpg --method improved --save results/

# 블록 기반
python run_otsu.py images/image.jpg --method block --block-size 32 --save results/

# 슬라이딩 윈도우
python run_otsu.py images/image.jpg --method sliding --block-size 32 --stride 16 --save results/
```

### 3. 종합 데모
```bash
python demo.py  # 모든 기능 자동 테스트
```

## 핵심 구현 특징 (최종 버전)

### 🎨 히스토그램 평활화
1. **다양한 알고리즘 지원**
   - Global HE: 전역 히스토그램 평활화
   - AHE: 적응적 히스토그램 평활화
   - CLAHE: 대비 제한 적응적 평활화

2. **🆕 격자 아티팩트 해결**
   - **Bilinear Interpolation**: 타일 경계에서 보간으로 부드러운 전환
   - **OpenCV 연동**: 최적화된 성능과 품질 비교 제공

3. **색공간 처리 최적화**
   - YUV 색공간에서 Y(휘도) 채널만 처리
   - 자연스러운 색감 보존 (U, V 채널 유지)

### 🔍 Local Otsu Thresholding
1. **🆕 블록 아티팩트 해결**
   - 겹치는 블록 처리로 96.3% 아티팩트 감소
   - 가중 블렌딩을 통한 부드러운 임계값 전환

2. **다양한 지역적 방법**
   - 블록 기반: 균등 분할 후 독립 처리
   - 슬라이딩 윈도우: 중첩 윈도우로 부드러운 처리
   - 개선된 방법: 최적화된 블렌딩과 후처리

## 성능 벤치마크 결과

### 히스토그램 평활화 성능 (480×640 이미지)
| 방법 | 처리 시간 | 격자 아티팩트 | 품질 점수 | 권장 용도 |
|------|----------|---------------|-----------|-----------|
| 기본 CLAHE | 0.35초 | 명확함 | 7/10 | 알고리즘 학습 |
| 보간 CLAHE | 4.26초 | 감소됨 | 8/10 | 품질 우선 |
| OpenCV CLAHE | 0.001초 | 최소화 | 9/10 | 실용적 사용 |

### Local Otsu 개선 효과
| 지표 | 기존 블록 방법 | 개선된 방법 | 개선율 |
|------|---------------|-------------|--------|
| 블록 경계 불연속성 | 109.04 | 4.04 | 96.3% 감소 |
| 처리 영역 커버리지 | 68.2% | 91.7% | 34.4% 증가 |
| 텍스트 가독성 점수 | 6.8/10 | 8.9/10 | 30.9% 향상 |

## 이론적 배경 강화

### 1. Histogram Equalization
- **CDF 변환의 물리적 의미**: 픽셀값의 누적 분포를 균등 분포로 재배치
- **수학적 공식**: `y' = 255 × CDF(x)`
- **YUV 색공간 선택 이유**: 인간 시각 인지와 색상 보존의 최적 균형

### 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
- **Clip Limit 효과**: 2~4 범위에서 노이즈 방지와 대비 개선의 균형
- **타일 기반 처리**: 지역적 적응으로 세밀한 대비 개선
- **🆕 Bilinear Interpolation**: 타일 경계의 불연속성 해결

### 3. Otsu Thresholding
- **Inter-class variance 최대화**: `σ²(between) = w₀ × w₁ × (μ₀ - μ₁)²`
- **수학적 관계**: `σ²(total) = σ²(within) + σ²(between)`
- **🆕 개선된 블렌딩**: 겹치는 블록과 가중평균으로 경계 아티팩트 해결

## 최종 코드 구조

### 핵심 함수/클래스 명세
1. **src/he.py**
   - `histogram_equalization_color()`: 컬러 이미지 HE
   - `clahe_implementation()`: CLAHE 구현
   - `clahe_with_bilinear_interpolation()`: 🆕 보간 CLAHE
   - `clahe_opencv_implementation()`: 🆕 OpenCV CLAHE

2. **src/otsu.py**
   - `otsu_threshold()`: 기본 Otsu 임계값 계산
   - `local_otsu_block()`: 블록 기반 Local Otsu
   - `local_otsu_sliding()`: 슬라이딩 윈도우 방법

3. **src/improved_local_otsu.py**
   - `improved_local_otsu()`: 🆕 개선된 Local Otsu
   - `apply_weighted_blending()`: 가중 블렌딩 함수

## 문서화 강화

### 수업 자료 매핑
- 각 알고리즘의 수학적 공식과 물리적 의미 연결
- 수업 자료 "docs/02_L2_Image_Processing_1.pdf"의 핵심 개념들을 코드 구현과 매핑
- 이론과 실제 구현 결과의 차이점 분석

### 종합 문서 제공
- **README.md**: 프로젝트 전체 개요와 사용법
- **docs/06_Python_Execution_Guide.md**: 🆕 상세한 실행 방법과 성능 비교
- **docs/04_Principle_and_Implementation.md**: 알고리즘 원리와 구현 상세
- **docs/05_Experimental_Design_and_Analysis.md**: 실험 설계와 결과 분석

이 프롬프트를 기반으로 Top 1 수준의 비쥬얼컴퓨팅 과제를 완성해 주세요.