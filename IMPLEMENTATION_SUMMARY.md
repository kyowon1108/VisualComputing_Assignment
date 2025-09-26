# 비쥬얼컴퓨팅 과제1 - 구현 코드 위치 요약

## 1. HE 비교강조 및 과정 시각화

### 📍 위치: `tests/test_he_step_visualization.py`
**라인 34-90**: `test_he_step_visualization()` 함수
```python
def test_he_step_visualization(image_path: str, save_figure: bool = True):
```

**구현 내용**:
- **4단계 과정**: 원본 RGB → Y채널 추출 → HE 적용 → 최종 RGB 결과
- **CDF 계산 및 시각화**: `calculate_cdf()`, `histogram_equalization_grayscale()` 사용
- **히스토그램 비교**: 원본 vs 평활화된 히스토그램을 bar chart로 표시
- **3x4 서브플롯** 구성으로 단계별 시각화

### 📍 추가 위치: `scripts/video_otsu_exact_pipeline.py`
**라인 124-180**: HE 과정을 애니메이션으로 시각화

---

## 2. DeltaE, Diff, SSIM 비교

### 📍 위치: `scripts/make_metrics.py`
**라인 45-120**: HE 메트릭 계산 함수들
```python
def calculate_delta_e_lab(img1, img2):  # 라인 45
def calculate_ssim_metrics(img1, img2): # 라인 78
def create_diff_visualization(img1, img2, output_path): # 라인 95
```

**구현 내용**:
- **DeltaE 계산**: RGB → LAB 변환 후 유클리드 거리 계산
- **SSIM 계산**: `skimage.metrics.structural_similarity` 사용
- **Diff 시각화**: 차이 이미지를 컬러맵으로 시각화
- **메트릭 집계**: 평균, 표준편차, 분포 히스토그램 생성

### 📍 결과 저장: `results/he_metrics_fixed/`
- `deltaE_*.png`: DeltaE 분석 이미지들
- `ssim_*.png`: SSIM 분석 이미지들
- `diff_*.png`: 차이 시각화 이미지들
- `he_metrics_stats.csv`: 정량적 메트릭 데이터

---

## 3. CLAHE 종합 분석

### 📍 위치: `src/he.py`
**라인 615-680**: `histogram_equalization_color()` 함수 내 CLAHE 구현
```python
elif algorithm == 'clahe':
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    if method == 'yuv':
        yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
```

**구현 내용**:
- **OpenCV CLAHE** 사용: `cv2.createCLAHE()`
- **파라미터 조정**: `clip_limit` (2.0), `tile_size` (8x8)
- **YUV 색공간** 적용: Y채널에만 CLAHE 적용하여 색상 보존
- **격자 아티팩트 방지**: OpenCV 내장 보간법 활용

### 📍 위치: `scripts/cli/run_he.py`
**라인 60-80**: CLAHE 명령행 실행 인터페이스

---

## 4. Improved Local Otsu 설계 전체 과정

### 📍 위치: `src/otsu.py`
**라인 1313-1355**: `improved_otsu()` 메인 함수
```python
def improved_otsu(image, window_size=75, stride=24, preblur=1.0, morph_ops=['open,3', 'close,3']):
```

**파이프라인 구현**:

#### 4.1 전처리: **라인 1282-1286** `apply_preprocessing()`
```python
def apply_preprocessing(image, preblur=1.0):
    if preblur > 0:
        return cv2.GaussianBlur(image, (0, 0), preblur)
```

#### 4.2 슬라이딩 윈도우: **라인 1222-1280** `sliding_window_otsu()`
```python
def sliding_window_otsu(image, window_size=75, stride=24):
    # 그리드 생성: window_size//2에서 시작, stride 간격
    # scipy.interpolate.RectBivariateSpline로 양선형 보간
```

#### 4.3 임계값 계산: **라인 168-220** `compute_otsu_threshold()`
```python
def compute_otsu_threshold(histogram):
    # Inter-class variance 최대화
    # between_variance = w0 * w1 * (mean0 - mean1) ** 2
```

#### 4.4 후처리: **라인 1288-1311** `apply_morphological_operations()`
```python
def apply_morphological_operations(binary_image, operations):
    # cv2.morphologyEx() 사용: MORPH_OPEN, MORPH_CLOSE
    # 3x3 타원형 커널, iterations 조정 가능
```

### 📍 애니메이션: `scripts/video_otsu_exact_pipeline.py`
**전체 파일**: 5단계 과정을 프레임별로 시각화

---

## 5. Global Otsu vs Improved Otsu ROI 비교

### 📍 위치: `scripts/create_otsu_roi_comparison.py`
**라인 15-90**: `create_otsu_roi_comparison()` 함수

**구현 내용**:
- **ROI 정의**: 3개 영역 `[(448,48,160,144), (64,144,256,192), (32,24,128,384)]`
- **2x3 서브플롯**: 상단(전체 이미지), 하단(ROI별 상세 비교)
- **색상 코딩**: ROI 1(빨강), ROI 2(녹색), ROI 3(파랑)
- **레이아웃**: 원본 | Global | Improved 나란히 배치

### 📍 CLI 실행: `scripts/cli/run_otsu.py`
**라인 80-90**: 기본 ROI 설정 및 분석

---

## 6. Glare ROI 히스토그램 분석

### 📍 위치: `otsu_analysis_final.py`
**라인 15-35**: `find_glare_roi()` 함수
```python
def find_glare_roi(image, percentile=95):
    high_intensity = image > np.percentile(image, percentile)
    # 연결 성분 분석으로 glare 영역 검출
```

**라인 40-85**: ROI별 히스토그램 생성
```python
def analyze_roi_histograms(original, global_result, improved_result, glare_roi):
    # 각 ROI에서 히스토그램 계산 및 시각화
    # matplotlib.pyplot.hist() 사용
```

**구현 내용**:
- **Glare 검출**: 상위 5% 밝기 픽셀 기준
- **연결 성분**: `cv2.connectedComponentsWithStats()` 사용
- **히스토그램 비교**: 원본/Global/Improved 3가지 방법 비교
- **ROI 오버레이**: glare 영역을 빨간 사각형으로 표시

---

## 7. XOR로 Global과 Improved Otsu 비교

### 📍 위치: `create_otsu_metrics.py`
**라인 37-52**: `create_xor_map()` 함수
```python
def create_xor_map(global_result, improved_result):
    global_binary = (global_result > 127).astype(np.uint8)
    improved_binary = (improved_result > 127).astype(np.uint8)
    xor_map = cv2.bitwise_xor(global_binary, improved_binary)
    disagreement_ratio = (disagreement_pixels / total_pixels) * 100
```

### 📍 시각화: `scripts/make_metrics.py`
**라인 316-335**: `create_xor_map()` 시각화 버전
```python
def create_xor_map(global_img, improved_img, output_path):
    xor_map = np.bitwise_xor(global_bin, improved_bin) * 255
    plt.imshow(xor_map, cmap='Reds')  # 빨간색 컬러맵으로 차이 강조
```

**구현 내용**:
- **이진화**: 127 임계값으로 0/1 변환
- **XOR 연산**: `cv2.bitwise_xor()` 사용
- **불일치 비율**: 전체 픽셀 대비 차이 픽셀 비율 계산
- **컬러맵**: 'Reds' 사용하여 차이점을 빨간색으로 강조

### 📍 애니메이션 전환: `scripts/create_otsu_transition_gif.py`
**전체 파일**: Global → Improved 부드러운 전환 + XOR 차이 표시

---

## 실행 명령어 요약

```bash
# 1. HE 과정 시각화
python tests/test_he_step_visualization.py images/he_dark_indoor.jpg

# 2. HE 메트릭 생성
python scripts/make_metrics.py he --force

# 3. CLAHE 실행
python scripts/cli/run_he.py images/image.jpg --algorithm clahe --method yuv

# 4. Improved Otsu 파이프라인 애니메이션
PYTHONPATH=. python scripts/video_otsu_exact_pipeline.py --src images/otsu_shadow_doc_02.jpg

# 5. ROI 비교 이미지 생성
PYTHONPATH=. python scripts/create_otsu_roi_comparison.py

# 6. Glare ROI 분석
python otsu_analysis_final.py

# 7. Global→Improved 전환 GIF
PYTHONPATH=. python scripts/create_otsu_transition_gif.py
```

## 주요 결과 파일 위치

- **HE 시각화**: `results/`*_he_4steps_analysis.png`
- **메트릭 분석**: `results/he_metrics_fixed/`
- **Otsu 비교**: `results/video/otsu_roi_comparison.png`
- **XOR 맵**: `results/video/otsu_xor_comparison.png`
- **애니메이션**: `results/video/*.gif`, `results/video/*.mp4`
- **히스토그램**: `results/glare_roi_histogram_analysis.png`