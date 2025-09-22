# 컬러 이미지 히스토그램 평활화와 Local Otsu Thresholding: 수학적 원리와 구현

## 목차
1. [컬러 이미지 히스토그램 평활화](#컬러-이미지-히스토그램-평활화)
2. [Otsu Thresholding 수학적 원리](#otsu-thresholding-수학적-원리)
3. [Local Otsu Thresholding](#local-otsu-thresholding)
4. [구현 코드 분석](#구현-코드-분석)
5. [OpenCV를 이용한 구현 방법](#opencv를-이용한-구현-방법)

---

## 컬러 이미지 히스토그램 평활화

### 수학적 원리

히스토그램 평활화(Histogram Equalization)는 이미지의 히스토그램을 균등분포에 가깝게 변환하여 대비(contrast)를 개선하는 기법입니다.

#### 기본 수학적 공식

1. **히스토그램 계산**
   ```
   h(i) = 픽셀값 i의 빈도수
   ```

2. **누적분포함수(CDF) 계산**
   ```
   CDF(i) = Σ(k=0 to i) h(k) / N
   ```
   여기서 N은 총 픽셀 수

3. **변환 공식**
   ```
   y' = Scale × CDF(x)
   ```
   여기서 Scale = 255 (8비트 이미지의 경우)

#### 물리적 의미

CDF는 특정 픽셀값 이하의 픽셀들이 전체에서 차지하는 비율을 나타내며, 이를 통해 어두운 영역과 밝은 영역을 전체 강도 범위에 고르게 분배합니다.

### 컬러 이미지에서의 문제점

RGB 컬러 이미지에서 각 채널(R, G, B)을 개별적으로 히스토그램 평활화하면 다음과 같은 문제가 발생합니다:

- **색상 왜곡**: 각 채널의 상대적 분포가 변경되어 부자연스러운 색상 변화
- **색감 손실**: 원본 이미지의 색조(hue)와 채도(saturation) 정보 손실

### YUV 색공간을 이용한 해결방법

#### YUV 색공간의 특징

- **Y 채널**: 휘도(Luminance) 정보, 인간의 시각 인지와 밀접한 관련
- **U, V 채널**: 색차(Chrominance) 정보, 색상과 채도 정보 포함

#### 처리 과정

1. RGB → YUV 색공간 변환
2. Y 채널에만 히스토그램 평활화 적용
3. YUV → RGB 색공간 역변환

이 방법의 장점:
- 색상 정보 보존
- 자연스러운 밝기 개선
- 인간의 시각 특성과 일치

### 본 구현에서의 코드 매핑

```python
# src/he.py의 histogram_equalization_color 함수 (라인 110-179)
def histogram_equalization_color(image: np.ndarray, method: str = 'yuv', show_process: bool = True):
    if method == 'yuv':
        # YUV 색공간으로 변환 (라인 140)
        yuv_image = rgb_to_yuv(image)

        # Y 채널에만 평활화 적용 (라인 143-144)
        y_channel = yuv_image[:, :, 0]
        y_equalized, process_info = histogram_equalization_grayscale(y_channel, show_process=False)

        # RGB로 역변환 (라인 151)
        rgb_equalized = yuv_to_rgb(yuv_equalized)
```

### CLAHE (Contrast Limited Adaptive Histogram Equalization)

#### 핵심 원리

1. **타일 분할**: 이미지를 작은 타일로 분할
2. **히스토그램 클리핑**: 과도한 증폭 방지
3. **로컬 평활화**: 각 타일에서 독립적 처리
4. **보간**: 타일 경계에서 부드러운 전환

#### 클리핑 공식

```
클립 임계값 = (총 픽셀 수 / 256) × clip_limit
```

#### 본 구현에서의 코드 매핑

```python
# src/he.py의 clahe_implementation 함수 (라인 181-284)
def clahe_implementation(image: np.ndarray, clip_limit: float = 2.0, tile_size: Tuple[int, int] = (8, 8)):
    # 히스토그램 클리핑 (라인 245)
    clipped_hist = clip_histogram(hist, clip_limit, tile_h * tile_w)

    # 클리핑된 히스토그램으로 CDF 계산 (라인 248)
    cdf = calculate_cdf(clipped_hist)
```

---

## Otsu Thresholding 수학적 원리

### 기본 개념

Otsu 방법은 이미지를 전경(foreground)과 배경(background) 두 클래스로 분할하는 최적의 임계값을 자동으로 찾는 알고리즘입니다.

### 수학적 공식

#### Within-class Variance (클래스 내 분산)

```
σ²w(t) = ω₀(t) × σ²₀(t) + ω₁(t) × σ²₁(t)
```

여기서:
- ω₀(t), ω₁(t): 각 클래스의 확률 (픽셀 비율)
- σ²₀(t), σ²₁(t): 각 클래스의 분산

#### Between-class Variance (클래스 간 분산)

```
σ²b(t) = ω₀(t) × ω₁(t) × (μ₀(t) - μ₁(t))²
```

여기서:
- μ₀(t), μ₁(t): 각 클래스의 평균값

#### 핵심 관계식

```
σ²total = σ²w(t) + σ²b(t)
```

총 분산은 임계값에 무관하므로, **클래스 간 분산 최대화**는 **클래스 내 분산 최소화**와 동일합니다.

### 최적화 목표

Otsu 방법은 다음을 최대화하는 임계값 t*를 찾습니다:

```
t* = argmax(σ²b(t))
```

### 본 구현에서의 코드 매핑

```python
# src/otsu.py의 calculate_otsu_threshold 함수 (라인 21-113)
def calculate_otsu_threshold(histogram: np.ndarray, show_process: bool = False):
    for threshold in range(256):
        # 클래스 확률 계산 (라인 66, 72)
        w0 = np.sum(histogram[:threshold + 1]) / total_pixels
        w1 = np.sum(histogram[threshold + 1:]) / total_pixels

        # 클래스 평균 계산 (라인 79, 84)
        mean0 = np.sum(pixel_values[:threshold + 1] * histogram[:threshold + 1]) / np.sum(histogram[:threshold + 1])
        mean1 = np.sum(pixel_values[threshold + 1:] * histogram[threshold + 1:]) / np.sum(histogram[threshold + 1:])

        # Inter-class variance 계산 (라인 90)
        inter_class_variance = w0 * w1 * (mean0 - mean1) ** 2
```

---

## Local Otsu Thresholding

### 동기

전역 Otsu 방법의 한계:
- 불균등한 조명 조건에서 성능 저하
- 지역적 특성을 반영하지 못함
- 복잡한 배경에서 부정확한 분할

### Block-based Local Otsu

#### 원리

1. 이미지를 고정 크기 블록으로 분할
2. 각 블록마다 독립적으로 Otsu 임계값 계산
3. 해당 블록에 임계값 적용

#### 장단점

**장점:**
- 계산 효율성 높음
- 지역적 조명 변화에 적응
- 구현이 단순함

**단점:**
- 블록 경계에서 불연속성 발생 가능
- 작은 블록에서 부정확한 임계값 계산 가능

#### 본 구현에서의 코드 매핑

```python
# src/otsu.py의 local_otsu_block_based 함수 (라인 171-260)
def local_otsu_block_based(image: np.ndarray, block_size: Tuple[int, int] = (32, 32)):
    for i in range(0, height, block_h):
        for j in range(0, width, block_w):
            # 블록 추출 (라인 218)
            block = image[i:end_i, j:end_j]

            # 블록별 Otsu 임계값 계산 (라인 225)
            block_threshold, block_calc_info = calculate_otsu_threshold(block_hist, show_process=False)

            # 임계값 적용 (라인 231-232)
            block_binary = apply_threshold(block, block_threshold)
            binary_image[i:end_i, j:end_j] = block_binary
```

### Sliding Window Local Otsu

#### 원리

1. 지정된 stride로 윈도우를 이동
2. 윈도우 영역에서 Otsu 임계값 계산
3. 중앙 픽셀에 해당 임계값 적용

#### 장단점

**장점:**
- 부드러운 임계값 전환
- 윈도우 겹침으로 인한 연속성
- 더 정확한 지역적 적응

**단점:**
- 높은 계산 복잡도
- 메모리 사용량 증가
- 처리 시간 증가

#### 본 구현에서의 코드 매핑

```python
# src/otsu.py의 local_otsu_sliding_window 함수 (라인 262-377)
def local_otsu_sliding_window(image: np.ndarray, window_size: Tuple[int, int] = (32, 32), stride: int = 8):
    for i in range(half_h, height - half_h, stride):
        for j in range(half_w, width - half_w, stride):
            # 윈도우 영역 정의 (라인 309-312)
            start_i = max(0, i - half_h)
            end_i = min(height, i + half_h + 1)

            # 윈도우 임계값 계산 (라인 322)
            window_threshold, window_calc_info = calculate_otsu_threshold(window_hist, show_process=False)

            # 중앙 영역에 적용 (라인 329-340)
            center_binary = apply_threshold(center_region, window_threshold)
            binary_image[center_start_i:center_end_i, center_start_j:center_end_j] = center_binary
```

---

## 구현 코드 분석

### 핵심 함수들

#### 1. 히스토그램 계산

```python
# src/utils.py
def compute_histogram(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # 256개 bin을 가진 히스토그램 계산
    histogram = np.bincount(image.flatten(), minlength=256)
    bin_edges = np.arange(257)
    return histogram, bin_edges
```

#### 2. CDF 계산

```python
# src/he.py (라인 20-49)
def calculate_cdf(histogram: np.ndarray) -> np.ndarray:
    # 누적 합계 계산
    cdf = np.cumsum(histogram)
    # 정규화 (0-1 범위로)
    cdf_normalized = cdf / cdf[-1]
    return cdf_normalized
```

#### 3. 색공간 변환

```python
# src/utils.py
def rgb_to_yuv(rgb_image: np.ndarray) -> np.ndarray:
    # RGB to YUV 변환 행렬 적용
    # Y = 0.299*R + 0.587*G + 0.114*B
    # U = -0.14713*R - 0.28886*G + 0.436*B
    # V = 0.615*R - 0.51499*G - 0.10001*B
```

#### 4. 입력 이미지 처리 전략

```python
# run_he.py - 컬러 이미지 직접 처리
def main():
    image = load_image(args.image_path)  # RGB 컬러 이미지 유지
    if args.method == 'yuv':
        result, info = histogram_equalization_color(image, method='yuv')
    # 컬러 정보를 보존하여 자연스러운 결과
```

```python
# run_otsu.py - 그레이스케일 변환 후 처리
def main():
    image = load_image(args.image_path)
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 필수 변환
    result, info = global_otsu_thresholding(gray_image)
    # 이진화는 본질적으로 흑백 결과
```

### 처리 방식의 핵심 차이점

#### 히스토그램 평활화 vs Otsu Thresholding

| 특성 | 히스토그램 평활화 | Otsu Thresholding |
|------|------------------|-------------------|
| **입력 처리** | RGB 컬러 이미지 직접 처리 | RGB → 그레이스케일 변환 필수 |
| **색공간 전략** | YUV 변환으로 색상 보존 | 그레이스케일 변환으로 단순화 |
| **출력 결과** | 개선된 컬러 이미지 | 이진 이미지 (흑백) |
| **색상 정보** | 보존됨 (U, V 채널 유지) | 손실됨 (이진화 특성상 불필요) |

#### 구현상 고려사항

```python
# 히스토그램 평활화에서 색상 보존 전략
def histogram_equalization_color(image, method='yuv'):
    if method == 'yuv':
        yuv_image = rgb_to_yuv(image)
        # Y 채널만 처리, U/V 채널은 보존
        y_equalized = histogram_equalization_grayscale(yuv_image[:,:,0])
        yuv_image[:,:,0] = y_equalized
        return yuv_to_rgb(yuv_image)  # 컬러 복원
```

```python
# Otsu에서 그레이스케일 변환이 필수인 이유
def global_otsu_thresholding(image):
    if len(image.shape) != 2:
        raise ValueError("그레이스케일 이미지가 필요합니다")
    # 이진화 알고리즘은 단일 채널에서만 의미가 있음
    threshold = calculate_otsu_threshold(histogram)
    return apply_threshold(image, threshold)  # 0 또는 255 값만 출력
```

### 실무적 함의

**왜 이런 차이가 중요한가?**

1. **알고리즘의 본질적 특성**
   - HE: 대비 개선 (컬러 정보 유지 필요)
   - Otsu: 객체 분할 (형태 정보만 필요)

2. **처리 효율성**
   - HE: 3채널 → YUV → 1채널 처리 → 3채널 복원
   - Otsu: 3채널 → 1채널 변환 → 1채널 이진화

3. **사용자 기대와 일치**
   - HE 사용자: "더 밝고 선명한 컬러 사진"을 기대
   - Otsu 사용자: "객체와 배경이 분리된 흑백 이미지"를 기대

### 시각화 기능

모든 주요 함수들은 중간 과정을 시각화하는 기능을 포함:

- `visualize_he_process`: 히스토그램 평활화 과정 시각화
- `visualize_color_he_process`: 컬러 이미지 처리 과정 시각화
- `visualize_otsu_calculation`: Otsu 계산 과정 시각화
- `visualize_local_otsu_process`: Local Otsu 과정 시각화

---

## OpenCV를 이용한 구현 방법

### 그레이스케일 히스토그램 평활화

```python
import cv2

# 기본 히스토그램 평활화
gray_image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
equalized = cv2.equalizeHist(gray_image)
```

### 컬러 이미지 히스토그램 평활화

#### YUV 색공간 사용

```python
import cv2

# BGR to YUV 변환
color_image = cv2.imread('image.jpg')
yuv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2YUV)

# Y 채널에만 히스토그램 평활화 적용
yuv_image[:,:,0] = cv2.equalizeHist(yuv_image[:,:,0])

# BGR로 역변환
result = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)
```

#### HSV 색공간 사용

```python
import cv2

# BGR to HSV 변환
color_image = cv2.imread('image.jpg')
hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

# V 채널에만 히스토그램 평활화 적용
hsv_image[:,:,2] = cv2.equalizeHist(hsv_image[:,:,2])

# BGR로 역변환
result = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
```

### CLAHE 구현

```python
import cv2

# CLAHE 객체 생성
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

# 그레이스케일 이미지에 적용
gray_image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
clahe_result = clahe.apply(gray_image)

# 컬러 이미지에 적용 (YUV 색공간 사용)
color_image = cv2.imread('image.jpg')
yuv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2YUV)
yuv_image[:,:,0] = clahe.apply(yuv_image[:,:,0])
result = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)
```

### Global Otsu Thresholding

```python
import cv2

# 그레이스케일 이미지 읽기
gray_image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Otsu 방법으로 임계값 자동 계산
ret, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

print(f"Otsu threshold: {ret}")
```

### Adaptive Thresholding

OpenCV에서는 Local Otsu 대신 Adaptive Thresholding을 제공합니다:

```python
import cv2

gray_image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Mean-based adaptive thresholding
thresh_mean = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

# Gaussian-based adaptive thresholding
thresh_gaussian = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
```

### 매개변수 설명

#### CLAHE 매개변수
- **clipLimit**: 클리핑 한계값 (일반적으로 2.0-4.0)
- **tileGridSize**: 타일 그리드 크기 (기본값: 8x8)

#### Adaptive Thresholding 매개변수
- **maxValue**: 임계값 조건을 만족하는 픽셀에 할당할 값
- **adaptiveMethod**: 임계값 계산 방법
  - `ADAPTIVE_THRESH_MEAN_C`: 주변 영역의 평균
  - `ADAPTIVE_THRESH_GAUSSIAN_C`: 가우시안 가중 평균
- **blockSize**: 임계값 계산에 사용할 영역 크기
- **C**: 평균에서 차감할 상수
