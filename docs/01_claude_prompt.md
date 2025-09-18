비쥬얼컴퓨팅 과제 목적의 Python 프로젝트를 생성하고 싶습니다.
과제를 제출한 사람들 중 Top 1이 될만한 학부생의 구조로 작성해 주세요.
아래 요구사항을 만족하는 코드를 단계별로 작성해 주세요. assign1 폴더에 전체적으로 구현을 해주셔야 합니다.

### 요구사항 원본
```
============== 과제1
주제1: 컬러 이미지 HE
주제2: Local Otsu Thresholding
동작 코드 구현 '확인' 및 중간 과정 리포트 (Presentation Slide 형식)
 제출물: 상세 작성 리포트 (PDF 제출)
 채점 가산점: 프로세싱 중간 과정 및 상태를 출력하여 확인하고 수업 내용과 최대한 매핑하여 보고
 Top 3 선정 (가산점 +3점), Top 1 선정 (가산점 +5점 및 수업 시간 발표)
 Due Date: ~9/26
==================================
```

### 환경
- conda로 python 3.13버전 등록 (이름은 python313)

### 주요 요구사항(요약)
- 주제1: 컬러 이미지 HE
- 주제2: Local Otsu Thresholding
- 동작 코드 구현 '확인' 및 중간 과정 리포트 (Presentation Slide 형식),
- 수업 자료는 "assign1/docs/02_L2_Image_Processing_1.pdf"에 위치함.

### 요구
① 컬러 이미지용 직관적 low-level Histogram Equalization  
② 직접 구현 Local Otsu Thresholding (블록별/슬라이딩 윈도우 기반, numpy 등만 사용)
- OpenCV의 built-in equalizeHist, threshold 함수 사용 금지 (최대한 과정 자체를 코드로 구현해 히스토그램 출력 등으로 진행하는 것이 좋음. 다만, opencv를 사용했을 때 어떤 명령어를 쓰면 되는지를 주석으로 추가 요망.)
- 코드와 설명은 한글, 영어 둘 다 포함
- 깃허브 튜토리얼로 올릴 수 있게 파일 구조/프로젝트 구조도와 각 파일 역할까지 명확하게 소개
- 예시 이미지 테스트를 위해 Unsplash, OpenCV 샘플 등 공개 이미지를 다운로드해 사용할 수 있도록 안내
- README.md 예시(프로젝트 개요, 설치법, 사용 예시, 기능 나열, 예시 이미지/스크린샷 등)와 함께 제공

### 프로젝트 구조 (업데이트됨)
```
assign1/
├── README.md           # 프로젝트 설명서
├── requirements.txt    # 의존성 패키지 목록
├── run_gui.py         # GUI 실행 스크립트 (간소화된 GUI)
├── run_he.py          # HE 명령줄 실행 스크립트
├── run_otsu.py        # Local Otsu 명령줄 실행 스크립트
├── demo.py            # 종합 데모 스크립트
├── download_images.py # 테스트 이미지 다운로드
├── src/               # 소스 코드 패키지
│   ├── __init__.py    # 패키지 초기화
│   ├── main.py        # 간소화된 GUI 애플리케이션
│   ├── he.py          # Histogram Equalization 구현
│   ├── otsu.py        # Local Otsu Thresholding 구현
│   └── utils.py       # 공통 유틸리티 함수
├── docs/              # 문서 및 과제 요구사항
├── images/            # 테스트 이미지 폴더
└── results/           # 처리 결과 저장 폴더
```

### 실행 방법 (업데이트됨)
1. **간소화된 GUI 실행**: `python run_gui.py`
   - 이미지 로드, HE/Otsu 처리, 결과 확인/저장 가능
   - 복잡한 GUI 대신 핵심 기능에 집중

2. **명령줄 실행**:
   - HE: `python run_he.py <image_path> --method yuv --show-process --save results/`
   - Otsu: `python run_otsu.py <image_path> --method compare --save results/`
   - 종합 데모: `python demo.py` (모든 기능 자동 테스트)

3. **코드 구조**:
   - 각 모듈의 역할/메인 함수 이름 간단 명세 추가
   - 코드 각 부분(HE, Otsu)에 '동작 원리와 수식', '중간 상태 시각화', '실행 예시' 설명 주석 포함
   - src/ 패키지 구조로 모듈화된 설계
   - README.md에는 과제 목표, 핵심 구현내용, 실행 방법, 주요 스크린샷, 테스트 이미지 출처 안내까지 포함

위와 같은 구조와 요구를 바탕으로--  
(1) 전체 코드를 파일별로 나누어 작성해주고,
(2) 각 핵심 함수/클래스/메인 루프에 한글 주석을 넣어주며,
(3) 최종적으로 깃허브 튜토리얼 예시(파일 구조표, 설치 방법, 실행법, 주요 스크린샷 위치 등)까지 안내해 주세요.
답변은 코드(파일별 구분), README 예시, 한글 구현 설명/알고리즘 흐름, 등으로 세션을 분리해 작성해 주세요.

---

[이론적 배경 강화]
- Otsu 방법의 inter-class variance 최대화와 within-class variance 최소화의 수학적 관계를 코드 주석에 상세히 설명
- 히스토그램 평활화에서 CDF 변환의 물리적 의미와 "y' = Scale * CDF(x)" 공식의 도출 과정 포함  
- YUV 색공간에서 휘도(Y) 채널만 처리하는 이론적 근거 설명
- CLAHE의 clip limit 설정 기준(2~4)과 그 효과에 대한 설명

[문서화 강화]
- 각 알고리즘의 수학적 공식과 물리적 의미 연결
- 수업 자료의 핵심 개념들을 코드 구현과 매핑하여 설명
- 이론과 실제 구현 결과의 차이점 분석