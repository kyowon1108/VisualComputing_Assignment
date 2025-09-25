
# Visual Computing Assignment – HE & Local Otsu (Submission)

- **Branch**: `feature/presentation-guide`
- **Commit**: `dc30be0`
- **Bundle**: `dist/submission_bundle.zip`

## 1) 과제 요구 매핑
- 직접 촬영 데이터, **640×480 규격**(비율 유지) → `docs/capture_metadata.md`
- **비교 하이라이트(ROI 확대)** → HE/OTSU contact sheet 수록
- **프로세스 플롯** → HE: Histogram/CDF, Otsu: Threshold heatmap & local histogram
- **영상**(파라미터 스윕) → MP4 + GIF (발표 시 MP4 재생, PDF엔 GIF/이미지 삽입)

## 2) 데이터 & 촬영 조건
- `docs/capture_metadata.md` 참조(ISO/셔터/조리개/WB/Focal 등)
- 모든 원본은 640×480 캔버스에 레터박스/패딩으로 맞춤

## 3) 방법 요약
- **Color HE**: RGB 대신 **Y(휘도) 채널**에 HE 적용, **CLAHE**(clipLimit, tileGrid)로 과증폭 억제
- **Local Otsu (Improved)**: 슬라이딩 윈도우 임계값 맵 + **양선형 보간**, **가중 블렌딩** + 가우시안 전처리, 모폴로지 후처리

## 4) 핵심 결과 (대표 ROI)
- **HE – Dark ROI**: 평균 밝기 0.16 → **3.59**, 에지강도 **2.3k → 8.3k** (CLAHE)
- **HE – Highlight ROI**: RGB-HE 평균 **230**로 과상향, **CLAHE 151**로 하이라이트 보존
- **Otsu – Glare ROI**: components **12 → 5**, holes **124 → 22** (Improved Local)

## 5) 권장 파라미터 (슬라이드 박스 동일)
- **HE(실내 저조)**: `--space yuv --he-mode clahe --tile 8 8 --clip 3.0 --bins 256`
- **Local Otsu(문서 글레어)**: `--method improved --window 75 --stride 32 --preblur 1.0 --morph open,3 --morph close,3`

## 6) 재현 커맨드
```bash
python run_he.py images/he_dark_indoor.jpg       --he-mode clahe --space yuv --tile 8 8 --clip 3.0       --show-plots --save results/he/

python run_otsu.py images/otsu_shadow_doc_02.jpg       --method improved --window 75 --stride 32 --preblur 1.0       --morph open,3 --morph close,3 --show-plots --save results/otsu/
```

## 7) 제출물(핵심 아티팩트)
- 슬라이드 요약: `results/slides/he_summary.png`, `results/slides/otsu_summary.png`
- 비교 컨택트시트: `results/he/compare_he_contact_sheet.png`, `results/otsu/compare_otsu_contact_sheet.png`
- 프로세스/지표: `results/he_metrics/`, `results/otsu_metrics/`, `results/ablation/*`
- 영상: `results/video/he_sweep.mp4`, `results/video/otsu_sweep.mp4`
- 리포트: `docs/final_report.pdf`

## 8) 한계 & 주의
- CLAHE clip이 크면 과증폭/헤일로 가능 → `clip≈2.5–3.0` 권장
- Local Otsu stride가 너무 작으면 속도↓, 너무 크면 블록 경계↑ → `24–32` 권장

## Bundle file list (top-level)

- `dist/submission_bundle.zip` (26 entries)
 - docs/
 - results/
