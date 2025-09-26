# Visual Computing Assignment - Image Enhancement

## Overview
This submission contains implementations and analysis of histogram equalization (HE) and Otsu thresholding methods for image enhancement.

## Reproduction Instructions

### Histogram Equalization
```bash
# RGB-HE
python run_he.py images/he_dark_indoor.jpg --space rgb --he-mode global --save results/he/

# Y-HE (luminance only)
python run_he.py images/he_dark_indoor.jpg --space yuv --he-mode global --save results/he/

# CLAHE (recommended)
python run_he.py images/he_dark_indoor.jpg --space yuv --he-mode clahe --tile 8 8 --clip 2.5 --save results/he/
```

### Otsu Thresholding
```bash
# Global Otsu
python run_otsu.py images/otsu_shadow_doc_02.jpg --method global --save results/otsu/

# Improved Local Otsu (recommended)
python run_otsu.py images/otsu_shadow_doc_02.jpg --method improved --window 75 --stride 24 --preblur 1.0 --save results/otsu/
```

### Analysis Scripts
```bash
# Parameter ablation study
python scripts/ablation.py

# Generate summary slides
python scripts/make_slide_figs.py

# Create final report
python scripts/make_pdf.py
```

## Key Artifacts

### Images and Results
- `images/he_dark_indoor.jpg` - Test image for HE methods
- `images/otsu_shadow_doc_02.jpg` - Test image for Otsu methods
- `results/he/` - HE processing results
- `results/otsu/` - Otsu processing results

### Quality Metrics
- `results/he_metrics/` - HE quality assessment (SSIM, Delta E, difference maps)
- `results/otsu_metrics/` - Otsu quality assessment (XOR maps, component analysis)

### Visualizations
- `results/slides/he_summary.png` - HE methods comparison
- `results/slides/otsu_summary.png` - Otsu methods comparison
- `results/video/he_sweep.mp4|gif` - HE parameter sweep animation
- `results/video/otsu_sweep.mp4|gif` - Otsu parameter sweep animation

### Analysis Data
- `results/ablation/ablation_he.csv` - HE parameter study results
- `results/ablation/ablation_otsu.csv` - Otsu parameter study results
- `results/ablation/*_top3.json` - Best parameter combinations

### Final Report
- `docs/final_report.pdf` - Complete analysis report with figures

## Key Findings

### Histogram Equalization
- **CLAHE (YUV space)** provides best balance of enhancement and color preservation
- **Optimal parameters**: tile=8x8, clip=2.5 for indoor low-light scenes
- **Y-channel processing** avoids color distortion while improving contrast

### Otsu Thresholding
- **Local Otsu** significantly outperforms global method for uneven illumination
- **Improved method** with pre-blur and morphological post-processing reduces noise
- **Optimal parameters**: window=75, stride=24, preblur=1.0 for document images

## Dependencies
- OpenCV >= 4.10
- NumPy >= 2.0
- scikit-image >= 0.25
- matplotlib >= 3.5
- ReportLab >= 4.0
- imageio (for video generation)
