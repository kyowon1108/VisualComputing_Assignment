# Visual Computing Assignment 1 - Final Submission

## Corrected Histogram Equalization Analysis

## Path Changes

| Previous Path | New Path |
|---------------|----------|
| run_he.py | scripts/cli/run_he.py |
| run_otsu.py | scripts/cli/run_otsu.py |



The following metrics show the corrected distinction between Y-HE and CLAHE methods:

| Method | deltaE_mean | SSIM  | Interpretation             |
|--------|-------------|-------|----------------------------|
| RGB-HE | 34.72       | 0.265 | High color distortion      |
| Y-HE   | 34.36       | 0.212 | Better chroma preservation |
| CLAHE  | 6.61        | 0.605 | Most perceptually similar  |

## Otsu Thresholding Improvements

The Improved Otsu method shows significant improvements over Global Otsu:
- Components: −91.5% reduction (cleaner segmentation)
- Average area: ×26.6 increase (larger meaningful regions)
- Holes: −43.9% reduction (more solid objects)

## Reproduction Commands

```bash
# Histogram Equalization
python scripts/cli/run_he.py images/he_dark_indoor.jpg --he-mode he --space yuv --save results/he/
python scripts/cli/run_he.py images/he_dark_indoor.jpg --he-mode clahe --space yuv --save results/he/
python scripts/cli/run_he.py images/he_dark_indoor.jpg --he-mode he --space rgb --save results/he/

# Otsu Thresholding
python scripts/cli/run_otsu.py images/otsu_sample_text.jpg --method global --save results/otsu/
python scripts/cli/run_otsu.py images/otsu_sample_text.jpg --method improved --save results/otsu/

# Generate Metrics
python scripts/make_metrics.py he --force
python scripts/make_metrics.py otsu --force
```

## Dependencies
- OpenCV 4.x
- NumPy 1.24+
- scikit-image 0.21+
- matplotlib 3.7+
- ReportLab 4.0+
