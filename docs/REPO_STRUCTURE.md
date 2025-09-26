# Repository Structure

This document describes the organization of the Visual Computing Assignment 1 repository.

## Directory Tree

```
.
├── images/                         # Input images (640x480 samples)
├── src/                            # Core algorithm implementations
│   ├── __init__.py                # Package marker
│   ├── he.py                      # Histogram equalization algorithms
│   ├── otsu.py                    # Otsu thresholding methods
│   └── utils.py                   # Shared utilities and color space conversions
├── scripts/                       # Build and utility scripts
│   ├── cli/                       # Command-line interfaces (moved from root)
│   │   ├── run_he.py             # HE tool (global/AHE/CLAHE; RGB/YUV/LAB/HSV)
│   │   └── run_otsu.py           # Otsu tool (global/block/sliding/improved)
│   ├── make_metrics.py           # Quality metrics generator (SSIM/ΔE/diff)
│   ├── make_videos.py            # Video/GIF creators and timelapses
│   ├── make_slide_figs.py        # Summary slide builder (PNG panels)
│   └── make_pdf.py               # Report PDF generator (ReportLab)
├── results/                      # Generated outputs and analysis
│   ├── he/                       # HE processed images (yuv_he/yuv_clahe/rgb_global)
│   ├── otsu/                     # Otsu binary outputs (global/improved)
│   ├── he_metrics_fixed/         # **Canonical HE quality metrics** (corrected Y-HE vs CLAHE)
│   ├── otsu_metrics/             # **Canonical Otsu analysis metrics** (compare/xor/table)
│   ├── slides/                   # Summary slide PNGs (he_summary/otsu_summary)
│   └── video/                    # MP4/GIF animations (sweep visualizations)
├── docs/                         # Documentation and reports
│   ├── final_report.pdf          # Generated comprehensive report
│   └── REPO_STRUCTURE.md         # This file
├── dist/                         # Distribution artifacts
│   ├── README_submission.md      # Submission documentation with metrics tables
│   └── submission_bundle_final.zip # **Final submission package**
└── archive/                      # Legacy/backup files (cleanup destination)
```

## Canonical Metric Directories

- **HE Metrics**: `results/he_metrics_fixed/` - Contains corrected Y-HE vs CLAHE distinction
  - Includes: diff/ssim/deltaE maps, collages, and `he_metrics_stats.csv`
- **Otsu Metrics**: `results/otsu_metrics/` - Contains comparative analysis artifacts
  - Includes: compare_panel, xor_map, metrics table, and statistics

## Final Artifacts

- **Final ZIP**: `dist/submission_bundle_final.zip` (SHA256: `3d9e4f1fef7aa1d4bd31420536b330152502959b5e173c107744596d47512511`, 18.5MB)
- **Report PDF**: `docs/final_report.pdf`
- **Corrected HE Stats**: `results/he_metrics_fixed/he_metrics_stats.csv`

## Key Metrics Summary

| Method | deltaE_mean | SSIM  | Interpretation             |
|--------|-------------|-------|----------------------------|
| RGB-HE | 34.72       | 0.265 | High color distortion      |
| Y-HE   | 34.36       | 0.212 | Better chroma preservation |
| CLAHE  | 6.61        | 0.605 | Most perceptually similar  |

## Usage Examples

```bash
# Histogram Equalization (new paths)
python scripts/cli/run_he.py images/he_dark_indoor.jpg --he-mode clahe --space yuv
python scripts/cli/run_he.py images/he_dark_indoor.jpg --he-mode he --space yuv

# Otsu Thresholding
python scripts/cli/run_otsu.py images/otsu_sample_text.jpg --method improved
python scripts/cli/run_otsu.py images/otsu_sample_text.jpg --method global

# Generate Quality Metrics
python scripts/make_metrics.py he --force
python scripts/make_metrics.py otsu --force
```

## Compatibility Notes

- Root-level `run_he.py` and `run_otsu.py` are now compatibility stubs
- Use the new paths in `scripts/cli/` for all new work
- All metrics now reference the canonical directories above
