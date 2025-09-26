#!/usr/bin/env python3
"""Rebuild slides, PDF, and ZIP bundle using existing artifacts"""

import os
import json
import hashlib
import zipfile
from pathlib import Path
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.enums import TA_CENTER

def ensure_dirs():
    """Create necessary directories"""
    dirs = ['results/slides', 'dist', 'docs']
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

def create_pdf_report():
    """Create PDF report using existing artifacts"""
    doc = SimpleDocTemplate("docs/final_report.pdf", pagesize=A4,
                           rightMargin=72, leftMargin=72,
                           topMargin=72, bottomMargin=72)

    elements = []
    styles = getSampleStyleSheet()

    # Custom title style
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=24,
        alignment=TA_CENTER,
        spaceAfter=30
    )

    # Custom caption style
    caption_style = ParagraphStyle(
        'Caption',
        parent=styles['Normal'],
        fontSize=10,
        alignment=TA_CENTER,
        spaceBefore=6,
        spaceAfter=12
    )

    # Title page
    elements.append(Paragraph("Visual Computing Assignment Report", title_style))
    elements.append(Paragraph("Image Enhancement Analysis", styles['Heading2']))
    elements.append(Spacer(1, 0.5*inch))
    elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    elements.append(PageBreak())

    # HE Summary page
    elements.append(Paragraph("Histogram Equalization Analysis", styles['Heading1']))
    elements.append(Spacer(1, 0.2*inch))

    if os.path.exists("results/slides/he_summary.png"):
        he_img = Image("results/slides/he_summary.png", width=7*inch, height=4.2*inch)
        elements.append(he_img)
        elements.append(Spacer(1, 0.2*inch))
        elements.append(Paragraph("Figure 1: Histogram Equalization comparison showing original vs CLAHE enhancement with ROI analysis", caption_style))

    elements.append(PageBreak())

    # Otsu Summary page
    elements.append(Paragraph("Otsu Thresholding Analysis", styles['Heading1']))
    elements.append(Spacer(1, 0.2*inch))

    if os.path.exists("results/slides/otsu_summary.png"):
        otsu_img = Image("results/slides/otsu_summary.png", width=7*inch, height=4.2*inch)
        elements.append(otsu_img)
        elements.append(Spacer(1, 0.2*inch))
        elements.append(Paragraph("Figure 2: Otsu thresholding comparison with threshold heatmap and ROI analysis", caption_style))

    elements.append(PageBreak())

    # HE Metrics page
    elements.append(Paragraph("HE Quality Metrics", styles['Heading1']))
    elements.append(Spacer(1, 0.2*inch))

    if os.path.exists("results/he_metrics/he_metrics_collage.png"):
        metrics_img = Image("results/he_metrics/he_metrics_collage.png", width=7*inch, height=5.6*inch)
        elements.append(metrics_img)
        elements.append(Spacer(1, 0.2*inch))
        elements.append(Paragraph("Figure 3: HE quality assessment using difference maps, SSIM, and Delta E metrics", caption_style))

    elements.append(PageBreak())

    # Otsu Metrics page
    elements.append(Paragraph("Otsu Quality Metrics", styles['Heading1']))
    elements.append(Spacer(1, 0.2*inch))

    if os.path.exists("results/otsu_metrics/metrics_table.png"):
        table_img = Image("results/otsu_metrics/metrics_table.png", width=5*inch, height=2*inch)
        elements.append(table_img)
        elements.append(Spacer(1, 0.2*inch))

    if os.path.exists("results/otsu_metrics/xor_map.png"):
        xor_img = Image("results/otsu_metrics/xor_map.png", width=6*inch, height=4*inch)
        elements.append(xor_img)
        elements.append(Spacer(1, 0.2*inch))
        elements.append(Paragraph("Figure 4: Otsu method comparison table and disagreement (XOR) map", caption_style))

    elements.append(PageBreak())

    # Videos page
    elements.append(Paragraph("Generated Animations", styles['Heading1']))
    elements.append(Spacer(1, 0.2*inch))

    video_content = """
    The following parameter sweep animations were generated:

    • HE Parameter Sweep: results/video/he_sweep.mp4, he_sweep.gif
      - Demonstrates CLAHE parameter variations (tile size and clip limit)

    • Otsu Parameter Sweep: results/video/otsu_sweep.mp4, otsu_sweep.gif
      - Shows local Otsu parameter effects (window, stride, pre-blur)

    These animations visualize how different parameter settings affect
    the enhancement results in real-time comparison format.
    """

    elements.append(Paragraph(video_content, styles['Normal']))

    # Build PDF
    doc.build(elements)
    return True

def create_readme():
    """Create submission README"""
    readme_content = """# Visual Computing Assignment - Image Enhancement

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
"""

    with open("dist/README_submission.md", "w", encoding="utf-8") as f:
        f.write(readme_content)

    return True

def create_submission_zip():
    """Create submission bundle ZIP"""
    zip_path = "dist/submission_bundle.zip"

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add PDF report
        if os.path.exists("docs/final_report.pdf"):
            zipf.write("docs/final_report.pdf", "final_report.pdf")

        # Add README
        if os.path.exists("dist/README_submission.md"):
            zipf.write("dist/README_submission.md", "README_submission.md")

        # Add key result directories
        for root, dirs, files in os.walk("results"):
            for file in files:
                file_path = os.path.join(root, file)
                arc_name = file_path
                zipf.write(file_path, arc_name)

        # Add input images
        for root, dirs, files in os.walk("images"):
            for file in files:
                if file.endswith(('.jpg', '.png', '.jpeg')):
                    file_path = os.path.join(root, file)
                    arc_name = file_path
                    zipf.write(file_path, arc_name)

        # Add key source files
        key_files = [
            "run_he.py",
            "run_otsu.py",
            "src/he.py",
            "src/otsu.py",
            "src/utils.py"
        ]

        for file in key_files:
            if os.path.exists(file):
                zipf.write(file, file)

    return zip_path

def calculate_sha256(file_path):
    """Calculate SHA256 hash of file"""
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    except:
        return "error"

def main():
    # 1. Ensure directories
    ensure_dirs()

    # 2. Check existing slides (should exist per audit)
    slides_exist = {
        "he": os.path.exists("results/slides/he_summary.png"),
        "otsu": os.path.exists("results/slides/otsu_summary.png")
    }

    # 3. Create PDF report
    pdf_created = create_pdf_report()

    # 4. Create README
    readme_created = create_readme()

    # 5. Create ZIP bundle
    zip_path = create_submission_zip()
    zip_exists = os.path.exists(zip_path)
    zip_hash = calculate_sha256(zip_path) if zip_exists else "error"

    # Final JSON output
    result = {
        "task": "slides_pdf_zip_refresh",
        "slides": slides_exist,
        "pdf": pdf_created and os.path.exists("docs/final_report.pdf"),
        "zip": {
            "path": zip_path,
            "exists": zip_exists,
            "sha256": zip_hash
        },
        "notes": ["reused existing assets; no heavy recompute"]
    }

    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()