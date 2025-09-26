#!/usr/bin/env python3
"""Build clean submission ZIP using existing artifacts"""

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
    dirs = ['results', 'docs', 'dist', 'results/slides', 'results/otsu_metrics', 'results/he_metrics', 'results/video']
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

def verify_artifacts():
    """Check which artifacts exist"""
    artifacts = {
        'slides': {
            'he': os.path.exists('results/slides/he_summary.png'),
            'otsu': os.path.exists('results/slides/otsu_summary.png')
        },
        'he_metrics': {
            'collage': os.path.exists('results/he_metrics/he_metrics_collage.png'),
            'individual': len([f for f in os.listdir('results/he_metrics') if f.startswith(('deltaE_', 'ssim_', 'diff_')) and f.endswith('.png')]) > 0 if os.path.exists('results/he_metrics') else False
        },
        'otsu_metrics': {
            'table': os.path.exists('results/otsu_metrics/metrics_table.png'),
            'xor': os.path.exists('results/otsu_metrics/xor_map.png'),
            'compare': os.path.exists('results/otsu_metrics/compare_panel.png')
        },
        'videos': {
            'he_mp4': os.path.exists('results/video/he_sweep.mp4'),
            'he_gif': os.path.exists('results/video/he_sweep.gif'),
            'otsu_mp4': os.path.exists('results/video/otsu_sweep.mp4'),
            'otsu_gif': os.path.exists('results/video/otsu_sweep.gif')
        }
    }
    return artifacts

def get_lib_versions():
    """Get library versions"""
    try:
        import cv2, numpy, skimage, reportlab, matplotlib
        return {
            'opencv': cv2.__version__,
            'numpy': numpy.__version__,
            'skimage': skimage.__version__,
            'reportlab': reportlab.Version,
            'matplotlib': matplotlib.__version__
        }
    except:
        return {'opencv': 'unknown', 'numpy': 'unknown', 'skimage': 'unknown', 'reportlab': 'unknown', 'matplotlib': 'unknown'}

def create_readme():
    """Create submission README"""
    versions = get_lib_versions()

    readme_content = f"""# Visual Computing Assignment - Image Enhancement

## Overview
This submission contains implementations and analysis of histogram equalization (HE) and Otsu thresholding methods.

## Environment
- OpenCV: {versions['opencv']}
- NumPy: {versions['numpy']}
- scikit-image: {versions['skimage']}
- ReportLab: {versions['reportlab']}
- matplotlib: {versions['matplotlib']}

## Reproduction Commands

### Histogram Equalization
```bash
# RGB-HE (global on each channel)
python run_he.py images/he_dark_indoor.jpg --he-mode he --space rgb --save results/he/

# Y-HE (luminance only in YUV space)
python run_he.py images/he_dark_indoor.jpg --he-mode he --space yuv --save results/he/

# CLAHE (recommended for low-light enhancement)
python run_he.py images/he_dark_indoor.jpg --he-mode clahe --space yuv --tile 8 8 --clip 2.5 --save results/he/
```

### Otsu Thresholding
```bash
# Global Otsu
python run_otsu.py images/otsu_shadow_doc_02.jpg --method global --save results/otsu/

# Improved Local Otsu (recommended for documents)
python run_otsu.py images/otsu_shadow_doc_02.jpg --method improved --window 75 --stride 24 --preblur 1.0 --save results/otsu/
```

## Artifacts Included

### Analysis Results
- `results/slides/he_summary.png` - HE methods comparison summary
- `results/slides/otsu_summary.png` - Otsu methods comparison summary
- `results/he_metrics/he_metrics_collage.png` - HE quality metrics (SSIM, Delta E, difference maps)
- `results/otsu_metrics/metrics_table.png` - Otsu comparison table
- `results/otsu_metrics/xor_map.png` - Method disagreement visualization
- `results/otsu_metrics/compare_panel.png` - Side-by-side comparison

### Parameter Sweep Videos
- `results/video/he_sweep.mp4|gif` - HE parameter sweep animation
- `results/video/otsu_sweep.mp4|gif` - Otsu parameter sweep animation

### Final Report
- `docs/final_report.pdf` - Complete analysis report with embedded figures

## Key Findings
- **CLAHE in YUV space** provides optimal enhancement for low-light images
- **Local Otsu with preprocessing** significantly outperforms global method for uneven illumination
- **Parameter optimization** shows clear quality improvements over default settings
"""

    with open("dist/README_submission.md", "w", encoding="utf-8") as f:
        f.write(readme_content)

    return True

def create_minimal_pdf():
    """Create minimal PDF if missing"""
    if os.path.exists("docs/final_report.pdf"):
        return False  # Already exists, don't overwrite

    doc = SimpleDocTemplate("docs/final_report.pdf", pagesize=A4,
                           rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)

    elements = []
    styles = getSampleStyleSheet()

    # Title
    title_style = ParagraphStyle('Title', parent=styles['Title'], fontSize=24, alignment=TA_CENTER, spaceAfter=30)
    elements.append(Paragraph("Image Enhancement Analysis Report", title_style))
    elements.append(Spacer(1, 0.5*inch))
    elements.append(PageBreak())

    # HE Summary
    elements.append(Paragraph("Histogram Equalization Analysis", styles['Heading1']))
    if os.path.exists("results/slides/he_summary.png"):
        elements.append(Image("results/slides/he_summary.png", width=7*inch, height=4.2*inch))
    elements.append(PageBreak())

    # Otsu Summary
    elements.append(Paragraph("Otsu Thresholding Analysis", styles['Heading1']))
    if os.path.exists("results/slides/otsu_summary.png"):
        elements.append(Image("results/slides/otsu_summary.png", width=7*inch, height=4.2*inch))
    elements.append(PageBreak())

    # HE Metrics
    elements.append(Paragraph("HE Quality Metrics", styles['Heading1']))
    if os.path.exists("results/he_metrics/he_metrics_collage.png"):
        elements.append(Image("results/he_metrics/he_metrics_collage.png", width=7*inch, height=5.6*inch))
    elements.append(PageBreak())

    # Otsu Metrics
    elements.append(Paragraph("Otsu Quality Metrics", styles['Heading1']))
    if os.path.exists("results/otsu_metrics/metrics_table.png"):
        elements.append(Image("results/otsu_metrics/metrics_table.png", width=5*inch, height=2*inch))
        elements.append(Spacer(1, 0.2*inch))
    if os.path.exists("results/otsu_metrics/xor_map.png"):
        elements.append(Image("results/otsu_metrics/xor_map.png", width=6*inch, height=4*inch))

    doc.build(elements)
    return True

def create_zip():
    """Create submission ZIP"""
    zip_path = "dist/submission_bundle.zip"

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add results directory
        if os.path.exists("results"):
            for root, dirs, files in os.walk("results"):
                for file in files:
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, file_path)

        # Add PDF
        if os.path.exists("docs/final_report.pdf"):
            zipf.write("docs/final_report.pdf", "docs/final_report.pdf")

        # Add README
        if os.path.exists("dist/README_submission.md"):
            zipf.write("dist/README_submission.md", "README_submission.md")

        # Add images if present
        if os.path.exists("images"):
            for root, dirs, files in os.walk("images"):
                for file in files:
                    if file.endswith(('.jpg', '.png', '.jpeg')):
                        file_path = os.path.join(root, file)
                        zipf.write(file_path, file_path)

    return zip_path

def get_file_info(file_path):
    """Get file size and SHA256"""
    if not os.path.exists(file_path):
        return 0, "error"

    size = os.path.getsize(file_path)

    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return size, sha256_hash.hexdigest()
    except:
        return size, "error"

def main():
    # Track what we reuse/generate/skip
    reused = []
    generated = []
    skipped = []

    # 1. Ensure directories
    ensure_dirs()

    # 2. Verify artifacts
    artifacts = verify_artifacts()

    # Track existing slides
    if artifacts['slides']['he']:
        reused.append('results/slides/he_summary.png')
    if artifacts['slides']['otsu']:
        reused.append('results/slides/otsu_summary.png')

    # Track existing metrics
    if artifacts['he_metrics']['collage']:
        reused.append('results/he_metrics/he_metrics_collage.png')
    if artifacts['otsu_metrics']['table']:
        reused.append('results/otsu_metrics/metrics_table.png')
    if artifacts['otsu_metrics']['xor']:
        reused.append('results/otsu_metrics/xor_map.png')
    if artifacts['otsu_metrics']['compare']:
        reused.append('results/otsu_metrics/compare_panel.png')

    # 3. Create README
    readme_created = create_readme()
    if readme_created:
        generated.append('dist/README_submission.md')

    # 4. Create PDF if needed
    pdf_generated = create_minimal_pdf()
    pdf_exists = os.path.exists("docs/final_report.pdf")

    if pdf_generated:
        generated.append('docs/final_report.pdf')
    elif pdf_exists:
        reused.append('docs/final_report.pdf')

    # 5. Create ZIP
    zip_path = create_zip()
    zip_exists = os.path.exists(zip_path)

    if zip_exists:
        generated.append(zip_path)
        size, sha256 = get_file_info(zip_path)
    else:
        size, sha256 = 0, "error"

    # Final JSON
    result = {
        "task": "build_and_package",
        "slides": {
            "he": artifacts['slides']['he'],
            "otsu": artifacts['slides']['otsu']
        },
        "pdf": pdf_exists,
        "zip": {
            "path": zip_path,
            "exists": zip_exists,
            "sha256": sha256,
            "size_bytes": size
        },
        "reused": reused,
        "generated": generated,
        "skipped": skipped
    }

    print(json.dumps(result))

if __name__ == '__main__':
    main()