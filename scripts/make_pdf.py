#!/usr/bin/env python3
"""Lightweight PDF report generation script"""

import os
import argparse
from pathlib import Path
from datetime import datetime

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import inch
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
except ImportError:
    print("Error: ReportLab not installed. Please install with: pip install reportlab")
    exit(1)

def setup_fonts():
    """Setup fonts with Helvetica fallback"""
    # Use built-in Helvetica fonts for ASCII-safe rendering
    pass

def create_pdf_report(output_path, force=False):
    """Create PDF report embedding slide PNGs and metrics tables"""
    if os.path.exists(output_path) and not force:
        print(f"PDF already exists: {output_path}")
        print("Use --force to regenerate")
        return False

    # Create document with A4 page size
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )

    # Get styles
    styles = getSampleStyleSheet()

    # Custom title style
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=24,
        alignment=TA_CENTER,
        spaceAfter=30
    )

    story = []

    # Title page
    story.append(Paragraph("Visual Computing Assignment Report", title_style))
    story.append(Paragraph("Image Enhancement Analysis", styles['Heading2']))
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(PageBreak())

    # HE Summary page
    story.append(Paragraph("Histogram Equalization Analysis", styles['Heading1']))
    story.append(Spacer(1, 0.2*inch))

    he_slide = "results/slides/he_summary.png"
    if os.path.exists(he_slide):
        he_img = Image(str(he_slide), width=7*inch, height=4.2*inch)
        story.append(he_img)
        story.append(Spacer(1, 0.2*inch))
    else:
        story.append(Paragraph("HE summary slide not available", styles['Normal']))

    story.append(PageBreak())

    # Otsu Summary page
    story.append(Paragraph("Otsu Thresholding Analysis", styles['Heading1']))
    story.append(Spacer(1, 0.2*inch))

    otsu_slide = "results/slides/otsu_summary.png"
    if os.path.exists(otsu_slide):
        otsu_img = Image(str(otsu_slide), width=7*inch, height=4.2*inch)
        story.append(otsu_img)
        story.append(Spacer(1, 0.2*inch))
    else:
        story.append(Paragraph("Otsu summary slide not available", styles['Normal']))

    story.append(PageBreak())

    # HE Metrics page
    story.append(Paragraph("HE Quality Metrics", styles['Heading1']))
    story.append(Spacer(1, 0.2*inch))

    he_metrics = "results/he_metrics/he_metrics_collage.png"
    if os.path.exists(he_metrics):
        metrics_img = Image(str(he_metrics), width=7*inch, height=5.6*inch)
        story.append(metrics_img)
        story.append(Spacer(1, 0.2*inch))
    else:
        story.append(Paragraph("HE metrics not available", styles['Normal']))

    story.append(PageBreak())

    # Otsu Metrics page
    story.append(Paragraph("Otsu Quality Metrics", styles['Heading1']))
    story.append(Spacer(1, 0.2*inch))

    otsu_table = "results/otsu_metrics/metrics_table.png"
    if os.path.exists(otsu_table):
        table_img = Image(str(otsu_table), width=5*inch, height=2*inch)
        story.append(table_img)
        story.append(Spacer(1, 0.2*inch))

    otsu_xor = "results/otsu_metrics/xor_map.png"
    if os.path.exists(otsu_xor):
        xor_img = Image(str(otsu_xor), width=6*inch, height=4*inch)
        story.append(xor_img)
        story.append(Spacer(1, 0.2*inch))

    if not os.path.exists(otsu_table) and not os.path.exists(otsu_xor):
        story.append(Paragraph("Otsu metrics not available", styles['Normal']))

    # Build PDF
    try:
        doc.build(story)
        return True
    except Exception as e:
        print(f"Error building PDF: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Generate PDF report from slide PNGs and metrics')
    parser.add_argument('--force', action='store_true', help='Force regeneration even if PDF exists')
    args = parser.parse_args()

    # Ensure output directory
    docs_dir = Path('docs')
    docs_dir.mkdir(exist_ok=True)
    pdf_path = docs_dir / 'final_report.pdf'

    # Setup fonts
    setup_fonts()

    # Create PDF
    if create_pdf_report(pdf_path, args.force):
        print(f"PDF created: {pdf_path}")
        return 0
    else:
        print("Failed to create PDF")
        return 1

if __name__ == '__main__':
    exit(main())