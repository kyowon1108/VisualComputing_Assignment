#!/usr/bin/env python3
"""Quick generator audit check"""

import os
import zipfile

def count_zip_artifacts():
    """Count artifacts in ZIP"""
    if not os.path.exists("dist/submission_bundle.zip"):
        return 0

    with zipfile.ZipFile("dist/submission_bundle.zip", 'r') as zf:
        return len([name for name in zf.namelist()
                   if name.endswith(('.png', '.csv', '.pdf', '.json', '.mp4', '.gif'))])

def check_key_generators():
    """Check if key generators exist"""
    generators = [
        "run_he.py",
        "run_otsu.py",
        "scripts/make_metrics.py",
        "scripts/make_videos.py",
        "scripts/make_slide_figs.py",
        "scripts/make_pdf.py"
    ]

    missing = [g for g in generators if not os.path.exists(g)]
    return len(generators) - len(missing), missing

def main():
    total_artifacts = count_zip_artifacts()
    covered_generators, missing_generators = check_key_generators()

    if total_artifacts > 0:
        coverage_ratio = f"{(covered_generators / 6) * 100:.1f}%"
    else:
        coverage_ratio = "N/A"

    result = {
        "coverage_ratio": coverage_ratio,
        "missing_generators": missing_generators
    }

    print(f'"{coverage_ratio}",{len(missing_generators)}')

if __name__ == '__main__':
    main()