#!/usr/bin/env python3
"""Update submission package with corrected HE metrics"""

import os
import json
import zipfile
import hashlib
from pathlib import Path

def create_submission_zip():
    """Create updated submission ZIP with corrected HE metrics"""

    # Core files to include
    files_to_zip = [
        # HE results
        "results/he/result_rgb_he.png",
        "results/he/result_yuv_he.png",
        "results/he/result_yuv_clahe.png",

        # Corrected HE metrics
        "results/he_metrics_fixed/he_metrics_stats.csv",
        "results/he_metrics_fixed/deltaE_collage.png",
        "results/he_metrics_fixed/ssim_collage.png",
        "results/he_metrics_fixed/diff_collage.png",

        # Individual metric maps
        "results/he_metrics_fixed/deltaE_rgb_he.png",
        "results/he_metrics_fixed/deltaE_y_he.png",
        "results/he_metrics_fixed/deltaE_clahe.png",
        "results/he_metrics_fixed/ssim_rgb_he.png",
        "results/he_metrics_fixed/ssim_y_he.png",
        "results/he_metrics_fixed/ssim_clahe.png",
        "results/he_metrics_fixed/diff_rgb_he.png",
        "results/he_metrics_fixed/diff_y_he.png",
        "results/he_metrics_fixed/diff_clahe.png",

        # Otsu results (if they exist)
        "results/otsu/global.png",
        "results/otsu/improved.png",

        # Source code
        "run_he.py",
        "run_otsu.py",
        "src/he.py",
        "src/otsu.py",
        "src/utils.py",

        # Scripts
        "scripts/make_metrics.py",
        "scripts/make_rgb_he_metrics.py",
    ]

    # Optional files (include if they exist)
    optional_files = [
        "results/slides/he_summary.png",
        "results/slides/otsu_summary.png",
        "results/report.pdf",
        "repro.lock.json",
    ]

    zip_path = "dist/submission_corrected_he.zip"
    os.makedirs("dist", exist_ok=True)

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        files_added = 0

        # Add core files
        for file_path in files_to_zip:
            if os.path.exists(file_path):
                zf.write(file_path, file_path)
                files_added += 1
                print(f"Added: {file_path}")

        # Add optional files
        for file_path in optional_files:
            if os.path.exists(file_path):
                zf.write(file_path, file_path)
                files_added += 1
                print(f"Added (optional): {file_path}")

    # Calculate SHA256
    with open(zip_path, 'rb') as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()

    zip_size = os.path.getsize(zip_path)

    result = {
        "status": "success",
        "zip_path": zip_path,
        "files_count": files_added,
        "size_bytes": zip_size,
        "sha256": sha256,
        "note": "Corrected HE metrics - Y-HE and CLAHE now properly distinguished"
    }

    print(json.dumps(result, indent=2))
    return result

if __name__ == "__main__":
    create_submission_zip()