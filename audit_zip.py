#!/usr/bin/env python3
import json
import zipfile
import os

def audit_zip():
    zip_path = "dist/submission_bundle_final.zip"

    # Key files to check
    key_files = [
        "results/slides/he_summary.png",
        "results/slides/otsu_summary.png",
        "results/he_metrics_fixed/diff_collage.png",
        "results/he_metrics_fixed/ssim_collage.png",
        "results/he_metrics_fixed/deltaE_collage.png",
        "results/he_metrics_fixed/he_metrics_stats.csv",
        "results/otsu_metrics/compare_panel.png",
        "results/otsu_metrics/xor_map.png",
        "results/otsu_metrics/metrics.csv",
        "results/otsu_metrics/metrics_table.png",
        "results/video/he_sweep.mp4",
        "results/video/he_sweep.gif",
        "results/video/otsu_sweep.mp4",
        "results/video/otsu_sweep.gif",
        "docs/final_report.pdf",
        "dist/README_submission.md"
    ]

    found = []
    missing = []

    if not os.path.exists(zip_path):
        return {"task": "zip_audit", "found": [], "missing": key_files}

    with zipfile.ZipFile(zip_path, 'r') as zf:
        zip_contents = set(zf.namelist())

        for file_path in key_files:
            if file_path in zip_contents:
                found.append(file_path)
            else:
                missing.append(file_path)

    result = {
        "task": "zip_audit",
        "found": found,
        "missing": missing
    }

    print(json.dumps(result))

if __name__ == "__main__":
    audit_zip()