#!/usr/bin/env python3
"""Build final submission bundle ZIP"""

import os
import argparse
import zipfile
from pathlib import Path

def build_final_zip(output_path, force=False):
    """Create final submission bundle ZIP"""
    if os.path.exists(output_path) and not force:
        print(f"ZIP already exists: {output_path}")
        print("Use --force to regenerate")
        return False

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Define items to include (if they exist)
    include_patterns = [
        "results/slides/*.png",
        "results/he_metrics_fixed/*",
        "results/otsu_metrics/*",
        "results/video/*",
        "docs/final_report.pdf",
        "dist/README_submission.md"
    ]

    files_to_zip = []
    for pattern in include_patterns:
        if '*' in pattern:
            # Handle glob patterns
            from glob import glob
            matches = glob(pattern)
            files_to_zip.extend(matches)
        else:
            # Handle individual files
            if os.path.exists(pattern):
                files_to_zip.append(pattern)

    if not files_to_zip:
        print("No files found to include in ZIP")
        return False

    try:
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file_path in files_to_zip:
                # Use relative path in ZIP
                rel_path = os.path.relpath(file_path)
                zf.write(file_path, rel_path)
                print(f"Added: {rel_path}")

        print(f"Final bundle created: {output_path}")
        print(f"Total files: {len(files_to_zip)}")
        return True

    except Exception as e:
        print(f"Error creating ZIP: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Build final submission bundle ZIP')
    parser.add_argument('--force', action='store_true', help='Force regeneration even if ZIP exists')
    args = parser.parse_args()

    output_path = Path('dist/submission_bundle_final.zip')

    if build_final_zip(output_path, args.force):
        return 0
    else:
        return 1

if __name__ == '__main__':
    exit(main())