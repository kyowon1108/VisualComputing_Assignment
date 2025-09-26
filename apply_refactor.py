#!/usr/bin/env python3
import os, json, glob, shutil, subprocess, pathlib, stat
from pathlib import Path

def apply_refactor():
    """Apply the repository refactor"""
    moves = [
        ["run_he.py", "scripts/cli/run_he.py"],
        ["run_otsu.py", "scripts/cli/run_otsu.py"]
    ]

    mkdirs = ["scripts/cli", "docs", "archive"]
    moves_applied = []
    created = []

    # 1) Create directories
    for d in mkdirs:
        if not os.path.exists(d):
            Path(d).mkdir(parents=True, exist_ok=True)
            created.append(d)

    # 2) Move files and create compat stubs
    for src, dst in moves:
        if os.path.exists(src):
            # Ensure destination directory exists
            Path(dst).parent.mkdir(parents=True, exist_ok=True)

            # Try git mv first, fallback to regular move
            try:
                subprocess.run(["git", "mv", src, dst], check=True, capture_output=True)
            except:
                shutil.move(src, dst)

            moves_applied.append([src, dst])

            # Create compatibility stub
            script_name = "run_he" if "run_he" in src else "run_otsu"
            stub_content = f"""#!/usr/bin/env python3
import runpy, sys, pathlib
sys.stderr.write("[WARN] Deprecated entrypoint. Use: python scripts/cli/{script_name}.py\\n")
runpy.run_path(str(pathlib.Path("scripts/cli/{script_name}.py")), run_name="__main__")
"""
            with open(src, 'w') as f:
                f.write(stub_content)

            # Make executable
            try:
                st = os.stat(src)
                os.chmod(src, st.st_mode | stat.S_IEXEC)
            except:
                pass

            created.append(f"{src}(stub)")

    # 3) Ensure src/__init__.py exists
    if not os.path.exists("src/__init__.py"):
        with open("src/__init__.py", "w") as f:
            f.write("# Visual Computing Assignment Package\n")
        created.append("src/__init__.py")
        try:
            subprocess.run(["git", "add", "src/__init__.py"], capture_output=True)
        except:
            pass

    # 4) Update READMEs and docs
    readme_files = ["README.md", "dist/README_submission.md"]
    docs_files = glob.glob("docs/*.md")
    replacements_count = {"run_he": 0, "run_otsu": 0}
    updated_files = []

    all_files = readme_files + docs_files
    for filepath in all_files:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                content = f.read()

            original_content = content

            # Replace paths
            content = content.replace("python run_he.py", "python scripts/cli/run_he.py")
            content = content.replace("python run_otsu.py", "python scripts/cli/run_otsu.py")

            # Normalize metric paths
            content = content.replace("results/he_metrics/", "results/he_metrics_fixed/")

            # Count replacements
            replacements_count["run_he"] += original_content.count("python run_he.py")
            replacements_count["run_otsu"] += original_content.count("python run_otsu.py")

            # Add path changes table for README files
            if filepath in readme_files and moves_applied:
                path_changes = "\n## Path Changes\n\n"
                path_changes += "| Previous Path | New Path |\n"
                path_changes += "|---------------|----------|\n"
                for old, new in moves_applied:
                    path_changes += f"| {old} | {new} |\n"
                path_changes += "\n"

                # Insert after first heading or at end
                lines = content.split('\n')
                inserted = False
                for i, line in enumerate(lines):
                    if line.startswith('#') and i > 0:
                        lines.insert(i+1, path_changes)
                        inserted = True
                        break
                if not inserted:
                    lines.append(path_changes)
                content = '\n'.join(lines)

            # Write updated content
            with open(filepath, 'w') as f:
                f.write(content)

            if content != original_content:
                updated_files.append(filepath)

    # 5) Generate docs/REPO_STRUCTURE.md
    create_repo_structure_doc()
    created.append("docs/REPO_STRUCTURE.md")

    # 6) Fix import issues in moved CLIs
    for _, dst in moves_applied:
        if os.path.exists(dst):
            with open(dst, 'r') as f:
                content = f.read()

            # Add path fix at top if not already present
            path_fix = "import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))"
            if path_fix not in content:
                lines = content.split('\n')
                # Insert after shebang and before other imports
                insert_pos = 1 if lines[0].startswith('#!') else 0
                lines.insert(insert_pos, path_fix)

                with open(dst, 'w') as f:
                    f.write('\n'.join(lines))

    # 7) Smoke tests
    smoke_results = run_smoke_tests()

    return {
        "moves_applied": moves_applied,
        "created": created,
        "readme_updates": {
            "files": [f"README.md", f"dist/README_submission.md", f"docs/*.md (updated {len([f for f in updated_files if f.startswith('docs/')])})"],
            "replacements_count": replacements_count
        },
        "smoke": smoke_results
    }

def create_repo_structure_doc():
    """Generate repository structure documentation"""

    # Get ZIP info if available
    zip_path = "dist/submission_bundle_final.zip"
    zip_info = ""
    if os.path.exists(zip_path):
        import hashlib
        with open(zip_path, 'rb') as f:
            sha256 = hashlib.sha256(f.read()).hexdigest()
        size_mb = os.path.getsize(zip_path) / (1024*1024)
        zip_info = f"- **Final ZIP**: `{zip_path}` (SHA256: `{sha256}`, {size_mb:.1f}MB)"

    content = f"""# Repository Structure

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

{zip_info}
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
"""

    with open("docs/REPO_STRUCTURE.md", "w") as f:
        f.write(content)

def run_smoke_tests():
    """Run non-destructive smoke tests"""
    results = {"compiled": False, "he_help": "", "otsu_help": ""}

    # Python compilation test
    try:
        result = subprocess.run(["python", "-m", "compileall", "-q", "."],
                              capture_output=True, text=True)
        results["compiled"] = result.returncode == 0
    except:
        results["compiled"] = False

    # CLI help tests
    for script_name, key in [("run_he.py", "he_help"), ("run_otsu.py", "otsu_help")]:
        script_path = f"scripts/cli/{script_name}"
        if os.path.exists(script_path):
            try:
                result = subprocess.run(["python", script_path, "--help"],
                                      capture_output=True, text=True, timeout=10)
                if result.stdout:
                    results[key] = result.stdout.split('\n')[0][:80]
                elif result.stderr:
                    results[key] = f"Error: {result.stderr[:50]}"
            except Exception as e:
                results[key] = f"Error: {str(e)[:50]}"

    return results

def main():
    result_data = apply_refactor()

    final_result = {
        "task": "repo_refactor_apply",
        "moves_applied": result_data["moves_applied"],
        "created": result_data["created"],
        "readme_updates": result_data["readme_updates"],
        "smoke": result_data["smoke"],
        "notes": [
            "compat stubs created",
            "no heavy recompute",
            "final zip unchanged"
        ]
    }

    print(json.dumps(final_result))

if __name__ == "__main__":
    main()