#!/usr/bin/env python3
import os, json, glob, shutil, subprocess, pathlib
from pathlib import Path

def discover_plan():
    """Build file move plan and detect references"""
    moves = [
        ["run_he.py", "scripts/cli/run_he.py"],
        ["run_otsu.py", "scripts/cli/run_otsu.py"]
    ]

    moves_planned = []
    for src, dst in moves:
        if os.path.exists(src) and src != dst:
            moves_planned.append([src, dst])

    # Check if src/__init__.py exists
    src_init_missing = not os.path.exists("src/__init__.py")

    # Count references in README files
    readme_files = ["README.md", "dist/README_submission.md"]
    ref_counts = {"run_he": 0, "run_otsu": 0}

    for readme in readme_files:
        if os.path.exists(readme):
            with open(readme, 'r') as f:
                content = f.read()
                ref_counts["run_he"] += content.count("python run_he.py")
                ref_counts["run_otsu"] += content.count("python run_otsu.py")

    # Count docs/*.md files
    docs_md_count = len(glob.glob("docs/*.md"))

    return {
        "moves_planned": moves_planned,
        "src_init_missing": src_init_missing,
        "readme_files": readme_files + [f"docs/*.md ({docs_md_count})"],
        "ref_counts": ref_counts
    }

def apply_refactor(plan):
    """Apply the refactor plan"""
    moves_applied = []
    created = []

    # Create directories
    mkdirs = ["scripts/cli", "docs", "archive"]
    for d in mkdirs:
        if not os.path.exists(d):
            Path(d).mkdir(parents=True, exist_ok=True)
            created.append(d)

    # Apply moves
    for src, dst in plan["moves_planned"]:
        Path(dst).parent.mkdir(parents=True, exist_ok=True)

        # Try git mv first, fallback to regular move
        try:
            subprocess.run(["git", "mv", src, dst], check=True, capture_output=True)
        except:
            shutil.move(src, dst)

        moves_applied.append([src, dst])

    # Create src/__init__.py if missing
    if plan["src_init_missing"]:
        with open("src/__init__.py", "w") as f:
            f.write("# Visual Computing Assignment Package\n")
        created.append("src/__init__.py")
        try:
            subprocess.run(["git", "add", "src/__init__.py"], capture_output=True)
        except:
            pass

    # Update references in README files
    replacements_made = {"run_he": 0, "run_otsu": 0}

    for readme in ["README.md", "dist/README_submission.md"]:
        if os.path.exists(readme):
            with open(readme, 'r') as f:
                content = f.read()

            original_content = content
            content = content.replace("python run_he.py", "python scripts/cli/run_he.py")
            content = content.replace("python run_otsu.py", "python scripts/cli/run_otsu.py")

            # Count actual replacements
            replacements_made["run_he"] += original_content.count("python run_he.py")
            replacements_made["run_otsu"] += original_content.count("python run_otsu.py")

            # Add path changes table
            if moves_applied:
                path_changes = "\n## Path Changes\n\n"
                path_changes += "| Previous Path | New Path |\n"
                path_changes += "|---------------|----------|\n"
                for old, new in moves_applied:
                    path_changes += f"| {old} | {new} |\n"
                path_changes += "\n"

                # Insert after first heading
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if line.startswith('#') and i > 0:
                        lines.insert(i+1, path_changes)
                        break
                else:
                    lines.append(path_changes)
                content = '\n'.join(lines)

            with open(readme, 'w') as f:
                f.write(content)

    # Generate REPO_STRUCTURE.md
    create_repo_structure_doc()
    created.append("docs/REPO_STRUCTURE.md")

    return moves_applied, created, replacements_made

def create_repo_structure_doc():
    """Generate repository structure documentation"""
    content = """# Repository Structure

This document describes the organization of the Visual Computing Assignment 1 repository.

## Directory Tree

```
.
├── images/                         # Input images (640x480 samples)
├── src/                            # Core algorithm implementations
│   ├── __init__.py
│   ├── he.py                      # Histogram equalization algorithms
│   ├── otsu.py                    # Otsu thresholding methods
│   └── utils.py                   # Shared utilities
├── scripts/                       # Build and utility scripts
│   ├── cli/                       # Command-line interfaces
│   │   ├── run_he.py             # HE tool (global/AHE/CLAHE)
│   │   └── run_otsu.py           # Otsu tool (global/improved)
│   ├── make_metrics.py           # Quality metrics generator
│   ├── make_videos.py            # Video/GIF creators
│   ├── make_slide_figs.py        # Summary slide builder
│   └── make_pdf.py               # Report PDF generator
├── results/                      # Generated outputs
│   ├── he/                       # HE processed images
│   ├── otsu/                     # Otsu binary outputs
│   ├── he_metrics_fixed/         # Canonical HE quality metrics
│   ├── otsu_metrics/             # Otsu analysis metrics
│   ├── slides/                   # Summary slide PNGs
│   └── video/                    # MP4/GIF animations
├── docs/                         # Documentation
│   ├── final_report.pdf          # Generated report
│   └── REPO_STRUCTURE.md         # This file
├── dist/                         # Distribution artifacts
│   ├── README_submission.md      # Submission documentation
│   └── submission_bundle_final.zip # Final package
└── archive/                      # Legacy/backup files
```

## Canonical Metric Directories

- **HE Metrics**: `results/he_metrics_fixed/` - Contains corrected Y-HE vs CLAHE distinction
- **Otsu Metrics**: `results/otsu_metrics/` - Contains comparative analysis artifacts

## Key Artifacts

- **Final ZIP**: `dist/submission_bundle_final.zip`
- **Report PDF**: `docs/final_report.pdf`
- **Corrected HE Stats**: `results/he_metrics_fixed/he_metrics_stats.csv`

## Usage

```bash
# Histogram Equalization
python scripts/cli/run_he.py images/he_dark_indoor.jpg --he-mode clahe --space yuv

# Otsu Thresholding
python scripts/cli/run_otsu.py images/otsu_sample_text.jpg --method improved

# Generate Metrics
python scripts/make_metrics.py he --force
```
"""

    with open("docs/REPO_STRUCTURE.md", "w") as f:
        f.write(content)

def run_smoke_tests(dry_run=True):
    """Run non-destructive smoke tests"""
    results = {"compiled": False, "he_help": "", "otsu_help": ""}

    # Python compilation test
    try:
        result = subprocess.run(["python", "-m", "compileall", "-q", "."],
                              capture_output=True, text=True)
        results["compiled"] = result.returncode == 0
    except:
        results["compiled"] = False

    # CLI help tests (only if files exist at expected locations)
    he_path = "scripts/cli/run_he.py" if not dry_run else "run_he.py"
    otsu_path = "scripts/cli/run_otsu.py" if not dry_run else "run_otsu.py"

    if os.path.exists(he_path):
        try:
            result = subprocess.run(["python", he_path, "--help"],
                                  capture_output=True, text=True, timeout=10)
            if result.stdout:
                results["he_help"] = result.stdout.split('\n')[0][:50]
        except:
            results["he_help"] = "Error getting help"

    if os.path.exists(otsu_path):
        try:
            result = subprocess.run(["python", otsu_path, "--help"],
                                  capture_output=True, text=True, timeout=10)
            if result.stdout:
                results["otsu_help"] = result.stdout.split('\n')[0][:50]
        except:
            results["otsu_help"] = "Error getting help"

    return results

def main():
    dry_run = True  # Default to true as specified

    # Discover and plan
    plan = discover_plan()

    # Apply if not dry run
    moves_applied = []
    created = []
    replacements_made = {"run_he": 0, "run_otsu": 0}

    if not dry_run:
        moves_applied, created, replacements_made = apply_refactor(plan)

    # Run smoke tests
    smoke_results = run_smoke_tests(dry_run)

    # Get final ZIP info if available
    zip_path = "dist/submission_bundle_final.zip"
    zip_info = ""
    if os.path.exists(zip_path):
        import hashlib
        with open(zip_path, 'rb') as f:
            sha256 = hashlib.sha256(f.read()).hexdigest()[:16]
        size_mb = os.path.getsize(zip_path) / (1024*1024)
        zip_info = f"submission_bundle_final.zip (SHA256: {sha256}..., {size_mb:.1f}MB)"

    result = {
        "task": "repo_refactor",
        "dry_run": dry_run,
        "moves_planned": plan["moves_planned"],
        "moves_applied": moves_applied,
        "created": created,
        "readme_updates": {
            "files": plan["readme_files"],
            "replacements_count": replacements_made if not dry_run else plan["ref_counts"]
        },
        "smoke": smoke_results,
        "notes": [
            "Canonical metrics dirs: he -> results/he_metrics_fixed, otsu -> results/otsu_metrics",
            "No recomputation performed",
            f"Final package: {zip_info}" if zip_info else "No final package detected"
        ]
    }

    print(json.dumps(result))

if __name__ == "__main__":
    main()