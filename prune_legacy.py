#!/usr/bin/env python3
"""Identify and prune unrelated legacy code files"""

import os
import ast
import sys
import json
import glob
import zipfile
import subprocess
from pathlib import Path
from datetime import datetime
from collections import defaultdict, deque

# Configuration
TARGET_DIRS = ["scripts", "src"]
KEEP_SCRIPTS = [
    "run_he.py", "run_otsu.py",
    "scripts/make_slide_figs.py", "scripts/make_pdf.py",
    "scripts/make_metrics.py", "scripts/make_videos.py", "scripts/run_ablation.py"
]
ZIP_PATH = "dist/submission_bundle.zip"
DRY_RUN = True
APPLY_MODE = "archive"  # "archive" or "delete"

def find_python_files(dirs):
    """Find all Python files in target directories"""
    py_files = set()
    for target_dir in dirs:
        if os.path.exists(target_dir):
            pattern = os.path.join(target_dir, "**", "*.py")
            files = glob.glob(pattern, recursive=True)
            py_files.update(files)

    # Exclude __init__.py, tests/, and notebooks
    filtered = set()
    for f in py_files:
        if "__init__.py" in f or "/test" in f or "test_" in f or f.endswith(".ipynb"):
            continue
        filtered.add(f)

    return filtered

def extract_imports(filepath):
    """Extract local imports from a Python file using AST"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        tree = ast.parse(content)
        imports = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    if node.level > 0:  # Relative import
                        imports.add(f".{node.module}")
                    else:
                        imports.add(node.module)
                else:
                    imports.add(".")  # from . import something

        return imports
    except:
        return set()

def has_main_block(filepath):
    """Check if file has if __name__ == '__main__': block"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        return 'if __name__ == "__main__"' in content or "if __name__ == '__main__'" in content
    except:
        return False

def resolve_import_to_file(import_name, current_file, target_dirs):
    """Resolve import to actual file path"""
    current_dir = os.path.dirname(current_file)

    # Handle relative imports
    if import_name.startswith('.'):
        # Relative import
        parts = import_name.split('.')
        if parts[0] == '':  # Leading dot
            parts = parts[1:]  # Remove empty first element

        # Try different combinations
        candidates = []
        if len(parts) == 0:
            # from . import something - check current directory
            candidates.append(current_dir)
        else:
            rel_path = os.path.join(current_dir, *parts)
            candidates.extend([
                f"{rel_path}.py",
                os.path.join(rel_path, "__init__.py")
            ])
    else:
        # Absolute import - check if it's a local module
        parts = import_name.split('.')
        candidates = []

        for target_dir in target_dirs:
            if os.path.exists(target_dir):
                # Try as direct file
                file_path = os.path.join(target_dir, f"{parts[0]}.py")
                candidates.append(file_path)

                # Try as package
                pkg_path = os.path.join(target_dir, *parts)
                candidates.extend([
                    f"{pkg_path}.py",
                    os.path.join(pkg_path, "__init__.py")
                ])

    # Return first existing candidate
    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.normpath(candidate)

    return None

def build_dependency_graph(py_files, target_dirs):
    """Build dependency graph between Python files"""
    graph = defaultdict(set)

    for filepath in py_files:
        imports = extract_imports(filepath)

        for imp in imports:
            resolved = resolve_import_to_file(imp, filepath, target_dirs)
            if resolved and resolved in py_files:
                graph[filepath].add(resolved)

    return graph

def find_reachable_files(entrypoints, graph):
    """Find all files reachable from entrypoints using BFS"""
    reachable = set()
    queue = deque(entrypoints)

    while queue:
        current = queue.popleft()
        if current in reachable:
            continue

        reachable.add(current)

        # Add dependencies
        for dep in graph.get(current, set()):
            if dep not in reachable:
                queue.append(dep)

    return reachable

def get_zip_listed_files(zip_path):
    """Get Python files listed in ZIP"""
    zip_files = set()
    if os.path.exists(zip_path):
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                for name in zf.namelist():
                    if name.endswith('.py'):
                        # Convert ZIP path to local path
                        local_path = name.replace('/', os.sep)
                        if os.path.exists(local_path):
                            zip_files.add(local_path)
        except:
            pass

    return zip_files

def get_current_branch():
    """Get current git branch"""
    try:
        result = subprocess.run(['git', 'branch', '--show-current'],
                              capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except:
        return "main"

def create_archive_dir():
    """Create archive directory with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    archive_dir = f"archive/legacy_{timestamp}"
    os.makedirs(archive_dir, exist_ok=True)
    return archive_dir

def archive_files(candidates, archive_dir):
    """Move files to archive directory"""
    for filepath in candidates:
        # Preserve directory structure in archive
        rel_path = filepath
        dest_path = os.path.join(archive_dir, rel_path)
        dest_dir = os.path.dirname(dest_path)
        os.makedirs(dest_dir, exist_ok=True)

        # Move file
        os.rename(filepath, dest_path)

def delete_files(candidates):
    """Delete files using git rm or rm"""
    for filepath in candidates:
        try:
            # Try git rm first
            subprocess.run(['git', 'rm', filepath], check=True, capture_output=True)
        except:
            # Fallback to regular rm
            try:
                os.remove(filepath)
            except:
                pass

def commit_changes(branch_name, message):
    """Create branch and commit changes"""
    try:
        # Create new branch
        subprocess.run(['git', 'checkout', '-b', branch_name], check=True, capture_output=True)

        # Add changes
        subprocess.run(['git', 'add', '-A'], check=True, capture_output=True)

        # Commit
        subprocess.run(['git', 'commit', '-m', message], check=True, capture_output=True)

        # Get commit hash
        result = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                              capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except:
        return None

def main():
    # Find all Python files
    py_files = find_python_files(TARGET_DIRS)

    # Normalize paths
    py_files = {os.path.normpath(f) for f in py_files}
    keep_scripts = {os.path.normpath(f) for f in KEEP_SCRIPTS if os.path.exists(f)}

    # Find entrypoints
    entrypoints = set(keep_scripts)
    for filepath in py_files:
        if has_main_block(filepath):
            entrypoints.add(filepath)

    # Build dependency graph
    graph = build_dependency_graph(py_files, TARGET_DIRS)

    # Find reachable files
    reachable = find_reachable_files(entrypoints, graph)

    # Get ZIP-listed files
    zip_files = get_zip_listed_files(ZIP_PATH)
    zip_files = {os.path.normpath(f) for f in zip_files}

    # Calculate candidates for removal
    protected = reachable | keep_scripts | zip_files
    candidates = py_files - protected

    # Sort for consistent output
    candidates = sorted(list(candidates))

    # Prepare result
    result = {
        "task": "legacy_prune",
        "branch": get_current_branch(),
        "dry_run": DRY_RUN,
        "apply_mode": APPLY_MODE,
        "reachable_count": len(reachable),
        "candidates_count": len(candidates),
        "candidates": candidates
    }

    if not DRY_RUN and candidates:
        timestamp = datetime.now().strftime("%Y%m%d")
        branch_name = f"chore/prune-legacy-{timestamp}"

        if APPLY_MODE == "archive":
            archive_dir = create_archive_dir()
            archive_files(candidates, archive_dir)
            result["archived_dir"] = archive_dir
        else:  # delete
            delete_files(candidates)

        # Commit changes
        commit_hash = commit_changes(branch_name, f"Prune legacy code files ({len(candidates)} files)")
        if commit_hash:
            result["commit"] = commit_hash

    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()