#!/usr/bin/env python3
"""Check AST imports for improved_local_otsu references"""

import ast
import glob
import os

def check_file_imports(filepath):
    """Check if file imports improved_local_otsu"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        tree = ast.parse(content)
        references = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if 'improved_local_otsu' in alias.name:
                        references.append(f"import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module and 'improved_local_otsu' in node.module:
                    references.append(f"from {node.module} import ...")

        return references
    except:
        return []

def main():
    # Check all Python files in scripts/ and src/
    py_files = []
    for pattern in ["scripts/*.py", "src/*.py", "src/**/*.py"]:
        py_files.extend(glob.glob(pattern, recursive=True))

    all_references = {}
    for filepath in py_files:
        if not os.path.exists(filepath):
            continue

        refs = check_file_imports(filepath)
        if refs:
            all_references[filepath] = refs

    if all_references:
        print("Found references:")
        for filepath, refs in all_references.items():
            print(f"  {filepath}: {refs}")
        return 1
    else:
        print("No AST imports found for improved_local_otsu")
        return 0

if __name__ == '__main__':
    exit(main())