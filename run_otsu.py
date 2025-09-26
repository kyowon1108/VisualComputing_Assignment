#!/usr/bin/env python3
import runpy, sys, pathlib
target = "run_he.py" if __file__.endswith("run_he.py") else "run_otsu.py"
sys.stderr.write(f"[WARN] Deprecated entrypoint. Use: python scripts/cli/{target}\n")
runpy.run_path(str(pathlib.Path("scripts/cli")/target), run_name="__main__")