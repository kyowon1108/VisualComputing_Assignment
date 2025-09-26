#!/usr/bin/env python3
import runpy, sys, pathlib
sys.stderr.write("[WARN] Deprecated entrypoint. Use: python scripts/cli/run_he.py\n")
runpy.run_path(str(pathlib.Path("scripts/cli/run_he.py")), run_name="__main__")
