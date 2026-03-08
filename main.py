from __future__ import annotations

import subprocess
import sys


def run():
    cmd = [sys.executable, "-m", "streamlit", "run", "frontend/ui.py"]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    run()
