"""Start Flask API + Vite React dashboard locally (offline)."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DASHBOARD = Path(__file__).resolve().parent
BACKEND = DASHBOARD / "backend"
FRONTEND = DASHBOARD / "frontend"
VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"
if not VENV_PY.is_file():
    VENV_PY = Path(sys.executable)


def main() -> int:
    print("=" * 60)
    print("Cattle Weight Estimation Dashboard")
    print("Backend:  http://127.0.0.1:5001")
    print("Frontend: http://127.0.0.1:5173")
    print("=" * 60)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(BACKEND) + os.pathsep + env.get("PYTHONPATH", "")

    backend = subprocess.Popen(
        [str(VENV_PY), str(BACKEND / "app.py")],
        cwd=str(BACKEND),
        env=env,
    )
    time.sleep(1.5)

    npm = "npm.cmd" if os.name == "nt" else "npm"
    frontend = subprocess.Popen(
        [npm, "run", "dev", "--", "--host", "127.0.0.1", "--port", "5173"],
        cwd=str(FRONTEND),
        shell=(os.name == "nt"),
    )

    try:
        backend.wait()
        frontend.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
        backend.terminate()
        frontend.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
