import sys
import subprocess
from pathlib import Path


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=root,
        check=False,
    )
    raise SystemExit(result.returncode)
