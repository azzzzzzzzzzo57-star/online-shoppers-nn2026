"""Run Week 10 preprocessing and validation baselines; does not evaluate test."""
import subprocess
import sys

for args in ([sys.executable, "-m", "src.data"], [sys.executable, "-m", "src.baselines"]):
    subprocess.run(args, check=True)
