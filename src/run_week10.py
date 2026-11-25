"""Run Week 10 preprocessing, both baselines, then Model A; no test evaluation."""
import subprocess
import sys

steps = [
    [sys.executable, "-m", "src.data"],
    [sys.executable, "-m", "src.baselines"],
    [sys.executable, "-m", "src.train"],
]
for command in steps:
    subprocess.run(command, check=True)
