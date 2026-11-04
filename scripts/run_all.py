"""Run every script in turn. Total run time: about 1-2 hours."""
import subprocess
import sys

SCRIPTS = ["rest_angle.py", "tip_moment.py", "finite_sheet_path.py", "finite_sheet_stages.py",
           "code_deflection.py", "code_release_rate.py", "linear_range.py", "checks.py"]
for s in SCRIPTS:
    print(f"\n===== {s}", flush=True)
    subprocess.run([sys.executable, s], check=True)
