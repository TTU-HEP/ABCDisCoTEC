import os
import subprocess
import sys
import venv

VENV_DIR = "venv"
MAIN_SCRIPT = "abcdexample.py"  # Your main script

# Set platform-dependent paths
if os.name == "nt":  # Windows
    python_bin = os.path.join(VENV_DIR, "Scripts", "python.exe")
else:  # macOS/Linux
    python_bin = os.path.join(VENV_DIR, "bin", "python")

# Step 1: Check if venv exists
if not os.path.exists(python_bin):
    print("❌ Virtual environment not found. Run the full setup script first.")
    sys.exit(1)

# Step 2: Run your script using the virtual environment
print(f"▶ Running {MAIN_SCRIPT} using {python_bin}...")
subprocess.check_call([python_bin, MAIN_SCRIPT])
