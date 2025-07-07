import os
import subprocess
import sys
import venv

# === Configuration ===
VENV_DIR = "venv"
REQUIREMENTS = "requirements.txt"
MAIN_SCRIPT = "abcdexample.py"

# === Step 1: Check Files Exist ===
if not os.path.exists(REQUIREMENTS):
    print(f"❌ requirements.txt not found: {REQUIREMENTS}")
    sys.exit(1)

if not os.path.exists(MAIN_SCRIPT):
    print(f"❌ Main script not found: {MAIN_SCRIPT}")
    sys.exit(1)

# === Step 2: Create Virtual Environment ===
if not os.path.exists(VENV_DIR):
    print("🚧 Creating virtual environment...")
    venv.create(VENV_DIR, with_pip=True)
    print("✅ Virtual environment created.")
else:
    print("✅ Virtual environment already exists.")

# === Step 3: Set Paths Based on Platform ===
if os.name == "nt":
    python_bin = os.path.join(VENV_DIR, "Scripts", "python.exe")
    pip_bin = os.path.join(VENV_DIR, "Scripts", "pip.exe")
else:
    python_bin = os.path.join(VENV_DIR, "bin", "python")
    pip_bin = os.path.join(VENV_DIR, "bin", "pip")

# Step 4: Install requirements in two phases
try:
    print(f"📦 Upgrading pip...")
    subprocess.check_call([pip_bin, "install", "--upgrade", "pip"])

    # Step 4a: install regular requirements first (excluding mdmm)
    print("📦 Installing core dependencies (excluding mdmm)...")
    with open(REQUIREMENTS, "r") as f:
        lines = f.readlines()

    mdmm_lines = [line for line in lines if "mdmm" in line and "git+" in line]
    base_lines = [line for line in lines if line not in mdmm_lines]

    # write temporary reduced requirements.txt
    with open("temp_requirements.txt", "w") as f:
        f.writelines(base_lines)

    subprocess.check_call([pip_bin, "install", "-r", "temp_requirements.txt"])
    os.remove("temp_requirements.txt")

    # Ensure wheel is installed first
    subprocess.check_call([pip_bin, "install", "wheel"])

    # Step 4b: install mdmm now that torch is present
    print("📦 Installing mdmm from GitHub...")
    for line in mdmm_lines:
        subprocess.check_call([pip_bin, "install", "--no-build-isolation", "--no-deps", line.strip()])

    print("✅ All dependencies installed.")

except subprocess.CalledProcessError:
    print("❌ Failed to install dependencies.")
    sys.exit(1)

# === Step 5: Run Main Script ===
try:
    print(f"🚀 Running {MAIN_SCRIPT}...")
    subprocess.check_call([python_bin, MAIN_SCRIPT])
    print("✅ Script finished.")
except subprocess.CalledProcessError:
    print("❌ Error running main script.")
    sys.exit(1)
