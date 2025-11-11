#!/usr/bin/env bash
set -e

echo "=== 🚀 FLEET Auto Setup Script ==="

# -------------------------------
# Flags
# -------------------------------
FORCE_ALL=false
FORCE_STEPS=()
CLEANUP=false

show_help () {
    cat <<EOF
Usage: $0 [options]

Options:
  --force <steps>     Re-run specific steps even if already done.
                      <steps> can be a comma-separated list:
                      venv,reqs,torch,containernet,docker
                      or "all" to force everything.

                      Example:
                        $0 --force reqs,torch
                        $0 --force all

  --cleanup            Remove all generated files (virtualenv, docker images)
  --help, -h           Show this help message and exit

Typical usage:
  $0                # Run normally, skipping steps already done
  $0 --force all    # Force all steps to run again
  $0 --cleanup      # Wipe everything
EOF
}

if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    show_help
    exit 0
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)
            if [[ -z "$2" ]]; then
                echo "[!] --force requires an argument (e.g., 'reqs,torch' or 'all')"
                exit 1
            fi
            if [[ "$2" == "all" ]]; then
                FORCE_ALL=true
            else
                IFS=',' read -r -a FORCE_STEPS <<< "$2"
            fi
            shift
            ;;
        --cleanup) CLEANUP=true ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

# -------------------------------
# Cleanup
# -------------------------------
if $CLEANUP; then
    echo "🧹 Cleaning up everything..."
    rm -rf .venv
    sudo docker rmi -f fleet-fl fleet-bg 2>/dev/null || true
    echo "✅ Cleanup done."
    exit 0
fi

# -------------------------------
# Helper
# -------------------------------
confirm_step () {
    echo
    read -p "[*] $1 (press Enter to continue, Ctrl+C to abort)"
}

should_force () {
    local step="$1"
    if $FORCE_ALL; then return 0; fi
    for s in "${FORCE_STEPS[@]}"; do
        if [[ "$s" == "$step" ]]; then
            return 0
        fi
    done
    return 1
}

# -------------------------------
# 1. Check dependencies
# -------------------------------
echo "=== 🔍 Checking dependencies ==="

if command -v python3 >/dev/null 2>&1; then
    PY_VERSION=$(python3 -V 2>&1 | awk '{print $2}')
    if [[ "$(printf '%s\n' "3.10" "$PY_VERSION" | sort -V | head -n1)" != "3.10" ]]; then
        echo "[!] Python >= 3.10 required, found $PY_VERSION"
        exit 1
    fi
else
    echo "[!] Python3.10+ is required but not found."
    exit 1
fi

python3 -m venv --help >/dev/null 2>&1 || { echo "[!] Python venv module not available."; exit 1; }
command -v docker >/dev/null 2>&1 || { echo "[!] Docker is required but not installed."; exit 1; }
command -v ovs-vsctl >/dev/null 2>&1 || { echo "[!] Open vSwitch is required but not installed"; exit 1; }

echo "✅ All required dependencies are installed."

# -------------------------------
# 2. Virtual environment
# -------------------------------
if [[ ! -d ".venv" ]] || should_force venv; then
    confirm_step "Creating Python virtual environment"
    rm -rf .venv
    python3 -m venv .venv
    echo "[+] Virtual environment created."
else
    echo "[*] Virtual environment already exists, skipping."
fi
source .venv/bin/activate

# -------------------------------
# 3. Install Python dependencies
# -------------------------------
if [[ ! -f ".venv/.reqs_done" ]] || should_force reqs; then
    confirm_step "Installing Python dependencies (excluding torch)"
    if [ -f "requirements.txt" ]; then
        pip install --upgrade pip
        pip install -r requirements.txt
        touch .venv/.reqs_done
        echo "[+] Requirements installed."
    else
        echo "[!] requirements.txt not found, Aborting."
        exit 1
    fi
else
    echo "[*] Requirements already installed, skipping."
fi

# -------------------------------
# 4. Install Torch
# -------------------------------
if [[ ! -f ".venv/.torch_done" ]] || should_force torch; then
    confirm_step "Installing PyTorch (GPU if available, otherwise CPU)"
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "[+] NVIDIA GPU detected, installing CUDA-enabled torch"
        pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu118
        TORCH_BASE="--index-url https://download.pytorch.org/whl/cu118"
    else
        echo "[*] No GPU detected, installing CPU-only torch"
        pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cpu
        TORCH_BASE="--index-url https://download.pytorch.org/whl/cpu"
    fi
    touch .venv/.torch_done
else
    echo "[*] Torch already installed, skipping."
    TORCH_BASE="--index-url https://download.pytorch.org/whl/cpu"
fi

# -------------------------------
# 5. Install Containernet
# -------------------------------
if [[ ! -f ".venv/.containernet_done" ]] || should_force containernet; then
    confirm_step "Cloning and installing Containernet"
    git clone https://github.com/oabuhamdan/containernet.git /tmp/containernet
    cd /tmp/containernet
    pip install .
    cd -
    rm -rf /tmp/containernet
    touch .venv/.containernet_done
    echo "[+] Containernet installed."
else
    echo "[*] Containernet already installed, skipping."
fi

# -------------------------------
# 6. Build Docker images
# -------------------------------
confirm_step "Building Docker images for FL and BG nodes (needs sudo)"
if [[ -z $(sudo docker images -q fleet-fl) || -z $(sudo docker images -q fleet-bg) ]] || should_force docker; then
    sudo docker build \
        --build-arg TORCH_BASE="$TORCH_BASE" \
        -t fleet-fl -f static/docker/Dockerfile-FL .

    sudo docker build \
        --build-arg TORCH_BASE="$TORCH_BASE" \
        -t fleet-bg -f static/docker/Dockerfile-BG .
    echo "[+] Docker images built."
else
    echo "[*] Docker images already exist, skipping."
fi

# -------------------------------
# 7. Done
# -------------------------------
echo
echo "✅ FLEET setup completed successfully!"
echo "To start using FLEET run:"
echo "sudo .venv/bin/python3 main.py"
