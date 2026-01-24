#!/bin/bash
# Build CTranslate2 with CUDA support on Jetson Orin Nano
# Run this script on the Jetson

set -e

echo "=== CTranslate2 Build Script for Jetson Orin Nano ==="

# Check CUDA version
echo "Checking CUDA installation..."
if [ -f /usr/local/cuda/version.txt ]; then
    cat /usr/local/cuda/version.txt
elif command -v nvcc &> /dev/null; then
    nvcc --version
else
    echo "ERROR: CUDA not found. Please install CUDA toolkit."
    exit 1
fi

# Install build dependencies
echo ""
echo "=== Installing build dependencies ==="
sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    git \
    libopenblas-dev \
    libopenblas64-dev \
    libonednn-dev \
    python3-dev \
    python3-pip \
    pybind11-dev

# Upgrade pip and install Python build tools
echo ""
echo "=== Installing Python build tools ==="
pip install --upgrade pip setuptools wheel

# Clone CTranslate2
echo ""
echo "=== Cloning CTranslate2 ==="
cd ~
if [ -d "CTranslate2" ]; then
    echo "CTranslate2 directory exists, pulling latest..."
    cd CTranslate2
    git pull
else
    git clone --recursive https://github.com/OpenNMT/CTranslate2.git
    cd CTranslate2
fi

# Get the latest stable release
git fetch --tags
LATEST_TAG=$(git describe --tags $(git rev-list --tags --max-count=1))
echo "Using version: $LATEST_TAG"
git checkout $LATEST_TAG
git submodule update --init --recursive

# Create build directory
echo ""
echo "=== Building CTranslate2 ==="
rm -rf build
mkdir build
cd build

# Configure with CMake
# Note: Adjust CUDA_ARCH based on your Jetson model:
# - Jetson Orin Nano: compute_87 (Ampere)
# - Jetson Xavier NX: compute_72 (Volta)
# - Jetson Nano: compute_53 (Maxwell)

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DWITH_CUDA=ON \
    -DWITH_CUDNN=ON \
    -DWITH_MKL=OFF \
    -DWITH_OPENBLAS=ON \
    -DCUDA_ARCH_LIST="8.7" \
    -DCMAKE_INSTALL_PREFIX=/usr/local

# Build with all available cores
echo "Building with $(nproc) cores..."
make -j$(nproc)

# Install
echo ""
echo "=== Installing CTranslate2 ==="
sudo make install
sudo ldconfig

# Build Python bindings
echo ""
echo "=== Building Python bindings ==="
cd ../python

# Install in the virtual environment
if [ -n "$VIRTUAL_ENV" ]; then
    echo "Installing to virtual environment: $VIRTUAL_ENV"
    pip install -r install_requirements.txt
    pip install .
else
    echo "No virtual environment detected. Activate your venv first!"
    echo "Run: source ~/voice-assistant-2/venv/bin/activate"
    echo "Then run this script again."
    exit 1
fi

# Verify installation
echo ""
echo "=== Verifying installation ==="
python -c "import ctranslate2; print(f'CTranslate2 version: {ctranslate2.__version__}'); print(f'CUDA available: {\"cuda\" in ctranslate2.get_supported_compute_types(\"cuda\")}')"

echo ""
echo "=== Build complete! ==="
echo "You can now run the voice assistant server."
