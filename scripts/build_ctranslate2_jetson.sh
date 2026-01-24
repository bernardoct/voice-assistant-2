#!/bin/bash
# Build CTranslate2 with CUDA support on Jetson Orin Nano
# Run this script on the Jetson

set -e

echo "=== CTranslate2 Build Script for Jetson Orin Nano ==="

# Find CUDA installation
CUDA_PATH=""
for cuda_dir in /usr/local/cuda-12.6 /usr/local/cuda-12 /usr/local/cuda; do
    if [ -d "$cuda_dir" ] && [ -f "$cuda_dir/bin/nvcc" ]; then
        CUDA_PATH="$cuda_dir"
        break
    fi
done

if [ -z "$CUDA_PATH" ]; then
    echo "ERROR: CUDA not found. Please install JetPack SDK."
    exit 1
fi

echo "Found CUDA at: $CUDA_PATH"
$CUDA_PATH/bin/nvcc --version

# Export CUDA paths for this script
export PATH="$CUDA_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$LD_LIBRARY_PATH"
export CUDA_HOME="$CUDA_PATH"

# Install build dependencies
echo ""
echo "=== Installing build dependencies ==="
sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    git \
    libopenblas-dev \
    python3-dev \
    python3-pip \
    pybind11-dev

# Clone CTranslate2
echo ""
echo "=== Cloning CTranslate2 ==="
cd ~
if [ -d "CTranslate2" ]; then
    echo "CTranslate2 directory exists, pulling latest..."
    cd CTranslate2
    git fetch --all
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
# Jetson Orin Nano uses SM 8.7 (Ampere architecture)
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DWITH_CUDA=ON \
    -DWITH_CUDNN=ON \
    -DWITH_MKL=OFF \
    -DWITH_OPENBLAS=ON \
    -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_PATH" \
    -DCMAKE_CUDA_COMPILER="$CUDA_PATH/bin/nvcc" \
    -DCUDA_ARCH_LIST="8.7" \
    -DCMAKE_INSTALL_PREFIX=/usr/local

# Build with all available cores
echo ""
echo "Building with $(nproc) cores... (this takes 15-30 minutes)"
make -j$(nproc)

# Install
echo ""
echo "=== Installing CTranslate2 ==="
sudo make install
sudo ldconfig

echo ""
echo "=== Build complete! ==="
echo ""
echo "Now install the Python bindings by running:"
echo "  cd ~/CTranslate2/python"
echo "  source ~/voice-assistant-2/venv/bin/activate"
echo "  pip install -r install_requirements.txt"
echo "  pip install ."
