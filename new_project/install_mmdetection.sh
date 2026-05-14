#!/bin/bash
# Install MMDetection and dependencies for DSNet+DINO training

set -e

echo "======================================================================"
echo "Installing MMDetection for DSNet+DINO"
echo "======================================================================"

# Activate environment
ENV_PATH="/home/as19197/new_project/wunet_yolo_env"

echo ""
echo "[1/4] Installing MMCV..."
$ENV_PATH/bin/pip install -U openmim
$ENV_PATH/bin/mim install mmcv-full

echo ""
echo "[2/4] Installing MMDetection..."
$ENV_PATH/bin/pip install mmdet

echo ""
echo "[3/4] Installing additional dependencies..."
$ENV_PATH/bin/pip install pycocotools

echo ""
echo "[4/4] Verifying installation..."
$ENV_PATH/bin/python -c "import mmdet; print('MMDetection version:', mmdet.__version__)"
$ENV_PATH/bin/python -c "import mmcv; print('MMCV version:', mmcv.__version__)"

echo ""
echo "======================================================================"
echo "✓ MMDetection installed successfully!"
echo "======================================================================"
