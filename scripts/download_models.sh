#!/bin/bash
# Download model weights for crowd density estimation

set -e

echo "Downloading model weights..."

# Create models directory if it doesn't exist
mkdir -p models

# YOLO11n weights
if [ ! -f "models/yolo11n.pt" ]; then
    echo "Downloading YOLO11n weights..."
    # Note: Update this URL when YOLO11 is officially released
    # For now, using YOLO8n as placeholder
    wget -O models/yolo11n.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
    echo "✓ YOLO11n downloaded"
else
    echo "✓ YOLO11n already exists"
fi

# YOLOv8n weights (backup)
if [ ! -f "models/yolov8n.pt" ]; then
    echo "Downloading YOLOv8n weights..."
    wget -O models/yolov8n.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
    echo "✓ YOLOv8n downloaded"
else
    echo "✓ YOLOv8n already exists"
fi

# CSRNet weights
echo ""
echo "CSRNet weights need to be trained or obtained separately."
echo "Please place csrnet_best.pth in the models/ directory."
echo ""
echo "To train CSRNet:"
echo "  1. Download ShanghaiTech dataset"
echo "  2. Run training script (see docs/training.md)"
echo ""
echo "Or download pre-trained weights from:"
echo "  https://github.com/leeyeehoo/CSRNet-pytorch"

echo ""
echo "Model download complete!"
echo "Note: CSRNet weights must be added manually."
