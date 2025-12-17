# Crowd Density Estimation

A computer vision project for accurate crowd counting and density estimation using YOLO detection and CSRNet density maps.

## 🎯 Project Overview

This project implements multiple approaches for crowd density estimation:
- **YOLO + CSRNet Fusion**: Combines object detection with density estimation
- **Tiled YOLO Detection**: Handles high-resolution images with overlapping tiles
- **CSRNet Density Estimation**: Generates crowd density heatmaps

### Current Performance
- **Average Count**: 33.8 people per frame
- **Detection**: ~19 people (YOLO)
- **Density Estimation**: ~15 people (CSRNet)
- **Video**: 1280x720, 24fps

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/crowd-density-estimation.git
cd crowd-density-estimation

# Install dependencies
pip install -r requirements.txt

# Download model weights
bash scripts/download_models.sh
```

### Usage

#### 1. YOLO + CSRNet Fusion (Recommended)
```bash
python src/detection/fusion_count.py \
    --video data/videos/ppl.mp4 \
    --yolo models/yolo11n.pt \
    --csr-weights models/csrnet_best.pth \
    --output outputs/videos/result.mp4 \
    --csv outputs/csv/counts.csv
```

#### 2. Tiled YOLO Detection Only
```bash
python src/detection/yolo_tiled_count.py \
    --video data/videos/ppl.mp4 \
    --yolo models/yolo11n.pt \
    --output outputs/videos/yolo_result.mp4
```

#### 3. CSRNet Density Only
```bash
python src/detection/csrnet_count.py \
    --video data/videos/ppl.mp4 \
    --weights models/csrnet_best.pth \
    --output outputs/videos/density_result.mp4
```

## 🔬 Methodology

### YOLO + CSRNet Fusion
1. **YOLO Detection**: Detects visible people with bounding boxes
2. **CSRNet Density**: Estimates density in occluded/distant regions
3. **Fusion**: Masks density inside detected boxes, calibrates remaining density
4. **Result**: Combined count from both methods

### Key Features
- Multi-scale inference for better accuracy
- Temporal smoothing for video consistency
- Automatic calibration using detection confidence
- FP16 support for GPU memory efficiency

## 📊 Results

| Method | Detection | Density | Total | Accuracy |
|--------|-----------|---------|-------|----------|
| YOLO Only | 19.2 | - | 19.2 | Baseline |
| CSRNet Only | - | 33.8 | 33.8 | Baseline |
| **Fusion** | 19.2 | 14.6 | **33.8** | **Best** |

## 🛠️ Requirements

- Python 3.8+
- PyTorch 1.12+
- CUDA 11.0+ (for GPU acceleration)
- 6GB+ VRAM recommended

See `requirements.txt` for full dependencies.

## 📖 Documentation

- [Implementation Plan](docs/implementation_plan.md) - Detailed technical approach
- [Video Analysis](docs/video_analysis.md) - Performance analysis
- [Model Comparison](docs/model_comparison.md) - Benchmark results



## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


## 🙏 Acknowledgments

- **CSRNet**: [Paper](https://arxiv.org/abs/1802.10062) | [Original Implementation](https://github.com/leeyeehoo/CSRNet-pytorch)
- **YOLO**: [Ultralytics](https://github.com/ultralytics/ultralytics)


---

**Note**: Model weights are not included in the repository due to size. Run `scripts/download_models.sh` to download them automatically.
