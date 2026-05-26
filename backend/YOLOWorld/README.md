# YOLO-World + SAHI Detection Component

## 📁 Directory Structure

```
YOLOWorld/
├── README.md                    # Component overview and installation
├── QUICK_START.md              # Quick start guide
├── USAGE_GUIDE_CN.md           # Detailed usage guide (Chinese)
├── requirements.txt            # Python dependencies
├── install.sh                  # Installation script
├── yolo_world_detector.py      # Main detector implementation
├── test_yolo_world.py          # Test script
└── benchmark_detectors.py      # Performance comparison tool
```

## 🚀 Quick Start

### 1. Install
```bash
bash install.sh
```

### 2. Configure
Edit `backend/.env`:
```bash
DETECTION_MODEL=yolo_world
YOLO_WORLD_MODEL=yolo11l-world.pt
```

### 3. Test
```bash
python test_yolo_world.py --image ../uploads/test.jpg --prompt "person . car"
```

## 📚 Documentation

- **QUICK_START.md** - Get started in 3 steps
- **USAGE_GUIDE_CN.md** - Complete guide with troubleshooting (中文)
- **requirements.txt** - Package dependencies

## 🔧 Scripts

### test_yolo_world.py
Test the detector on a single image:
```bash
python test_yolo_world.py --image path/to/image.jpg --prompt "person . car . dog"
```

### benchmark_detectors.py
Compare performance with other detectors:
```bash
python benchmark_detectors.py \
    --image path/to/image.jpg \
    --prompt "person . car" \
    --detectors grounding_dino yolo_world \
    --iterations 5
```

### install.sh
Automated installation of dependencies:
```bash
bash install.sh
```

## 🎯 Features

- **Open-vocabulary detection** - Detect any object described in text
- **SAHI integration** - Sliced inference for small objects
- **Adaptive slicing** - Automatically adjusts based on image size
- **Multiple model sizes** - Choose speed vs accuracy tradeoff
- **Easy integration** - Drop-in replacement for existing detectors

## 📦 Models

| Model | Size | Speed | Accuracy | Recommended For |
|-------|------|-------|----------|-----------------|
| yolo11s-world.pt | ~40MB | ⚡⚡⚡ | ⭐⭐ | Real-time, edge devices |
| yolo11m-world.pt | ~80MB | ⚡⚡ | ⭐⭐⭐ | Balanced performance |
| yolo11l-world.pt | ~120MB | ⚡ | ⭐⭐⭐⭐ | High accuracy tasks |

Models are auto-downloaded to `~/.cache/ultralytics/` on first use.

## 🔄 Integration

The YOLO-World detector integrates seamlessly with the existing pipeline:

```python
# In detector.py, the factory automatically loads YOLO-World
# when DETECTION_MODEL=yolo_world

from app.services.detector import get_detector

detector = get_detector()  # Returns YOLOWorldDetector
detections = detector.predict(image, prompt, box_threshold, text_threshold)
```

## ⚙️ Configuration

All configuration is done via environment variables in `.env`:

```bash
# Model selection
YOLO_WORLD_MODEL=yolo11l-world.pt

# SAHI parameters
SAHI_SLICE_HEIGHT=640
SAHI_SLICE_WIDTH=640
SAHI_OVERLAP_HEIGHT_RATIO=0.2
SAHI_OVERLAP_WIDTH_RATIO=0.2

# Detection thresholds
BOX_THRESHOLD=0.25
TEXT_THRESHOLD=0.25

# Device
DEVICE=cuda:0
```

## 🐛 Troubleshooting

### Installation Issues
```bash
# If install.sh fails, try manual installation:
pip install ultralytics>=8.3.0
pip install sahi>=0.11.18
```

### GPU Memory Issues
```bash
# Use smaller model
YOLO_WORLD_MODEL=yolo11m-world.pt

# Increase slice size
SAHI_SLICE_HEIGHT=800

# Or use CPU
DEVICE=cpu
```

### No Detections
```bash
# Lower threshold
BOX_THRESHOLD=0.15

# Check prompt format (use dots)
prompt="person . car . dog"  # ✓ Correct
prompt="person, car, dog"    # ✗ Wrong
```

## 📊 Performance

Typical performance on RTX 3090 (1920x1080 image):

| Detector | Load Time | Inference | FPS | Small Objects |
|----------|-----------|-----------|-----|---------------|
| Grounding DINO | ~8s | ~0.8s | 1.25 | Good |
| YOLO-World (L) | ~3s | ~0.3s | 3.33 | Excellent (SAHI) |
| YOLO-World (M) | ~2s | ~0.2s | 5.00 | Excellent (SAHI) |
| YOLO-World (S) | ~1s | ~0.1s | 10.00 | Very Good (SAHI) |

*Note: SAHI adds overhead but significantly improves small object detection*

## 🔗 References

- [YOLO-World Paper](https://arxiv.org/abs/2401.17270)
- [Ultralytics Documentation](https://docs.ultralytics.com/)
- [SAHI GitHub](https://github.com/obss/sahi)
- [YOLO-World GitHub](https://github.com/AILab-CVC/YOLO-World)

## 📝 License

This component uses:
- Ultralytics YOLO (AGPL-3.0)
- SAHI (MIT)

## 🤝 Contributing

To add new features or fix bugs:
1. Modify `yolo_world_detector.py`
2. Test with `test_yolo_world.py`
3. Update documentation
4. Run benchmark to verify performance

## 💡 Tips

1. **For small objects**: Use SAHI with smaller slices (512x512)
2. **For speed**: Use yolo11s-world with larger slices (800x800)
3. **For accuracy**: Use yolo11l-world with more overlap (0.3)
4. **For memory**: Increase slice size or use smaller model
5. **For debugging**: Check logs in `backend/logs/`
