# BG-Remover: Intelligent Background Removal Tool

A powerful background removal and image segmentation tool powered by **BiRefNet** (Bilateral Reference for High-Resolution Dichotomous Image Segmentation), a state-of-the-art AI model designed for precise object extraction and background removal.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Configuration](#configuration)
- [API & Components](#api--components)
- [Requirements](#requirements)
- [Model Information](#model-information)
- [Output Format](#output-format)
- [Troubleshooting](#troubleshooting)

---

## 📌 Overview

**BG-Remover** is an automated image processing pipeline that intelligently removes backgrounds from images using the BiRefNet neural network. The tool performs:

1. **Background Detection**: Uses AI to identify and segment the foreground object
2. **Alpha Channel Creation**: Generates precise transparency masks
3. **Auto-Crop**: Automatically crops images to object boundaries with configurable padding
4. **Portrait Formatting**: Converts images to portrait ratio (4:5) with intelligent canvas sizing
5. **Batch Processing**: Processes multiple images efficiently with GPU acceleration

## ✨ Features

- **AI-Powered Segmentation**: State-of-the-art BiRefNet model for accurate background removal
- **CUDA GPU Support**: Automatic GPU detection and acceleration for faster processing
- **Batch Processing**: Process entire folders of images simultaneously
- **Auto-Crop with Padding**: Intelligent object detection with configurable internal padding
- **Portrait Ratio Conversion**: Automatically scales images to 4:5 portrait format
- **Multiple Image Formats**: Supports PNG, JPG, JPEG, WebP input formats
- **Memory Optimization**: Automatic cleanup and GPU cache management
- **Transparent Output**: RGBA PNG format with preserved transparency

---

## 📂 Project Structure

```
BG-Remover/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── server.py                         # Main batch processing pipeline
├── birefnet_repo.py                  # Model downloader from HuggingFace
├── test.py                           # System information and dependency checker
├── inputs/                           # Input images directory (place images here)
├── outputs/                          # Processed images directory (results saved here)
│
└── BiRefNet/                         # BiRefNet model directory
    ├── BiRefNet_config.py            # Model configuration class
    ├── birefnet.py                   # BiRefNet model implementation (2249 lines)
    ├── handler.py                    # Image preprocessing and inference handler
    ├── config.json                   # Model metadata and configuration
    ├── model.safetensors             # Pre-trained model weights (~335MB)
    └── README.md                     # Original BiRefNet documentation
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA 11.7 (optional, for GPU acceleration)
- 8GB+ RAM for CPU inference, 4GB+ VRAM for GPU
- ~350MB disk space for model weights

### Step 1: Clone/Setup Project

```bash
cd BG-Remover
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Download BiRefNet Model

The model is automatically downloaded when needed, but you can pre-download it:

```bash
python birefnet_repo.py
```

This will download the BiRefNet model and dependencies to the `BiRefNet/` directory.

### Step 4: Verify Installation

```bash
python test.py
```

You should see output confirming your Python environment, CUDA availability, and installed package versions.

---

## 📖 Quick Start

### Basic Usage

1. **Place images** in the `inputs/` directory:
   ```bash
   cp /path/to/your/image.png inputs/
   ```

2. **Run the processor**:
   ```bash
   python server.py
   ```

3. **Retrieve results** from the `outputs/` directory

The script will:
- Automatically detect all image files in `inputs/`
- Process each image with the BiRefNet model
- Apply automatic cropping and portrait formatting
- Save processed images as PNG files in `outputs/`

---

## 🔧 Usage

### Main Processing Pipeline (`server.py`)

The main script processes all images in bulk mode with the following workflow:

```
Input Image → RGB Conversion → AI Segmentation → Mask Creation 
→ Alpha Channel Application → Auto-Crop → Portrait Formatting → PNG Output
```

#### Configuration Parameters (in `server.py`):

```python
INPUT_DIR = "./inputs"              # Source directory for images
OUTPUT_DIR = "./outputs"            # Destination directory for processed images
LOCAL_MODEL_PATH = "./BiRefNet"     # Path to model files
TARGET_RATIO = 0.8                  # Target aspect ratio (width/height = 4:5)
INTERNAL_PADDING = 0.10             # Internal padding as % of object size (10%)
device = "cuda" or "cpu"            # Auto-detected based on availability
```

#### Processing Function: `process_to_portrait_with_crop()`

This core function performs:

1. **Bounding Box Detection**: Finds the tightest box around non-transparent pixels
2. **Padding Calculation**: Adds internal padding around the object (default 10%)
3. **Aspect Ratio Calculation**: Determines portrait canvas size (4:5 ratio)
4. **Canvas Creation**: Creates final RGBA canvas with transparent backdrop
5. **Centered Placement**: Places padded object in center of canvas

**Parameters:**
- `img`: PIL Image with alpha channel (RGBA)
- `target_ratio`: Target width/height ratio (default 0.8 for 4:5 portrait)
- `padding_pct`: Percentage of object size to pad (default 0.10 for 10%)

**Returns:** PIL Image in RGBA format ready for output

### Advanced Usage

#### Modify Target Ratio

For different aspect ratios, edit `server.py`:

```python
# For square format (1:1 ratio):
TARGET_RATIO = 1.0

# For 16:9 landscape:
TARGET_RATIO = 16/9

# For 3:4 portrait:
TARGET_RATIO = 0.75
```

#### Adjust Padding

```python
INTERNAL_PADDING = 0.15  # 15% padding around object
```

#### CPU-Only Mode

The script auto-detects GPU, but to force CPU:

```python
device = "cpu"  # Remove cuda detection line
```

---

## 📋 Configuration

### BiRefNet Model Configuration (`BiRefNet/config.json`)

The model uses `transformers` library configuration:

```json
{
  "architectures": ["BiRefNet"],
  "auto_map": {
    "AutoConfig": "BiRefNet_config.BiRefNetConfig",
    "AutoModelForImageSegmentation": "birefnet.BiRefNet"
  }
}
```

### Image Preprocessing (`BiRefNet/handler.py`)

Handles image normalization using ImageNet standards:

```python
transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
```

---

## 🔌 API & Components

### Core Components

#### 1. **Model Loading (`server.py` lines 18-24)**
```python
model = AutoModelForImageSegmentation.from_pretrained(
    LOCAL_MODEL_PATH,
    trust_remote_code=True,
    local_files_only=True
).to(device).eval()
```

#### 2. **Image Inference (`server.py` lines 82-88)**
```python
input_tensor = transform_image(input_image).unsqueeze(0).to(device)
with torch.cuda.amp.autocast():
    preds = model(input_tensor)[-1].sigmoid().cpu()
```

#### 3. **Mask Application (`server.py` lines 89-92)**
```python
mask = transforms.ToPILImage()(preds[0].squeeze())
mask = mask.resize(input_image.size, Image.BILINEAR)
input_image.putalpha(mask)
```

#### 4. **Batch Processing (`server.py` lines 76-98)**
- Iterates through all images in `INPUT_DIR`
- Shows progress with `tqdm` progress bar
- Automatically handles memory cleanup with `torch.cuda.empty_cache()`

### Handler Module (`BiRefNet/handler.py`)

Provides alternative processing with advanced foreground refinement:

- **ImagePreprocessor Class**: Handles consistent image normalization
- **Foreground Refinement**: `refine_foreground()` function for enhanced edge quality
- **Multiple Model Variants**: Support for different model weights (General, HR, Lite, Portrait, etc.)

---

## 📦 Requirements

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| torch | 2.0.1 | Deep learning framework |
| torchvision | - | Image processing utilities |
| transformers | - | HuggingFace model loading |
| Pillow | 12.1.0 | Image I/O and manipulation |
| huggingface_hub | 0.36.2 | Model downloading |
| numpy | 1.26.4 | Numerical computing |
| opencv-python | 4.8.1.78 | CV operations for refinement |
| safetensors | 0.7.0 | Model format support |
| tqdm | - | Progress bars |

### CUDA Components (GPU acceleration)

- nvidia-cuda-runtime-cu11 11.7.99
- nvidia-cudnn-cu11 8.5.0.96
- nvidia-cublas-cu11 11.10.3.66
- onnxruntime-gpu 1.16.3

See [requirements.txt](requirements.txt) for complete dependencies.

---

## 🧠 Model Information

### BiRefNet

**Paper**: "Bilateral Reference for High-Resolution Dichotomous Image Segmentation"

**Authors**: Peng Zheng, Dehong Gao, Deng-Ping Fan, and others

**Publication**: CAAI AIR 2024

**Repository**: https://github.com/ZhengPeng7/BiRefNet

### Model Capabilities

- **Task**: Dichotomous Image Segmentation (object vs. background)
- **Input Resolution**: 1024×1024 pixels (configurable)
- **Output**: Binary segmentation mask (0-255 alpha channel)
- **Accuracy**: High-precision object detection suitable for:
  - Portrait extraction
  - Product photography
  - Document scanning
  - Professional image editing

### Available Model Variants

The framework supports multiple specialized models:

- `General` (default): Universal object segmentation
- `General-HR`: High-resolution variant (2048×2048)
- `General-Lite`: Lightweight, faster inference
- `Portrait`: Optimized for portrait/face extraction
- `Matting`: Advanced matting for hair/fur edges
- `COD`: Camouflaged Object Detection
- `HRSOD`: High-Resolution Salient Object Detection
- `DIS5K`: Trained on DIS5K dataset

---

## 📸 Output Format

### Image Specifications

| Property | Value |
|----------|-------|
| Format | PNG (RGBA) |
| Channels | 4 (Red, Green, Blue, Alpha) |
| Color Space | RGB |
| Transparency | Yes (Alpha channel) |
| Naming | Original filename preserved |
| Location | `./outputs/` directory |

### Example Output

**Input**: `product.jpg` (1080×1080, opaque background)

**Output**: `product.png` (1280×1600, transparent background)
- Automatically cropped to object
- 10% internal padding applied
- Scaled to 4:5 portrait ratio
- PNG with transparency

---

## 🔍 Troubleshooting

### Issue: "CUDA out of memory"

**Solution**: Reduce batch size or use CPU mode
```python
device = "cpu"  # Force CPU in server.py
```

### Issue: "FileNotFoundError: model not found"

**Solution**: Download the model first
```bash
python birefnet_repo.py
```

### Issue: Slow Processing

**Check GPU availability**:
```bash
python test.py
```

If CUDA not found, install appropriate CUDA drivers for your GPU.

### Issue: Poor Edge Quality

**Enable foreground refinement** in `BiRefNet/handler.py`:
```python
result = refine_foreground(result, mask, r=90)
```

### Issue: Images Not Processing

**Verify input** folder:
```bash
ls -la inputs/
# Should contain .png, .jpg, .jpeg, or .webp files
```

**Check output** folder permissions:
```bash
chmod 755 outputs/
```

---

## 📝 Notes

- **First Run**: Initial run downloads and caches the BiRefNet model (~335MB)
- **GPU Memory**: Recommended 4GB+ VRAM for optimal performance
- **Processing Speed**: ~0.5-2 seconds per image on modern GPUs
- **Accuracy**: BiRefNet achieves state-of-the-art accuracy on DIS5K and other benchmarks

---

## 📄 License

This project uses BiRefNet, which is licensed under the MIT License. See [BiRefNet/README.md](BiRefNet/README.md) for original model license and citations.

---

## 🔗 Related Resources

- **HuggingFace Model Card**: https://huggingface.co/zhengpeng7/BiRefNet
- **Paper**: https://arxiv.org/pdf/2401.03407
- **Official Repository**: https://github.com/ZhengPeng7/BiRefNet
- **HuggingFace Demo**: https://huggingface.co/spaces/ZhengPeng7/BiRefNet_demo

---

## 📞 Support

For issues or questions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review the test output: `python test.py`
3. Check BiRefNet original repository for model-specific issues
4. Ensure all dependencies are correctly installed: `pip install -r requirements.txt`

---

**Last Updated**: February 2026  
**Version**: 1.0  
**Status**: Functional & Ready for Production
