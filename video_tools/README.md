# Video and Image Processing Tools

A collection of GPU-accelerated tools for extracting, analyzing, and organizing high-quality frames from videos and images using aesthetic scoring.

## Tools

### 1. Video Sampler (videoSampler0x08.py)

A GPU-accelerated video frame extraction tool that samples and aesthetically scores frames from videos.

#### Features
- GPU-accelerated frame analysis using PyTorch
- Aesthetic scoring based on multiple criteria:
  - Brightness analysis
  - Color diversity
  - Edge detection
  - Blur detection
  - Deep learning feature extraction
- Year-based organization
- Parallel processing
- Handles special characters in filenames
- Memory-efficient batch processing

#### Usage
```bash
python videoSampler0x08.py --input_dir /path/to/videos --output_dir /path/to/output --frames_per_video 30 --aesthetic_threshold 0.75
```

#### Parameters
- `--input_dir`: Directory containing video files
- `--output_dir`: Output directory for extracted frames
- `--frames_per_video`: Number of frames to sample per video (default: 10)
- `--aesthetic_threshold`: Minimum score for frame selection (0.0 to 1.0, default: 0.75)

### 2. Image Scorer (imageScorer0x03.py)

A tool for aesthetically scoring and organizing existing image collections.

#### Features
- GPU-accelerated image analysis
- Flexible organization options:
  - Year-based organization
  - Flat directory output
  - Sequential naming
- File limit controls:
  - Total file limit
  - Per-directory limit
- Detailed statistics

#### Usage Examples

1. Basic scoring and organization:
```bash
python imageScorer0x03.py --input_dir /path/to/images --output_dir /path/to/output --aesthetic_threshold 0.85
```

2. With file limits:
```bash
python imageScorer0x03.py --input_dir /path/to/images --output_dir /path/to/output --aesthetic_threshold 0.85 --max_files_total 999 --max_files_per_dir 25
```

3. Flat directory with sequential naming:
```bash
python imageScorer0x03.py --input_dir /path/to/images --output_dir /path/to/output --aesthetic_threshold 0.85 --flat_dir --sequential_naming
```

#### Parameters
- `--input_dir`: Directory containing images
- `--output_dir`: Output directory for organized images
- `--aesthetic_threshold`: Minimum score for image selection (0.0 to 1.0, default: 0.75)
- `--sequential_naming`: Use sequential numbering (e.g., 000001.jpg)
- `--flat_dir`: Store all outputs in a single directory
- `--max_files_total`: Maximum total number of output files
- `--max_files_per_dir`: Maximum files to process from each input directory

## Installation

1. Requirements:
```bash
pip install torch torchvision opencv-python pillow tqdm numpy scikit-image
```

2. CUDA Support (Optional but recommended):
- Install CUDA Toolkit from NVIDIA website
- Install PyTorch with CUDA support:
```bash
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu117
```

## Aesthetic Scoring System

The tools use a sophisticated scoring system that combines multiple factors:

1. Brightness Score (20%)
   - Analyzes image brightness distribution
   - Penalizes too dark or too bright images
   - Optimal around middle brightness

2. Color Diversity (20%)
   - Measures color histogram entropy
   - Higher scores for images with diverse color palettes
   - Uses HSV color space for better analysis

3. Edge Detection (20%)
   - Evaluates image detail and composition
   - Uses Sobel filters for edge detection
   - Higher scores for images with clear subjects/features

4. Blur Detection (20%)
   - Measures image sharpness
   - Uses Laplacian variance
   - Penalizes blurry or out-of-focus images

5. Deep Learning Features (20%)
   - Uses ResNet18 for feature extraction
   - Evaluates high-level image characteristics
   - Trained on millions of images

## Output Organization

### Video Sampler Output Structure
```
output_dir/
├── all_frames/
│   ├── 1987/
│   ├── 2002/
│   └── ...
└── 0.75_aesthetic/
    ├── 1987/
    ├── 2002/
    └── ...
```

### Image Scorer Output Structure (with --flat_dir)
```
output_dir/
└── 0.75_aesthetic_flat/
    ├── 000001.jpg
    ├── 000002.jpg
    └── ...
```

### Image Scorer Output Structure (without --flat_dir)
```
output_dir/
└── 0.75_aesthetic/
    ├── 1987/
    │   ├── score_0.892_image1.jpg
    │   └── score_0.873_image2.jpg
    ├── 2002/
    │   └── score_0.901_image3.jpg
    └── unknown/
        └── score_0.881_image4.jpg
```

## Tips

1. Choosing Aesthetic Threshold:
   - 0.75-0.80: More inclusive selection
   - 0.80-0.85: Moderate selection
   - 0.85-0.90: More selective
   - 0.90+: Very selective, only the highest quality frames

2. Optimizing Performance:
   - Use GPU acceleration when available
   - Adjust batch size based on available memory
   - Use appropriate file limits for manageable outputs

3. Organization Strategy:
   - Use --flat_dir for easier sequential browsing
   - Use year-based organization for historical collections
   - Use --sequential_naming for clean, numbered outputs

## Notes

- The tools automatically use GPU acceleration if available
- Progress bars show processing status
- Detailed statistics are provided after processing
- All operations maintain original files unchanged
- Error handling ensures graceful failure recovery