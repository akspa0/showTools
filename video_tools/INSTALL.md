# Installation Instructions

## Prerequisites

1. NVIDIA GPU with CUDA support
2. Python 3.8 or newer
3. NVIDIA GPU Drivers
4. CUDA Toolkit

## Step-by-Step Setup

1. Install NVIDIA GPU Drivers
   - Visit: https://www.nvidia.com/download/index.aspx
   - Select your GPU model and download the latest driver
   - Install the driver

2. Install CUDA Toolkit
   - Visit: https://developer.nvidia.com/cuda-downloads
   - Download CUDA 11.8 (recommended for latest PyTorch)
   - Follow the installation instructions for your OS

3. Create and activate a Python virtual environment (recommended):
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

4. Install dependencies:
```bash
# Install PyTorch and other dependencies
pip install -r requirements.txt
```

5. Verify GPU Support:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

## Troubleshooting

1. If CUDA is not available:
   - Ensure NVIDIA drivers are properly installed
   - Verify CUDA Toolkit installation
   - Try reinstalling PyTorch with specific CUDA version:
     ```bash
     pip uninstall torch torchvision torchaudio
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
     ```

2. For older GPUs:
   - Use CUDA 11.7 or 11.6 if your GPU doesn't support CUDA 11.8
   - Modify the installation command:
     ```bash
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
     ```

3. Common Issues:
   - "CUDA not available" - Check NVIDIA driver installation
   - "CUDA version mismatch" - Ensure CUDA Toolkit version matches PyTorch CUDA version
   - "Out of memory" - Reduce batch_size in the script (default is 32)

## System Requirements

Minimum:
- NVIDIA GPU with 4GB VRAM
- 8GB System RAM
- CUDA-compatible GPU (Compute Capability 3.5 or higher)

Recommended:
- NVIDIA GPU with 8GB+ VRAM
- 16GB System RAM
- Recent NVIDIA GPU (RTX 2000 series or newer)

## Performance Notes

1. GPU Memory Usage:
   - Default batch size (32) requires ~4GB VRAM
   - Reduce batch_size if you encounter memory errors:
     ```bash
     # Example with smaller batch size
     python imageScorer0x04.py --input_dir ... --batch_size 16
     ```

2. Processing Speed:
   - GPU acceleration typically provides 10-20x speedup
   - Performance scales with GPU capability
   - RTX series GPUs will see additional speedup from Tensor Cores

## Quick Test

After installation, run this test script to verify GPU support:

```python
import torch
import torchvision

# Check CUDA availability
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
```

If everything is set up correctly, you should see your GPU details and available memory.