# Environment Setup Guide

## Setting up a Python Environment for CLAP Annotator

### 1. Create a new Python virtual environment

#### Using venv (recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

#### Using conda
```bash
conda create -n clap-annotator python=3.10
conda activate clap-annotator
```

### 2. Install dependencies

First, install the core dependencies:

```bash
pip install numpy>=2.0.0 scipy>=1.12.0 scikit-learn>=1.4.0
pip install torch>=2.0.0 transformers[torch]>=4.30.0
```

Then install python-audio-separator directly from GitHub:

```bash
pip install git+https://github.com/nomadkaraoke/python-audio-separator.git
```

Finally, install the remaining dependencies:

```bash
pip install python-dotenv>=0.19.0 gradio>=3.32.0 ffmpeg-python>=0.2.0 librosa>=0.10.0 soundfile>=0.12.1
```

Alternatively, you can install all dependencies at once using the requirements.txt file:

```bash
pip install -r requirements.txt
```

### 3. Set up Hugging Face credentials

You have two options for setting up your Hugging Face credentials:

#### Option 1: Using huggingface-cli (Recommended)

1. Install the Hugging Face Hub package if not already installed:
   ```bash
   pip install huggingface_hub
   ```

2. Log in to your Hugging Face account:
   ```bash
   huggingface-cli login
   ```
   
3. When prompted, enter your Hugging Face token. You can get one from https://huggingface.co/settings/tokens

This will automatically set up the necessary environment variables and store your token securely.

#### Option 2: Creating a .env file

Create a file named `.env` in the project root with the following content:

```
HF_TOKEN="your_huggingface_token_here"
LOG_LEVEL="INFO"
```

Replace `your_huggingface_token_here` with your actual Hugging Face token from https://huggingface.co/settings/tokens

### 4. Test the environment

```bash
python cli.py --check-env
```

### 5. Run the CLI with a test file

```bash
python cli.py test_audio/2014-06-19-aculo.mp3 --prompts "voice,music"
```

## Troubleshooting

### Python-Audio-Separator Installation Issues

If you encounter issues installing python-audio-separator from GitHub:

1. Make sure you have Git installed on your system.

2. Try installing the dependencies required by python-audio-separator first:
   ```bash
   pip install torch onnx onnxruntime numpy>=2.0.0 librosa requests six tqdm pydub
   ```

3. Then try installing python-audio-separator again:
   ```bash
   pip install git+https://github.com/nomadkaraoke/python-audio-separator.git
   ```

4. If you're still having issues, try cloning the repository and installing it manually:
   ```bash
   git clone https://github.com/nomadkaraoke/python-audio-separator.git
   cd python-audio-separator
   pip install -e .
   ```

### NumPy/SciPy Compatibility Issues

If you encounter errors related to NumPy or SciPy:

```bash
pip uninstall numpy scipy scikit-learn -y
pip install numpy>=2.0.0 scipy>=1.12.0 scikit-learn>=1.4.0
```

### FFmpeg Missing

Make sure FFmpeg is installed on your system and available in your PATH:

- Windows: Download from https://ffmpeg.org/download.html and add to PATH
- Linux: `sudo apt-get install ffmpeg` (Ubuntu/Debian) or `sudo yum install ffmpeg` (CentOS/RHEL)
- Mac: `brew install ffmpeg` (requires Homebrew)

### Transformers Compatibility Issues

If you encounter issues with the transformers library and NumPy 2.0+:

```bash
pip install transformers==4.38.0
```

This specific version of transformers is known to work better with NumPy 2.0+. 