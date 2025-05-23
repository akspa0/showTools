# CLAP Audio Annotator

## Overview

This tool provides a user interface (built with Gradio) to annotate audio files using the CLAP (Contrastive Language-Audio Pretraining) model. It first separates the input audio into 'Vocals' and 'Instrumental' stems using a model from the `python-audio-separator` library, then processes each stem with CLAP based on user-provided text prompts.

This version focuses on processing single audio files.

## Features (Phase 1)

*   **Audio Separation**: Choose from available models (e.g., UVR-MDX-NET, Demucs) to separate vocals and instrumentals.
*   **CLAP Annotation**: Annotate separated audio stems based on text prompts.
*   **Chunking**: Processes long audio files efficiently by analyzing them in chunks.
*   **Prompt Presets**: Load predefined text prompts or save your own frequently used prompts to text files in the `_presets/clap_prompts/` directory.
*   **Configurable Threshold**: Adjust the confidence threshold for CLAP detections via a slider.
*   **Gradio UI**: Easy-to-use web interface for uploading audio, managing presets, selecting models, and viewing results.
*   **CLI Interface**: Command-line interface for processing single files or batch processing multiple files.
*   **Standardized Output**: Saves results in a structured JSON format within a unique, timestamped directory for each input file under `ClapAnnotator_Output/`.

## Prerequisites

*   **Python**: 3.10+
*   **ffmpeg**: You **must** have `ffmpeg` installed on your system and accessible in your system's PATH. This is required for audio resampling. You can download it from [https://ffmpeg.org/](https://ffmpeg.org/).
*   **(Optional) CUDA**: For GPU acceleration (recommended for performance), ensure you have a compatible NVIDIA GPU, CUDA Toolkit, and cuDNN installed. PyTorch should be installed with CUDA support (this is handled by the `transformers[torch]` dependency if CUDA is detected).

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd ClapAnnotator
    ```

2.  **Create a Python virtual environment (recommended):**
    ```bash
    python -m venv venv
    # Activate the environment
    # Windows:
    # venv\Scripts\activate
    # macOS/Linux:
    # source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Hugging Face Token:**
    
    You have two options for setting up your Hugging Face token:
    
    **Option 1: Using the setup script (Recommended)**
    
    Run the setup script with your token:
    ```bash
    python setup_hf_token.py --token "YOUR_HUGGING_FACE_TOKEN_HERE"
    ```
    
    Or use interactive mode:
    ```bash
    python setup_hf_token.py --interactive
    ```
    
    **Option 2: Manual setup**
    *   Create a file named `.env` in the project root directory (where `requirements.txt` is located).
    *   Add your Hugging Face API token to this file. You need a token (even a read-only one) to download the CLAP model. Get one from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
    *   The content of the `.env` file should be:
        ```
        HF_TOKEN="YOUR_HUGGING_FACE_TOKEN_HERE"
        ```
        Replace `YOUR_HUGGING_FACE_TOKEN_HERE` with your actual token.

## Configuration

*   **Environment Variables**: `HF_TOKEN` is loaded from the `.env` file.
*   **Application Settings**: Most other settings (model names, default thresholds, paths, chunk sizes) are defined in `config/settings.py`. You can modify this file directly if needed.

## Usage

### Gradio Web Interface

1.  Ensure your virtual environment is activated.
2.  Run the Gradio application:
    ```bash
    python gradio_app/app.py
    ```
3.  Open the displayed local URL (usually `http://127.0.0.1:7860`) in your web browser.
4.  Upload an audio file.
5.  Select an Audio Separation model.
6.  Choose a CLAP Prompt Preset from the dropdown or enter your own prompts (one per line) in the text box.
7.  (Optional) Adjust the CLAP Confidence Threshold slider.
8.  Click "Analyze Audio".
9.  View the status updates, the resulting JSON, and download the results file.

### Command Line Interface (CLI)

The CLI allows you to process audio files directly from the command line, with support for both single files and batch processing.

#### Basic Usage

Process a single audio file:

```bash
python cli.py path/to/audio.mp3 --prompts "voice,speech,singing"
```

Use a preset for prompts:

```bash
python cli.py path/to/audio.mp3 --preset my_preset_name
```

Use prompts from a text file (one prompt per line):

```bash
python cli.py path/to/audio.mp3 --prompt-file path/to/prompts.txt
```

#### Additional Options

Specify a separator model:

```bash
python cli.py path/to/audio.mp3 --prompts "voice,speech" --separator-model "UVR-MDX-NET Main"
```

Set a custom confidence threshold:

```bash
python cli.py path/to/audio.mp3 --prompts "voice,speech" --confidence 0.6
```

Specify a custom output directory:

```bash
python cli.py path/to/audio.mp3 --prompts "voice,speech" --output-dir path/to/output
```

Keep temporary files after processing:

```bash
python cli.py path/to/audio.mp3 --prompts "voice,speech" --keep-temp
```

#### Batch Processing

Process all audio files in a directory:

```bash
python cli.py path/to/audio/folder --batch --prompts "voice,speech"
```

#### Help

For a complete list of options:

```bash
python cli.py --help
```

## Presets

*   You can add your own CLAP prompt presets by creating `.txt` files in the `_presets/clap_prompts/` directory.
*   Each line in the `.txt` file should contain a single text prompt.
*   The filename (without the `.txt` extension) will be used as the preset name in the Gradio dropdown.
*   You can also save the prompts currently entered in the UI as a new preset using the "Save Current Prompts as Preset Name" input and the "Save Preset" button.

## Output

*   Results for each analysis run are saved in a unique subdirectory within the `ClapAnnotator_Output/` folder.
*   The subdirectory name is based on the sanitized input filename and a timestamp (e.g., `MyAudioFile_20231027_103000`).
*   Inside this directory, you will find the main `results.json` file containing the analysis.
*   Temporary files used during processing are created in a `temp/` subdirectory within the run's output folder and are deleted automatically upon completion or error.

## Future Work (Phase 2)

*   Implement batch processing by allowing folder inputs.
*   Develop a job queue system to manage multiple analyses.
*   Add UI controls to view, pause, and cancel queued jobs.

## License

(Placeholder - Consider adding an appropriate open-source license like MIT)

## Acknowledgements

*   This tool utilizes the excellent `python-audio-separator` library: [https://github.com/nomadkaraoke/python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator)
*   CLAP model and `transformers` library by Hugging Face and the original CLAP authors.
*   `ffmpeg` for audio processing.
*   `Gradio` for the user interface. 