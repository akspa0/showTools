# Technical Context

## Technologies
- Python 3.10+
- `python-audio-separator`
- Hugging Face `transformers[torch]`
- `gradio`
- `ffmpeg-python` (requires system `ffmpeg`)
- `python-dotenv`
- `numpy`
- `librosa`
- `soundfile`

## Project Structure
- Core modules: `audio_separation/`, `clap_annotation/`, `gradio_app/`
- Utilities: `utils/` (logging, file I/O, audio processing, presets)
- Configuration: `config/`, `.env`
- Presets: `_presets/clap_prompts/` (Stores user-defined CLAP prompt lists, e.g., as `.txt` or `.json` files)
- Outputs: Standardized outputs generated under `ClapAnnotator_Output/`
- CLI: `cli.py` (Command-line interface)

## Setup
- Install Python dependencies (e.g., `pip install -r requirements.txt`).
- **Install `ffmpeg`**: Ensure `ffmpeg` is installed on the system and available in the PATH.
- **Configuration**: Create a `.env` file (from `.env.example`) for `HF_TOKEN`.
- **Presets**: The `_presets/clap_prompts/` directory will be created if it doesn't exist. Users can add preset files here.

## Settings (`config/settings.py`)
Key parameters managed here include:
- `HF_TOKEN` (loaded from `.env`)
- `CLAP_MODEL_NAME = "laion/clap-htsat-fused"`
- `AUDIO_SEPARATOR_MODEL_DIR` (Path for downloaded separator models)
- `AUDIO_SEPARATOR_AVAILABLE_MODELS` (Dict defining available models for the UI)
- `TEMP_OUTPUT_DIR = "./temp_output/"`
- `BASE_OUTPUT_DIR = "ClapAnnotator_Output"` (Base for final results)
- `LOG_LEVEL = "INFO"`
- `CLAP_CHUNK_DURATION_S = 3` (Reduced from 10 seconds for better granularity)
- `CLAP_EXPECTED_SR = 48000`
- `DEFAULT_CLAP_CONFIDENCE_THRESHOLD = 0.55` (Increased from 0.45 based on testing)
- `CLAP_PRESETS_DIR = "_presets/clap_prompts/"`

## CLI Parameters
The command-line interface (`cli.py`) supports the following parameters:

```
usage: cli.py [-h] [--check-env] [--batch]
              [--separator-model {Mel Band RoFormer Vocals,...}]
              [--prompts PROMPTS | --preset {preset_name} | --prompt-file PROMPT_FILE]
              [--confidence CONFIDENCE] [--chunk-duration CHUNK_DURATION]
              [--output-dir OUTPUT_DIR] [--keep-temp] [--keep-audio] [--no-keep-audio]
              [input]
```

- `--check-env`: Check if the environment is properly configured
- `--batch`: Process all audio files in the input directory
- `--separator-model`: Audio separator model to use
- `--prompts`: Comma-separated list of CLAP prompts
- `--preset`: Name of CLAP prompt preset to use
- `--prompt-file`: Path to a text file containing prompts (one per line)
- `--confidence`: Confidence threshold for CLAP detections (default: 0.55)
- `--chunk-duration`: Duration in seconds for each CLAP analysis chunk (default: 3)
- `--output-dir`: Custom output directory
- `--keep-temp`: Keep temporary files after processing
- `--keep-audio`: Save separated audio files to output directory (default: True)
- `--no-keep-audio`: Don't save separated audio files, only keep JSON results 

## mhrpTools Integration
- **Dependencies:**
  - Python 3.10+
  - ClapAnnotator (as subprocess/module)
  - preFapMix (stereo mixing logic, imported or refactored as a module)
  - WhisperBite (as subprocess/module, with demucs step disabled)
- **Orchestration:**
  - mhrpTools will call each tool in sequence, passing outputs between them as files.
  - No changes to existing codebases; all integration is external.
- **Interfaces:**
  - CLI and Gradio UI for user interaction.
  - Modular pipeline for batch and single-file processing.
- **Output:**
  - Structured, timestamped directories for results, compatible with all downstream tools. 