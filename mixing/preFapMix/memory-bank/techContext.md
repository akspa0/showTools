# Technical Context

## Technologies Used
- **Python**: Core programming language
- **FFmpeg**: Audio processing and manipulation (summing, format conversion, etc., often called via `subprocess`).
- **audiomentations**: Python library for audio data augmentation and stem-specific loudness normalization/limiting.
- **audio-separator (CLI tool)**: For vocal and instrumental separation (invoked via `subprocess`).
- **soundfile**: Python library for reading and writing audio files.
- **JSON**: For workflow definitions (e.g., `default_audio_analysis_workflow.json`) and data interchange between stages.
- **numpy**: Numerical processing.
- **openai-whisper**: Python library for OpenAI's speech recognition model (used in `transcription_module.py`).
- **pyannote.audio**: Python library for speaker diarization (used in `diarization_module.py`).
- **lmstudio**: Python SDK (`lmstudio-python`) for interacting with local LLMs via LM Studio desktop application (used in `llm_module.py`).
- **torch**: Deep learning framework, a dependency for `openai-whisper` and `pyannote.audio`.
- **LLM Interaction**: Via local LM Studio server for text analysis and summarization (integrated in `llm_module.py`).
- **Gradio**: Potential for web-based user interface (longer-term goal).
- **pydub**: Audio manipulation (use to be minimized in favor of FFmpeg/audiomentations).
- **argparse**: Python library for command-line argument parsing (used in `workflow_executor.py`).

## Development Setup
- Python environment with required dependencies.
- FFmpeg installed and accessible in PATH.
- LM Studio Desktop application installed and running with desired models loaded/served for `llm_module.py`.
- Sufficient computing resources for audio processing, transcription, and local LLM inference.
- Optional GPU support for faster Whisper transcription and local LLM inference.

## Technical Constraints
- Must handle 8kHz audio files properly.
- FFmpeg must be installed on the system for audio processing.
- `llm_module.py` requires a running LM Studio instance with the configured model identifier available.
- Processing large audio files requires efficient memory usage.
- Metadata format must be consistent for show file generation.
- File naming must preserve chronological ordering.

## Dependencies
- Python 3.x
- FFmpeg (accessible in PATH)
- Required Python packages:
  - `audiomentations`
  - `soundfile`
  - `numpy`
  - `yt_dlp` (if URL input is used by workflow executor or a stage)
  - `openai-whisper`
  - `pyannote.audio`
  - `torch`
  - `lmstudio`
  - `audio-separator` (as a command-line tool, ensure it's installed and in PATH, also used as a library by ClapAnnotator)
  - `gradio` (if/when UI is developed)
  - `pydub` (current usage to be reviewed and minimized)
  - `argparse` (standard library, but noted for its use in `workflow_executor.py`)

## Execution
- The main entry point for running a full audio processing pipeline is `workflow_executor.py`.
- It is driven by command-line arguments parsed using `argparse`, allowing specification of:
    - Workflow JSON configuration file (e.g., `default_audio_analysis_workflow.json`)
    - Either a single `--input_audio_file` or an `--input_dir` for batch processing of all supported audio files (e.g., .wav, .mp3, .flac, .m4a) within that directory. These two input arguments are mutually exclusive.
    - Base output directory (e.g., `workflow_runs/`)
    - Log level
- If `--input_dir` is used, the executor iterates through each found audio file and runs the complete workflow for it.
- It manages the execution of different processing stages (modules) sequentially, passing outputs of one stage as inputs to the next based on the workflow definition.
- Outputs for each run are stored in a unique timestamped directory under the specified base output directory.

## Key Modules and their Roles
- **`audio_preprocessor.py`**: Handles initial audio preparation, including stem separation (vocals/instrumental) using `audio-separator` and loudness normalization using `audiomentations`.
- **`clap_module.py`**: Performs sound event detection using a CLAP (Contrastive Language-Audio Pretraining) model, likely via the `ClapAnnotator` external project.
- **`diarization_module.py`**: Conducts speaker diarization using `pyannote.audio` to identify who spoke when.
- **`transcription_module.py`**: Transcribes speech to text using `openai-whisper`, incorporating speaker information from the diarization stage.
- **`llm_module.py`**: Leverages a local Large Language Model (via LM Studio and the `lmstudio-python` SDK) for analysis, summarization, or other text-based tasks on the transcript and sound events.
- **`workflow_executor.py`**: Orchestrates the entire pipeline, calling the various modules in sequence as defined by a workflow JSON file. Handles input/output path resolution between stages.
- **`call_processor.py`**: (New) Processes the outputs of multiple `workflow_executor.py` runs. It identifies `recv_out` and `trans_out` pairs belonging to the same call, retrieves their processed outputs (like audio stems, transcripts), and then combines them (e.g., mixes audio, merges analyses) to produce a final output per call.

*(Note: Other utility modules or placeholder modules might exist but these are central to the core audio analysis pipeline.)* 