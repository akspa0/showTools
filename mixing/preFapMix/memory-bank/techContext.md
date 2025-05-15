# Technical Context

## Technologies Used
- **Python**: Core programming language
- **FFmpeg**: Audio processing and manipulation (summing, format conversion, etc., often called via `subprocess`).
- **audiomentations**: Python library for audio data augmentation and stem-specific loudness normalization/limiting.
- **audio-separator (CLI tool & library)**: For vocal and instrumental separation. Used as CLI by `audio_preprocessor.py` and as a library by `ClapAnnotator` (which is used by `clap_module.py`).
- **soundfile**: Python library for reading and writing audio files.
- **JSON**: For workflow definitions (e.g., `default_audio_analysis_workflow.json`) and data interchange between stages.
- **numpy**: Numerical processing.
- **openai-whisper**: Python library for OpenAI's speech recognition model (used in `transcription_module.py`).
- **pyannote.audio**: Python library for speaker diarization (used in `diarization_module.py`).
- **pyannote.database.util.RTTMParser**: Used in `transcription_module.py` for parsing RTTM files.
- **lmstudio-python**: Python SDK (`lmstudio-python`) for interacting with local LLMs via LM Studio desktop application (used in `llm_module.py`).
- **torch**: Deep learning framework, a dependency for `openai-whisper` and `pyannote.audio`.
- **LLM Interaction**: Via local LM Studio server for text analysis and summarization (integrated in `llm_module.py`).
- **Gradio**: Potential for web-based user interface (longer-term goal).
- **pydub**: Audio manipulation (used in `transcription_module.py` for slicing audio segments).
- **argparse**: Python library for command-line argument parsing (used in `workflow_executor.py` and `call_processor.py`).

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
  - `lmstudio-python`
  - `audio-separator` (as a command-line tool, ensure it's installed and in PATH, also used as a library by ClapAnnotator)
  - `gradio` (if/when UI is developed)
  - `pydub` (current usage to be reviewed and minimized)
  - `argparse` (standard library, but noted for its use in `workflow_executor.py`)

## Execution Flow
1. **`workflow_executor.py`** is the main entry point for processing individual audio streams.
    * Driven by CLI arguments: `--config_file`, `--input_dir` (or `--input_audio_file`), `--output_dir`, `--final_call_output_dir` (passed to `call_processor`), `--cp_llm_model_id` (passed to `call_processor`), `--log_level`.
    * Processes each audio file in `--input_dir` through a JSON-defined workflow (e.g., clap -> preprocess -> diarize -> transcribe -> llm_summary_per_stream).
    * Generates PII-safe run directories and passes `pii_safe_file_prefix` to modules.
    * If `--input_dir` is a "call folder" (contains `recv_out-*` or `trans_out-*` files), it invokes `call_processor.py` via `subprocess.run()` after all individual files in that directory are processed.
2. **`call_processor.py`** is invoked by `workflow_executor.py` for call folders.
    * Receives `--input_run_dir` (pointing to `workflow_executor.py`'s output directory for that run), `--output_call_dir`, and `--llm_model_id`.
    * Identifies pairs of `recv_out` and `trans_out` processed streams using PII-safe call IDs.
    * Aggregates data: mixes audio, merges transcripts, **(soon)** copies structured soundbite outputs from the transcription stage, and generates a combined LLM summary.
    * Saves final outputs to a PII-safe call-specific directory (e.g., `processed_calls_test/call_YYYYMMDD-HHMMSS/`).

## Key Modules and their Roles
- **`audio_preprocessor.py`**: Handles initial audio preparation, including stem separation (vocals/instrumental) using `audio-separator` and loudness normalization using `audiomentations`.
- **`clap_module.py`**: Performs sound event detection using a CLAP (Contrastive Language-Audio Pretraining) model, likely via the `ClapAnnotator` external project.
- **`diarization_module.py`**: Conducts speaker diarization using `pyannote.audio` to identify who spoke when.
- **`transcription_module.py`**: Transcribes speech to text using `openai-whisper`, incorporating speaker information from the diarization stage.
- **`llm_module.py`**: Leverages a local Large Language Model (via LM Studio and the `lmstudio-python` SDK) for analysis, summarization, or other text-based tasks on the transcript and sound events.
- **`workflow_executor.py`**: Orchestrates the entire pipeline, calling the various modules in sequence as defined by a workflow JSON file. Handles input/output path resolution between stages.
- **`call_processor.py`**: (New) Processes the outputs of multiple `workflow_executor.py` runs. It identifies `recv_out` and `trans_out` pairs belonging to the same call, retrieves their processed outputs (like audio stems, transcripts), and then combines them (e.g., mixes audio, merges analyses) to produce a final output per call.

*(Note: Other utility modules or placeholder modules might exist but these are central to the core audio analysis pipeline.)*

## Key Modules and Data Formats
- **`audio_preprocessor.py`**: Outputs PII-safe named `.wav` stems (e.g., `call_XYZ_vocals_normalized.wav`).
- **`diarization_module.py`**: Outputs PII-safe named `.rttm` files (e.g., `call_XYZ_vocals_normalized_diarization.rttm`).
- **`transcription_module.py`**:
    - Current: Outputs a main PII-safe named JSON transcript (e.g., `call_XYZ_transcription.json`) containing all speaker segments.
    - **Target**: Will also output individual soundbites as `.wav` and `.txt` files into speaker-specific subdirectories (e.g., `S0/0001_words.wav`, `S0/0001_words.txt`) within its stage output directory.
- **`llm_module.py`**: `run_llm_summary` outputs per-stream `.txt` summaries. `generate_llm_summary` (called by `call_processor`) outputs combined call `.txt` summaries.
- **PII-Safe Identifiers**: Typically `call_YYYYMMDD-HHMMSS` derived from original filenames, used as prefixes for files and parts of directory names to ensure privacy and organization. 