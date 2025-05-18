# Tech Context: Unified mhrpTools Pipeline

## Core Language
- Python 3.10+

## Key Libraries
- `gradio`: Web UI
- `argparse`: CLI
- `python-audio-separator`: Audio separation (ClapAnnotator)
- `transformers[torch]`: CLAP annotation
- `ffmpeg-python` and system `ffmpeg`: Resampling, normalization, audio manipulation
- `pydub`, `librosa`, `soundfile`, `numpy`: Audio processing
- `pyannote.audio`: Speaker diarization (WhisperBite)
- `openai-whisper`: Speech-to-text transcription (WhisperBite)
- `yaml`, `json`: Metadata output
- `logging`, `os`, `shutil`, `datetime`, `re`, `subprocess`: Standard utilities

## External Dependencies
- `ffmpeg` (system): Required for audio/video processing
- CUDA-compatible GPU recommended for performance (CPU supported)
- Hugging Face account and API token required for diarization

## Project Structure
- `unified_pipeline/`: Main orchestrator, data model, CLI, UI
- `ClapAnnotator/`, `preFapMix/`, `WhisperBite/`: Subproject logic, imported as modules
- `memory-bank/`: Documentation and project intelligence

## Configuration
- All relevant options for each stage are exposed in both CLI and Gradio UI, including advanced/edge-case parameters
- Show-edit mode is a workflow preset
- Output structure: Root output folder <show_name>_<timestamp>, each call in a numbered subfolder, all outputs/metadata centralized

## Development Setup
- Install Python dependencies from `requirements.txt` in each subproject and unified_pipeline
- Ensure `ffmpeg` is installed and in PATH
- Set up Hugging Face token for diarization
- Internet access required for model/audio downloads

## Execution
- CLI: `python -m unified_pipeline.cli --input ... --show-name ... --output-dir ...`
- Web UI: `python -m unified_pipeline.ui`

## Hardware
- CUDA GPU recommended, CPU supported

## Constraints
- No subprocess/CLI orchestration; all integration is via direct Python imports
- Modular, extensible architecture for future features and integrations
- Robust error handling and logging at every stage 