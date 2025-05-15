# Technical Context

## Technologies Used
- **Python**: Core programming language
- **FFmpeg**: Audio processing and manipulation (summing, final mix loudnorm, format conversion).
- **audiomentations**: Python library for audio data augmentation and stem-specific loudness normalization/limiting.
- **Gradio**: Web-based user interface
- **Whisper**: OpenAI's speech recognition model (replacing fap)
- **python-audio-separator**: Vocal and instrumental separation
- **numpy**: Numerical processing
- **pydub**: Audio manipulation (to be reduced in favor of FFmpeg)
- **json/yaml**: For metadata storage and retrieval

## Development Setup
- Python environment with required dependencies
- FFmpeg installed and accessible in PATH
- Sufficient computing resources for audio processing and transcription
- Optional GPU support for faster Whisper transcription

## Technical Constraints
- Must handle 8kHz audio files properly
- FFmpeg must be installed on the system for audio processing
- Processing large audio files requires efficient memory usage
- Metadata format must be consistent for show file generation
- File naming must preserve chronological ordering

## Dependencies
- Python 3.x
- FFmpeg
- Required Python packages:
  - gradio
  - numpy
  - pydub (to be minimized)
  - subprocess (for calling FFmpeg)
  - openai-whisper (for transcription)
  - audio-separator (for vocal/instrumental separation)
  - torch (for neural network models)
  - audiomentations (for stem normalization and effects) 