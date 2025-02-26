# Audio Processor

## Overview
A comprehensive audio processing script that transcribes audio files, optionally separates vocals, detects DTMF tones, and generates granular audio segments.

## Features
- Transcription using Whisper AI
- Optional vocal separation with Demucs
- Audio normalization
- DTMF tone detection
- Automatic sentence and paragraph segmentation
- Configurable audio slicing and processing

## Requirements
- Python 3.8+
- Libraries:
  - whisper
  - librosa
  - soundfile
  - numpy
  - demucs (optional)

## Installation
```bash
pip install whisper librosa soundfile numpy
# Optional: install demucs for vocal separation
pip install demucs
```

## Usage
```bash
python audio_processor.py /path/to/audio/folder [options]
```

### Options
- `--buffer FLOAT`: Buffer duration in seconds (default: 0.25)
- `--use-llm`: Use Local Language Model for paragraph inference (placeholder)
- `--split`: Split vocals from audio using Demucs
- `--normalize`: Normalize audio after splitting

## Output Structure
```
output_folder/
├── sentences/         # Individual sentence audio files
├── paragraphs/        # Paragraph-level audio files
└── transcription.json # Full transcription metadata
```

## Example
```bash
python audio_processor.py ./recordings --split --normalize
```

## Limitations
- Requires Whisper model download
- Local LLM paragraph inference is a placeholder
- Vocal separation requires Demucs installation

## Contributing
Contributions welcome. Please open an issue or submit a pull request.