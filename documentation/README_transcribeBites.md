# Audio Soundbites Transcription Script

This script transcribes audio files using Google Cloud Speech-to-Text API. It segments the audio files based on silence intervals, transcribes each segment, and saves the transcriptions along with the segmented audio files.

## Features

- Transcribes audio files (.mp3 or .wav format) to text using Google Cloud Speech-to-Text API.
- Segments audio files based on silence intervals to process manageable chunks.
- Supports multi-speaker diarization for accurate transcriptions of conversations.
- Generates FLAC and MP3 files for each segmented audio along with their transcriptions.
- Handles duplicate segments and prevents duplicate transcriptions.

## Prerequisites

- Python 3.x
- Google Cloud Platform account with Speech-to-Text API enabled.
- Google Cloud service account credentials with access to Speech-to-Text API.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/akspa0/showTools.git
   ```

2. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

```bash
python transcribeBites0F.py --input /path/to/input/folder --output /path/to/output/folder --speakers 1 --credentials /path/to/credentials.json
```

### Arguments

- `--input`: Path to the folder containing audio files to transcribe.
- `--output`: Path to the folder to save transcriptions and segmented audio.
- `--speakers`: Number of speakers in the audio (default: 1).
- `--credentials`: Path to Google Cloud service account credentials JSON file.
- `--max-duration`: Maximum duration of each segment in milliseconds (default: 10000).
- `--silence-thresh`: Silence threshold in dB (default: -40).
- `--compression-ratio`: Compression ratio for dynamic compression (default: 10.0).
- `--dry-run`: Simulate the process without actually transcribing the audio.

## Acknowledgments

- This script utilizes the Google Cloud Speech-to-Text API for audio transcription.
- It makes use of the pydub library for audio processing.
- The tqdm library is used for displaying progress bars during processing.

## Authors

- [akspa](https://github.com/akspa0)
