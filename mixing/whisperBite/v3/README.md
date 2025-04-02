# WhisperBite

WhisperBite is an audio processing tool that automatically extracts and transcribes spoken content from audio files. It identifies different speakers, separates the audio into bite-sized segments, and provides accurate transcriptions with timestamps.

## Features

- **Speaker Diarization**: Automatically identifies and separates different speakers
- **Word-Level Segmentation**: Creates individual audio clips for each spoken word
- **Transcription**: Uses OpenAI's Whisper models for high-quality transcriptions
- **Vocal Separation**: Optional removal of background music and noise
- **Customizable Processing**: Multiple options for fine-tuning results
- **Sequential Ordering**: All outputs maintain original chronological order with sequence prefixes

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/whisperBite.git
cd whisperBite

# Install required packages
pip install -r requirements.txt

# For vocal separation, install Demucs
pip install demucs
```

## Required Dependencies

- Python 3.8+
- PyTorch
- Whisper
- pyannote.audio (requires Hugging Face token)
- PyDub
- FFmpeg
- Gradio (for web interface)
- yt-dlp (for downloading audio from URLs)

## Usage

### Command Line

```bash
python whisperBite.py --input_file path/to/audio.mp3 --output_dir ./output --model base --num_speakers 2 --auto_speakers --enable_vocal_separation
```

### Web Interface

```bash
python app.py
```

Then open http://localhost:7860 in your browser

To make the interface publicly accessible:

```bash
python app.py --public
```

### Options

- `--input_file`: Path to input audio file
- `--input_dir`: Directory containing multiple audio files
- `--url`: URL to download audio from (YouTube, etc.)
- `--output_dir`: Directory to save output files
- `--model`: Whisper model size (tiny, base, small, medium, large)
- `--num_speakers`: Expected number of speakers
- `--auto_speakers`: Automatically detect optimal speaker count
- `--enable_vocal_separation`: Remove background music and noise

## Output Structure

WhisperBite creates a timestamped output directory for each processed file:

```
output_dir/
├── filename_[timestamp]/
│   ├── master_transcript.txt          # Full chronological transcript with speaker labels
│   ├── Speaker_X_full_transcript.txt  # Complete transcript for each speaker
│   ├── word_timings.json              # JSON with metadata for all extracted words
│   ├── Speaker_X_transcriptions/      # Speaker segments with transcripts
│   │   ├── 0000_segment_text.wav      # Audio segment
│   │   └── 0000_segment_text.txt      # Segment transcript
│   ├── Speaker_X_words/               # Individual word audio clips
│   │   ├── 0000_word.wav
│   │   ├── 0001_word.wav
│   │   └── ...
│   └── filename_results.zip           # All outputs packaged for sharing
```

## Example

Process a YouTube interview:

```bash
python whisperBite.py --url https://www.youtube.com/watch?v=example --output_dir ./output --model small --auto_speakers --enable_vocal_separation
```

## Requirements for Hugging Face

To use the speaker diarization feature, you need a Hugging Face token with access to the pyannote/speaker-diarization model:

1. Create an account on Hugging Face
2. Generate an access token at https://huggingface.co/settings/tokens
3. Accept the user agreement for pyannote/speaker-diarization
4. Provide your token when using the web interface

## License

MIT
