# Audio Processing Pipeline

This script processes audio files through a unified pipeline, including normalization, vocal separation, speaker diarization, and transcription using OpenAI's Whisper.

## Features
- Normalize audio to -14 LUFS loudness.
- Optionally separate vocals using Demucs.
- Perform speaker diarization to split audio by speakers.
- Transcribe audio using Whisper.

## Requirements
### Python Version
- Python 3.8+

### Required Python Packages
Install the required Python packages using pip:
```bash
pip install -r requirements.txt
```

#### `requirements.txt`
```
pydub
torch
whisper
pyannote.audio
termcolor
```

Additionally, for Demucs:
```bash
pip install demucs
```

## Usage
Run the script with the following command:
```bash
python whisperBite.py [OPTIONS]
```

### Command-Line Options
| Option                     | Description                                                                 | Default         |
|----------------------------|-----------------------------------------------------------------------------|-----------------|
| `--input_dir`              | Directory containing input audio files.                                     | None            |
| `--input_file`             | Single audio file for processing.                                           | None            |
| `--output_dir`             | Directory to save output files.                                             | Required        |
| `--model`                  | Whisper model to use (`base`, `small`, `medium`, `large`, `turbo`).         | `turbo`         |
| `--num_speakers`           | Expected number of speakers in the audio.                                   | `2`             |
| `--enable_vocal_separation`| Enable vocal separation using Demucs.                                       | Disabled        |

### Examples
#### Process a Single Audio File
```bash
python whisperBite.py --input_file path/to/audio.wav --output_dir path/to/output
```

#### Process a Directory of Audio Files
```bash
python whisperBite.py --input_dir path/to/audio_directory --output_dir path/to/output
```

#### Use Vocal Separation
```bash
python whisperBite.py --input_file path/to/audio.wav --output_dir path/to/output --enable_vocal_separation
```

#### Use a Specific Whisper Model
```bash
python whisperBite.py --input_file path/to/audio.wav --output_dir path/to/output --model small
```

## Output Structure
The script saves output files in the following structure:
```
output_dir/
    normalized/
        audio_normalized.wav
    demucs/
        vocals.wav
    speakers/
        Speaker_1/
            segment_0_10.wav
            segment_10_20.wav
        Speaker_2/
            segment_0_10.wav
    Speaker_1_transcriptions/
        transcription_1.txt
    Speaker_2_transcriptions/
        transcription_2.txt
```

## Notes
- Ensure Demucs is installed and accessible if enabling vocal separation.
- Whisper requires a GPU for efficient transcription. Use `--model` to specify a smaller model for lower-resource environments.
- Ensure sufficient disk space for intermediate files.

## License
MIT License
