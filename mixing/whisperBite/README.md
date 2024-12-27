# whisperBite

This tool processes audio files into transcribed soundbites, leveraging OpenAI's Whisper for transcription and pyannote.audio for speaker diarization. It includes features like vocal separation, silence-based slicing, and downloadable audio from URLs.

## Features
- **Audio Transcription**: Uses Whisper to transcribe audio into text.
- **Speaker Diarization**: Splits audio by speakers (default: 2 speakers, configurable).
- **Vocal Separation**: Optionally isolate vocals using Demucs.
- **Silence-Based Slicing**: Automatically slices audio into smaller chunks.
- **Audio Download**: Downloads audio from URLs and names files after video titles.

## Setup

### Requirements
- Python 3.8+
- GPU recommended for efficient processing (optional).

### Install Dependencies
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install FFmpeg (required for pydub):
   ```bash
   sudo apt install ffmpeg
   ```

4. Install Demucs (for vocal separation):
   ```bash
   pip install demucs
   ```

### Setup for pyannote.audio
1. Create a Hugging Face account if you don’t have one: [Hugging Face](https://huggingface.co/).

2. Obtain your Hugging Face token:
   - Navigate to [Settings > Access Tokens](https://huggingface.co/settings/tokens).
   - Create a token with `read` access.

3. Set the environment variable for pyannote:
   ```bash
   export HF_TOKEN=<your_hugging_face_token>
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
| `--url`                    | URL to download audio from.                                                 | None            |
| `--output_dir`             | Directory to save output files.                                             | Required        |
| `--model`                  | Whisper model to use (`base`, `small`, `medium`, `large`, `turbo`).         | `turbo`         |
| `--num_speakers`           | Number of speakers for diarization.                                          | `2`             |
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

#### Download and Process Audio from a URL
```bash
python whisperBite.py --url "<audio_url>" --output_dir path/to/output
```

#### Use Vocal Separation
```bash
python whisperBite.py --input_file path/to/audio.wav --output_dir path/to/output --enable_vocal_separation
```

#### Specify Number of Speakers
```bash
python whisperBite.py --input_file path/to/audio.wav --output_dir path/to/output --num_speakers 3
```

## Output Structure
The script saves output files in the following structure:
```
output_dir/
    <audio_file_name>/
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
    <audio_file_name>_results.zip
```

## Notes
- Ensure Demucs and pyannote dependencies are installed if using vocal separation or diarization.
- GPU is recommended for Whisper and pyannote.audio for faster processing.
- Ensure sufficient disk space for intermediate files and results.

## License
Creative Commons Attribution-ShareAlike 4.0 