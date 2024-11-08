# fapMixPlus

This project provides an end-to-end audio processing pipeline to automate the extraction, separation, slicing, transcription, and renaming of audio files. The resulting files are saved in a structured output directory with cleaned filenames.

## Features

- **Download Audio**: Fetches audio files from a URL or uses local input files.
- **Convert to WAV**: Converts audio files to WAV format.
- **Separate Vocals**: Isolates vocal tracks from the WAV files.
- **Slice Audio**: Segments the separated vocal track for transcription.
- **Transcribe**: Generates transcriptions from audio slices.
- **Sanitize and Rename Files**: Creates sanitized filenames with a numerical prefix, limited to 128 characters.

## Prerequisites

- Python 3.x
- Install required Python packages:
  ```bash
  pip install yt-dlp
  ```
- `fap` (Fish Audio Preprocessor) should be installed and available in the PATH.

## Usage

### Command-line Arguments

| Argument        | Description                                                          |
|-----------------|----------------------------------------------------------------------|
| `--url`         | URL of the audio source (YouTube or other supported link).           |
| `--output_dir`  | Directory for saving all outputs. Default is `output/`.              |
| `input_dir`     | Path to a local directory of input files (optional if `--url` used). |

### Example Command

```bash
python fapMixPlus.py --url https://youtu.be/example_video --output_dir my_output
```

This command will download the audio from the URL, process it, and save the results in the `my_output` folder.

### Output Structure

The output directory will contain a timestamped folder with the following structure:

```
output_<timestamp>/
├── wav_conversion/            # WAV-converted audio files
├── separation_output/         # Separated vocal track files
├── slicing_output/            # Sliced segments from separated audio
├── final_output/              # Final, sanitized, and renamed .wav and .lab files
```

## Functionality Details

1. **Download Audio**: Downloads audio from a URL, saving it in `.m4a` format.
2. **WAV Conversion**: Converts audio to WAV using `fap to-wav`.
3. **Separation**: Separates vocals from the WAV files using `fap separate`.
4. **Slicing**: Segments the separated vocal track into smaller audio slices.
5. **Transcription**: Uses `fap transcribe` to transcribe each slice.
6. **Sanitization and Renaming**:
   - Extracts the first 10 words from each `.lab` file.
   - Replaces spaces with underscores, removes special characters, and limits to 128 characters.
   - Applies a numerical prefix if no valid content is in the `.lab` file.

## Example File Names in Final Output

Final output files in `final_output` will be structured like:
- `0001_Hello_this_is_a_sample_transcription.wav`
- `0001_Hello_this_is_a_sample_transcription.lab`

Files without usable `.lab` content will retain the numerical prefix, e.g., `0002.wav` and `0002.lab`.
