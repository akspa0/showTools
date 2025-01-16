# fapMixPlus

This project provides an end-to-end audio processing pipeline to automate the extraction, separation, slicing, transcription, and renaming of audio files. The resulting files are saved in a structured output directory with cleaned filenames and optional ZIP archives for easier distribution or storage.

## Features

- **Download Audio**: Fetches audio files from a URL or uses local input files.
- **Convert to WAV**: Converts audio files to WAV format.
- **Separate Vocals**: Isolates vocal tracks from the WAV files.
- **Slice Audio**: Segments the separated vocal track for transcription.
- **Transcribe**: Generates transcriptions from audio slices.
- **Sanitize and Rename Files**: Creates sanitized filenames with a numerical prefix, limited to 128 characters.
- **Generate ZIP Files**: Compresses processed files into ZIP archives for easy storage and distribution.

## Prerequisites

- **Python 3.x**
- Install required Python packages:
  ```bash
  pip install yt-dlp
  ```
- **Fish Audio Preprocessor (`fap`)** should be installed and available in the PATH.

### Installing the Fish Audio Preprocessor (`fap`)

1. Clone the [Fish Audio Preprocessor repository](https://github.com/fishaudio/audio-preprocess):
   ```bash
   git clone https://github.com/fishaudio/audio-preprocess.git
   ```

2. Navigate to the repository directory:
   ```bash
   cd audio-preprocess
   ```

3. Install the package from the cloned repository:
   ```bash
   pip install -e .
   ```

This step installs `fap` and makes it accessible as a command-line tool, which is essential for `fapMixPlus.py` to function correctly.

4. Verify the installation by checking the version:
   ```bash
   fap --version
   ```

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
├── zip_files/                 # Compressed ZIP archives of processed files
```

### ZIP File Details

In addition to organizing output files by processing stages, `fapMixPlus` can generate ZIP archives for convenience. Each ZIP file in the `zip_files/` directory will contain a set of processed audio and transcription files, with names based on their content and timestamp. The ZIP filenames will follow this format:

```
output_<timestamp>.zip
```

Each ZIP file will include:
- The WAV and `.lab` files from `final_output/`, with sanitized filenames.
- These ZIP files are ideal for transferring or archiving processed audio.

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
7. **ZIP File Creation**:
   - After processing, the final `.wav` and `.lab` files are compressed into ZIP archives in `zip_files/` for each session, making it easy to organize or share the output.

## Example File Names in Final Output

Final output files in `final_output` will be structured like:
- `0001_Hello_this_is_a_sample_transcription.wav`
- `0001_Hello_this_is_a_sample_transcription.lab`

Files without usable `.lab` content will retain the numerical prefix, e.g., `0002.wav` and `0002.lab`.