# preFapMix.py

**preFapMix.py** is an audio processing script that applies soft limiting, optional loudness normalization, and optional slicing for transcription. It can also produce stereo-mixed outputs with optional audio appended to the end. The script organizes processed files into structured folders with sanitized filenames and retains original timestamps for continuity.

## Features

1. **Soft Limiting**: Reduces loud peaks in audio to prevent clipping.
2. **Optional Loudness Normalization**: Adjusts audio levels to achieve consistent loudness.
3. **Conditional Slicing and Transcription**: Options to slice and transcribe files in the left or right channels separately, or both channels together.
4. **Stereo Mixing with Optional Tone Appending**: Optionally appends a custom tone (`tones.wav`) to the end of stereo-mixed audio.
5. **Organized Output Structure**: Outputs are saved in structured folders with sanitized filenames.
6. **Timestamp Preservation**: Maintains the original timestamps for all output files.

## Installation Requirements

- **Python 3.x**
- **Pydub** for audio processing
  ```bash
  pip install pydub
  ```
- **FFmpeg**: Required by Pydub for handling audio files
  ```bash
  sudo apt-get install ffmpeg
  ```
- **fap**: The transcription tool, assumed to be installed and accessible via the command line.

## Usage

### Command Line

Run the script from the command line with the following arguments:

```bash
python preFapMix.py --input-dir <input_directory> --output-dir <output_directory> [options]
```

### Options

- **`--input-dir`**: Directory containing input audio files (required).
- **`--output-dir`**: Directory where processed files will be saved (required).
- **`--transcribe`**: Enables transcription for both left and right channels. Implies both `--transcribe_left` and `--transcribe_right`.
- **`--transcribe_left`**: Enables transcription only for the left channel.
- **`--transcribe_right`**: Enables transcription only for the right channel.
- **`--normalize`**: Enables loudness normalization on the audio.
- **`--tones`**: Appends the contents of `tones.wav` to the end of each stereo output file.
- **`--num-workers`**: Specifies the number of workers to use for transcription (default is 2).

### Workflow

1. **Pre-Processing**:
   - Applies a soft limiter at -6 dB to control peaks.
   - If `--normalize` is enabled, normalizes loudness to -23 LUFS for consistency.

2. **Conditional Slicing and Transcription**:
   - If `--transcribe` is enabled, slices audio files to smaller segments and transcribes each segment, generating `.lab` files.
   - With `--transcribe_left` or `--transcribe_right`, transcribes only files in the left or right folders, respectively.

3. **Stereo Mixing with Optional Tone Appending**:
   - Combines left and right channels into a stereo file.
   - If `--tones` is enabled, appends `tones.wav` to the end of each stereo file.

4. **File Naming and Organization**:
   - Names each sliced audio file with its original numeric name, followed by the first 12 words (or fewer) from its `.lab` file.
   - All filenames are sanitized for UTF-8 compliance.

### Output Structure

The output structure is organized within `<output_directory>/run_<timestamp>` as follows:

- **`normalized/`**: Contains normalized versions of the input audio files.
- **`left/`** and **`right/`**: Contains sliced (and optionally transcribed) audio files in respective left and right channel folders.
- **`stereo/`**: Contains stereo-mixed files with optional tone appended to the end.
- **`transcribed-and-sliced/`**:
  - Root: Contains combined `.lab` files for each original input.
  - **`left/`** and **`right/`**: Contains subfolders of sliced audio files and corresponding `.lab` files.

### Example Command

```bash
python preFapMix.py --input-dir ./my_audio_files --output-dir ./processed_audio --transcribe --normalize --tones --num-workers 3
```

This command will:
1. Process the audio files in `./my_audio_files` with soft limiting and loudness normalization.
2. Slice and transcribe each file in the left and right channels.
3. Mix each pair of left and right channels into a stereo file and append `tones.wav` to the end of each stereo output.
