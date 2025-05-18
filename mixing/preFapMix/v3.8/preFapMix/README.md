# preFapMix.py

## Overview

`preFapMix.py` is a Python script designed to process audio files through a series of steps including soft limiting, optional loudness normalization, channel splitting, stereo mixing, slicing, transcription, and renaming. The script is particularly useful for preparing audio files for further processing or analysis, especially in contexts where transcription and channel management are important.

## Features

- **Soft Limiting**: Applies a soft limiter to prevent clipping and reduce peaks in the audio.
- **Loudness Normalization**: Optionally normalizes the loudness of the audio files to a consistent level.
- **Channel Splitting**: Separates stereo audio files into left and right mono channels.
- **Stereo Mixing**: Combines left and right channels back into stereo with adjustable pan settings.
- **Slicing**: Splits audio files into smaller segments based on silence detection.
- **Transcription**: Generates transcriptions for sliced audio segments using an external tool.
- **Renaming**: Renames audio files based on their transcriptions while ensuring filenames do not exceed 128 characters.

## Dependencies

- **Python 3.6+**
- **Pydub**: Audio processing library (`pip install pydub`)
- **FFmpeg**: Required by Pydub for audio encoding/decoding
- **External Tools**:
  - **fap**: Command-line tool for slicing and transcribing audio (`fap slice-audio-v2`, `fap transcribe`)
  - **tones.wav**: An optional audio file to append tones at the end of stereo outputs (if using the `--tones` option)

## Installation

1. **Install Python 3.6 or higher** if it's not already installed on your system.
2. **Install Pydub**:
   ```bash
   pip install pydub
   ```
3. **Install FFmpeg**:
   - **Windows**: Download from [FFmpeg website](https://ffmpeg.org/download.html) and add to your system PATH.
   - **macOS**: Install via Homebrew:
     ```bash
     brew install ffmpeg
     ```
   - **Linux**: Install via your package manager:
     ```bash
     sudo apt-get install ffmpeg
     ```
4. **Install fap**:
   - Follow the installation instructions provided by the `fap` tool.

## Usage

```bash
python preFapMix.py --input-dir <input_directory> --output-dir <output_directory> [options]
```

### Required Arguments

- `--input-dir`: Directory containing the input audio files to be processed.
- `--output-dir`: Directory where the processed output will be saved.

### Optional Arguments

- `--transcribe`: Enable transcription for both left and right channels.
- `--transcribe_left`: Enable transcription for the left channel only.
- `--transcribe_right`: Enable transcription for the right channel only.
- `--tones`: Append `tones.wav` to the end of stereo output files.
- `--normalize`: Enable loudness normalization.
- `--num-workers`: Number of worker threads for transcription (default: `2`).

### Example Commands

1. **Process Audio with Transcription and Normalization**:

   ```bash
   python preFapMix.py --input-dir ./input_audio --output-dir ./output_audio --transcribe --normalize
   ```

2. **Process Left Channel Only with Transcription**:

   ```bash
   python preFapMix.py --input-dir ./input_audio --output-dir ./output_audio --transcribe_left
   ```

3. **Process Audio and Append Tones to Stereo Outputs**:

   ```bash
   python preFapMix.py --input-dir ./input_audio --output-dir ./output_audio --tones
   ```

## Detailed Description

### Processing Steps

1. **Normalization and Soft Limiting**:
   - Each audio file in the input directory is loaded.
   - A soft limiter is applied to reduce peaks without introducing distortion.
   - If the `--normalize` flag is set, loudness normalization is applied to achieve a consistent volume level across all files.
   - The processed files are saved in the `normalized` subdirectory within the output directory.

2. **Channel Splitting**:
   - Normalized audio files are split into left and right channels.
   - The channels are saved as separate mono audio files in the `left` and `right` subdirectories, respectively.

3. **Stereo Mixing**:
   - Left and right channel files are combined back into stereo audio files.
   - A slight pan is applied: left channel is panned 20% to the left, and right channel is panned 20% to the right.
   - If the `--tones` flag is set, `tones.wav` is appended to the end of the stereo audio.
   - Stereo files are saved in the `stereo` subdirectory.

4. **Slicing and Transcription**:
   - If transcription is enabled (`--transcribe`, `--transcribe_left`, or `--transcribe_right`), the script slices the audio files into segments based on silence detection using `fap slice-audio-v2`.
   - Transcriptions are generated for each sliced audio segment using `fap transcribe`.

5. **Renaming and Copying**:
   - Sliced audio files are renamed based on their transcriptions.
   - Filenames are sanitized to remove special characters and spaces.
   - To prevent filesystem issues, filenames are truncated to ensure they do not exceed 128 characters.
   - The renamed files are copied to the `transcribed-and-sliced` subdirectory under `left` or `right` as appropriate.

### Command-Line Options Explained

- `--input-dir`:
  - Path to the directory containing input audio files.
  - The script processes all files in this directory.

- `--output-dir`:
  - Path to the directory where all output files and subdirectories will be saved.
  - A timestamped subdirectory is created for each run.

- `--transcribe`:
  - Enables transcription for both left and right channels.
  - Equivalent to using both `--transcribe_left` and `--transcribe_right`.

- `--transcribe_left`:
  - Enables transcription for the left channel only.

- `--transcribe_right`:
  - Enables transcription for the right channel only.

- `--tones`:
  - Appends the contents of `tones.wav` to the end of each stereo output file.
  - Useful for adding signaling tones or markers to the audio.

- `--normalize`:
  - Applies loudness normalization to the audio files after soft limiting.
  - Helps achieve a consistent volume level across all processed files.

- `--num-workers`:
  - Specifies the number of worker threads to use for transcription.
  - Default is `2`, but you can increase this number to speed up transcription if you have more CPU cores.

## Output Directory Structure

- `<output_dir>/run_<timestamp>/normalized/`:
  - Contains normalized audio files.

- `<output_dir>/run_<timestamp>/left/`:
  - Contains mono audio files for the left channel.

- `<output_dir>/run_<timestamp>/right/`:
  - Contains mono audio files for the right channel.

- `<output_dir>/run_<timestamp>/stereo/`:
  - Contains mixed stereo audio files.

- `<output_dir>/run_<timestamp>/transcribed-and-sliced/left/`:
  - Contains sliced and transcribed audio files from the left channel.

- `<output_dir>/run_<timestamp>/transcribed-and-sliced/right/`:
  - Contains sliced and transcribed audio files from the right channel.

## Notes

- **File Naming Conventions**:
  - Filenames are sanitized to remove special characters and spaces.
  - Filenames are truncated to ensure they do not exceed 128 characters, to prevent filesystem issues.
  - When transcriptions are used in filenames, only the first 12 words are included.

- **Transcription Accuracy**:
  - The quality of transcriptions depends on the capabilities of the `fap transcribe` tool and the clarity of the audio.
  - Ensure that the audio quality is sufficient for accurate transcription.

- **Resource Usage**:
  - Processing audio files can be CPU and memory-intensive.
  - Adjust the `--num-workers` option based on your system's capabilities.

- **Error Handling**:
  - The script includes logging to help identify and troubleshoot issues.
  - If an error occurs with a specific file, the script logs the error and continues processing the remaining files.

- **Dependencies**:
  - Ensure all dependencies are installed and accessible from the command line.
  - The `fap` tool must be installed and properly configured.

## Troubleshooting

- **FFmpeg Not Found**:
  - If you receive an error related to FFmpeg, ensure it is installed and added to your system PATH.

- **Permission Errors**:
  - Ensure you have read and write permissions for the input and output directories.

- **Transcription Failures**:
  - Verify that the `fap` tool is installed and that you have the necessary permissions and configurations.

- **Long Filenames**:
  - If you still encounter issues with long filenames, check the logic in the `rename_and_copy_sliced_files` function.

## Contributing

Contributions are welcome! If you have suggestions for improvements or encounter any issues, feel free to open an issue or submit a pull request.

## License

Creative Commons non-commercial use with attribution