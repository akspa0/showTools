# fapMix0x00.py

`fapMix0x00.py` is a Python script for processing audio files using `fish-audio-preprocess` (fap) tools. This script automates the process of converting audio files to WAV format, normalizing their loudness, slicing them into manageable chunks, and transcribing the audio. It then renames the audio files based on their transcriptions and organizes the output in a structured directory.

## Requirements

- Python 3.x
- `fish-audio-preprocess` (fap) tools (https://github.com/fishaudio/audio-preprocess)
- Necessary dependencies should be installed (`argparse`, `subprocess`, `shutil`, `tempfile`, `logging`).

## Usage

```bash
python fapMix0x00.py <input_dir> <output_dir>
```

### Arguments

- `<input_dir>`: The directory containing the input audio files. This directory can include files in various formats.
- `<output_dir>`: The directory where the processed files will be saved. The script will create this directory if it does not exist.

## Steps Performed by the Script

1. **Check and Convert to WAV**:
   - The script checks if there are WAV files in the input directory.
   - If no WAV files are found, it converts the audio files to WAV format.

2. **Loudness Normalization**:
   - Normalizes the loudness of all WAV files in the input directory.

3. **Audio Slicing**:
   - Slices the normalized audio files into chunks of a minimum duration of 3 seconds.

4. **Transcription**:
   - Transcribes the sliced audio files into text.

5. **Renaming and Copying**:
   - Renames the WAV files based on the first 40 characters from their corresponding `.lab` files.
   - Copies the renamed WAV files and `.lab` files into the output directory, preserving the directory structure.

6. **Temporary Files**:
   - Temporary files and directories are kept intact for manual inspection after script execution.

## Example

```bash
python fapMix0x00.py /path/to/input_dir /path/to/output_dir
```

This command will process the audio files located in `/path/to/input_dir` and save the results in `/path/to/output_dir`.

## Troubleshooting

- **No `.lab` files found**: Verify that the transcription step completed successfully. The script logs messages if `.lab` files are not found.
- **Output issues**: Ensure that `fish-audio-preprocess` (fap) tools are properly installed and accessible in your PATH.

## License

This script is provided as-is without any warranty. Use it at your own risk.

## Contact

For issues or questions, please reach out to the author or maintainers.
