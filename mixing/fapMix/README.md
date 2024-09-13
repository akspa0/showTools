# Audio File Processing Script

This script, `fapMix0x00.py`, processes audio files by converting them to WAV format, normalizing their loudness, slicing them, and transcribing them. It also renames files based on transcription text and can package the final output into a zip file.

## Features

- **Convert to WAV**: Converts non-WAV audio files to WAV format.
- **Normalize Loudness**: Applies loudness normalization to the WAV files.
- **Slice Audio**: Slices the audio files into shorter segments.
- **Transcribe Audio**: Transcribes the sliced audio files.
- **Rename Files**: Renames WAV files based on transcription text and preserves the folder structure.
- **Zip Output**: Optionally packages the processed output into a zip file.
- **Temporary Files**: Keeps temporary files for debugging and inspection if needed.

## Requirements

- Python 3.x
- `fap` command-line tool (for audio processing tasks) - https://github.com/fishaudio/audio-preprocess

## Usage

### Basic Usage

```bash
python fapMix0x00.py <input_dir> [--zip]
```

- `<input_dir>`: Path to the directory containing audio files.
- `--zip` (optional): Create a zip file of the output folder.

### Example

```bash
python fapMix0x00.py /path/to/audio/files --zip
```

This will process the files in `/path/to/audio/files`, and if the `--zip` option is provided, it will create a zip file of the output folder.

## Script Details

1. **Convert to WAV**: If no WAV files are found in the input directory, the script converts all audio files to WAV format.
2. **Normalize Loudness**: Normalizes the loudness of the WAV files.
3. **Slice Audio**: Slices the normalized audio files into segments of at least 3 seconds.
4. **Transcribe Audio**: Transcribes the sliced audio files using the `fap` tool.
5. **Rename and Copy Files**: Renames WAV files based on the transcription text and copies them to the output directory while preserving the folder structure.
6. **Zip Output**: Creates a zip file of the output directory if the `--zip` option is specified.

## Logging

- Logs are generated for debugging purposes and include detailed information about the processing steps. The logs are saved with timestamps to aid in tracking issues.

## Temporary Files

- Temporary files are stored in a timestamped temporary directory created by the script. This directory is preserved for manual inspection if debugging is needed. If the `--debug` flag is not used, the temporary files will not be deleted.
