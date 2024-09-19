# Audio Merging Script

This Python script recursively merges audio files from subfolders. It processes audio files whose filenames start with a prefix between `1-9999` followed by a dash (`-`). The valid audio files are merged into one long track for each subfolder and saved in a corresponding output directory.

## Features

- **Recursively processes subfolders**: The script traverses through all subfolders in the specified input folder.
- **Merges valid audio files**: Only files with a numeric prefix (e.g., `1-file.wav`, `234-description.wav`) are merged.
- **Flexible audio formats**: Supports `.wav`, `.mp3`, `.ogg`, and `.flac` formats.
- **Automatic directory creation**: Ensures that output directories are created if they don't already exist.
- **Organized output**: Each merged track is saved in a corresponding output folder, preserving the subfolder structure from the input directory.

## Prerequisites

1. **Python 3.x**
2. **pydub** for handling audio merging:
   ```bash
   pip install pydub
   ```
3. **FFmpeg** installed on your system (required by `pydub` for encoding/decoding audio files).
   - [FFmpeg Installation Guide](https://ffmpeg.org/download.html)

## Usage

### Command Line

Run the script from the terminal with the following syntax:

```bash
python glueAudio.py /path/to/input/folder /path/to/output/folder
```

### Arguments

- **`/path/to/input/folder`**: The root folder containing subfolders of audio files to merge.
- **`/path/to/output/folder`**: The root folder where merged audio files will be saved. The output folder structure mirrors the input folder structure.

### Example

Given the following directory structure:

```
input_folder/
    subfolder1/
        1-audio.wav
        2-audio.mp3
        invalid.wav
    subfolder2/
        10-intro.wav
        234-part.wav
```

After running the command:

```bash
python glueAudio.py /input_folder /output_folder
```

The output directory will contain:

```
output_folder/
    subfolder1/
        merged_audio.wav
    subfolder2/
        merged_audio.wav
```

### Error Handling

- If no valid files are found in a subfolder, that subfolder will be skipped.
- If any intermediate directory (e.g., `Full-Audio`) does not exist, it will be created automatically.

## Code Overview

### Main Functions

1. **`is_valid_audio(filename)`**: 
   - Returns `True` if the file starts with a numeric prefix (1-9999 followed by `-`).
   - Example valid filename: `1-audio.wav`.

2. **`merge_audio_in_subfolders(input_folder, output_folder)`**:
   - Walks through the input folder and merges valid audio files in each subfolder.
   - Saves the merged output in a subfolder under the specified output folder.

3. **`ensure_directory_exists(file_path)`**:
   - Ensures the directory for the output file exists before trying to write the merged audio file.

## Dependencies

- [pydub](https://pydub.com/): A simple and easy-to-use library for audio file manipulation.
- FFmpeg: Required for handling audio file formats.

## Installation

1. Clone or download this repository to your local machine.

2. Install the required Python dependencies:

   ```bash
   pip install pydub
   ```

3. Install FFmpeg on your system, following the instructions [here](https://ffmpeg.org/download.html).

