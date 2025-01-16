# Audio Merger Script

This script merges audio files from subfolders into single merged files. By default, it processes all audio files found in a folder hierarchy and merges them into a single audio file for each subfolder. Optionally, the script can restrict processing to only audio files that have a numerical prefix (1-9999) followed by a dash (`-`).

## Features

- **Process all audio files**: The script processes all found audio files by default, regardless of their names.
- **Optional prefix filter**: Restrict merging to files with a 1-9999 numerical prefix by using the `--use-prefix` option.
- **Supports multiple formats**: The script can process `.wav`, `.mp3`, `.ogg`, and `.flac` files.
- **Error handling**: Files that cannot be decoded (e.g., corrupted or invalid files) are skipped, and a warning message is shown.
- **Automatic directory creation**: The script ensures that output directories are created as needed, avoiding errors due to missing directories.

## Requirements

- Python 3.6+
- [pydub](https://github.com/jiaaro/pydub)
- [ffmpeg](https://www.ffmpeg.org/)

### Install dependencies

```bash
pip install pydub
```

Make sure `ffmpeg` is installed and accessible via your system's PATH. You can download it from [here](https://www.ffmpeg.org/download.html).

## Usage

```bash
python glueAudio.py <input_folder> <output_folder> [--use-prefix]
```

### Arguments:

- `input_folder`: The root folder containing subfolders with audio files to merge.
- `output_folder`: The root folder where the merged audio files will be saved.

### Options:

- `--use-prefix`: If provided, only audio files with a 1-9999 prefix (e.g., `1-some_audio.wav`) will be processed. By default, all audio files are processed regardless of the filename prefix.

### Example:

To merge all audio files in subfolders under `/path/to/input/folder`, and save the merged files to `/path/to/output/folder`:

```bash
python glueAudio.py /path/to/input/folder /path/to/output/folder
```

To only process files with a prefix (e.g., `1-audio.wav`, `9999-audio.mp3`) and skip other files:

```bash
python glueAudio.py /path/to/input/folder /path/to/output/folder --use-prefix
```

### Handling Errors

If any file cannot be decoded (e.g., corrupted or unsupported files), the script will skip that file and print a warning message:

```
Warning: Could not decode /path/to/file.wav, skipping this file.
```

The rest of the valid files will still be merged as intended.

## Output Structure

For each subfolder in the `input_folder`, the script will create a corresponding folder in the `output_folder`. Inside each output subfolder, a file named `merged_audio.wav` will be generated, containing the merged audio of all valid files from the corresponding input subfolder.

### Example Directory Structure:

**Input Directory:**

```
/path/to/input/folder
    ├── subfolder1
    │   ├── 1-file1.wav
    │   ├── 2-file2.mp3
    │   └── invalid_file.wav
    ├── subfolder2
        ├── 1-file1.ogg
        └── 2-file2.flac
```

**Output Directory:**

```
/path/to/output/folder
    ├── subfolder1
    │   └── merged_audio.wav
    └── subfolder2
        └── merged_audio.wav
```

Invalid or problematic files (e.g., `invalid_file.wav`) will be skipped, and their content will not be merged.
