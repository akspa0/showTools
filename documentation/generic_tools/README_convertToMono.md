# Audio Converter to Mono

This script recursively converts audio files from a specified directory, including all subdirectories, to mono format. It can also convert a single file.

## Prerequisites

- Python 3.x
- PyDub
- FFmpeg

Ensure FFmpeg is installed on your system. Install PyDub using pip:

```bash
pip install pydub
```

## Usage

To convert a directory and its subdirectories:

```bash
python convertToMono.py <input_directory> <output_directory>
```

To convert a single file:

```bash
python convertToMono.py <input_file> <output_file>
```

`<input_directory>`: Path to the directory containing audio files.

`<output_directory>`: Path where the converted mono audio files will be saved, maintaining the original directory structure.

`<input_file>`: Path to the single audio file to convert.

`<output_file>`: Path to save the converted mono audio file.

## Notes

- The script handles audio files in the input directory and its subdirectories.
- Ensure the script has necessary permissions to access the specified directories and files.
- The script maintains the original directory structure in the output directory.
```

Replace `<input_directory>`, `<output_directory>`, `<input_file>`, and `<output_file>` with your actual paths when using the script.
