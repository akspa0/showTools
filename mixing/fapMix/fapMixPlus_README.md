# Efficient Audio Processing Script

This script enables the structured processing of audio files, including downloading from a URL, converting to WAV format, normalizing loudness, slicing, and transcribing audio content.

## Prerequisites

Ensure you have the following dependencies installed:

- Python 3.x
- `yt-dlp`
- `ffmpeg`
- `shutil`

Install the required Python packages using `pip`:
```bash
pip install yt-dlp

## Usage
Run the script with the following options:

Download Audio from URL and Process
```bash
python fapMixPlus0x10.py --url <audio_url> --output_dir <output_directory>

## Process Existing Audio Directory
```bash
python fapMixPlus0x10.py <input_directory> --output_dir <output_directory>

## Arguments
--url: URL to download audio from.

--output_dir: Directory for storing output files (default: "output").

input_directory: Directory of input files if no URL is provided.

## Steps
* Audio Download: Downloads audio from the specified URL (if provided) and saves it in the specified download directory.

* WAV Conversion: Converts audio files to WAV format.

* Loudness Normalization: Normalizes the loudness of audio files.

* Audio Slicing: Slices audio files into smaller segments.

* Transcription: Transcribes the audio files and renames them based on the first few words of the transcription.

## Example
To download and process audio from a URL:

```bash
python fapMixPlus0x10.py --url https://example.com/audio --output_dir /path/to/output

To process audio from an existing directory:

```bash
python fapMixPlus0x10.py /path/to/input --output_dir /path/to/output

## Logging
The script provides detailed logging for each stage of the process, including errors and progress updates. Logs are displayed in the console for easy monitoring.

## Notes
Ensure the ffmpeg binary is accessible in your system PATH for the script to run successfully.