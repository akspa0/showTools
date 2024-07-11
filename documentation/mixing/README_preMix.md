# preMix.py

## Overview
This Python script processes stereo audio files to balance the volume levels of the left and right channels, amplifies the overall volume without clipping, and exports the adjusted audio as separate left and right channel files as well as a combined stereo file. It ensures neither channel overpowers the other, maintaining a clear stereo effect with a 40% middle separation.

## Requirements
- Python 3.6 or later
- `pydub` library
- `ffmpeg`

## Installation
1. Ensure Python and `pip` are installed on your system.
2. Install `pydub` using pip:
   ```
   pip install pydub
   ```
3. `ffmpeg` must be installed and accessible in your system's PATH. Visit [FFmpeg](https://ffmpeg.org/download.html) to download and install it.

## Usage
1. Place your audio files in a directory.
2. Run the script from the command line, specifying the input directory (or file) and the output directory where the processed files will be saved. Enable debug mode if needed.
   ```
   python preMix.py --input /path/to/input --output /path/to/output --debug
   ```

### Arguments
- `--input` - Path to the input file or directory containing audio files.
- `--output` - Path to the output directory where processed files will be saved.
- `--debug` (optional) - Enable debug mode to show detailed processing messages.

## Features
- **Volume Balancing**: Analyzes and adjusts the peak levels of the left and right audio channels for balanced volume.
- **Stereo Separation**: Maintains a clear stereo effect by panning the channels 20% left and right, respectively.
- **Output Generation**: Exports the processed audio as separate left and right channel files and a combined stereo file, all saved in the specified output directory.
- **Folder Creation**: Automatically creates the output directory if it does not exist.

## Contribution
Feel free to fork the repository and submit pull requests to enhance the script's functionality or address any issues. For bugs and feature requests, please open an issue in the project repository.
