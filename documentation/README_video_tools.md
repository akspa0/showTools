# Video Sampler 2x0

`videosampler2x0.py` is a Python script designed to sample and process video frames efficiently. This tool is part of the `showTools` repository and is intended to assist users in extracting and analyzing frames from video files for various purposes such as video editing, analysis, or further processing.

## Features

- Extract frames from video files at specified intervals.
- Save extracted frames to a designated directory.
- Option to resize frames to desired dimensions.
- Easy-to-use command-line interface.

## Requirements

- Python 3.x
- OpenCV
- NumPy

You can install the required packages using the following command:

```sh
pip install opencv-python numpy
```

## Usage

The script can be run from the command line with the following syntax:

```sh
python videosampler2x0.py -i <input_video> -o <output_directory> -s <sample_interval> [-r <resize_dimensions>]
```

### Arguments

- `-i, --input_video`: Path to the input video file.
- `-o, --output_directory`: Path to the directory where the extracted frames will be saved.
- `-s, --sample_interval`: Interval at which frames are extracted (e.g., every 10 frames).
- `-r, --resize_dimensions` (optional): Dimensions to resize frames to, specified as `width,height` (e.g., `640,480`).

### Example

Extract frames from `example.mp4` every 30 frames and save them to the `output_frames` directory:

```sh
python videosampler2x0.py -i example.mp4 -o output_frames -s 30
```

Extract frames from `example.mp4` every 10 frames, resize them to 320x240, and save them to the `output_frames` directory:

```sh
python videosampler2x0.py -i example.mp4 -o output_frames -s 10 -r 320,240
```

## Contributing

Contributions are welcome! If you find a bug or have a feature request, please open an issue or submit a pull request.

## Acknowledgments

- This script utilizes OpenCV for video processing and NumPy for array manipulations.
- Thanks to all contributors and the open-source community for their invaluable support and tools.