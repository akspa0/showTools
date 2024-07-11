# 5.1 Channel Audio Splitter and Combiner

This tool allows you to split 5.1 channel FLAC audio files into individual channel files and combine separate channel files into a 5.1 surround FLAC file sampled at 48kHz.

## Features

- **Split 5.1 Channel FLAC Files**: Extracts each channel from a 5.1 FLAC file and saves them as separate 32-bit WAV files sampled at 48kHz.
- **Combine Channel Files**: Combines individual channel WAV files into a single 5.1 FLAC file sampled at 48kHz.
- **Progress Tracking**: Uses `tqdm` to display progress while processing directories of files.

## Requirements

- Python 3.6+
- pydub
- numpy
- tqdm

## Installation

1. Clone the repository:

   ```sh
   git clone https://github.com/akspa0/showTools/showTools.git
   cd generic_tools
   ```

2. Install the required Python packages:

   ```sh
   pip install pydub numpy tqdm
   ```

## Usage

### Split a Single File

To split a single 5.1 channel FLAC file into separate channel files:

```sh
python 5.1_tool.py split input_file.flac output_directory
```

### Split a Directory of Files

To split all 5.1 channel FLAC files in a directory:

```sh
python 5.1_tool.py split input_directory output_directory
```

### Combine Channel Files into a 5.1 FLAC File

To combine individual channel WAV files into a single 5.1 FLAC file:

```sh
python 5.1_tool.py combine input_directory output_file.flac
```

## Examples

### Splitting a Single File

```sh
python 5.1_tool.py split example.flac output
```

This will create a directory `output/example` containing the following files:

- `FC.wav`
- `FL.wav`
- `FR.wav`
- `BL.wav`
- `BR.wav`
- `LFE.wav`

### Splitting a Directory of Files

```sh
python 5.1_tool.py split flac_files output
```

This will process each FLAC file in the `flac_files` directory, creating a subdirectory in the `output` directory for each file with the respective channel files.

### Combining Channel Files

Assuming you have a directory `channels` with the following files:

- `FC.wav`
- `FL.wav`
- `FR.wav`
- `BL.wav`
- `BR.wav`
- `LFE.wav`

You can combine them into a single 5.1 FLAC file:

```sh
python 5.1_tool.py combine channels output.flac
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on GitHub.

## Author

- [akspa0](https://github.com/akspa0/showTools)

## Acknowledgments

- Thanks to the developers of `pydub`, `numpy`, and `tqdm` for their awesome libraries.
