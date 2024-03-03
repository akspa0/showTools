# preMix21.py Audio processing script

This Python script processes stereo audio files, splitting them into two mono streams, normalizing and compressing the audio slightly, then merging the two streams into a single output file.

## Prerequisites

Make sure you have Python installed. This script requires Python 3.

## Installation

1. Clone or download this repository to your local machine.
2. Navigate to the directory where you cloned/downloaded the repository.
3. Install the required dependencies by running:

```
pip install -r requirements.txt
```

## Usage

Run the script with the following command:

```
python preMix21.py <input> <output> [--append-audio] [--pre-split] [--debug]
```

- `<input>`: Path to the input file or directory containing audio files.
- `<output>`: Path to the output directory where processed files will be saved.
- `--append-audio`: Optional. Path to an audio file to append to the end of each input file.
- `--pre-split`: Optional. Treat input files as pre-split left and right channels.
- `--debug`: Optional. Enable debug mode to show debug messages.

## Examples

1. Process a single audio file:

```
python preMix21.py input_file.wav output_folder
```

2. Process all audio files in a directory:

```
python preMix21.py input_folder output_folder
```

3. Process audio files with appending another audio:

```
python preMix21.py input_file.wav output_folder --append-audio append_audio_file.wav
```

4. Process audio files with pre-split channels:

```
python preMix21.py input_folder output_folder --pre-split
```
