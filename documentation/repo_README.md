## Root README.md
```
# showTools

showTools is a collection of Python scripts designed for various purposes such as video sampling, audio manipulation, and transcription. This repository aims to provide tools that enhance media manipulation and analysis.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Scripts](#scripts)
  - [Generic Tools](#generic-tools)
  - [Mixing](#mixing)
  - [Transcription](#transcription)
  - [Video Tools](#video-tools)
- [Contributing](#contributing)
- [License](#license)

## Installation

To install the necessary dependencies, run:

```bash
pip install -r requirements.txt
```

## Usage

Each script has its own specific use case and can be run individually. Below are the details for each script organized by category.

## Scripts

### Generic Tools

- [5.1_tool.py](generic_tools/README_5.1_tool.md)
- [convertToMono.py](generic_tools/README_convertToMono.md)
- [random_audio_sampler.py](generic_tools/README_random_audio_sampler.md)
- [splitOrConcatAudio.py](generic_tools/README_splitOrConcatAudio.md)

### Mixing

- [preMix33C.py](mixing/README_preMix33C.md)

### Transcription

- [transcribeBites10-0.py](transcription/README_transcribeBites.md)

### Video Tools

- [videosampler0x4.py](video_tools/README_videosampler0x4.md)
- [videosampler2x0.py](video_tools/README_videosampler2x0.md)

## Contributing

Contributions are welcome! Please read the [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on contributing to this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

## generic_tools/README_5.1_tool.md
```markdown
# 5.1_tool.py

This script is designed to manipulate 5.1 audio files.

## Usage

```bash
python 5.1_tool.py --input <input_audio> --output <output_directory>
```

## Options

- `--input`: Path to the input 5.1 audio file.
- `--output`: Directory where the processed audio files will be saved.
```

## generic_tools/README_convertToMono.md
```markdown
# convertToMono.py

This script converts stereo audio files to mono.

## Usage

```bash
python convertToMono.py --input <input_audio> --output <output_audio>
```

## Options

- `--input`: Path to the input stereo audio file.
- `--output`: Path where the mono audio file will be saved.
```

## generic_tools/README_random_audio_sampler.md
```markdown
# random_audio_sampler.py

This script samples random segments from an audio file.

## Usage

```bash
python random_audio_sampler.py --input <input_audio> --output <output_directory> --duration <duration_in_seconds>
```

## Options

- `--input`: Path to the input audio file.
- `--output`: Directory where the sampled audio files will be saved.
- `--duration`: Duration of each audio sample in seconds.
```

## generic_tools/README_splitOrConcatAudio.md
```markdown
# splitOrConcatAudio.py

This script splits or concatenates audio files based on user input.

## Usage

```bash
python splitOrConcatAudio.py --input <input_audio> --mode <split_or_concat> --output <output_audio>
```

## Options

- `--input`: Path to the input audio file.
- `--mode`: Mode of operation, either 'split' or 'concat'.
- `--output`: Path where the processed audio file will be saved.
```

## mixing/README_preMix33C.md
```markdown
# preMix33C.py

This script is used for premixing audio tracks.

## Usage

```bash
python preMix33C.py --input <input_directory> --output <output_audio>
```

## Options

- `--input`: Directory containing the audio tracks to be premixed.
- `--output`: Path where the premixed audio file will be saved.
```

## transcription/README_transcribeBites.md
```markdown
# transcribeBites10-0.py

This script transcribes audio bites using a specified transcription service.

## Usage

```bash
python transcribeBites10-0.py --input <input_audio> --output <output_text>
```

## Options

- `--input`: Path to the input audio file.
- `--output`: Path where the transcribed text file will be saved.
```

## video_tools/README_videosampler0x4.md
```markdown
# videosampler0x4.py

This script generates glitch art from video samples.

## Usage

```bash
python videosampler0x4.py --input <input_video> --output <output_directory> --interval <interval_in_seconds>
```

## Options

- `--input`: Path to the input video file.
- `--output`: Directory where the glitch art images will be saved.
- `--interval`: Time interval in seconds between each frame sample.
```

## video_tools/README_videosampler2x0.md
```markdown
# videosampler2x0.py

This script is designed for video sampling. It allows users to extract frames from a video at specified intervals.

## Usage

```bash
python videosampler2x0.py --input <input_video> --output <output_directory> --interval <interval_in_seconds>
```

## Options

- `--input`: Path to the input video file.
- `--output`: Directory where the sampled frames will be saved.
- `--interval`: Time interval in seconds between each frame sample.
```
