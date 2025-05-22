# techContext.md

**Purpose:**
Describes technologies used, development setup, technical constraints, and dependencies.

## Technologies Used

- Python 3.x
- torchaudio: audio loading, resampling, and saving
- soundfile: audio file I/O
- numpy: array and audio processing
- librosa: resampling and audio utilities
- pyannote: speaker diarization
- parakeet (HuggingFace): ASR transcription
- OpenAI Whisper: optional ASR transcription
- pyloudnorm: loudness normalization
- rich: console output and tracebacks
- tqdm: progress bars
- requests: LLM API calls
- mutagen: audio metadata (ID3) reading/writing and propagation

## Development Setup

- Install dependencies via pip (requirements.txt provided)
- Requires access to HuggingFace models and PyPI packages
- CLI options for ASR engine, LLM config, and call tones
- Workflow logic and LLM tasks defined in workflows/ JSON files

## Technical Constraints

- Must handle large batches of files efficiently
- Must ensure privacy (no PII in outputs, logs, or manifests)
- File tracking and metadata lineage must be robust and lossless
- All output files and logs must be auditable and extensible

## Dependencies

- parakeet-tdt-0.6b-v2 (https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)
- audio-separator (https://pypi.org/project/audio-separator/)
- transformers (for CLAP, https://huggingface.co/docs/transformers/model_doc/clap)
- audiomentations (https://github.com/iver56/audiomentations)
- mutagen (https://mutagen.readthedocs.io/en/latest/)
- torchaudio, soundfile, numpy, librosa, pyannote, pyloudnorm, tqdm, rich, requests 