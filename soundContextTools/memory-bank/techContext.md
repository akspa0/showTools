# techContext.md

**Purpose:**
Describes technologies used, development setup, technical constraints, and dependencies.

## Technologies Used

- Python 3.x
- parakeet (HuggingFace): TDT model for audio context
- audio-separator: Source separation (vocals/instruments)
- CLAP (HuggingFace): Audio annotation
- audiomentations: Audio augmentations, loudness normalization
- mutagen: Audio metadata (ID3) reading/writing and propagation
- (Planned) pyannote: Speaker diarization

## Development Setup

- Install dependencies via pip (requirements.txt to be created)
- Requires access to HuggingFace models and PyPI packages

## Technical Constraints

- Must handle large batches of files efficiently
- Must ensure privacy (no PII in outputs)
- File tracking and metadata lineage must be robust and lossless

## Dependencies

- parakeet-tdt-0.6b-v2 (https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)
- audio-separator (https://pypi.org/project/audio-separator/)
- transformers (for CLAP, https://huggingface.co/docs/transformers/model_doc/clap)
- audiomentations (https://github.com/iver56/audiomentations)
- mutagen (https://mutagen.readthedocs.io/en/latest/) 