# Tech Context

## Technologies Used
- Python 3.10+
- PyTorch (for Whisper)
- NVIDIA NeMo/Parakeet (for Parakeet ASR)
- pyannote (for diarization)
- pydub, ffmpeg (for audio slicing/mixing)
- mutagen (for ID3 tagging)
- yt-dlp (for audio URL support)
- Robust logging (Python logging module)

## Development Setup
- Modular scripts for each pipeline stage
- CLI/config options for ASR engine, LLM endpoint, and input type
- Requirements.txt includes all dependencies

## Technical Constraints
- All outputs must be PII-safe and user-facing
- Output folders must be flat, sanitized, and future-proof
- Minimum output folder size enforced (192KB)
- All scripts must handle errors gracefully and log all actions

## Dependencies
- See requirements.txt for full list
- All major dependencies are open source and actively maintained

## CLAP Segmentation Technology
- Uses CLAP Annotator for event detection (e.g., tones, music, scene changes).
- Segments audio based on detected event boundaries.
- Each segment is saved as a new audio file and processed independently.
- Segmentation is configurable (prompts, thresholds, chunk duration, min/max segment length, overlap).
- Enables robust support for podcasts, radio, meetings, and other non-call audio.