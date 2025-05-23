# Progress

## What Works
- Whisper diarization-based segmenting and per-segment transcription (with soundbites and timestamps) is robust.
- Parakeet transcription works for full audio, but segment-based logic is being implemented.
- Pipeline output keys are now consistent for downstream stages.
- Modular pipeline: CLAP, Preprocess, Diarize, Transcribe, LLM Summary all functional
- Final output builder and Character.AI description generator are robust and compliant
- Output folders are sanitized, unique, and cleaned up if under 192KB
- Soundbites are generated for all speakers, with correct ID3 tags
- Both Whisper and Parakeet ASR engines are supported
- yt-dlp support for audio URLs is implemented
- Robust error handling and logging throughout

## What's Left to Build
- **TOP PRIORITY:** Full refactor of CLAP segmentation using Hugging Face transformers CLAP model directly (no legacy wrappers or CLI)
- Remove all stem separation and legacy CLAPAnnotatorWrapper logic from segmentation
- Implement robust, prompt-driven event detection and pairing logic
- Use ffmpeg/ffprobe for all audio I/O
- Output segments and metadata as before, but with a clean, maintainable codebase
- Provide a minimal test script for validation
- Ensure upstream pipeline always places files in correct folders for downstream modules
- Further improvements to error handling and edge case management
- Ongoing monitoring for output structure and naming issues

## Current Status
- Diarization-based segmenting and per-segment transcription (Whisper/Parakeet) is now the top priority.
- All transcription must use diarization-based segments, with per-segment TXT/JSON outputs including timestamps.
- Master transcript TXT is plain text; master JSON includes all segment details and timings.
- The Parakeet full-audio path is deprecated; all ASR is segment-based.
- CLAP segmentation (Hugging Face CLAP) is implemented but still needs thorough testing.
- Pipeline is robust, PII-safe, and user-facing
- All major features are implemented and functional
- Outstanding issues are mostly related to upstream pipeline structure

## Known Issues
- Occasional upstream misplacement of files can cause downstream failures
- Edge cases in output structure/naming may still arise and require monitoring

## CLAP Segmentation (In Progress)
- Now the top priority for development.
- Will enable robust, configurable segmentation of all input audio, not just phone calls.
- Each segment will be processed as a virtual call, greatly increasing pipeline flexibility and applicability 