# Product Context

## Problem Statement
Audio recordings from various sources (often at 8kHz) need to be processed, mixed (for calls), transcribed, and analyzed. Key challenges include:
1. Ensuring high-quality audio processing, including proper resampling, stem separation, and loudness normalization.
2. Using up-to-date transcription and diarization technologies for accuracy.
3. Organizing outputs logically, especially for paired call data (`recv_out`/`trans_out`), in a PII-safe manner.
4. Providing detailed, speaker-segmented transcriptions along with corresponding audio soundbites for easy review and use, similar to the output style of `whisperBite.py`.
5. Generating useful summaries or analyses from the transcribed conversations.

## Solution: The "PreFapMix" Workflow System
This project implements a modular, workflow-driven system (`workflow_executor.py` and `call_processor.py`) to:
1.  **Process Individual Audio Streams (`workflow_executor.py`):**
    *   Detect sound events (CLAP).
    *   Separate vocals from instrumentals (`audio_preprocessor.py`).
    *   Normalize vocal and instrumental stems to target LUFS values.
    *   Diarize speaker segments using `pyannote.audio` (`diarization_module.py`).
    *   Transcribe vocals using `openai-whisper`, guided by diarization (`transcription_module.py`).
        *   **Target Output:** This module is being enhanced to produce individual audio soundbites (`.wav`) and text files (`.txt`) for each speaker segment, organized into speaker-specific subdirectories (e.g., `S0/`, `S1/`) with content-derived filenames (e.g., `0001_some_words.wav`). This mirrors the desired output style of `whisperBite.py`.
    *   Generate per-stream summaries using a local LLM via LM Studio (`llm_module.py`).
    *   Maintain PII-safe naming for all intermediate outputs using timestamp-based identifiers (e.g., `call_YYYYMMDD-HHMMSS`).
2.  **Aggregate and Finalize Call Data (`call_processor.py`):**
    *   Identify and pair processed `recv_out` and `trans_out` streams based on their PII-safe call ID.
    *   Mix audio from paired vocal stems into a stereo file.
    *   Merge JSON transcripts from paired streams.
    *   **Target:** Copy the detailed soundbite structures (speaker subdirectories with `.wav` and `.txt` files) from the transcription stage of each stream into the final call output directory.
    *   Generate a combined LLM summary for the entire call.
    *   Save all final outputs into a PII-safe call-specific directory (e.g., `processed_calls_test/call_YYYYMMDD-HHMMSS/`).

## User Experience Goals
- **Reliable and Consistent Output Quality:** Clear audio, accurate transcription and diarization.
- **Detailed and Usable Transcription Output:** Beyond a single JSON, provide individual speaker audio soundbites and their text, mirroring `whisperBite.py`'s utility for review and further use.
- **PII Safety:** Ensure all generated filenames and directory names are free of personally identifiable information, relying on timestamps for identification.
- **Organized Outputs:** Logically structured output directories for individual stream processing runs and for final aggregated call data.
- **CLI-Driven Workflow:** Robust command-line interface for both `workflow_executor.py` and (indirectly) `call_processor.py`.
- **Minimal Configuration for Standard Use:** Sensible defaults for most workflow parameters.

## Target Users
- Analysts needing to process, transcribe, and review call recordings or other audio communications.
- Users requiring detailed, speaker-segmented audio soundbites alongside transcriptions.
- Anyone needing to apply a consistent audio processing pipeline (separation, normalization, diarization, transcription, LLM summary) to a batch of audio files or call data. 