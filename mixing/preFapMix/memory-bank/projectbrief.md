# Project Brief

This project is a robust, modular audio processing pipeline designed for call recordings. Its core requirements are:

- **PII-safe output**: All outputs are sanitized for privacy and compliance.
- **User-facing, future-proof structure**: Output folders are flat, sanitized, and organized for end users.
- **Stereo mixing**: Final MP3s are mixed from vocal and instrumental stems, with correct panning and volume.
- **Soundbite extraction**: Per-speaker soundbites are generated, with ID3 tags and content-derived filenames.
- **LLM-based metadata**: Summaries, titles, and other metadata are generated using LLMs, with flexible backend selection.
- **Robust error handling and logging**: All scripts are designed for reliability and traceability.
- **Flexible ASR backend**: Supports both Whisper and NVIDIA Parakeet TDT 0.6B V2, selectable via CLI/config.
- **Modular pipeline**: Each stage (CLAP, Preprocess, Diarize, Transcribe, LLM) is a separate, robust module.

The pipeline is designed to be extensible, maintainable, and safe for sensitive data.

## Core Requirements
- Process audio files (often 8kHz per channel) for transcription, PII-safe storage, and analysis.
- Match `recv_out` and `trans_out` file pairs from call data.
- Process individual audio streams through a defined workflow: audio separation (vocals/instrumentals), loudness normalization, speaker diarization, transcription.
- For paired call data, combine processed streams: mix audio, merge transcripts (including references to individual soundbites), and generate LLM-derived call name, synopsis, and categorical hashtags.
- For transcription, provide detailed output including individual audio soundbites (`.wav`) and text (`.txt`) for each speaker segment, with PII-safe, content-derived filenames, organized into speaker-specific subdirectories.
- Create a user-friendly "final output" directory for each processed call, featuring a sanitized call name as the folder name, MP3 compressed primary audio, a plain text transcript, the LLM-generated synopsis, and hashtags.
- For processed paired calls, create a chronologically ordered "show" file, including a single concatenated MP3 audio of all calls, a timestamp file detailing the start of each call with its name, and a combined plain text transcript of all calls in the show.
- Ensure all output filenames and directory structures are PII-safe, primarily using timestamp-based identifiers for intermediate files and sanitized LLM-generated names for final outputs where appropriate.
- Generate complete conversation transcripts with accurate speaker attribution.
- All transcription (Whisper and Parakeet) must use diarization-based segmenting: audio is always sliced into diarization-based segments (soundbites), and each segment is transcribed individually, with per-segment TXT/JSON outputs including timestamps.
- Master transcript TXT is plain text; master JSON includes all segment details and timings.
- The "full-audio" Parakeet transcription path is deprecated; all ASR is diarization-based, segment-by-segment.
- CLAP segmentation (using Hugging Face CLAP) is a key feature for event detection and future rules-based segmentation, but is not the current top priority.

## LLM-Generated Content Requirements
- **Purpose:** To provide concise, actionable insights and organizational metadata from transcribed audio content.
- **Key Outputs:** 
    - **Call Name:** A concise (under 10 words), witty/absurd, PII-safe title with no punctuation, suitable for folder/file naming and display.
    - **Call Synopsis:** A structured, objective, PII-safe summary highlighting the main reason for the call, key topics, decisions/outcomes, and action items.
    - **Hashtag Categories:** 3-5 space-separated, PII-safe hashtags (e.g., #OrderInquiry #ProductDefect) summarizing main subjects and themes for organizational purposes.
- **Audience:** Users who need to quickly understand and organize audio content (e.g., call center supervisors, researchers, individuals reviewing past meetings).
- **Value:** Saves time, improves accessibility and organization of audio content, and enables quicker decision-making.

## Goals
- Implement a robust, modular, workflow-driven system using `workflow_executor.py` (for per-file processing), `call_processor.py` (for call data aggregation and finalization), and a new `show_compiler.py` (for multi-call show file generation).
- Utilize `pyannote.audio` for speaker diarization and `openai-whisper` for transcription.
- Leverage `audio-separator` (CLI/library) and `audiomentations` for effective stem separation and loudness normalization.
- Achieve a well-balanced mix for call audio with prominent vocals.
- Ensure consistent loudness across all outputs.
- Provide detailed, `whisperBite.py`-style transcription outputs (individual soundbites).
- Integrate local LLM processing via LM Studio for generating call names, synopses, and hashtag categories.
- Output primary audio files in MP3 format in the final user-facing directories.
- Maintain a straightforward, efficient, and configurable CLI-driven workflow.
- Support batch processing of audio files and call folders.
- Ensure all intermediate and final outputs are organized logically and named in a PII-safe manner.

## Future Enhancements
- **Utilize Structured JSON Transcripts:** Leverage the detailed, structured JSON output from the transcription stage (containing word-level timestamps and granular segment data) for more advanced LLM tasks. This could include creating highly detailed, chronologically accurate show-level synopses by combining multiple call transcripts, enabling fine-grained Q&A about *when* specific things were said, improving event correlation, and providing richer data for potential LLM fine-tuning in the future.

## CLAP-Based Segmentation
- The pipeline supports CLAP-based segmentation for all input audio, not just phone calls.
- CLAP event annotation is used to detect boundaries (e.g., ringing, tones, music, scene changes) and split audio into logical segments.
- Each segment is processed as a separate "call" through the full pipeline (preprocessing, diarization, transcription, LLM, etc.).
- Segmentation is configurable (prompts, min/max segment length, overlap, etc.) and robust to different audio types.
- This enables the pipeline to handle podcasts, radio, long recordings, and arbitrary audio, not just call center data. 