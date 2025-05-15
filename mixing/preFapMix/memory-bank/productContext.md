# Product Context

## Problem Statement
Audio recordings from various sources (at 8kHz) need to be properly processed, mixed, and transcribed for analysis. The current implementation has several limitations:
1. It mixes 8kHz audio channels and then resamples the mix to 44.1kHz, resulting in quality loss
2. It relies on fish-audio-processor for transcription, which uses an outdated Whisper model
3. Output organization is not optimized for chronological review
4. There's no separation between vocals and instrumentals, limiting clarity and transcription quality
5. It doesn't properly pair and process both sides of a conversation (recv_out/trans_out files)
6. There's no mechanism to create a unified "show" file with all calls in sequence

## Solution
PreFapMix provides a streamlined pipeline to:
1. Match recv_out/trans_out file pairs that belong to the same conversation
2. Process each channel separately and chronologically
3. Separate vocals from instrumentals using python-audio-separator
4. Normalize vocal and instrumental stems to target LUFS values using `audiomentations` for clear differentiation.
5. Properly resample each channel to 44.1kHz
6. Transcribe vocals using OpenAI Whisper (large-v3 model)
7. Mix the resampled channels into stereo output with adjustable instrumental volume
8. Normalize the final mixed output to a consistent LUFS target using `ffmpeg loudnorm`.
9. Generate complete conversation transcripts with speaker attribution
10. Organize outputs chronologically
11. Generate consolidated "show files" with metadata including timestamps
12. Provide both command-line and GUI interfaces

## User Experience Goals
- Simple, intuitive interface with only essential controls
- Clear visibility of processing options
- Reliable and consistent output quality, especially regarding audio levels and vocal clarity.
- Chronological organization of processed files
- Complete conversation transcripts for both sides of communication
- Adjustable mixing options for instrumental/vocal balance
- Consolidated show output with detailed metadata
- Minimal configuration required for standard workflows

## Target Users
- Audio analysts who need to process and transcribe conversations
- Researchers working with audio data from different sources
- Users who need to mix and normalize audio recordings for analysis
- Content creators organizing communication recordings chronologically
- Analysts who need both transcript content and full audio in an organized format 