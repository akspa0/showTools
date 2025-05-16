# PreFapMix Project Brief

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