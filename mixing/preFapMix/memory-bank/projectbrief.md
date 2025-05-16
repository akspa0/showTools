# PreFapMix Project Brief

## Core Requirements
- Process audio files (often 8kHz per channel) for transcription, PII-safe storage, and analysis.
- Match `recv_out` and `trans_out` file pairs from call data.
- Process individual audio streams through a defined workflow: audio separation (vocals/instrumentals), loudness normalization, speaker diarization, transcription, and LLM-based analysis.
- For transcription, provide detailed output including individual audio soundbites (`.wav`) and text (`.txt`) for each speaker segment, with PII-safe, content-derived filenames, organized into speaker-specific subdirectories.
- For paired call data, combine processed streams: mix audio, merge transcripts (including references to individual soundbites), and generate combined LLM summaries.
- Ensure all output filenames and directory structures are PII-safe, primarily using timestamp-based identifiers.
- Generate complete conversation transcripts with accurate speaker attribution.
- (Future) Create a chronologically ordered "show" file containing all processed calls with metadata.

## LLM Summarization & Analysis Requirements (Initial Draft - User Input Needed)
- **Purpose:** To provide concise, actionable insights from transcribed audio content, reducing the need for users to listen to full recordings or read lengthy transcripts.
- **Key Outputs:** 
    - **Concise Summary:** A brief overview of the main topics discussed, key decisions made, and action items identified in the audio.
    - **(Optional/Future) Entity Extraction:** Identification of key entities (people, organizations, locations, etc.).
    - **(Optional/Future) Sentiment Analysis:** Overall sentiment of the conversation or specific segments.
- **Audience:** Users who need to quickly understand the content of an audio recording without manual review (e.g., call center supervisors, researchers, individuals reviewing past meetings).
- **Value:** Saves time, improves accessibility of audio content, and enables quicker decision-making based on recorded conversations.

## Goals
- Implement a robust, modular, workflow-driven system using `workflow_executor.py` (for per-file processing) and `call_processor.py` (for call data aggregation).
- Utilize `pyannote.audio` for speaker diarization and `openai-whisper` for transcription.
- Leverage `audio-separator` (CLI/library) and `audiomentations` for effective stem separation and loudness normalization.
- Achieve a well-balanced mix for call audio with prominent vocals.
- Ensure consistent loudness across all outputs.
- Provide detailed, `whisperBite.py`-style transcription outputs (individual soundbites).
- Integrate local LLM processing via LM Studio for summaries and analysis.
- Maintain a straightforward, efficient, and configurable CLI-driven workflow.
- Support batch processing of audio files and call folders.
- Ensure all intermediate and final outputs are organized logically and named in a PII-safe manner. 