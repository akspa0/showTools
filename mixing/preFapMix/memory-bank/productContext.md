# Product Context

## Why This Project Exists

Organizations and individuals need to process call recordings for insights, compliance, and user-facing applications, while ensuring privacy and future-proofing. This pipeline solves the problem of extracting meaningful, PII-safe, and well-organized outputs from raw call audio.

## Problems Solved
- Ensures all outputs are sanitized and PII-safe.
- Provides a flat, user-friendly output structure for easy access and archiving.
- Automates extraction of soundbites, transcripts, and LLM-generated metadata.
- Supports robust error handling and logging for reliability.
- Allows flexible selection of ASR/transcription backend.

## How It Should Work
- User provides a call recording (file or URL).
- Pipeline processes audio through modular stages: CLAP event detection, preprocessing, diarization, transcription, LLM summary/metadata.
- Outputs are organized in a flat, sanitized folder structure, with all relevant files (final mix, soundbites, transcripts, metadata) easily accessible.
- Output folders are cleaned up if under a minimum size, ensuring only meaningful results are kept.

## User Experience Goals
- Outputs are easy to find, understand, and use.
- All files are named and organized for clarity and privacy.
- The system is robust to errors and edge cases, with clear logs for troubleshooting.

## Problem Statement

Processing multi-speaker audio recordings, especially paired call center data (`recv_out`, `trans_out`), for analysis and review is often a manual, time-consuming, and error-prone process. Key challenges include:

*   **Speaker Attribution:** Accurately identifying who said what in a conversation.
*   **PII Safety:** Ensuring sensitive information from original filenames or content is not exposed in processed outputs.
*   **Actionable Outputs:** Generating outputs that are immediately useful for various stakeholders (e.g., quality assurance, content creation, compliance). This includes needing clear, mixed audio, precise transcriptions, and quick summaries.
*   **Navigability & Organization:** Difficulty in pinpointing specific moments or utterances within a long recording without detailed, segmented transcriptions, and challenges in organizing large volumes of processed calls.
*   **Consistency:** Ensuring a consistent level of quality and format across all processed audio.

## Solved Problems & Goals

`PreFapMix` aims to solve these problems by providing an automated pipeline that:

1.  **Ensures PII Safety:** Implements a robust PII-safe naming convention for all intermediate and final files, typically based on a generated call timestamp (e.g., `call_YYYYMMDD-HHMMSS`) for processing stages, and sanitized LLM-generated names for final user-facing outputs.
2.  **Automates Processing:** Automates the workflow of audio separation, normalization, speaker diarization, transcription, and LLM-based content generation.
3.  **Provides Detailed Transcriptions with Soundbites:** Generates not only full transcripts but also individual soundbite audio files (`.wav`) and corresponding text files (`.txt`) for each speaker segment. These soundbites are named using a PII-safe convention and organized into speaker-specific subfolders.
4.  **Facilitates Call Review & Organization:** For paired recordings, it mixes the `recv` and `trans` vocal stems into a clear stereo audio file (to be MP3 compressed), merges their respective transcripts into JSON and plain text formats, and produces:
    *   **LLM-Generated Call Name:** A concise, witty, PII-safe title for easy identification.
    *   **LLM-Generated Call Synopsis:** A structured summary of the call's purpose, key topics, decisions, and action items.
    *   **LLM-Generated Hashtag Categories:** Tags for thematic organization and quick content filtering.
5.  **Delivers Actionable & Organized Outputs:** 
    *   Provides the above LLM-generated content for individual audio streams and combined calls.
    *   Creates a "final output" directory for each call, named with the sanitized call name, containing the primary audio (MP3), plain text transcript, synopsis, and hashtags in a user-friendly flat structure.
    *   For batches of paired calls, generates a "show file" consisting of a concatenated MP3 audio of all calls, a timestamp file (with call names) for navigation, and a combined transcript for the entire show.
    *   **Value Proposition:** These outputs drastically reduce review time, allow for quicker information retrieval, improve call discoverability and organization through names and hashtags, and enable efficient review of multiple calls via the "show file".
    *   **Use Cases:** 
        *   Quickly grasping the gist and key details of a customer service call via its name and synopsis.
        *   Organizing and searching/filtering calls using hashtags.
        *   Efficiently reviewing a sequence of related calls using the "show file."
        *   Identifying calls that require more detailed manual review based on summary content.
6.  **Standardizes Outputs:** Ensures all outputs are consistently formatted (including MP3 for audio) and organized logically.

## Desired User Experience

*   **Reliability:** The pipeline should consistently process audio files and produce accurate outputs.
*   **Ease of Use:** The system should be operable via a straightforward command-line interface with clear configuration options.
*   **Navigable & Usable Outputs:** The "final output" directory structure, file naming (using LLM-generated names), and file formats (MP3 audio, plain text transcripts) should be logical, easy to understand, and directly usable. The "show file" should provide an efficient way to review multiple calls.
*   **Configurability:** Users should be able to configure aspects of the workflow, such as the models used and output directories.
*   **Efficiency:** The pipeline should process audio in a reasonably timely manner.
*   **Transparency:** Clear logging should provide insight into the processing steps and any issues encountered.

## Target Users

*   Call center quality assurance teams.
*   Content analysts and creators who need to review and extract information from audio recordings.
*   Compliance officers needing to review call data.
*   Anyone needing to process, understand, organize, and efficiently review multi-speaker audio recordings.

## CLAP Segmentation Context
- CLAP segmentation allows the pipeline to process any long-form or non-call audio (e.g., podcasts, radio, meetings) by splitting it into logical segments based on detected events (e.g., tones, music, scene changes).
- Each segment is treated as a virtual call, enabling independent analysis, transcription, and metadata generation.
- This greatly expands the pipeline's applicability and value beyond call center data.