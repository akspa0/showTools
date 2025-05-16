# Product Context for PreFapMix

## Problem Statement

Processing multi-speaker audio recordings, especially paired call center data (`recv_out`, `trans_out`), for analysis and review is often a manual, time-consuming, and error-prone process. Key challenges include:

*   **Speaker Attribution:** Accurately identifying who said what in a conversation.
*   **PII Safety:** Ensuring sensitive information from original filenames or content is not exposed in processed outputs.
*   **Actionable Outputs:** Generating outputs that are immediately useful for various stakeholders (e.g., quality assurance, content creation, compliance). This includes needing clear, mixed audio, precise transcriptions, and quick summaries.
*   **Navigability:** Difficulty in pinpointing specific moments or utterances within a long recording without detailed, segmented transcriptions.
*   **Consistency:** Ensuring a consistent level of quality and format across all processed audio.

## Solved Problems & Goals

`PreFapMix` aims to solve these problems by providing an automated pipeline that:

1.  **Ensures PII Safety:** Implements a robust PII-safe naming convention for all intermediate and final files, typically based on a generated call timestamp (e.g., `call_YYYYMMDD-HHMMSS`).
2.  **Automates Processing:** Automates the workflow of audio separation, normalization, speaker diarization, transcription, and summarization.
3.  **Provides Detailed Transcriptions with Soundbites:** Generates not only full transcripts but also individual soundbite audio files (`.wav`) and corresponding text files (`.txt`) for each speaker segment. These soundbites are named using a PII-safe convention (sequence number and sanitized first few words of the utterance) and organized into speaker-specific subfolders, similar to the output of the `whisperBite.py` tool. This allows for easy access to and verification of specific utterances.
4.  **Facilitates Call Review:** For paired recordings, it mixes the `recv` and `trans` vocal stems into a clear stereo audio file, merges their respective transcripts (maintaining speaker identity and soundbite references), and produces a combined LLM-generated summary of the entire call.
5.  **Delivers Actionable Insights (Initial Draft - User Input Needed):** 
    *   Provides LLM-generated summaries for individual audio streams and combined calls. 
    *   **Value Proposition:** These summaries aim to give users a rapid understanding of the audio's core content, key discussion points, decisions, and action items without needing to listen to the entire recording or sift through a full transcript. This drastically reduces review time and allows for quicker information retrieval and response.
    *   **Use Cases:** 
        *   Quickly grasping the gist of a customer service call.
        *   Reviewing key takeaways from a meeting or interview.
        *   Identifying calls that require more detailed manual review based on summary content.
    *   **(Future Enhancements):** Could be expanded to include automated topic tagging, sentiment analysis, or extraction of specific entities (e.g., product names, customer IDs) mentioned in the audio, further enhancing the analytical value.
6.  **Standardizes Outputs:** Ensures all outputs are consistently formatted and organized.

## Desired User Experience

*   **Reliability:** The pipeline should consistently process audio files and produce accurate outputs.
*   **Ease of Use:** The system should be operable via a straightforward command-line interface with clear configuration options (e.g., through JSON workflow files).
*   **Navigable Outputs:** The output directory structure and file naming should be logical and easy to understand, allowing users to quickly find the information they need (e.g., mixed audio, specific speaker soundbites, full transcripts, summaries).
*   **Configurability:** Users should be able to configure aspects of the workflow, such as the models used for separation, diarization, transcription, and LLM summarization, as well as parameters like target loudness.
*   **Efficiency:** The pipeline should process audio in a reasonably timely manner.
*   **Transparency:** Clear logging should provide insight into the processing steps and any issues encountered.

## Target Users

*   Call center quality assurance teams.
*   Content analysts and creators who need to review and extract information from audio recordings.
*   Compliance officers needing to review call data.
*   Anyone needing to process and understand multi-speaker audio recordings efficiently and safely.