# System Patterns for PreFapMix

## Overall Architecture: Multi-Tier Processing

The system employs a multi-tier architecture to handle audio processing:

1.  **`workflow_executor.py` (Individual File Processing Tier):**
    *   **Responsibility:** Processes single audio files through a series of configurable stages (e.g., stem separation, normalization, diarization, transcription, per-stream LLM analysis if configured).
    *   **Input:** An individual audio file path and a workflow definition JSON file.
    *   **Output:** A structured run directory containing all intermediate and final outputs for that single audio file, named using a PII-safe identifier.
    *   **PII Safety:** Generates a `pii_safe_identifier_stem` passed to modules.
    *   **Call Folder Awareness:** If processing files within a "call folder" (containing `recv_out` or `trans_out` named files), it invokes `call_processor.py` as a final step after processing all relevant individual files in that folder.

2.  **`call_processor.py` (Call Aggregation & Finalization Tier):**
    *   **Responsibility:** Aggregates outputs from `workflow_executor.py` for calls (pairs or singles), performs mixing, transcript merging, generates multiple LLM-derived outputs (name, synopsis, hashtags), and creates a structured "final output" directory for each call.
    *   **Input:** A base directory containing `workflow_executor.py` run directories, and a `--final_output_dir` path.
    *   **Primary Output (`processed_calls` directory - e.g., `workspace/processed_calls/call_YYYYMMDD-HHMMSS/`):
        *   Mixed stereo audio (`.wav`) for pairs.
        *   Single stream audio (`.wav`) for singles.
        *   Merged transcript JSON (`.json`).
        *   LLM-generated call name (`_suggested_name.txt`).
        *   LLM-generated call synopsis (`_combined_call_summary.txt`).
        *   LLM-generated hashtags (`_hashtags.txt`).
        *   Copied soundbite directories (e.g., `RECV_S0/`, `TRANS_S0/`).
        *   Individual stream summaries if applicable.
    *   **Secondary Output ("Final Output" Directory - e.g., `final_processed_calls/[Sanitized_Call_Name]/`):
        *   Sanitized, LLM-derived call name used as the folder name.
        *   Primary call audio (`[Sanitized_Call_Name].mp3`) - MP3 compressed.
        *   Plain text transcript (`transcript.txt`).
        *   Synopsis (`synopsis.txt`).
        *   Hashtags (`hashtags.txt`).
        *   `call_type.txt` (indicates "pair" or "single_recv" etc., for `show_compiler.py`).
    *   **LLM Interaction:** Makes three distinct calls to `llm_module.generate_llm_summary` for call naming, synopsis, and hashtag categories using specific system prompts.

3.  **`show_compiler.py` (Show Compilation Tier - Planned):
    *   **Responsibility:** Compiles multiple processed *paired* calls from the "final output" directory structure into a single browsable "show."
    *   **Input:** The base "final output" directory (e.g., `final_processed_calls/`).
    *   **Output (in a dedicated `show_output` directory):
        *   `show_audio.mp3`: A single MP3 file concatenating the primary audio of all selected paired calls, in chronological order.
        *   `show_timestamps.txt`: A text file listing the start time of each call within `show_audio.mp3`, along with the call's name.
        *   `show_transcript.txt`: A single text file concatenating the plain text transcripts of all calls included in the show.
    *   **Key Logic:** 
        *   Scans the input directory for call folders.
        *   Filters for calls marked as "pair" (using `call_type.txt`).
        *   Sorts these paired calls chronologically (based on their original `call_id` timestamp).
        *   Uses `ffmpeg` for audio concatenation and potentially final MP3 compression if sources are WAV.
        *   Parses call durations and individual plain text transcripts to generate metadata files.

## Key Design Patterns & Principles
*   Modular Design: Core tasks in specialized modules.
*   Configuration-Driven Workflow: `workflow_executor.py` uses JSON configs.
*   PII-Safe by Design: Throughout intermediate stages and for final outputs.
*   CLI-Driven: All main scripts (`workflow_executor.py`, `call_processor.py`, planned `show_compiler.py`) are CLI operable.
*   Tiered Processing: Individual files -> Call aggregation/finalization -> Show compilation.

## Data Flow & Output Structure

*   **`workflow_executor.py` Run Directory Structure (Example):**
    ```
    workspace/workflow_runs/WorkflowName_InputFileStem_Timestamp/
    ├── 00_stage_name/
    │   └── pii_safe_prefix_output_file.ext
    ├── 01_another_stage/
    │   └── pii_safe_prefix_another_file.ext
    ├── ...
    └── WorkflowName_InputFileStem_Timestamp_workflow_summary.json
    └── WorkflowName_InputFileStem_Timestamp_executor.log
    ```
*   **`transcription_module.py` Soundbite Output Structure (within its stage directory):**
    ```
    .../03_transcription/
    ├── pii_safe_prefix_transcription.json  (Main transcript with references)
    ├── S0/                                 (Speaker 0 soundbites)
    │   ├── 0000_first_few_words.wav
    │   ├── 0000_first_few_words.txt
    │   ├── 0001_next_segment_words.wav
    │   ├── 0001_next_segment_words.txt
    │   └── ...
    ├── S1/                                 (Speaker 1 soundbites)
    │   ├── 0000_some_other_words.wav
    │   ├── 0000_some_other_words.txt
    │   └── ...
    └── ...
    ```
*   **`call_processor.py` Primary Output Directory Structure (Example - `processed_calls/call_YYYYMMDD-HHMMSS/`):
    ```
    workspace/processed_calls/call_20231026-103000/
    ├── call_20231026-103000_mixed_vocals.wav
    ├── call_20231026-103000_merged_transcript.json
    ├── call_20231026-103000_suggested_name.txt
    ├── call_20231026-103000_combined_call_summary.txt
    ├── call_20231026-103000_hashtags.txt
    ├── RECV_S0/
    │   ├── ... (soundbites)
    └── TRANS_S0/
        ├── ... (soundbites)
    ```
*   **`call_processor.py` "Final Output" Directory Structure (Example - `final_processed_calls/[Sanitized_Call_Name]/`):
    ```
    final_processed_calls/A_Witty_Call_About_Penguins/
    ├── A_Witty_Call_About_Penguins.mp3
    ├── transcript.txt
    ├── synopsis.txt
    ├── hashtags.txt
    └── call_type.txt 
    ```
*   **`show_compiler.py` Output Directory Structure (Example - `show_output/`):
    ```
    show_output/
    ├── show_audio.mp3
    ├── show_timestamps.txt
    └── show_transcript.txt
    ```

## Important Considerations
*   Dependency Management (`requirements.txt`, system `ffmpeg`).
*   Resource Usage (CPU, GPU, RAM for various stages).
*   Model Management (downloading/caching for diarization, transcription, separation).