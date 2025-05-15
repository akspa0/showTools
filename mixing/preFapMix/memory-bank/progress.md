# Progress Tracking

## What Works

### Core Architecture & Workflow
- **`workflow_executor.py`:**
    - Successfully executes a pipeline of audio processing stages defined in a JSON file.
    - Manages input/output paths, unique run directories, and data passing between stages.
    - Supports batch processing from an `--input_dir`.
    - Handles PII-safe run ID generation (`_generate_pii_safe_run_identifier_stem`) for intermediate workflow run directories (e.g., `DefaultAudioAnalysisWorkflow_call_YYYYMMDD-HHMMSS_TIMESTAMP`).
    - Passes `pii_safe_file_prefix` to each stage module for consistent temporary/output file naming within stages.
    - Identifies "call folders" (containing `recv_out-*` or `trans_out-*` files) and skips general `out-*` files from main workflow processing if `--input_dir` is a call folder.
    - **Subprocess Invocation:** If `--input_dir` is identified as a call folder, after processing individual `recv_out-*` and `trans_out-*` files, it calls `call_processor.py` as a subprocess, passing necessary arguments (`--input_run_dir` pointing to its own output directory, `--output_call_dir`, `--llm_model_id`).
- **`call_processor.py` (Initial Version - Evolving):**
    - **Purpose:** To process the outputs of `workflow_executor.py` for "call folders", identify `recv_out`/`trans_out` file pairs belonging to the same call, and combine their processed data.
    - **Current State:**
        - Basic file structure, logging, and command-line argument parsing (`--input_run_dir`, `--output_call_dir`, `--log_level`, `--llm_model_id`).
        - Logic for `generate_call_id_from_workflow_dir_name` (extracting PII-safe call identifiers like `call_YYYYMMDD-HHMMSS` from `workflow_executor.py` output directory names).
        - `find_call_pairs` groups related workflow run directories.
        - `process_call` function:
            - Retrieves `vocals_normalized.wav`, `transcript.json`, and `final_analysis_summary.txt` from stage output directories within each workflow run.
            - Copies these files to a call-specific output directory (e.g., `processed_calls_test/call_YYYYMMDD-HHMMSS/`).
            - **Audio Mixing:** Mixes `recv_vocal_stem_path` and `trans_vocal_stem_path` into a stereo WAV file.
            - **Transcript Merging:** Merges `recv_transcript_path` and `trans_transcript_path` JSON files chronologically, prefixing speaker labels.
            - **Combined LLM Summary:** If a merged transcript exists, calls `llm_module.generate_llm_summary` with a user-defined system prompt ("Based on the following conversation, write a story about what is going on.") and the `llm_model_id` from CLI.
- **Modules (within `workflow_executor.py` pipeline):**
    - **`clap_module.py`:**
        - Runs CLAP annotation.
        - Accepts `pii_safe_file_prefix` and uses it for temporary PII-safe input audio copy and output JSON naming (e.g. `call_XYZ_temp_clap_input_clap_results.json`).
    - **`audio_preprocessor.py`:**
        - Performs stem separation and normalization.
        - Accepts `pii_safe_file_prefix` and uses it for all temporary and final output stem filenames (e.g. `call_XYZ_vocals_normalized.wav`).
    - **`diarization_module.py`:**
        - Performs speaker diarization using `pyannote.audio`.
        - Accepts `pii_safe_file_prefix` (though not actively used for naming outputs if input vocal stem is already correctly prefixed). Produces RTTM file (e.g. `call_XYZ_vocals_normalized_diarization.rttm`).
    - **`transcription_module.py` (Partially Fixed):**
        - Uses `openai-whisper` for transcription, guided by RTTM.
        - Accepts `pii_safe_file_prefix` and uses it for the main output JSON (e.g., `call_XYZ_transcription.json`).
        - **RTTM Parsing:** Correctly uses `pyannote.database.util.RTTMParser` to load RTTM files.
    - **`llm_module.py` (Signature Fixed):**
        - Integrates with LM Studio for summarization.
        - `run_llm_summary` (called by workflow executor) now accepts `pii_safe_file_prefix` (though not actively used by it).
        - `generate_llm_summary` (called by `call_processor.py`) takes transcript, system prompt, and LLM config.

### Overall
- PII-safe naming for intermediate workflow run directories (e.g., `DefaultAudioAnalysisWorkflow_call_20250512-020958_20250515_042033`).
- PII-safe naming for temporary and output files within `clap_module`, `audio_preprocessor`, and `transcription_module` using the `pii_safe_file_prefix`.

## What's Left to Build / Refine

### Current High Priority
1.  **`transcription_module.py` - Soundbite Generation (Align with `whisperBite.py`):**
    *   **Goal:** Output individual audio soundbites (`.wav`) and their corresponding text files (`.txt`) for each speaker segment, with filenames based on sequence and content (e.g., `0001_some_transcribed_words.wav/.txt`).
    *   **Tasks:**
        *   Add a `sanitize_filename` utility.
        *   Modify `slice_audio_for_transcription` to include a unique `sequence_id` for each slice.
        *   Modify `transcribe_sliced_segments`:
            *   Accept `persistent_output_dir` (the stage's main output dir) and `pii_safe_file_prefix`.
            *   For each transcribed temporary slice:
                *   Create speaker-specific subdirectories (e.g., `S0/`, `S1/`) within `persistent_output_dir`.
                *   Generate soundbite filenames (e.g., `{sequence_id:04d}_{sanitized_first_words}.wav/.txt`).
                *   Copy the temporary audio slice to this new persistent path.
                *   Create the corresponding `.txt` file.
                *   Update the segment data in `all_transcribed_segments` to include `soundbite_audio_path` and `soundbite_text_path` pointing to these persistent files.
        *   Update `run_transcription` to facilitate this and ensure the main JSON output (`{pii_safe_file_prefix}_transcription.json`) includes these new paths in its segment list.
2.  **`call_processor.py` - Handling Richer Transcription Output:**
    *   Modify `process_call` to look for and copy the new speaker-specific subdirectories (e.g., `S0/`, `S1/`) containing soundbites from the transcription stage output into the final processed call directory (e.g., `processed_calls_test/<call_id>/S0/`).

### Lower Priority / Future Enhancements
- **Refine LLM Integration:**
    - Improve prompt engineering for `call_processor.py`'s combined summary.
    - Investigate token limits for LM Studio and implement transcript chunking if necessary for `generate_llm_summary` in `llm_module.py` and `call_processor.py`.
- **CLAP Module Redundancy:** `ClapAnnotator` performs its own stem separation. `audio_preprocessor.py` also performs stem separation. Evaluate if the CLAP module's separation is necessary or if it can directly use the main audio, and if the `audio_preprocessor.py` separation is sufficient for all downstream tasks. The `default_audio_analysis_workflow.json` has `clap_separator_model: null` which *should* make ClapAnnotator use the main audio, but the logs show it still defaulting to "Mel Band RoFormer Vocals". This needs verification.
- **Error Handling & Logging:** Continue to refine.
- **Configuration & Usability:** Review defaults, improve documentation.
- **Testing:** Comprehensive testing with diverse real-world call audio.

## Current Status
- **Debugging workflow integration:** Primarily focusing on `transcription_module.py` to ensure it produces detailed soundbite outputs as per `whisperBite.py`'s functionality.
- Key modules (`workflow_executor`, `call_processor`, `clap_module`, `audio_preprocessor`, `diarization_module`, `llm_module`) have undergone significant development and PII-safety refactoring.
- RTTM parsing in `transcription_module.py` and `pii_safe_file_prefix` argument handling in `llm_module.py` have been addressed.

## Known Issues
- **Transcription Output:** `transcription_module.py` does not yet save individual speaker soundbites (`.wav` and `.txt`) with content-derived names to persistent speaker-specific directories; it only produces a single JSON output. This is the **current primary focus of development.**
- **`call_processor.py`'s `_find_output_file`:** Currently retrieves specific filenames. It will need to be adapted or new logic added to copy entire speaker-specific directories (e.g., `S0/`, `S1/`) once `transcription_module.py` creates them.
- **CLAP Separator Model:** Despite setting `clap_separator_model: null` in workflow JSON, logs indicate `ClapAnnotator` may still be using its default internal separator ("Mel Band RoFormer Vocals"). This behavior needs to be confirmed and addressed if CLAP analysis on the raw input (or specific stems from `audio_preprocessor`) is desired.
- **LM Studio Token Limits:** Not yet handled explicitly; long transcripts might exceed context windows.
- **Test Coverage:** More extensive testing with varied call data is needed.

## Current Status
- Memory bank updated with refined, focused approach
- Core architecture design completed
- Implementation plan established for essential functionality
- Components from v3.8 projects identified for integration

## Known Issues
- Current implementation doesn't match recv_out/trans_out pairs
- No vocal/instrumental separation
- Relies on fish-audio-processor for transcription
- No mechanism for generating consolidated show files
- No metadata for call timestamps
- No mechanism for adjusting instrumental volume in the final mix 