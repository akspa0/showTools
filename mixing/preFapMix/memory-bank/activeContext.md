# Active Context

## Current Focus
- **Aligning `transcription_module.py` with `whisperBite.py` soundbite output:** This is the primary active task. The goal is for `transcription_module.py` to output individual audio soundbites (`.wav`) and their corresponding text files (`.txt`) for each speaker segment. Filenames should be based on segment sequence and transcribed content (e.g., `0001_some_transcribed_words.wav/.txt`), and these files should be organized into speaker-specific subdirectories (e.g., `S0/`, `S1/`) within the transcription stage's output directory.
- **Updating `call_processor.py`:** Once `transcription_module.py` produces the detailed soundbite structure, `call_processor.py` will need to be updated to correctly find and copy these (e.g., the `S0/`, `S1/` subdirectories) into the final processed call output directory.

## Recent Changes
- **PII-Safe Identifiers & Workflow Integration:**
    - `workflow_executor.py` now generates PII-safe run IDs (e.g., `DefaultAudioAnalysisWorkflow_call_YYYYMMDD-HHMMSS_TIMESTAMP`) and passes a `pii_safe_file_prefix` to each module for consistent internal naming.
    - Modules (`clap_module.py`, `audio_preprocessor.py`, `transcription_module.py`, `diarization_module.py`) updated to accept and utilize `pii_safe_file_prefix` for their temporary and output files.
    - `workflow_executor.py` now correctly identifies "call folders" (containing `recv_out-*` or `trans_out-*` files) when processing an `--input_dir`. If it's a call folder, it skips general `out-*` files from the main workflow and subsequently calls `call_processor.py` as a subprocess with appropriate arguments.
- **`call_processor.py` Development:**
    - Implemented `generate_call_id_from_workflow_dir_name` for PII-safe call IDs (e.g., `call_YYYYMMDD-HHMMSS`).
    - Implemented `find_call_pairs` to group `recv_out` and `trans_out` workflow runs.
    - `process_call` now retrieves main outputs (vocals, transcript JSON, summary TXT) from workflow runs, mixes audio, merges transcripts, and calls `llm_module.generate_llm_summary` for combined summaries using a user-defined prompt and a CLI-provided model ID.
- **Module-Specific Fixes:**
    - **`transcription_module.py`:** Corrected RTTM file loading to use `pyannote.database.util.RTTMParser` instead of the deprecated `Annotation.read_rttm()`.
    - **`llm_module.py`:** The `run_llm_summary` function (called by `workflow_executor.py`) signature was updated to accept `pii_safe_file_prefix` to resolve a `TypeError`.
    - **`clap_module.py`:** Updated to accept `pii_safe_file_prefix` and use it for naming its temporary PII-safe input audio copy and output JSON.

## Next Steps (Immediate)
1.  **Modify `transcription_module.py` for Soundbite Generation:**
    *   Add a `sanitize_filename` utility function (similar to `whisperBite.py`).
    *   Ensure `slice_audio_for_transcription` includes a unique `sequence_id` in the data dictionary it returns for each slice.
    *   Modify `transcribe_sliced_segments`:
        *   It must accept `persistent_output_dir` (the main output directory for the transcription stage) and `pii_safe_file_prefix`.
        *   For each successfully transcribed temporary audio slice:
            *   Create speaker-specific subdirectories (e.g., `S0/`, `S1/`) within `persistent_output_dir`.
            *   Generate a soundbite base name using the `sequence_id` and sanitized first few words of the transcription (e.g., `{sequence_id:04d}_{sanitized_first_words}`).
            *   Copy the temporary audio slice (e.g., from `temp_slices_dir / S0 / ...wav`) to the new persistent path (e.g., `persistent_output_dir / S0 / {sequence_id:04d}_{sanitized_first_words}.wav`).
            *   Create a corresponding text file (e.g., `persistent_output_dir / S0 / {sequence_id:04d}_{sanitized_first_words}.txt`) containing the timestamp and transcription for that soundbite.
            *   The segment dictionaries added to the `all_transcribed_segments` list must be updated to include `soundbite_audio_path` and `soundbite_text_path` pointing to these new persistent soundbite files.
    *   Update `run_transcription` to correctly pass `output_dir` (as `persistent_output_dir`) and `pii_safe_file_prefix` to `transcribe_sliced_segments`.
    *   Ensure the main JSON output of `run_transcription` (e.g., `{pii_safe_file_prefix}_transcription.json`) includes these new paths in its list of segments.
2.  **Test `transcription_module.py` Standalone (if feasible) or via `workflow_executor.py`:** Verify that it produces the correct directory structure and individual soundbite files (`.wav` and `.txt`) as per the `whisperBite.py` model.

## Future Steps (Post Soundbite Generation)
1.  **Update `call_processor.py`:** Modify `process_call` (and potentially `_find_output_file` or add new helper) to correctly locate and copy the new speaker-specific soundbite subdirectories (e.g., `S0/`, `S1/`, containing the `.wav` and `.txt` files) from the transcription stage's output within the workflow run directory to the final call output directory (e.g., `processed_calls_test/<call_id>/S0/`).
2.  **Comprehensive Testing:** After the above changes, conduct thorough end-to-end testing of the `workflow_executor.py` -> `call_processor.py` pipeline with real call audio data.
3.  **Address Lower Priority Items:** Revisit LLM token limits, CLAP module redundancy, logging, etc., as listed in `progress.md`.

## Active Decisions
- **Prioritize `whisperBite.py` Parity for Transcription Output:** The immediate goal is to replicate the detailed soundbite output structure (individual `.wav` and `.txt` files per speaker segment, named by content/sequence) within the `transcription_module.py`.
- **Phased Approach for `call_processor.py`:** Update `call_processor.py` to handle the new transcription output *after* `transcription_module.py` is confirmed to be generating it correctly.
- **Continue PII-Safe Practices:** Maintain PII-safe naming conventions throughout the workflow and in all generated outputs.