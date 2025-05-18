# Progress Tracking

## What Works

### Core Architecture & Workflow
- **`workflow_executor.py`:**
    - Successfully executes a pipeline of audio processing stages defined in a JSON file.
    - Manages input/output paths, unique run directories, and data passing between stages.
    - Supports batch processing from an `--input_dir`.
    - Handles PII-safe run ID generation for intermediate workflow run directories.
    - Passes `pii_safe_file_prefix` to each stage module.
    - Identifies "call folders" and correctly invokes `call_processor.py` as a subprocess, passing necessary arguments.
- **`call_processor.py`:**
    - **Call Identification & File Retrieval:**
        - Generates `call_id` from workflow directory names.
        - `find_call_pairs` correctly groups related workflow run directories for pairs and singles.
        - `_find_output_file` helper robustly locates necessary files (vocals, transcripts, individual summaries) from stage outputs.
    - **Core Processing for Pairs:**
        - Mixes `recv_vocal_stem_path` and `trans_vocal_stem_path` into a stereo `.wav` file.
        - Merges `recv_transcript_path` and `trans_transcript_path` JSON files chronologically, prefixing speaker labels correctly.
    - **Soundbite Directory Handling:**
        - `_copy_soundbite_directories` correctly copies speaker-specific soundbite folders (e.g., `S0/`, `S1/`) from transcription stage outputs to the primary call processing directory (e.g. `processed_calls/[call_id]/RECV_S0/`).
    - **LLM Integration (via `llm_module.generate_llm_summary`):**
        - Successfully makes three distinct LLM calls using approved, concise system prompts for:
            1.  Call Naming (output to `[call_id]_suggested_name.txt`)
            2.  Call Synopsis (output to `[call_id]_combined_call_summary.txt`)
            3.  Hashtag Categories (output to `[call_id]_hashtags.txt`)
    - **"Final Output" Directory Generation (User-Friendly Structure):
        - Takes `--final_output_dir` argument to specify a base path.
        - Creates a subdirectory for each call, named using a sanitized version of the LLM-generated call name (or `call_id` as fallback).
        - Populates this directory with:
            - The primary call audio as `.wav` (e.g., `[Sanitized_Call_Name].wav`).
            - `transcript.txt` (plain text version converted from the merged JSON transcript).
            - `synopsis.txt` (copied from LLM-generated synopsis).
            - `hashtags.txt` (copied from LLM-generated hashtags).
            - `call_type.txt` (e.g., "pair", "single_recv") for use by `show_compiler.py`.
- **Modules (within `workflow_executor.py` pipeline):**
    - `clap_module.py`, `audio_preprocessor.py`, `diarization_module.py` are functioning as per earlier updates.
    - `transcription_module.py`: RTTM parsing is resolved. Produces JSON transcript and speaker-specific soundbite folders.
    - `llm_module.py`: `generate_llm_summary` function correctly handles system prompts and interacts with LM Studio.
- **PII Safety:** Maintained through `pii_safe_file_prefix` in workflow stages and sanitized naming in `call_processor.py` outputs.
- **Resolved Issues:** Previous terminal echo problem, initial LLM import/usage issues, and RTTM parsing issues are resolved. Soundbite generation in `transcription_module.py` and copying in `call_processor.py` are now working.

### Final Output Builder
- The final output builder is now fully implemented and invoked unconditionally as the last step of the pipeline.
- It collects all call outputs, soundbites, LLM responses, transcripts, show files, and metadata, and zips the final output.
- It marks bad calls, makes all paths relative, and supports a flat call folder for convenience.
- All lineage, tagging, and extensibility requirements are met.
- The output is robust, user-facing, and ready for further content moderation and UI integration.

## Final Output Builder Redesign (May 2025)
- The previous output builder was insufficient: it did not compile final call audio, did not robustly collect or name outputs, and did not produce a user-facing product.
- A new, modular, and robust output builder is planned and requirements are confirmed.
- Implementation will begin in a new session.

## What's Left to Build / Refine (Current Development Cycle)

### Current High Priority
1.  **MP3 Compression:**
    *   Implement a utility function (e.g., in `call_processor.py` or `utils.py`) for WAV to MP3 conversion using `ffmpeg`.
    *   Integrate into `call_processor.py` to convert the primary `.wav` audio file in the `final_output_dir/[Sanitized_Call_Name]/` directory to `.mp3` (e.g., `[Sanitized_Call_Name].mp3`), then delete the source WAV.
    *   **Decision Needed:** Should soundbites copied to the `final_output_dir` (if this feature is added) also be MP3 compressed?

2.  **`show_compiler.py` (New Script Development):**
    *   **Input:** Base "final output" directory (e.g., `final_processed_calls/`).
    *   **Core Logic:**
        *   Scan input for call folders; read `call_type.txt` to filter for "pair" calls.
        *   Sort identified paired calls chronologically (based on original `call_id` timestamp. Consider embedding `call_id` in the final output folder name like `[call_id]_[Sanitized_Name]` or ensure `call_processor.py` writes a `call_metadata.json` with the original `call_id` into each final call directory).
        *   Concatenate the primary MP3 audio files from these calls into a single `show_audio.mp3` (using `ffmpeg`).
        *   Generate `show_timestamps.txt`: Calculate and write the start time and name for each call within `show_audio.mp3`.
        *   Generate `show_transcript.txt`: Concatenate the `transcript.txt` files from each call, with clear separators/headers.
    *   **Output:** Save these three files into a specified `show_output` directory.

### Clarifications & Decisions Needed for Current Cycle
- **Instrumental Stems for Show Audio:** Does `show_audio.mp3` require instrumental stems to be mixed with vocals? 
    *   *If yes:* This would necessitate changes in `audio_preprocessor.py` (to save instrumental stems if not already), `call_processor.py` (to create a vocal+instrumental mix for each call to be placed in the `final_output_dir`), and `show_compiler.py` would then use this full mix as source.
    *   *If no (current assumption):* `show_compiler.py` uses the vocal-only MP3s from `final_output_dir`.
- **Soundbites in "Final Output" Directory:** Are the individual soundbite folders (e.g., `RECV_S0/`) and their contents to be copied into the `final_output_dir/[Sanitized_Call_Name]/soundbites/` directory? If so, should these also be MP3 compressed?

### Next Focus
- Advanced content moderation (censoring or flagging problematic content)
- Show-level LLM synopses
- UI integration for LLM task management

### Broader Architectural Enhancements & Future Work (Next Development Cycle)
- **Configuration File for LLM Server:** Introduce a `config/settings.json` to specify LM Studio URL, model, and parameters. Modify `llm_module.py` and its callers.
- **Gradio App Update:** Overhaul `app.py` to reflect all new functionalities of the CLI tools and the new configuration system.
- **Workspace File Audit & Cleanup:** Review and remove/relocate any old, unused, or misplaced files/folders (e.g., `old_implementations/`, `v3.8/`, `audio_analyzer_phase2.py`, `audiotoolkit_phase1.py`, etc.).
- **CLAP Module Redundancy:** Further investigation if `ClapAnnotator`'s internal separation is truly disabled or if it can efficiently use pre-separated stems.
- **LM Studio Token Limits:** Explicit handling (e.g., transcript chunking) for very long transcripts if they become an issue.
- **Comprehensive Testing:** End-to-end testing with diverse real-world call audio across the full pipeline.

## Known Issues (Relevant to Current Cycle)
- **Soundbite Handling for Final Output:** Decision pending on whether soundbites are copied to the `final_output_dir` and if they should be MP3 compressed.
- **Instrumental Stems in Show Audio:** Clarification and potential implementation path needed if instrumentals are required for the show file.
- **Chronological Sorting for `show_compiler.py`:** Strategy needed for reliable chronological sorting (e.g., include `call_id` in final folder names or add a metadata file).

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