# Active Context

## Current Focus
- **Refining Integrated Modules:** Improving the recently integrated `diarization_module.py` (using `pyannote.audio`) and `transcription_module.py` (using `openai-whisper`).
    - Specifically, planning the implementation of `detect_optimal_speakers` functionality (inspired by `WhisperBite`) into `diarization_module.py`.
- **Developing LLM Module (`llm_module.py`):** Actively implementing LLM analysis and summarization stage, integrating with `lmstudio-python` SDK for local model interaction (e.g., using a model like 'NousResearch/Hermes-2-Pro-Llama-3-8B').
- **Testing with Real Audio:** Preparing to test the entire pipeline with diverse, real-world audio files to assess actual performance and output quality of CLAP, preprocessing, diarization, transcription, and LLM summarization (via LM Studio).
- **Dependency Management:** Ensuring `HF_TOKEN` usage is clear and documenting it. Ensuring users are aware of the dependency on a running LM Studio instance with appropriate models loaded for the `llm_module.py`.
- **Code Cleanup:** Reviewing and potentially removing extensive debug logging added to `workflow_executor.py` during the recent troubleshooting phase.
- **Developing `call_processor.py`:** 
    - Refining call ID generation and pairing logic.
    - Implementing retrieval of processed files (stems, transcripts) from `workflow_executor.py` output directories.
    - Porting and adapting audio mixing and tone appending logic from `audiotoolkit_phase1.py`.

## Recent Changes
- **Workflow Executor Stabilized:** Successfully debugged `workflow_executor.py`, resolving complex issues related to input path templating (`_resolve_input_path`) and context data storage/flow for nested output structures. The executor now reliably passes data between stages.
- **`ClapAnnotator` Integration:** `clap_module.py` now successfully incorporates the external `ClapAnnotator` project for sound event detection. This includes handling `ClapAnnotator`'s internal audio separation and its dummy file generation for problematic audio inputs.
- **Audio Preprocessing (`audio_preprocessor.py`):** This module is functioning correctly, performing stem separation (if needed, using `audio-separator` CLI) and loudness normalization (using `audiomentations`). It robustly creates valid silent dummy stems if underlying separation tools fail, ensuring pipeline continuity.
- **Diarization Module Enhanced (`diarization_module.py`):** Replaced placeholder with a functional version using `pyannote.audio.Pipeline` for speaker diarization. It now supports automatic speaker count detection if `num_speakers`, `min_speakers`, or `max_speakers` are not provided in the configuration. It continues to use a configured number of speakers if specified and relies on an external `HF_TOKEN`.
- **Transcription Module Implementation (`transcription_module.py`):** Replaced placeholder with a functional version using `openai-whisper`. It processes audio segments guided by RTTM files from the diarization stage.
- **Successful End-to-End Pipeline Test:** The complete pipeline (`clap_event_annotation` -> `audio_preprocessing` -> `speaker_diarization` -> `transcription` -> `llm_summary_and_analysis` (placeholder)) has been successfully run with a test audio file (`pipeline_test_input.wav`), with all functional stages producing their expected outputs and passing data correctly.
- **`llm_module.py` Refactored:** Switched from OpenAI API to `lmstudio-python` SDK for local LLM interaction. Configuration updated to use `lm_studio_model_identifier`. Corresponding memory bank files (`activeContext.md`, `progress.md`, `systemPatterns.md`, `techContext.md`) were updated to reflect this major change.
- **`workflow_executor.py` Enhanced:** Implemented `argparse` for robust command-line argument handling (`--config_file`, `--input_audio_file`, `--output_dir`, `--log_level`), replacing previous hardcoded test setup in `__main__`. This allows flexible execution of workflows with user-specified inputs.
- **Workflow Definition & Robustness Fixes (Post-Test):**
    - Corrected key access in `default_audio_analysis_workflow.json` for `vocal_stem_path` (from `vocals_normalized_path` to `vocals_normalized`) used by diarization and transcription stages.
    - Modified `clap_event_annotation` stage in workflow to set `clap_separator_model` to `null`, aiming to prevent `ClapAnnotator` from performing internal stem separation.
    - Enhanced `llm_module.py` (`_format_clap_events_for_prompt`) to be more robust in parsing CLAP event JSON data, with better error handling and logging for malformed or unexpected structures.
    - Adjusted `audio_preprocessing` stage configuration in `default_audio_analysis_workflow.json` to set `vocals_lufs` to `-3.0` to help prevent clipping.
    - Enhanced `workflow_executor.py` to support batch processing of audio files from a directory using a new `--input_dir` command-line argument, mutually exclusive with `--input_audio_file`.
    - **Created `call_processor.py` (Initial Version):**
        - Basic structure, logging, and argument parsing (`--input_run_dir`, `--output_call_dir`, `--log_level`).
        - Includes initial logic for `generate_call_id_from_workflow_dir_name` and `find_call_pairs` to identify related workflow runs.
        - Placeholder `process_call` function.

## Next Steps
1.  **Finalize API Key/Local Server Strategy:**
    *   Add an explicit error or stop for `diarization_module.py` if `HF_TOKEN` is missing and required by the chosen model.
    *   Ensure `llm_module.py` gracefully handles LM Studio connection errors or if the specified model is not available.
    *   Document prerequisites for `HF_TOKEN` and LM Studio setup (running server, model availability) for users.
2.  **Comprehensive Testing with Real Audio:** Execute the full pipeline with a variety of real audio inputs.
3.  **Refine `llm_module.py` (Post-Testing):** Based on testing, improve prompt engineering for local models, explore different local models via LM Studio, and potentially add more structured output or specific analysis tasks.
4.  **Refactor Debug Logging:** Remove or reduce the verbosity of temporary debug logs in `workflow_executor.py`.
5.  **Review and Refine Module Configurations:** Assess the default configurations in `default_audio_analysis_workflow.json` for each stage based on real audio testing (e.g., CLAP prompts, diarization model, Whisper model size).

## Active Decisions
- **Core Integrated Tools Confirmed:** `ClapAnnotator` for sound events, `audio_preprocessor.py` for stem prep, `pyannote.audio` for diarization (with auto speaker detection), `openai-whisper` for transcription, and `llm_module.py` (using `lmstudio-python`) for LLM tasks form the primary functional backbone.
- **Workflow-Driven Architecture is Stable:** The `workflow_executor.py` and JSON-defined pipelines are the established method for orchestration.
- **Focus on Quality and Robustness:** Future work will prioritize improving the quality of outputs from each stage and ensuring the pipeline handles diverse inputs gracefully.