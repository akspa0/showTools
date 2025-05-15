# Progress Tracking

## What Works

### Core Architecture & Workflow
- **Workflow Executor (`workflow_executor.py`):** Successfully executes a pipeline of audio processing stages defined in a JSON file (`default_audio_analysis_workflow.json`).
    - Dynamically loads modules and calls specified functions.
    - Manages input/output paths for each stage, creating unique timestamped run directories.
    - Handles complex data passing between stages using a context dictionary and robust path templating (e.g., `{workflow.original_input_audio_file}`, `{stages.stage_name[key1][key2]}`), including resolution of nested dictionary keys.
    - Successfully stores and retrieves stage outputs, including entire dictionaries returned by modules, and maps them correctly to the `workflow_context`.
    - **Path Resolution Fix:** Path templating logic now correctly accesses dictionary keys from previous stages (e.g., `[vocals_normalized]` directly, instead of assuming a `_path` suffix if not present), crucial for passing correct file paths to diarization and transcription.
    - Now driven by `argparse` for command-line execution (config file, input audio file/directory, output dir, log level).
        - Supports batch processing of all supported audio files (e.g., .wav, .mp3, .flac, .m4a) in a specified input directory (`--input_dir`).
        - `--input_audio_file` and `--input_dir` are mutually exclusive.
- **CLAP Event Annotation (`clap_module.py`):
    - Integrates and successfully runs `ClapAnnotator.cli.process_audio_file`.
    - Handles `ClapAnnotator`'s internal audio separation.
    - Produces CLAP event JSON files.
    - Includes fallback to create dummy silent stems within `ClapAnnotator` if its internal separation fails, allowing CLAP to proceed.
    - **Configuration Update:** The workflow configuration for `clap_event_annotation` now sets `clap_separator_model` to `null`. This is intended to instruct `ClapAnnotator` to analyze the original audio input directly, rather than performing its own internal stem separation first. The effectiveness of `null` for this purpose in `ClapAnnotator` is pending confirmation in the next test run.
- **Audio Preprocessor (`audio_preprocessor.py`):
    - `run_generic_preprocess` function is successfully called by the workflow executor.
    - Performs stem separation (vocals/instrumentals) using `audio-separator` CLI.
    - If `audio-separator` fails (e.g., with simple test audio), it creates *valid, silent* dummy WAV files for stems, preventing downstream errors.
    - Normalizes stems (vocals and instrumentals) to configured LUFS targets using `audiomentations.LoudnessNormalization`.
        - Default vocal stem normalization target is now `-3.0 LUFS` (changed from -18.0 LUFS) to help prevent clipping.
    - Output paths for processed stems are correctly passed to subsequent stages.
- **Speaker Diarization (`diarization_module.py`):
    - Successfully uses `pyannote.audio.Pipeline` for speaker diarization.
    - Takes a vocal stem path and configuration (model name, speaker count parameters) as input.
    - If speaker count parameters (`num_speakers`, `min_speakers`, `max_speakers`) are not provided in the config, the pipeline attempts to automatically detect the number of speakers.
    - If speaker count parameters are provided, they are used by the pipeline.
    - Produces an RTTM file as output.
    - Relies on an externally set `HF_TOKEN` environment variable (logs a warning if not found).
- **Transcription (`transcription_module.py`):
    - Successfully uses `openai-whisper` for transcription.
    - Takes a vocal stem path and an RTTM diarization file as input.
    - Segments the audio based on RTTM for speaker-aware transcription.
    - Produces a JSON transcript file with speaker labels and timestamps.
- **LLM Analysis (`llm_module.py` - Phase 1 Implemented):
    - Integrates with OpenAI API for conversation summarization.
    - Expects `OPENAI_API_KEY` environment variable.
    - Parses transcript (structured or plain text) and CLAP event data for prompt construction.
    - Uses a configurable LLM model (default: `gpt-3.5-turbo`) and `max_tokens`.
    - Outputs a plain text summary file (`final_analysis_summary.txt`).
    - Includes basic error handling for API calls and file operations.
- **LLM Analysis (`llm_module.py` - LM Studio Integration):
    - Integrates with local LM Studio server via `lmstudio-python` SDK for conversation summarization.
    - Expects `lm_studio_model_identifier` in config (e.g., 'NousResearch/Hermes-2-Pro-Llama-3-8B') which must be available in the running LM Studio instance.
    - Parses transcript (structured or plain text) and CLAP event data for prompt construction.
    - **CLAP Data Handling:** Now includes more robust parsing of CLAP event JSON. It attempts to handle various structures (e.g., different top-level keys for the event list like `"events"` or `"detections"`) and logs warnings if data is malformed or keys are missing, aiming to proceed with an empty/partial event list if full parsing fails.
    - Uses a configurable LLM model (default: `gpt-3.5-turbo`, though this should be an LM Studio identifier) and `max_tokens` (though `max_tokens` might be LM Studio specific).
    - Outputs a plain text summary file (`final_analysis_summary.txt`).
    - Includes error handling for LM Studio connection and model interaction.
- **Workflow Definition (`default_audio_analysis_workflow.json`):
    - Defines the full, working pipeline: `clap_event_annotation` -> `audio_preprocessing` -> `speaker_diarization` -> `transcription` -> `llm_summary_and_analysis`.
    - Input/output mappings between these stages are correctly configured and functioning.
- **End-to-End Pipeline Execution:**
    - The entire pipeline, with the above functional modules, runs successfully with test audio (`pipeline_test_input.wav`).
    - Data flows correctly between all stages.

### New Components Under Development
- **`call_processor.py` (Initial Version):**
    - **Purpose:** To process the outputs of multiple `workflow_executor.py` runs, identify `recv_out`/`trans_out` file pairs belonging to the same call, and combine their processed data (e.g., mix audio stems, eventually merge analyses).
    - **Current State:** 
        - Basic file structure, logging, and command-line argument parsing (`--input_run_dir`, `--output_call_dir`, `--log_level`) are implemented.
        - Initial logic for `generate_call_id_from_workflow_dir_name` (extracting call identifiers from `workflow_executor.py` output directory names) and `find_call_pairs` (grouping related workflow run directories) is in place.
        - A placeholder `process_call` function exists, which will house the logic for retrieving files and performing combination tasks.
    - **Next Steps:** Implement robust file retrieval from workflow output directories, port audio mixing logic, then later add analysis merging capabilities.

## What's Left to Build / Refine

### LLM Module Implementation
1.  **`llm_module.py` (`run_llm_summary`):**
    - Replace placeholder logic with actual calls to an LLM API (e.g., OpenAI).
    - Design and implement effective prompts that utilize the transcript, CLAP events, and diarization information to generate useful summaries or analyses.
    - Define the output structure for the LLM results (e.g., structured JSON or plain text summary file).

### Testing and Robustness
1.  **Comprehensive Testing with Real Audio:** Thoroughly test the entire pipeline with a diverse set of real-world audio files (various lengths, accents, noise levels, number of speakers) to identify and address any quality issues or failures.
2.  **`HF_TOKEN` Management:** Improve handling of the `HF_TOKEN`. Consider adding a more explicit check at the start of relevant modules and providing clearer error messages or guidance if it's not set.
3.  **Error Handling and Logging:** Review and enhance error handling in all modules. Refine logging to be informative for production use, potentially reducing default verbosity of debug logs from `workflow_executor.py`.

### Configuration and Usability
1.  **Review Default Configurations:** Based on real audio testing, fine-tune default configurations in `default_audio_analysis_workflow.json` (e.g., Whisper model size, CLAP confidence thresholds, diarization parameters).
2.  **Documentation:** Update or create documentation for running the workflow, including prerequisites (like `HF_TOKEN`) and configuration options.

## Current Status
- **Core pipeline is functional and integrated:** The `workflow_executor.py` successfully orchestrates `ClapAnnotator`, `audio_preprocessor.py`, `diarization_module.py` (with `pyannote.audio`), and `transcription_module.py` (with `openai-whisper`).
- **End-to-end tests with dummy audio pass reliably.**
- **Path resolution and data flow issues in the executor have been resolved.**
- The immediate focus is on **enhancing the diarization module (`detect_optimal_speakers`)**, fully implementing the **`llm_module.py`**, and conducting **extensive testing with real audio data**.

## Known Issues
- **`llm_module.py` is a placeholder.** No actual LLM processing occurs.
- **Real-world audio testing is pending:** Performance and quality on diverse, real audio have not yet been systematically evaluated.
- **LLM Summarization (Phase 1) is basic:** The current summarization is general. More advanced analysis, prompt engineering, and structured output may be needed based on testing.
- **Real-world audio testing is pending for the full pipeline including LLM summarization:** Performance and quality on diverse, real audio have not yet been systematically evaluated.
- **`HF_TOKEN` dependency:** Relies on an externally set environment variable; modules log a warning but proceed if it's not found, which might lead to failures if authenticated models are required by `pyannote.audio`.
- **API Key Dependencies:** Relies on `HF_TOKEN` (for diarization) and `OPENAI_API_KEY` (for LLM) environment variables. Modules log warnings or errors if these are not found.
- **LLM Summarization (LM Studio) is initial:** Summarization quality depends on the local model. Prompt engineering and model choice will need refinement based on testing.
- **Real-world audio testing is pending for the full pipeline including LM Studio LLM summarization.**
- **External Dependencies:** Relies on `HF_TOKEN` (for diarization if using gated models) and a running LM Studio instance with the specified model available (for LLM tasks).

## Current Status (Post User Feedback on Loudness Fixes)
- **Phase 1: Core audio processing is largely complete and producing good quality output, especially regarding loudness and vocal/instrumental balance, per user confirmation.**
- Key functionalities implemented: input handling (files, URLs, dir), stem separation, multi-stage loudness normalization (stems via audiomentations, final mix via ffmpeg loudnorm), stem volume adjustments, stereo mixing with correct channel mapping, timestamped outputs, and intermediate stem saving.
- The critical issues of quiet audio and buried vocals have been resolved with the new default LUFS targets (`0.0` for vocals, `-14.0` for instrumentals, `-8.0` for final mix).

## Known Issues
- Comprehensive testing across a wider variety of audio files is still needed to ensure robustness.
- Error handling, while present, could be further improved for edge cases.
- No specific vocal enhancement/EQ stage yet. (Note: Limiter is present, but specific EQ was removed previously, current focus is on loudness balance)
- No GUI yet.

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