# System Patterns

## Architecture Overview
Two main Python scripts orchestrate the audio processing:
1.  **`workflow_executor.py`**: This is the primary engine. It reads a workflow definition from a JSON file (e.g., `default_audio_analysis_workflow.json`) and executes a sequence of audio processing stages on individual audio files. It handles:
    *   Dynamic loading of modules and functions for each stage.
    *   Creation of unique, PII-safe run directories for each audio file processed (e.g., `workflow_runs_test/DefaultAudioAnalysisWorkflow_call_YYYYMMDD-HHMMSS_EXECUTOR_TIMESTAMP/`).
    *   Management of data flow between stages via a context dictionary and path templating.
    *   Passing a `pii_safe_file_prefix` (derived from the input audio filename, e.g., `call_YYYYMMDD-HHMMSS`) to each stage for consistent PII-safe naming of temporary files and primary outputs within that stage.
    *   Batch processing of audio files from an input directory.
    *   Conditional invocation of `call_processor.py` if the input directory is identified as a "call folder" (contains `recv_out-*` or `trans_out-*` files).
2.  **`call_processor.py`**: This script is invoked by `workflow_executor.py` *after* all relevant individual `recv_out-*` and `trans_out-*` files in a call folder have been processed by the main workflow. Its purpose is to:
    *   Identify pairs of processed `recv_out` and `trans_out` streams based on their PII-safe call ID (e.g., `call_YYYYMMDD-HHMMSS`).
    *   Aggregate and combine outputs from these paired streams (e.g., mix audio, merge transcripts, generate combined summaries).
    *   Save the final combined call data into a PII-safe directory structure (e.g., `processed_calls_test/call_YYYYMMDD-HHMMSS/`).

**Processing Modules (called by `workflow_executor.py`):**
*   **`clap_module.py`**: Detects sound events. Uses `pii_safe_file_prefix` for its temporary input copy and output JSON.
*   **`audio_preprocessor.py`**: Separates audio stems (vocals/instrumental) and normalizes them. All its outputs (e.g., `call_YYYYMMDD-HHMMSS_vocals_normalized.wav`) use the `pii_safe_file_prefix`.
*   **`diarization_module.py`**: Performs speaker diarization. Output RTTM file incorporates `pii_safe_file_prefix` (e.g., `call_YYYYMMDD-HHMMSS_vocals_normalized_diarization.rttm`).
*   **`transcription_module.py`**: Transcribes speech, guided by diarization. Uses `pii_safe_file_prefix` for its main JSON output. **(Currently being updated to also output individual speaker soundbites (.wav/.txt) into speaker-specific subfolders within its stage output directory, with content-derived filenames, similar to `whisperBite.py`).**
*   **`llm_module.py`**: Provides LLM functionalities. `run_llm_summary` (called by executor) produces per-stream summaries. `generate_llm_summary` (called by `call_processor.py`) produces combined call summaries.

## Key Technical Decisions
- **Two-Tier Processing:** `workflow_executor.py` handles per-file processing, and `call_processor.py` handles aggregation and finalization for call-type data.
- **PII Safety in Naming:** Consistent use of `pii_safe_file_prefix` (timestamp-based identifiers like `call_YYYYMMDD-HHMMSS`) for all intermediate and final output directories and filenames to avoid PII leakage.
- **Modular Design & Dynamic Loading:** Stages are independent Python modules, dynamically loaded by `workflow_executor.py`.
- **JSON Workflow Definition:** Pipelines are defined in JSON, allowing flexibility.
- **Path Templating for Data Flow:** For inter-stage data dependency management within `workflow_executor.py`.
- **Targeted Soundbite Output (Transcription):** `transcription_module.py` is being refactored to produce detailed, speaker-segmented audio and text files, mirroring `whisperBite.py`'s output style.

## Design Patterns
- **Pipeline / Pipes and Filters:** Core pattern for `workflow_executor.py`.
- **Orchestrator/Controller:** `workflow_executor.py` acts as an orchestrator for per-file processing; `call_processor.py` orchestrates call aggregation.
- **Interpreter:** `workflow_executor.py` interprets the JSON workflow.
- **Strategy Pattern:** Different modules can be used for similar tasks if configured in the workflow.
- **Facade Pattern:** Modules abstract underlying libraries (Whisper, Pyannote, LMStudio SDK).

## Component Relationships & Data Flow Example (Call Folder)
1.  User runs `workflow_executor.py --input_dir /path/to/call_folder ...`
2.  `workflow_executor.py` identifies it as a call folder.
3.  For each `recv_out-*.wav` and `trans_out-*.wav` in the call folder:
    a.  `workflow_executor.py` generates a `pii_safe_file_prefix` (e.g., `call_20250512-020958`).
    b.  It creates a run directory: `workflow_runs_test/DefaultAudioAnalysisWorkflow_call_20250512-020958_EXECUTION_TS/`.
    c.  It executes stages (clap, audio_preprocess, diarize, transcribe, llm_summary) on the input WAV.
        i.  `audio_preprocessor.py` outputs `.../01_audio_preprocessing/call_20250512-020958_vocals_normalized.wav`.
        ii. `diarization_module.py` outputs `.../02_speaker_diarization/call_20250512-020958_vocals_normalized_diarization.rttm`.
        iii. **(Target)** `transcription_module.py` will output:
            - `.../03_transcription/call_20250512-020958_transcription.json` (main JSON)
            - `.../03_transcription/S0/0000_some_words.wav`
            - `.../03_transcription/S0/0000_some_words.txt`
            - `.../03_transcription/S1/0001_other_words.wav` (etc.)
        iv. `llm_module.py` outputs `.../04_llm_summary_and_analysis/final_analysis_summary.txt` (per-stream summary).
4.  After processing all individual files, `workflow_executor.py` calls `call_processor.py --input_run_dir workflow_runs_test/ ...`.
5.  `call_processor.py`:
    a.  Uses `find_call_pairs` to group `workflow_runs_test/DefaultAudioAnalysisWorkflow_call_20250512-020958_EXECUTION_TS/` and its `trans_out` counterpart.
    b.  For each pair (e.g., `call_20250512-020958`):
        i.  Creates output dir: `processed_calls_test/call_20250512-020958/`.
        ii. Retrieves `vocals_normalized.wav` from both recv and trans workflow runs.
        iii. Retrieves `_transcription.json` from both.
        iv. **(Target)** Retrieves speaker subfolders (e.g., `S0/`, `S1/`) with soundbites from both.
        v.  Mixes audio -> `.../call_20250512-020958_mixed_vocals.wav`.
        vi. Merges JSONs -> `.../call_20250512-020958_merged_transcript.json`.
        vii.Copies soundbite folders -> `.../S0_recv/`, `.../S0_trans/` (needs careful naming/merging strategy for soundbites from paired calls - TBD).
        viii.Generates combined LLM summary -> `.../call_20250512-020958_combined_call_summary.txt`.

## Key Libraries & Tools
- `pyannote.audio`, `pyannote.database.util.RTTMParser`
- `openai-whisper`
- `pydub`
- `lmstudio-python` (for LM Studio SDK)
- `ffmpeg` (underlying utility)
- `audio-separator` (CLI and library)
- `audiomentations`

## Key Technical Decisions
- **Workflow-Driven Architecture:** Adopting a flexible system where processing pipelines are defined externally (JSON) and executed by a generic engine, allowing for easier modification and extension of pipelines.
- **Modular Design:** Separating distinct processing tasks into individual Python modules to promote reusability, testability, and independent development.
- **Dynamic Module Loading:** The workflow executor dynamically imports and calls functions from specified modules based on the workflow definition.
- **Path Templating for Data Flow:** Using a string templating mechanism (e.g., `{workflow.original_input_audio_file}`, `{stages.stage_name[output_key]}`) within the workflow JSON to manage data dependencies between stages.
- **Utilizing `audio-separator` CLI:** For vocal/instrumental stem separation within `audio_preprocessor.py`.
- **Utilizing `audiomentations`:** For stem-specific loudness normalization and potentially other augmentations in `audio_preprocessor.py`.
- **Placeholder Modules for Initial Development:** Using simplified placeholder modules that create dummy outputs to enable end-to-end testing of the workflow executor and data flow before full functionality of each stage is implemented. (Note: Most key modules are now beyond basic placeholders).
- **FFmpeg for Core Audio Utilities:** Retaining FFmpeg for underlying audio operations where appropriate (e.g., resampling, format conversion, called via `subprocess`).
- **Integration of `ClapAnnotator`**: `clap_module.py` directly calls `ClapAnnotator.cli.process_audio_file`.
- **Integration of `pyannote.audio`**: `diarization_module.py` uses `pyannote.audio.Pipeline`.
- **Integration of `openai-whisper`**: `transcription_module.py` uses `whisper.load_model` and transcribes segments based on RTTM.
- **Handling of `HF_TOKEN`**: Currently relies on an environment variable for Hugging Face authenticated models.

## Design Patterns
- **Pipeline Processing / Pipes and Filters:** The overall architecture where data flows through a sequence of processing stages.
- **Interpreter:** The `workflow_executor.py` interprets the JSON workflow definition to orchestrate the processing.
- **Strategy Pattern:** Different modules/functions can be swapped in for stages in the workflow JSON, allowing different strategies for, e.g., transcription or diarization.
- **Command Pattern:** Underlying tools like `ffmpeg` or `audio-separator` are often invoked as commands.
- **Facade Pattern:** Individual modules can act as facades to more complex underlying libraries or tools.
- **Builder (Conceptual):** The workflow executor, guided by the JSON, builds and executes a processing chain.

## Component Relationships
- **`workflow_executor.py`**: Orchestrates the entire pipeline, calling the various modules in sequence as defined by a workflow JSON file. Handles input/output path resolution between stages.
    - Can process a single audio file specified via `--input_audio_file` or batch process all supported audio files (e.g., .wav, .mp3, .flac, .m4a) in a directory specified via `--input_dir`.
    - Driven by command-line arguments parsed using `argparse`.
- **`default_audio_analysis_workflow.json` (and similar):** Configuration file defining specific pipelines and their stages.
- **`clap_module.py`**: Module for CLAP event annotation, wraps `ClapAnnotator`.
    - *Input:* Audio file path, configuration (prompts, confidence, separator model for `ClapAnnotator`).
    - *Output:* Path to CLAP events JSON file.
- **`audio_preprocessor.py`**: Module for stem separation and normalization.
    - *Input:* Audio file path, configuration (target LUFS, sample rates, separator model).
        - Default LUFS targets in `default_audio_analysis_workflow.json` for this stage are now typically `-3.0 LUFS` for vocals and `-23.0 LUFS` for instrumentals.
    - *Output:* Dictionary containing paths to processed vocal and instrumental stems (e.g., `vocals_normalized_path`, `instrumental_normalized_path`).
- **`diarization_module.py`**: Module for speaker diarization using `pyannote.audio`.
    - *Input:* Vocal stem path, configuration (diarization model name, and optionally `num_speakers`, `min_speakers`, `max_speakers`). If speaker count parameters are omitted, the module attempts automatic speaker count detection.
    - *Output:* Path to RTTM diarization file.
- **`transcription_module.py`**: Module for transcription using `openai-whisper`.
    - *Input:* Vocal stem path, RTTM diarization file path, configuration (Whisper model name).
    - *Output:* Path to JSON transcript file.
- **`llm_module.py`**: Leverages a local Large Language Model (via LM Studio and the `lmstudio-python` SDK) for analysis, summarization, or other text-based tasks on the transcript and sound events.
    - Parses transcript data and CLAP event data to construct a prompt for the LLM.
    - Features robust parsing of CLAP event JSON. It attempts to handle various structures (including potentially malformed ones or those with different top-level keys for the event list like `"events"` or `"detections"` if `"clap_events"` is not found) and logs warnings if data issues are encountered, aiming to proceed with partial data if possible.
    - Interacts with a configured LM Studio model server.
    - Outputs a plain text summary file.

- **External Tools/Libraries (Examples):**
    - `audio-separator` (CLI and library): Used by `audio_preprocessor.py` (CLI) and `ClapAnnotator` (library).
    - `audiomentations` (Python library): Used by `audio_preprocessor.py`.
    - `soundfile` (Python library): For reading/writing audio files in modules.
    - `ffmpeg` (CLI tool): For various audio manipulations, callable via `subprocess`.
    - `pyannote.audio` (Python library): Used by `diarization_module.py`.
    - `openai-whisper` (Python library): Used by `transcription_module.py`.
    - `torch` (Python library): Dependency for `pyannote.audio` and `openai-whisper`.
    - `lmstudio` (Python library): Used by `llm_module.py` for interacting with a local LM Studio server.
    - LLM Interaction: Via local LM Studio server (using `lmstudio-python` SDK).

## Workflow JSON Structure
- **Workflow JSON Structure (`default_audio_analysis_workflow.json` as example):**
    - `name`: A descriptive name for the workflow.
    - `stages`: An ordered list of processing stages. Each stage is a dictionary with:
        - `stage_name`: A unique identifier for the stage.
        - `module`: The Python module name (e.g., `"audio_preprocessor"`).
        - `function`: The function name within the module to call (e.g., `"run_generic_preprocess"`).
        - `inputs`: Defines a dictionary where keys are the parameter names for the stage's function, and values are strings that can be literal values or template strings (e.g., `"{workflow.original_input_audio_file}"`, `"{stages.previous_stage_name[output_key]}"`).
            - **Path Resolution Example:** The `audio_preprocessing` stage might output a dictionary like `{"vocals_normalized": "path/to/vocals.wav", "instrumental_normalized": "path/to_instrumental.wav"}` which is stored under its `stage_name` in the `workflow_context`. Downstream stages needing the vocal stem path would use an input template like `"vocal_stem_path": "{stages.audio_preprocessing[processed_stems_info][vocals_normalized]}"` to correctly access this path. Note the direct key access (e.g., `vocals_normalized`) rather than assuming a `_path` suffix if the producing module doesn't add one.
        - `config`: A dictionary of static configuration parameters specific to the stage's function.
            - **CLAP Module Configuration Example:** For the `clap_event_annotation` stage, the `config` object can include `clap_prompts` (list of text prompts for sound events), `clap_confidence_threshold`, and `clap_separator_model`. Setting `clap_separator_model` to `null` is the current strategy to request that the underlying `ClapAnnotator` tool processes the main input audio directly, rather than performing its own internal audio separation before analysis.
        - `outputs`: Defines how the return value (or parts of it, if a dictionary) of the stage's function should be stored in the `workflow_context` under the `stage_name` for use by subsequent stages. 