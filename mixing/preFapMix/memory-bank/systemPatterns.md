# System Patterns

## Architecture Overview
The system employs a modular, workflow-driven architecture for audio processing and analysis. Key components include:
1.  **Workflow Executor (`workflow_executor.py`):** The central orchestrator that reads a pipeline definition from a JSON file. It dynamically loads and executes a sequence of processing stages.
2.  **JSON Workflow Definition (e.g., `default_audio_analysis_workflow.json`):** Defines the stages of the pipeline, including the module and function to call for each stage, how inputs are sourced (from original input or previous stages), stage-specific configurations, and how outputs are stored.
3.  **Modular Processing Stages (Python Scripts):** Each distinct processing task is encapsulated in its own Python module/script. Current primary stages:
    *   **CLAP Event Annotation (`clap_module.py`):** Integrates `ClapAnnotator` project to detect sound events. Handles its own audio separation internally using `audio-separator` library code.
    *   **Audio Preprocessing (`audio_preprocessor.py`):** Handles further stem separation if needed (vocals/instrumentals using `audio-separator` CLI) and stem normalization (using `audiomentations`). Primary purpose is to prepare clean vocal and instrumental stems for subsequent stages.
    *   **Speaker Diarization (`diarization_module.py`):** Identifies speaker segments using `pyannote.audio`.
    *   **Transcription (`transcription_module.py`):** Converts speech to text using `openai-whisper`, guided by diarization output.
    *   **LLM Analysis (`llm_module.py`):** Performs analysis or summarization using a Large Language Model (currently a placeholder).
4.  **Input Audio File:** The initial audio file that the workflow processes.
5.  **Timestamped Run Directories:** The executor creates a unique directory for each run (e.g., `runs/run_YYYYMMDD_HHMMSS/`) containing subdirectories for each stage's outputs, ensuring organized and isolated results.

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