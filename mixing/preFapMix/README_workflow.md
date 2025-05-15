# Audio Processing Workflow (`README_workflow.md`)

This document describes the audio processing workflow for this project, detailing how to run analyses, what stages are involved, and how to process outputs.

The workflow is primarily managed by two Python scripts:
1.  `workflow_executor.py`: Executes a defined sequence of audio processing stages on a single input audio file.
2.  `call_processor.py`: Processes the outputs of one or two `workflow_executor.py` runs (typically for RECV/TRANS streams of a call) to produce a combined, final output for a call.

## 1. Core Workflow Execution (`workflow_executor.py`)

`workflow_executor.py` is responsible for running a series of audio analysis modules on a single audio input. The sequence of operations and their configurations are defined in a JSON workflow file.

### Prerequisites

*   **Python Environment:** Ensure all required Python packages are installed (see `requirements.txt` or project setup guide).
*   **LM Studio (for LLM Summarization):**
    *   LM Studio must be running.
    *   The language model specified in the workflow configuration (or `llm_module.py` default) must be downloaded and actively served by LM Studio.
*   **ffmpeg (for `call_processor.py` audio mixing):** `ffmpeg` must be installed and accessible in the system PATH if you intend to use `call_processor.py` to mix audio from paired calls.

### Running the Workflow

Execute the workflow from the command line:

```bash
python workflow_executor.py --input_file /path/to/your/audio.wav \\
                            --workflow_config /path/to/your/workflow.json \\
                            --output_dir /path/to/output_base_directory \\
                            --log_level DEBUG
```

**Key Command-Line Arguments:**

*   `--input_file <path>`: (Required) Path to the input audio file (e.g., `.wav`, `.mp3`).
*   `--workflow_config <path>`: (Required) Path to the JSON file defining the workflow stages (e.g., `default_audio_analysis_workflow.json`).
*   `--output_dir <path>`: (Required) Base directory where a unique sub-directory for this specific run will be created.
*   `--log_level <LEVEL>`: (Optional) Logging level (e.g., `DEBUG`, `INFO`, `WARNING`, `ERROR`). Defaults to `INFO`.
*   `--python_path <path>`: (Optional) Additional path to add to `sys.path` for module imports (e.g., `.` if running from project root and modules are in subdirectories).

### Workflow Configuration File (e.g., `default_audio_analysis_workflow.json`)

This JSON file defines the sequence of processing stages. Each stage specifies:
*   `stage_name`: A unique name for the stage.
*   `module`: The Python module to load (e.g., `transcription_module`).
*   `function`: The function to call within that module (e.g., `run_transcription`).
*   `inputs`: A dictionary mapping function argument names to their values. Values can be:
    *   Literal values (strings, numbers, booleans).
    *   Special tokens:
        *   `"{workflow.original_input_audio_file}"`: Path to the initial audio file provided to the executor.
        *   `"{workflow.main_run_dir}"`: Path to the main output directory for the current workflow run.
        *   `"{workflow.current_stage_output_dir}"`: Path to the dedicated output directory for the current stage.
    *   References to outputs from previous stages:
        *   Syntax: `"{stages.PREVIOUS_STAGE_NAME[key1][key2]}"`
        *   Example: `"{stages.audio_preprocessing[processed_stems_info][vocals_normalized]}"` retrieves the `vocals_normalized` path from the dictionary returned by the `audio_preprocessing` stage.
*   `config`: A dictionary of configuration parameters specific to the stage's function.
*   `outputs`: (Informational) Describes how to access specific return values from the stage function if needed for documentation or understanding (the executor stores the entire return value in its context).

**Default Workflow Stages (`default_audio_analysis_workflow.json`):**

1.  **`clap_event_annotation` (`clap_module.run_clap_annotation`):**
    *   Detects specified sound events (e.g., "telephone ringing") using a CLAP model.
    *   Outputs a JSON file listing detected events with timestamps and confidence.
2.  **`audio_preprocessing` (`audio_preprocessor.run_generic_preprocess`):**
    *   Performs audio preprocessing tasks like:
        *   Resampling to a target sample rate.
        *   Vocal separation (e.g., using UVR_MDXNET_Main).
        *   LUFS normalization of vocal and instrumental stems.
    *   Outputs processed audio stems (e.g., `vocals_normalized.wav`).
3.  **`speaker_diarization` (`diarization_module.run_diarization`):**
    *   Identifies different speakers and their speech segments in the vocal stem.
    *   Uses a diarization model (e.g., `pyannote/speaker-diarization-3.1`).
    *   Outputs an RTTM file detailing speaker turns.
4.  **`transcription` (`transcription_module.run_transcription`):**
    *   Transcribes the vocal stem based on the speaker diarization.
    *   Uses a Whisper model (e.g., "base").
    *   Generates individual audio soundbites per speaker segment (`.wav`, `.txt`).
    *   Outputs a main `transcript.json` file containing all segments, speaker labels, timestamps, text, and paths to soundbites.
5.  **`llm_summary_and_analysis` (`llm_module.run_llm_summary`):**
    *   Generates a summary of the conversation using an LLM (via LM Studio).
    *   Uses the transcript, CLAP events, and potentially diarization info as input for the prompt.
    *   Outputs a `final_analysis_summary.txt` file.

### Output Structure (per `workflow_executor.py` run)

A unique run directory is created under the specified `--output_dir`, e.g.:
`--output_dir / DefaultAudioAnalysisWorkflow_call_YYYYMMDD-HHMMSS_timestamp /`

Inside this run directory:
*   `DefaultAudioAnalysisWorkflow_call_YYYYMMDD-HHMMSS_timestamp_executor.log`: Main log file for the workflow run.
*   `00_clap_event_annotation/`: Output files from the CLAP annotation stage (e.g., `clap_events.json`).
*   `01_audio_preprocessing/`: Output files from audio preprocessing (e.g., `vocals_normalized.wav`, `instrumental_normalized.wav`).
*   `02_speaker_diarization/`: Output files from diarization (e.g., RTTM file).
*   `03_transcription/`: Output files from transcription (e.g., `transcript.json`, speaker soundbite subfolders like `S0/`, `S1/`).
*   `04_llm_summary_and_analysis/`: Output files from LLM summarization (e.g., `final_analysis_summary.txt`).

## 2. Combined Call Processing (`call_processor.py`)

`call_processor.py` takes the outputs from one or two `workflow_executor.py` runs (which might represent the RECV and TRANS audio streams of a single call, or just a single audio stream) and processes them to produce a consolidated output for that call.

### Purpose

*   Combine RECV and TRANS audio streams into a stereo file.
*   Merge RECV and TRANS transcripts, adjusting timestamps and speaker labels.
*   Consolidate soundbite directories from RECV and TRANS streams.
*   Generate a single LLM summary for the entire call (merged or single stream).

### Running `call_processor.py`

```bash
python call_processor.py --input_run_parent_dir /path/to/workflow_executor_outputs/ \\
                         --output_call_dir /path/to/final_processed_calls/ \\
                         --llm_config_file /path/to/llm_config.json \\
                         --log_level DEBUG
```

**Key Command-Line Arguments:**

*   `--input_run_parent_dir <path>`: (Required) Path to a directory containing one or more output directories generated by `workflow_executor.py`. `call_processor.py` will scan this directory to find pairs (RECV/TRANS) or single workflow runs based on derived `call_id`s from their directory names.
*   `--output_call_dir <path>`: (Required) Base directory where sub-directories for each processed `call_id` will be created.
*   `--llm_config_file <path>`: (Optional) Path to a JSON file containing configuration for the LLM summarization step (e.g., model identifier, prompt details). If not provided, defaults may be used from `llm_module.py`.
*   `--log_level <LEVEL>`: (Optional) Logging level. Defaults to `INFO`.

### How it Works

1.  **Scan for Call Pairs/Singles:**
    *   Scans the `--input_run_parent_dir` for `workflow_executor.py` output directories.
    *   Derives a `call_id` for each run (typically from a timestamp in the directory name like `call_YYYYMMDD-HHMMSS`).
    *   Groups runs by `call_id`. If two runs share a `call_id` and their names suggest RECV and TRANS streams (e.g., containing "recv_out" and "trans_out"), they are treated as a pair. Otherwise, they are processed as single streams.
2.  **Process Each Call:** For each unique `call_id`:
    *   Creates a new output directory: `--output_call_dir / <call_id> /`
    *   **If Paired (RECV & TRANS):**
        *   **Mix Vocals:** Uses `ffmpeg` to combine the `vocals_normalized.wav` from RECV (left channel) and TRANS (right channel) into `mixed_vocals.wav`.
        *   **Merge Transcripts:** Combines the `transcript.json` from RECV and TRANS. Timestamps in the TRANS transcript are adjusted to follow the RECV stream. Speaker labels are prefixed (e.g., `RECV_S0`, `TRANS_S1`). The result is `merged_transcript.json`.
        *   **Copy Soundbites:** Speaker soundbite directories (e.g., `S0/`, `S1/`) from both RECV and TRANS transcription outputs are copied and prefixed (e.g., `RECV_S0/`, `TRANS_S1/`) into the final call output directory.
    *   **If Single Stream:**
        *   Copies the `vocals_normalized.wav` and `transcript.json` directly (may rename for consistency).
        *   Copies soundbite directories, prefixing based on stream type if known (e.g., `RECV_S0/`) or using original names.
    *   **Combined LLM Summary:** Generates `combined_call_summary.txt` using `llm_module.generate_llm_summary` on the (merged or single) transcript.
    *   Copies the individual stream summary files (e.g., `recv_final_analysis_summary.txt`) for reference.

### Output Structure (per `call_id` in `output_call_dir`)

`--output_call_dir / <call_id> /`

*   `mixed_vocals.wav` (for pairs) or the original vocal stem (for singles, potentially renamed, e.g., `vocals.wav`).
*   `merged_transcript.json` (for pairs) or the original `transcript.json` (for singles, potentially renamed, e.g., `transcript.json`).
*   `combined_call_summary.txt`: LLM summary for the entire call.
*   `recv_final_analysis_summary.txt` (if RECV stream existed)
*   `trans_final_analysis_summary.txt` (if TRANS stream existed)
*   `RECV_S0/`, `RECV_S1/`, `TRANS_S0/`, ... : Prefixed soundbite directories containing `.wav` and `.txt` files per segment.

## 3. Overview of Key Modules

*   **`audio_preprocessor.py`:** Handles initial audio loading, resampling, vocal/instrumental separation, and loudness normalization.
*   **`clap_module.py`:** Detects specific sound events (e.g., phone ringing, DTMF tones) using CLAP models.
*   **`diarization_module.py`:** Performs speaker diarization to identify who spoke when, using models like PyAnnote.
*   **`transcription_module.py`:** Transcribes speech to text using models like Whisper, segmenting audio based on diarization and creating soundbites.
*   **`llm_module.py`:** Interacts with a Large Language Model (via LM Studio) to generate summaries or perform other text-based analysis.

## 4. Troubleshooting

*   **LM Studio Connection Issues:**
    *   Ensure LM Studio is running.
    *   Verify the correct model (e.g., `nidum-gemma-3-4b-it-uncensored`) is downloaded, loaded, and actively being served in LM Studio.
    *   Check that the LM Studio server is accessible on `http://localhost:1234` (or the configured endpoint).
    *   Review logs from `llm_module.py` and `workflow_executor.py` for specific error messages.
*   **`ffmpeg` Not Found:** If `call_processor.py` fails during audio mixing, ensure `ffmpeg` is installed and its executable is in your system's PATH.
*   **Module Import Errors:** Ensure your Python environment is set up correctly and that the `python_path` argument is used if necessary for `workflow_executor.py` to find project modules.

---
This README should provide a good starting point for understanding and using the audio processing workflow. 