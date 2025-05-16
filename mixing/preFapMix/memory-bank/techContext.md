# Technical Context for PreFapMix

This document outlines the key technologies, libraries, and technical decisions for the Audio Processing Pipeline project.

## Core Technologies

*   **Python:** Version 3.10+ is the primary programming language.
*   **FFmpeg:** Essential for various audio manipulation tasks, including mixing audio streams, resampling, and format conversion. It must be installed and accessible in the system PATH.

## Key Python Libraries & Frameworks

*   **`pyannote.audio`:**
    *   **Purpose:** Used for speaker diarization to determine speaker segments in an audio file.
    *   **Models:** Typically uses pretrained models like `pyannote/speaker-diarization-3.1` (or newer) from Hugging Face. Requires a Hugging Face token (`HF_TOKEN`) for access to gated models.
    *   **Output:** Generates speaker annotations, typically saved as an RTTM (Rich Transcription Time Marked) file.
    *   **RTTM File Handling (Critical Issue):**
        *   The current method `Annotation.read_rttm(diarization_file_path)` in `transcription_module.py` is causing an `AttributeError: type object 'Annotation' has no attribute 'read_rttm'`.
        *   This indicates an incorrect approach to parsing RTTM files with the current `pyannote.audio` version/setup.
        *   **Next Step:** Investigation of `whisperBite.py`'s RTTM handling mechanism is planned to find a working solution for parsing RTTM files into a `pyannote.core.Annotation` object or a compatible structure. Previous attempts to use `pyannote.database.util.RTTMParser` also resulted in import errors, suggesting it might be deprecated or relocated.
*   **`openai-whisper`:**
    *   **Purpose:** Used for speech-to-text transcription.
    *   **Models:** Supports various model sizes (e.g., `tiny.en`, `base`, `small`, `medium`).
    *   **Hardware:** Can leverage CUDA-enabled GPUs for significantly faster processing.
*   **`pydub`:**
    *   **Purpose:** Used for audio slicing (based on diarization timestamps) and potentially other audio manipulations like format conversion or simple effects if needed.
*   **`torch`:**
    *   **Purpose:** Core deep learning framework, a dependency for both `pyannote.audio` and `openai-whisper`.
*   **`audio-separator` (Library/CLI):**
    *   **Purpose:** Used for separating audio into vocal and instrumental stems. Can be invoked as a CLI tool or potentially as a Python library.
    *   **Models:** Utilizes various models like UVR_MDXNET_Main, mel_band_roformer_vocals_fv4_gabox.ckpt, etc.
*   **`audiomentations` (or similar for normalization):**
    *   **Purpose:** Used for loudness normalization of audio stems to target LUFS values. `audio_preprocessor.py` currently uses this.
*   **`requests` (Implied for LM Studio):**
    *   **Purpose:** To communicate with the LM Studio local server via HTTP POST requests for generating LLM summaries.
*   **`lmstudio-python` (Library for LM Studio):**
    *   **Purpose:** Provides a Python client to interact with a local LM Studio server.
    *   **Import Strategy (Confirmed):** 
        *   Use `import lmstudio as lms` for the main module.
        *   The `LMSClient` is accessed via `lms.LMSClient()` (if using the lower-level client) OR the convenience API `model = lms.llm("model-identifier")` is used, followed by `model.respond("prompt")`.
        *   Specific LM Studio exceptions (e.g., `lms.LMStudioModelNotFoundError`, `lms.LMStudioServerError`, `lms.LMStudioClientError`, `lms.LMStudioError`) are attributes of the main `lms` module object and should be caught directly (e.g., `except lms.LMStudioModelNotFoundError:`).
    *   **Submodules:** The `lmstudio.exceptions` or `lmstudio.errors` submodules were found *not* to exist or be importable in the current environment, despite some documentation suggesting them. Direct attribute access on the `lms` object is the confirmed method for exceptions.

## Development Environment & Setup

*   **Operating System:** Primarily developed and tested on a system with WSL (Windows Subsystem for Linux) for some components, but aims for cross-platform compatibility where Python and its libraries are supported.
*   **Python Version:** 3.10 or higher.
*   **Dependency Management:** A `requirements.txt` file should be maintained for Python package dependencies.
*   **LM Studio:**
    *   A local instance of LM Studio must be running with a compatible model loaded (e.g., `NousResearch/Hermes-2-Pro-Llama-3-8B`, `nidum-gemma-3-4b-it-uncensored`).
    *   The LM Studio server endpoint (typically `http://localhost:1234/v1/chat/completions`) is targeted by `llm_module.py`.

## Workflow Configuration

*   Defined in JSON files (e.g., `default_audio_analysis_workflow.json`).
*   Specifies stages, module function calls, input/output mappings between stages using a simple templating mechanism (e.g., `{stages.previous_stage_name[output_key]}`).

## PII Safety Strategy

*   A `pii_safe_file_prefix` (e.g., `call_YYYYMMDD-HHMMSS` or `file_YYYYMMDD-HHMMSS`) is generated by `workflow_executor.py` based on the input filename's timestamp or a generic timestamp if not available.
*   This prefix is passed to each processing module.
*   Modules use this prefix for naming all their output files and any significant temporary files to avoid PII leakage from original filenames.

## Important Technical Notes & Constraints

*   **`ffmpeg` Installation:** Must be installed and available in the system's PATH.
*   **Hugging Face Token:** A `HF_TOKEN` environment variable or other authentication method is required for downloading some `pyannote.audio` models.
*   **Internet Access:** Required for downloading models (Whisper, Pyannote, audio-separator) on first run or when models are not cached.
*   **GPU Availability:** Highly recommended for `openai-whisper` and `pyannote.audio` for acceptable performance. CPU-only execution will be significantly slower.
*   **Error Handling:** Robust error handling within modules and the workflow executor is crucial for identifying and managing issues during processing.
*   **File Paths:** Care must be taken with file paths, especially when operating between WSL and Windows environments if paths are passed directly. Using `Pathlib` and ensuring paths are correctly resolved is important.