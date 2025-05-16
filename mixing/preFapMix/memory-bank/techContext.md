# Technical Context for PreFapMix

This document outlines the key technologies, libraries, and technical decisions for the Audio Processing Pipeline project.

## Core Technologies

*   **Python:** Version 3.10+ is the primary programming language.
*   **FFmpeg:** Essential for various audio manipulation tasks, including mixing audio streams, resampling, format conversion (e.g., WAV to MP3), and audio concatenation (for the show file). It must be installed and accessible in the system PATH.

## Key Python Libraries & Frameworks

*   **`pyannote.audio`:**
    *   **Purpose:** Used for speaker diarization.
    *   **Models:** Uses Hugging Face pretrained models (e.g., `pyannote/speaker-diarization-3.1`). Requires `HF_TOKEN`.
    *   **Output:** RTTM files.
*   **`openai-whisper`:**
    *   **Purpose:** Used for speech-to-text transcription.
*   **`pydub`:**
    *   **Purpose:** Used for audio slicing and potentially other manipulations.
*   **`torch`:**
    *   **Purpose:** Core deep learning framework, dependency for `pyannote.audio` and `openai-whisper`.
*   **`audio-separator` (Library/CLI):**
    *   **Purpose:** Used for separating audio into vocal and instrumental stems.
*   **`audiomentations` (or similar for normalization):**
    *   **Purpose:** Used for loudness normalization.
*   **`lmstudio-python` (Library for LM Studio):**
    *   **Purpose:** Python client to interact with a local LM Studio server for generating call names, synopses, and hashtag categories.
    *   **Usage:** `call_processor.py` uses this library to make three distinct calls to the LLM via `llm_module.generate_llm_summary`.
    *   **Import Strategy:** `import lmstudio as lms`, exceptions are attributes of `lms` object.

## Development Environment & Setup

*   **Operating System:** Primarily developed and tested on a system with WSL, aims for cross-platform compatibility.
*   **Python Version:** 3.10 or higher.
*   **Dependency Management:** `requirements.txt` for Python packages. System-level `ffmpeg`.
*   **LM Studio:** Local instance required with a compatible model loaded (e.g., Gemma-3 based, NousResearch/Hermes-2-Pro-Llama-3-8B).

## Workflow Configuration

*   Defined in JSON files (e.g., `default_audio_analysis_workflow.json`) for `workflow_executor.py`.
*   `call_processor.py` and the planned `show_compiler.py` are primarily CLI argument-driven for paths and major model choices.

## PII Safety Strategy

*   `pii_safe_file_prefix` used in `workflow_executor.py` stages.
*   `call_processor.py` uses `call_id` for its primary output structure and sanitized LLM-generated names for the "final output" directory structure, ensuring PII from original filenames is not propagated to user-facing outputs.

## Important Technical Notes & Constraints

*   **`ffmpeg` Installation:** Crucial and must be in PATH.
*   **Hugging Face Token:** `HF_TOKEN` required for some `pyannote.audio` models.
*   **Internet Access:** For downloading models on first run.
*   **GPU Availability:** Highly recommended for `openai-whisper` and `pyannote.audio`.
*   **Error Handling:** Robust error handling is implemented and ongoing.
*   **File Paths:** `Pathlib` is used for path manipulations to improve cross-platform compatibility.