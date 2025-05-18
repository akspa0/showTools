# Project Brief: CLAP Annotator Tool

## Overview
A highly configurable tool to annotate audio files using CLAP after separating vocal and instrumental tracks with user-selected `python-audio-separator` models.

## Phase 1 Goals
- Provide an intuitive Gradio-based interface for easy interaction with **single audio files**.
- Allow user selection of audio separation models and relevant parameters.
- Enable users to define custom CLAP text prompts and manage them via a preset system.
- Offer user control over CLAP processing parameters, including the confidence threshold.
- Generate structured JSON outputs with timestamped CLAP detections in standardized, timestamped output folders.

## Phase 2 / Future Goals
- Implement a job queue system for batch processing of folders containing multiple audio files.
- Provide UI controls to view, pause, and cancel jobs within the queue.
- Potentially explore parallel processing for jobs.

## Requirements (Phase 1)
- Python 3.10+
- Core libraries: `python-audio-separator`, Hugging Face `transformers`, `gradio`, `ffmpeg-python`.
- System dependency: `ffmpeg` installed and in PATH.
- Configuration via `config/settings.py` and `.env`.
- A preset management system using the `_presets/clap_prompts/` directory.
- Standardized output directory structure under `ClapAnnotator_Output/`. 