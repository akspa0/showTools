# Project Brief: mhrpTools (Multi-Stage Hybrid Audio Processing Tools)

## Overview
mhrpTools is a unified audio processing toolkit that integrates the best features of ClapAnnotator, preFapMix, and WhisperBite into a single, user-friendly pipeline. It provides both a CLI and Gradio web interface for batch and single-file audio processing, automating complex workflows that previously required manual intervention and coordination between multiple tools.

## Core Goals
- Eliminate manual audio editing and file management by automating the full pipeline from audio separation to mixing and transcription.
- Leverage ClapAnnotator for state-of-the-art audio separation as the first step for all input files.
- Use preFapMix logic for stereo mixing of files with `recv_out`/`trans_out` naming conventions.
- Use WhisperBite for transcription and soundbite extraction, omitting redundant vocal separation (demucs) since ClapAnnotator provides superior results.
- Provide a modular, extensible architecture that does not alter the original subprojects, but orchestrates them externally.
- Support both command-line and web-based (Gradio) workflows for maximum flexibility.
- Ensure the Gradio UI exposes all relevant options for ClapAnnotator, preFapMix, and WhisperBite, including advanced and edge-case parameters, to provide full user control and configurability.
- Provide a 'show-edit' mode that compiles all calls into a single show file, inserting tones.wav only between calls, not at the end of each call.
- Ensure forward/backward compatibility: tones.wav is used as a marker so future CLAP-based tools can split shows back into calls, enabling robust archiving and reprocessing.

## Relationship to Subprojects
- **ClapAnnotator:** Used for all initial audio separation and annotation.
- **preFapMix:** Used for stereo mixing logic, specifically for files with `recv_out`/`trans_out` prefixes.
- **WhisperBite:** Used for transcription, diarization, and soundbite extraction, with demucs step omitted.

## Source of Truth
This document defines the core requirements, goals, and scope for the mhrpTools integration project at the root of the repository. 