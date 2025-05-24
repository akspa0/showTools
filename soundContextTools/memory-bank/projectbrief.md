# projectbrief.md

**Purpose:**
Foundation document that shapes all other files. Defines core requirements and goals. Source of truth for project scope.

## Project Name

Audio Context Tool

## Overview

A privacy-focused, modular pipeline for processing phone call audio and other recordings. The pipeline automates ingestion, PII removal, file tracking, audio separation, CLAP annotation, loudness normalization, speaker diarization, transcription, soundbite extraction, LLM integration, remixing, and show creation. All steps are orchestrated by a PipelineOrchestrator that enforces strict privacy, traceability, and manifest/logging requirements. The system is fully auditable, extensible, and robust to errors and malformed data.

## Core Requirements

- Handle multiple audio input types, especially phone call tuples (out-, trans_out-, recv_out-)
- Remove PII from filenames and re-index tuples using a unique, zero-padded chronological index
- Track files and metadata through all processing steps
- Audio separation (vocals/instruments)
- CLAP-based context annotation (with confidence threshold)
- Loudness normalization (to -14.0 LUFS)
- Speaker diarization (pyannote)
- Modular, extensible, and batch-oriented pipeline
- Robust error handling and defensive filtering
- LLM integration for downstream tasks
- All logging and manifest writing is strictly PII-free and only occurs after anonymization

## Goals

- Automate manual audio editing workflows for privacy and scale
- Enable dataset creation from phone call audio with full traceability
- Ensure privacy by removing PII and enforcing anonymized logging/manifesting
- Maintain traceability and auditability of files and processing lineage
- Support extensibility and user configuration via CLI and workflow JSONs

## Scope

- In scope: All pipeline stages, privacy enforcement, manifest/logging, LLM/CLAP integration, error handling, extensibility
- Out of scope: UI, deployment, advanced analytics (for now)

## 2024-06-XX: Project Milestone
- All core requirements and goals are satisfied.
- Pipeline is stable, robust, and production-ready. 