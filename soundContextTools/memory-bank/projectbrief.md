# projectbrief.md

**Purpose:**
Foundation document that shapes all other files. Defines core requirements and goals. Source of truth for project scope.

## Project Name

Audio Context Tool

## Overview

A tool for processing various types of audio (especially phone call recordings) through AI models to provide context, diarization, normalization, remixing, and transcription into soundbites. Focuses on automating the editing of raw phone call conversations into contextual datasets, with robust file tracking and PII removal.

## Core Requirements

- Handle multiple audio input types, especially phone call tuples (out-, trans_out-, recv_out-)
- Remove PII from filenames and re-index tuples
- Track files through all processing steps
- Audio separation (vocals/instruments)
- CLAP-based context annotation
- Loudness normalization (audiomentations)
- Speaker diarization (pyannote, planned)
- Modular, extensible pipeline

## Goals

- Automate painful manual audio editing workflows
- Enable dataset creation from phone call audio
- Ensure privacy by removing PII
- Maintain traceability of files throughout processing

## Scope

- In scope: File renaming, PII removal, tuple tracking, audio separation, CLAP annotation, normalization, diarization prep
- Out of scope: UI, deployment, advanced analytics (for now) 