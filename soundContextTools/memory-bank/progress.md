# progress.md

**Purpose:**
Tracks what works, what's left to build, current status, and known issues.

## What Works

- Project documentation and memory bank initialized
- Project requirements and technical stack defined
- Initial ingestion and manifest logic drafted
- Pipeline plan finalized: output structure, manifest privacy, diarization, soundbites, transcription, CLAP annotation integration

## What's Left to Build

- Pipeline orchestrator for job management and progress tracking
- Job and orchestrator abstractions (Job, PipelineOrchestrator)
- Refactor file input logic to integrate with orchestrator
- Stage-based subfolder logic for each pipeline step
- Manifest privacy for call tuples (no original filenames)
- Diarization, soundbite extraction, and transcript rebuilding with CLAP annotation integration
- Audio separation (vocals/instruments)
- Loudness normalization
- Speaker diarization (planned)

## Current Status

Ready to design and implement the pipeline orchestrator for robust, extensible, and debuggable pipeline management.

## Known Issues

- None yet (project just started) 