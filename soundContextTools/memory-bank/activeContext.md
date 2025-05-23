# activeContext.md

**Purpose:**
Tracks current work focus, recent changes, next steps, and active decisions/considerations.

## Current Focus

- All pipeline stages (ingestion, separation, CLAP, diarization, normalization, transcription, soundbite, remix, show, logging/manifest, LLM integration) are implemented and robust
- Privacy, manifest, and logging requirements are strictly enforced by the PipelineOrchestrator
- LLM and CLAP integration is complete, with master transcript and event merging
- Defensive code and error handling are in place for malformed or missing data
- System is fully auditable, extensible, and user-configurable
- **Finalization stage in progress:**
  - MP3 conversion for all soundbites (finalized/soundbites/), calls (finalized/calls/), and show audio (finalized/show/), all at 192kbps VBR
  - Embedding full metadata/lineage in ID3 tags (call index, speaker, timestamps, transcript, LLM titles, anonymized input file, etc.)
  - LLM-driven show title and description (family-friendly, comedic, fallback to default if invalid)
  - Show MP3 and .txt description named after LLM show title, fallback to completed-show.mp3 if needed
  - Manifest and logs updated with all finalized outputs and metadata
  - Two-stage LLM workflow: per-call titles/synopses, then show-level title/description
  - Show.txt includes the full show description and a list of call titles with their timestamps

## Recent Changes

- Completed implementation of all pipeline stages and orchestrator logic
- Integrated robust error handling and defensive filtering
- Finalized privacy-first manifest/logging and traceability patterns
- Added LLM and CLAP integration, master transcript, and extensible workflow config
- **Started finalization stage for MP3 outputs and show-level LLM metadata**

## Next Steps

- Complete and test finalization stage for MP3 outputs and metadata
- Ensure robust fallback logic for LLM output
- Update documentation and memory bank as the project evolves

## Active Decisions & Considerations

- All outputs and logs are strictly PII-free and fully auditable
- User preferences and workflow logic are extensible via CLI and workflow JSONs
- Defensive programming and robust error handling are required at every stage
- Show folder is always 'show/', but MP3 and .txt are named after LLM show title (fallback to completed-show.mp3 if needed)
- Show description is included in manifest, ID3 tags, and as a separate .txt file
- All soundbites are converted to MP3 with full metadata and included in finalized/soundbites/

## Current Focus: further hardening, extensibility, and user-driven improvements

- Current focus: further hardening, extensibility, and user-driven improvements 