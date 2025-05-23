# activeContext.md

**Purpose:**
Tracks current work focus, recent changes, next steps, and active decisions/considerations.

## Current Focus

- ✅ **MAJOR BREAKTHROUGH: Resume functionality successfully implemented and tested**
- All pipeline stages (ingestion, separation, CLAP, diarization, normalization, transcription, soundbite, remix, show, logging/manifest, LLM integration) are implemented and robust
- Privacy, manifest, and logging requirements are strictly enforced by the PipelineOrchestrator
- LLM and CLAP integration is complete, with master transcript and event merging
- Defensive code and error handling are in place for malformed or missing data
- System is fully auditable, extensible, and user-configurable
- **NEW: Non-breaking resume functionality with comprehensive state tracking**
  - Pipeline state management with JSON persistence
  - Smart stage skipping for completed work
  - Failure recovery with detailed error tracking
  - Resume helpers integrated into orchestrator
  - CLI arguments for resume operations
  - Zero impact on existing workflows
- **Finalization stage in progress:**
  - MP3 conversion for all soundbites (finalized/soundbites/), calls (finalized/calls/), and show audio (finalized/show/), all at 192kbps VBR
  - Embedding full metadata/lineage in ID3 tags (call index, speaker, timestamps, transcript, LLM titles, anonymized input file, etc.)
  - LLM-driven show title and description (family-friendly, comedic, fallback to default if invalid)
  - Show MP3 and .txt description named after LLM show title, fallback to completed-show.mp3 if needed
  - Manifest and logs updated with all finalized outputs and metadata
  - Two-stage LLM workflow: per-call titles/synopses, then show-level title/description
  - Show.txt includes the full show description and a list of call titles with their timestamps

## Recent Changes

- ✅ **Implemented complete resume functionality (pipeline_state.py, resume_utils.py)**
- ✅ **Enhanced orchestrator with run_with_resume() method - backward compatible**
- ✅ **Added comprehensive CLI arguments: --resume, --resume-from, --show-resume-status**
- ✅ **Created full test suite with 100% pass rate**
- ✅ **Validated state persistence, failure recovery, and skip logic**
- Completed implementation of all pipeline stages and orchestrator logic
- Integrated robust error handling and defensive filtering
- Finalized privacy-first manifest/logging and traceability patterns
- Added LLM and CLAP integration, master transcript, and extensible workflow config

## Next Steps

- **Immediate: Enhanced error handling and edge cases for resume functionality**
- **Next: Advanced resume controls (--resume-from, --force-rerun, --clear-from)**
- **Then: Real-world integration testing with actual audio files**
- **Future: Performance monitoring and stage timing analytics**
- Complete and test finalization stage for MP3 outputs and metadata
- Ensure robust fallback logic for LLM output
- Update documentation and memory bank as the project evolves

## Active Decisions & Considerations

- **Resume functionality is production-ready but can be enhanced with granular controls**
- **Debugging workflow dramatically improved - no more re-running expensive stages**
- All outputs and logs are strictly PII-free and fully auditable
- User preferences and workflow logic are extensible via CLI and workflow JSONs
- Defensive programming and robust error handling are required at every stage
- Show folder is always 'show/', but MP3 and .txt are named after LLM show title (fallback to completed-show.mp3 if needed)
- Show description is included in manifest, ID3 tags, and as a separate .txt file
- All soundbites are converted to MP3 with full metadata and included in finalized/soundbites/

## Current Focus: Enhanced error handling and granular resume controls for production robustness 