## Output Structure Refactor (2024-05)
- Root output folder is <show_name>_<timestamp>.
- Each call is in a numbered subfolder (001/, 002/, ...), in chronological order.
- All outputs for a call (stereo, left, right, soundbites, metadata) go in its numbered folder.
- Absolute paths are used for all file operations.
- All directories are created as needed (os.makedirs(..., exist_ok=True)).
- stereo/ directories are always present and populated after mixing.
- Robust error handling/logging for missing files/dirs (log and skip, do not crash batch).
- All metadata and master transcript are stored in the root output folder for easy access and future LLM naming.
- This fixes legacy/ad-hoc folder logic and enables future LLM-based automation and show-building.

# Progress

## Architectural Decision (2024-05)
- Decided to move away from subprocess/CLI orchestration of ClapAnnotator, preFapMix, and WhisperBite.
- Will build a new, unified tool that imports and orchestrates the core logic of all subprojects directly as Python modules.
- This will address inefficiency, error propagation, and fragmented UX, and enable a unified data model, centralized error handling, and seamless LLM/metadata integration.

## Previous State
- Output structure refactor and UI option inventory completed for the legacy pipeline.
- Robust error handling, folder structure, and metadata output implemented in the subprocess-based system.

## Next Milestone
- Draft a migration plan and architecture for the unified tool.
- Begin extracting and refactoring core logic from each subproject for direct import.
- Design unified data model and pipeline flow.
- Plan new UI and metadata/LLM integration. 