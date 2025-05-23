# Active Context

## Current Focus
- **Major architectural shift in progress:** The project is now actively extracting and refactoring the core logic from each legacy subproject (ClapAnnotator, preFapMix, WhisperBite) into new, clean, unified modules under `unified_pipeline/`.
- The goal is a true rewrite, not just orchestration:
  - All core logic (separation, annotation, mixing, transcription) will be re-implemented in the new codebase.
  - All legacy CLI/subprocess logic will be removed.
  - The new program will be self-contained, with unified configuration, data models, error handling, and extensibility.
  - UI/CLI and preset management will operate on the new, unified logic.
  - The new codebase will not depend on the old project structure at runtime.
- This is a major architectural shift and the current top priority.

## Next Steps
- Inventory and extract the true core logic from each subproject.
- Refactor and re-implement this logic as new modules under `unified_pipeline/` (e.g., `separation.py`, `annotation.py`, `mixing.py`, `transcription.py`).
- Unify configuration, data models, and error handling in the new codebase.
- Rebuild the pipeline, UI, and CLI to use only the new, unified modules.
- Implement extensible preset management on the new architecture.
- Deprecate all legacy project boundaries and dependencies.
- Test and validate the new program as a standalone, modern, extensible tool.

## Current Focus
- Refactoring output structure and file handling for reliability and clarity.
- Centralizing and enforcing output folder structure:
  - Root output folder: <show_name>_<timestamp>
  - Each call: numbered subfolder (001/, 002/, ...)
  - All outputs for a call (stereo, left, right, soundbites, metadata) go in its numbered folder.
- Always use absolute paths for all file operations.
- Ensure all directories exist before writing (os.makedirs(..., exist_ok=True)).
- Always create/populate stereo/ directories as needed.
- Add robust error handling for missing files/dirs (log and skip, do not crash batch).
- Store all metadata and master transcript in the root output folder for easy access and future LLM naming.
- This fixes legacy/ad-hoc folder logic and prepares for future LLM-based automation and show-building.
- UI/UX Enhancements: Ensuring the Gradio UI exposes all relevant options for ClapAnnotator, preFapMix, and WhisperBite, including advanced and edge-case parameters.
- Inventorying all available options for each tool, identifying which are missing from the current UI.
- Planning the UI layout (tabs, advanced/expandable sections) to keep the interface user-friendly while providing full control.
- Preparing for implementation and comprehensive testing of the new UI.
- Implementing 'show-edit' mode: a workflow preset that controls the entire pipeline for show compilation.
- In 'show-edit' mode:
  - Normalize incoming (trans_out) and outgoing (recv_out) audio separately using FFmpeg.
  - Mix to stereo with 40% separation (left = 0.6*incoming + 0.4*outgoing, right = 0.6*outgoing + 0.4*incoming) using FFmpeg pan filter.
  - Concatenate all calls in chronological order, inserting tones.wav only between calls (not at the end of each call).
  - Output a single show file, named after the input folder, with all calls and tones in order.
  - Output comprehensive YAML/JSON metadata for the show, including all call lineage, transcripts, diarization, timings, and tones info.
  - Expose 'show-edit' mode as a preset/option in the UI/CLI.
- This design is forward-thinking: tones.wav as a marker enables future CLAP-based tools to split shows back into calls, making the workflow reversible and robust for archiving and reprocessing.
- In show-edit mode, tones are not appended to the end of individual calls; tones.wav is only inserted between calls in the final show file. This is critical for future splitting and robust archiving.

## Next Steps
- Complete implementation of show-edit mode in the pipeline.
- Update UI/CLI to expose show-edit mode as a preset/option.
- Ensure tones.wav is only inserted between calls, not at the end of each call, in show-edit mode.
- Update metadata to reflect show-edit mode, call/tone order, and all relevant info for future splitting.
- Test with real data and validate forward/backward compatibility with CLAP-based tools.
- Complete the options inventory for each subtool.
- Update the Gradio UI to expose all missing options, grouped by tool and with appropriate help text.
- Test the enhanced UI with real-world and edge-case scenarios.
- Gather user feedback for further refinement.
- Update documentation and memory bank to reflect the new UI/UX patterns. 