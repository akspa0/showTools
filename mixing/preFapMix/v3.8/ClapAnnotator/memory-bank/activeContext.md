# Active Context

## Current Focus
- **Integration Project: mhrpTools**
  - Planning and design for a new tool (`mhrpTools`) that unifies the best of ClapAnnotator, preFapMix, and WhisperBite.
  - ClapAnnotator will serve as the first step for all audio pre-processing (separation/splitting) in the new pipeline.
  - The integration will not alter existing ClapAnnotator code, but will call it as a module or subprocess.
  - The new tool will support both CLI and Gradio UI workflows.

## Integration Context
- **Pipeline:**
  1. All audio is first processed by ClapAnnotator for separation.
  2. If output files are named with `recv_out`/`trans_out`, they are passed to preFapMix logic for stereo mixing.
  3. Otherwise, audio is passed to WhisperBite for transcription and soundbite extraction (with demucs step omitted).
- **No changes to existing ClapAnnotator logic or UI.**
- **Goal:** Eliminate manual audio editing by automating the best features of all three tools.

## Next Steps
- Finalize the design and interface for mhrpTools (CLI and Gradio UI).
- Implement integration logic to call ClapAnnotator as a subprocess/module.
- Ensure output compatibility for downstream mixing and transcription steps.
- Document the integration and update system patterns as needed.

## Decisions & Considerations
- Existing ClapAnnotator code remains unchanged; all integration is external.
- Output structure and naming conventions must be compatible with downstream tools.
- The new tool will provide a unified, user-friendly workflow for batch and single-file processing.

## Gradio Application (Phase 1) Implemented
- The Gradio application (`gradio_app/app.py`) for single audio file processing is now functionally complete. This includes:
  - UI elements for file upload, model selection (separator), CLAP prompt input (manual and preset-based), and parameter adjustments (confidence, chunk duration, audio preservation).
  - Backend logic orchestrating audio separation, resampling, CLAP annotation, result aggregation, JSON output, and temporary file cleanup.
  - CLAP prompt preset loading and saving functionality.
  - Progress updates within the UI.

## Comprehensive Testing (Gradio App)
- Test with various audio file types and lengths.
- Verify functionality of all UI controls (separator models, presets, prompt input, confidence, chunk duration, audio preservation).
- Test edge cases (e.g., very short audio, audio with no detectable events, invalid inputs).
- Evaluate error handling and UI feedback.

## UI/UX Refinement
- Based on testing, identify and implement any necessary improvements to the Gradio application's user interface and experience.

## Documentation
- Update `README.md` with detailed setup instructions and a comprehensive guide on using the Gradio application for single-file analysis.

## Phase 2 Planning
- Begin outlining requirements and design considerations for Phase 2, which includes batch processing capabilities.

## Decisions & Considerations
- Phase 1 (single-file processing via Gradio) is considered functionally complete.
- Batch processing remains a Phase 2 goal.
- Default CLAP chunk duration of 3 seconds and confidence threshold of 0.55 remain the standard.
- Audio file preservation is enabled by default in the Gradio UI. 