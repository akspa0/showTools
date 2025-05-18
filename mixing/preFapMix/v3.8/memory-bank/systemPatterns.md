# System Patterns: Unified mhrpTools Pipeline

## Overall Architecture
- Unified Python package orchestrates all stages: separation (ClapAnnotator), annotation (CLAP), mixing (preFapMix), transcription (WhisperBite), and show compilation.
- Direct function/class imports from each subproject; no subprocess/CLI orchestration.
- Unified data model: AudioFile, Call, Show (see data_model.py).
- Centralized error handling, logging, and reporting.
- Modular pipeline: each stage can fail independently; errors are reported in the UI/CLI.
- Output structure: Root output folder <show_name>_<timestamp>, each call in a numbered subfolder, all outputs/metadata centralized.
- Batch and single-file processing supported.

## Integration Pattern
- Each subproject's core logic is imported as a Python module.
- Pipeline stages:
  1. Separation (ClapAnnotator): AudioSeparatorWrapper
  2. Annotation (CLAP): CLAPAnnotatorWrapper
  3. Mixing (preFapMix): process_audio_files and helpers
  4. Transcription (WhisperBite): process_audio and helpers (demucs step omitted)
  5. Show-edit mode: Concatenate calls, insert tones.wav only between calls, not at the end
  6. Metadata/LLM: YAML/JSON output, call naming, image prompt generation
- All relevant options for each stage are exposed in both CLI and Gradio UI, including advanced/edge-case parameters.

## Error Handling
- Input validation at each stage (file existence, format, size, etc.).
- try/except blocks around all major processing steps; errors are logged and surfaced to the user.
- Robust handling of missing/corrupt files, model failures, and resource issues.
- UI/CLI disables interactive elements during processing and provides clear feedback.
- Temporary files are cleaned up after processing; final outputs are preserved.

## UI/UX Patterns
- Gradio UI and CLI both supported.
- UI groups options by tool (tabs or collapsible panels), with help text/tooltips for each option.
- Defaults are sensible but all can be overridden.
- Show-edit mode is a workflow preset, exposed in both UI and CLI.
- Progress indicators and error reporting at each stage.
- Output structure and metadata are designed for robust archiving and future tool integration.

## Show-Edit Mode
- In show-edit mode, tones.wav is inserted only between calls during show compilation, not at the end of each call.
- Enables future CLAP-based tools to split shows back into calls using tones as markers.
- Comprehensive YAML/JSON metadata is output for the show, including call lineage, transcripts, diarization, timings, and tones info.
- Designed for robust archiving, reprocessing, and future reverse engineering. 