# Product Context: mhrpTools

## Problem
- Manual audio editing, mixing, and transcription across multiple tools is time-consuming, error-prone, and inefficient.
- Users must coordinate between different programs (ClapAnnotator, preFapMix, WhisperBite), manage intermediate files, and manually trigger each step.
- Redundant processing (e.g., multiple vocal separations) wastes resources and can degrade quality.

## Solution
- mhrpTools automates the entire workflow, orchestrating the best features of each subproject in a single pipeline.
- Provides both CLI and Gradio UI for batch and single-file processing.
- Eliminates redundant steps (e.g., disables demucs in WhisperBite, always uses ClapAnnotator for separation).
- Handles all file management, naming, and output structuring automatically.

## User Experience Goals
- **Simplicity:** One command or UI action to process audio from start to finish.
- **Configurability:** Users can select processing modes (mixing, soundbites, or both), batch or single-file, and output locations.
- **Transparency:** Clear logs, progress indicators, and error reporting at each stage.
- **Extensibility:** Easy to add new processing steps or swap out subcomponents as technology evolves.
- **Reliability:** Robust error handling and recovery from failures at any stage.
- **Advanced Control:** All relevant options for each subtool (ClapAnnotator, preFapMix, WhisperBite) are exposed in the UI, including advanced and edge-case parameters, so power users can fully customize the pipeline.
- **Show-Edit Mode:** Enables efficient show compilation, robust archiving, and future reverse engineering using tones.wav and CLAP annotations. Designed to replace hours of manual labor and support future tool development. 