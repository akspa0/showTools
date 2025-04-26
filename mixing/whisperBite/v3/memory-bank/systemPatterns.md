# System Patterns

**Overall Architecture:**
*   Core processing pipeline (`whisperBite.py`).
*   Gradio web UI (`app.py`).
*   Conditional Two-Pass Pipeline.

**Key Components/Functions:**
*   `app.py`:
    *   `build_interface()`: Defines UI, accepting audio/video file types.
    *   `run_pipeline()`: Handles UI logic, calls `whisperBite.process_audio`.
*   `whisperBite.py`:
    *   `process_audio()`: Main orchestrator.
        *   Determines actual input file (handles directory input by finding newest audio/video).
        *   **Checks if input is video using `VIDEO_EXTENSIONS`.**
        *   **If video, calls `extract_audio_from_video` to create temporary WAV.**
        *   Calls normalization, optional vocal separation, diarization, slicing.
        *   Calls `transcribe_with_whisper`.
        *   Conditionally calls `run_second_pass_diarization`.
        *   Calls `zip_results`.
        *   **Cleans up temporary extracted WAV in `finally` block.**
    *   `extract_audio_from_video()`: Uses `ffmpeg` via `subprocess` to extract audio.
    *   `transcribe_with_whisper()`: Handles transcription, conditional word extraction.
    *   `run_second_pass_diarization()`: Implements refinement pass. Iterates first-pass segments, runs diarization, checks for overlap. If found, re-slices audio, transcribes sub-segments, **saves refined audio/text files using content-based naming (similar to first pass)**, and aggregates `2nd_pass/master_transcript.txt`.
    *   Other helpers (`normalize_audio`, `slice_audio_by_speaker`, etc.).
*   `utils.py`, `vocal_separation.py`: Helpers.

**Processing Flow (Video Input):**
UI Input (Video) -> `app.py:run_pipeline` -> `whisperBite.process_audio` -> Detect Video -> `extract_audio_from_video` (ffmpeg -> temp WAV) -> Normalize (temp WAV) -> ... [Rest of pipeline using normalized temp WAV] ... -> **`zip_results` (Recursively zips entire output dir excluding intermediates)** -> Cleanup (Delete temp WAV).

**Data Management:**
*   Timestamped output directory per input file.
*   Handles audio/video detection.
*   Temporary WAV file created inside output dir for video inputs, cleaned up afterwards.
*   Conditional `_words` dirs and `word_timings.json`.
*   Conditional `2nd_pass` subdirectory containing refined audio and transcripts.
*   `app.py` handles zip download via Gradio temp mechanism.
*   **`zip_results` creates archive containing the full output structure (including `2nd_pass`), excluding intermediate `normalized` and `downloads` folders.**

**Error Handling:**
*   `extract_audio_from_video` checks for `ffmpeg` errors.
*   `process_audio` checks extraction status, uses `finally` for cleanup.
*   `zip_results` includes basic metadata generation with error handling. 