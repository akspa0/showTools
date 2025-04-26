# Progress

**What Works:**
*   Main processing pipeline (normalization, optional vocal separation, diarization, slicing, transcription).
*   Gradio Web UI (`app.py`).
*   Command-line interface (`whisperBite.py`).
*   Handling of file (audio/video), directory (newest file), and URL inputs.
*   **Automatic audio extraction from video files using ffmpeg.**
*   Structured output generation (transcripts, segments, optional word data).
*   Basic error handling and logging.
*   Automatic speaker count detection.
*   Optional second-pass diarization refinement logic implemented.
*   Option to enable/disable word audio extraction implemented.
*   Result zipping.

**What Needs Work/Verification:**
*   Thorough testing of video input handling with various formats/codecs.
*   Robustness of `detect_optimal_speakers` heuristic.
*   Effectiveness and performance of the `run_second_pass_diarization` refinement, especially the fixed `num_speakers=2` assumption during the second pass analysis.
*   Handling of edge cases (corrupt files, ffmpeg failures, silence, etc.).
*   Performance optimization for long files.
*   Configurability of hardcoded parameters (fades, merge gaps, word padding, second pass thresholds like min duration and speaker count).
*   Dependency checking (ffmpeg/demucs availability).
*   Cleanup of temporary directories created by `app.py` for Gradio downloads.

**Current Status:** Core features including video input, second pass, and word toggle are implemented. Requires testing, especially for video formats and second pass accuracy.

**Known Issues:**
*   Folder input only processes the newest compatible file.
*   Potential filename parsing errors in `transcribe_with_whisper` (fallback exists).
*   Demucs check is basic. 