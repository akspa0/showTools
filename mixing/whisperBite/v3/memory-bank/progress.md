# Progress

**What Works:**
*   Main processing pipeline (normalization, optional vocal separation, diarization, slicing, transcription).
*   Gradio Web UI (`app.py`).
*   Command-line interface (`whisperBite.py`).
*   Handling of file (audio/video), directory (newest file), and URL inputs.
*   Automatic audio extraction from video files using `ffmpeg`.
*   Structured output generation (transcripts, segments, optional word data).
*   Basic error handling and logging.
*   Manual speaker count setting (now default).
*   Optional automatic speaker count detection (disabled by default).
*   Optional second-pass diarization refinement logic implemented.
*   Optional word audio extraction (disabled by default).
*   Optional sound detection using Whisper on non-vocal track (requires vocal separation).
*   Result zipping.
*   Correct Demucs execution (using `-n` flag).
*   Speaker labels formatted as `S0`, `S1`, etc.
*   Master transcript generation attempts merging first/second pass and sound events.

**What Needs Work/Verification:**
*   **Master transcript merging:** Ensure second pass correctly replaces, not duplicates, original segments (User noted potential remaining issue). Requires reviewing debug logs added.
*   **Sound Detection Accuracy:** Evaluate Whisper's effectiveness on `no_vocals` track, especially for specific sounds like phone ringing. Consider dedicated SED models if needed.
*   Robustness of `detect_optimal_speakers` heuristic (if enabled).
*   Effectiveness and performance of the `run_second_pass_diarization` refinement (beyond merging bug).
*   Handling of edge cases (corrupt files, ffmpeg failures, silence, etc.).
*   Performance optimization for long files.
*   Configurability of hardcoded parameters (fades, merge gaps, word padding, second pass thresholds, sound detection regex).
*   Dependency checking (ffmpeg/demucs availability).
*   Cleanup of temporary directories created by `app.py` for Gradio downloads.
*   Packaging for deployment (e.g., Pinokio).

**Current Status:** Core features refined, including Demucs integration, speaker formatting, defaults, and sound detection (basic). Master transcript merging needs verification.

**Known Issues:**
*   Potential duplication in master transcript when second pass is used (needs log verification).
*   Sound detection accuracy is dependent on Whisper model and may not be reliable for specific sounds like ringing.
*   Folder input only processes the newest compatible file.
*   Potential filename parsing errors in `transcribe_with_whisper` (fallback exists).
*   Demucs check is basic. 