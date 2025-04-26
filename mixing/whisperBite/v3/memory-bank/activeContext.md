# Active Context

**Current Work Focus:** Adding video input support and ensuring robustness.

**Recent Changes:**
*   Added `enable_second_pass` option to trigger re-diarization and transcription of initial segments.
*   Implemented `run_second_pass_diarization` function in `whisperBite.py`.
*   Added `enable_word_extraction` option (default off) to control word snippet generation.
*   Modified `transcribe_with_whisper` to handle `segment_info` dicts and the word extraction toggle.
*   Modified `slice_audio_by_speaker` to return `segment_info` dict.
*   Updated `app.py` UI with new checkboxes and passed options to backend.
*   Fixed `TypeError` in `app.py` related to duplicate `hf_token` input.
*   Fixed `NameError` in `whisperBite.py` call to `run_second_pass_diarization`.
*   Fixed `TypeError` in `run_second_pass_diarization` by ensuring it receives the correct `segment_info` data structure.
*   Restored folder input option in UI and CLI, clarifying it currently processes only the newest file.
*   **Added support for video file inputs:**
    *   Updated `app.py` `gr.File` component to accept video types.
    *   Added `VIDEO_EXTENSIONS` and `extract_audio_from_video` function in `whisperBite.py`.
    *   Modified `process_audio` to detect video files, call `ffmpeg` for audio extraction to a temporary WAV file, and use that for processing.
    *   Added `finally` block to `process_audio` to clean up the temporary WAV file.
*   Updated relevant Memory Bank files (`projectbrief.md`, `productContext.md`, `systemPatterns.md`, `techContext.md`, `progress.md`).

**Next Steps:**
*   Test video file input functionality.
*   Test second-pass refinement thoroughly.
*   Address items in `progress.md` ("Needs Work/Verification"). 