# Active Context

**Current Work Focus:** Refining output generation and adding sound detection.

**Recent Changes:**
*   Refactored Demucs integration:
    *   Removed faulty `--version` check.
    *   Simplified path handling in `vocal_separation.py`.
    *   Corrected command flag from `--model` to `-n`.
    *   `separate_vocals_with_demucs` now returns paths for both `vocals` and `no_vocals` tracks.
*   Implemented Speaker Label Formatting:
    *   Added `format_speaker_label` helper.
    *   Updated slicing, transcription, and refinement functions to use `S0`, `S1` format for outputs.
*   Refactored Master Transcript Generation:
    *   Removed transcript writing from `transcribe_with_whisper` and `run_second_pass_diarization`.
    *   Implemented merging logic in `process_audio` to combine first-pass and second-pass results, attempting to replace originals with refined segments.
    *   Added sorting and final writing of `master_transcript.txt` within `process_audio`.
*   Added Optional Sound Detection:
    *   Added UI checkbox in `app.py`, enabled only when vocal separation is on.
    *   Passed option to `process_audio`.
    *   If enabled, `process_audio` transcribes the `no_vocals.wav` track.
    *   Uses regex to identify bracketed/parenthesized tags (e.g., `[ music ]`, `(noise)`) as sound events.
    *   Adds these events (labeled `SOUND`) to the final segment list before sorting and writing the master transcript.
*   Changed Auto Speaker Detection Default:
    *   Set default to `False` in UI (`app.py`) and backend (`process_audio`).
*   Updated `README.md` with new features and notes.
*   Added detailed logging for transcript merging to aid debugging potential duplication issues.

**Next Steps / Potential Issues:**
*   Verify the master transcript merging logic thoroughly to ensure duplicates are correctly eliminated when the second pass is used. (Initial user feedback suggests this might still need work). Logs were added to help diagnose.
*   Evaluate the effectiveness of sound detection using Whisper on the `no_vocals` track. Consider alternative methods (e.g., dedicated SED models like YAMNet) if Whisper is insufficient.
*   Test phone ringing detection specifically.
*   Package application for multi-platform deployment (e.g., using Pinokio). 