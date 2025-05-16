# Active Context for PreFapMix

## Current Overall Goal

Finalize all core audio processing and output generation features, including MP3 compression for individual calls, and implement the `show_compiler.py` script for creating aggregated "show files" from processed paired calls. Ensure robust PII safety and deliver user-friendly, organized outputs as defined in `projectbrief.md` and `productContext.md`.

## Current Focus & Immediate Tasks

1.  **Implement MP3 Compression:** 
    *   Add a utility function (likely in `call_processor.py` or a shared `utils.py`) to compress WAV files to MP3 using `ffmpeg`.
    *   Integrate this into `call_processor.py` to convert the primary audio file in the "final output" directory (e.g., `final_processed_calls/[Sanitized_Call_Name]/[Sanitized_Call_Name].wav`) to MP3 format.
    *   Decide on and implement MP3 compression for soundbites within the "final output" structure (if desired).

2.  **Design and Implement `show_compiler.py`:**
    *   **Inputs:** The base "final output" directory (e.g., `final_processed_calls/`).
    *   **Outputs (in a `show_output` directory):** 
        *   `show_audio.mp3` (concatenated MP3s of paired calls).
        *   `show_timestamps.txt` (start times and names of calls in the show).
        *   `show_transcript.txt` (concatenated plain text transcripts).
    *   **Core Logic:** Filter for "pair" type calls using `call_type.txt`, sort chronologically, use `ffmpeg` for audio concatenation, and generate metadata files.

## Recent Changes & Discoveries

*   **Advanced LLM Integration in `call_processor.py`:** 
    *   Successfully implemented three distinct LLM calls (via `llm_module.generate_llm_summary`) for generating call names, synopses, and hashtag categories using approved, concise system prompts.
    *   Outputs are saved to respective `.txt` files within the primary processed call directory (e.g., `processed_calls/[call_id]/`).
*   **"Final Output" Structure Initiated in `call_processor.py`:**
    *   Added `--final_output_dir` argument.
    *   Creates a flatter, user-friendly directory for each call (e.g., `final_processed_calls/[Sanitized_Call_Name]/`).
    *   This directory includes:
        *   The primary call audio as `.wav` (awaiting MP3 compression).
        *   A plain text `transcript.txt` (converted from JSON).
        *   Copied `synopsis.txt` and `hashtags.txt`.
        *   A `call_type.txt` file (e.g., containing "pair" or "single_recv") to aid `show_compiler.py`.
*   **Previous Blockers Resolved:** Terminal echo issue and initial LLM import/usage issues are considered resolved.

## Next Steps (Sequential)

1.  Implement MP3 compression functionality in `call_processor.py` for the main audio file in the "final output" directories.
2.  Decide and implement if soundbites in the "final output" directories also require MP3 compression.
3.  Develop the `show_compiler.py` script with audio concatenation, timestamp generation, and transcript aggregation features.
4.  Conduct thorough end-to-end testing of the complete workflow from `workflow_executor.py` through `call_processor.py` to `show_compiler.py`.
5.  Update all memory bank files to reflect these final additions and the overall completed state of this feature set.

## Active Decisions & Considerations

*   **Instrumental Stems for Show Audio:** Needs clarification – should `show_audio.mp3` include instrumental stems, or is the current vocal-only mix from `call_processor.py` sufficient? (Current assumption: vocal-only based on existing `mixed_vocals.wav`).
*   **MP3 Compression for Soundbites:** Decision pending on whether to compress the individual soundbite `.wav` files in the "final output" directory structure.

## Open Questions

*   Should soundbites in the `final_output_dir/[call_name]/soundbites/` also be MP3 compressed? (Assuming soundbites are planned to be copied there eventually).
*   Does the `show_audio.mp3` need to include instrumental stems, or is the current vocal-only mix from `call_processor.py` (i.e., `mixed_vocals.wav`) the correct source?